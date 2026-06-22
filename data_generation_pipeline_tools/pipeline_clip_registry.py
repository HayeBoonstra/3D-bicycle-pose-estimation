"""Inventory raw Blender clips and MuJoCo trajectories for incremental dataset builds."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAJECTORY_ID_RE = re.compile(r"(mujoco_[a-z]+_\d{5})")

RAW_MANIFEST_FIELDS = [
    "clip_id",
    "scene_id",
    "blend",
    "trajectory_id",
    "trajectory_csv",
    "trajectory_sha256",
    "camera_seed",
    "n_frames",
    "status",
]

TRAJECTORY_MANIFEST_FIELDS = [
    "trajectory_id",
    "trajectory_csv",
    "pattern",
    "seed",
    "frames",
    "fps",
    "sha256",
]


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    clip_dir: Path
    scene_id: str
    camera_seed: str
    trajectory_csv: str
    trajectory_sha256: str
    n_frames: int
    has_keypoints_3d: bool
    has_dynamics_gt: bool

    @property
    def trajectory_id(self) -> str:
        match = TRAJECTORY_ID_RE.search(self.clip_id)
        return match.group(1) if match else ""


def trajectory_file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_raw_clip(clip_dir: Path) -> ClipRecord | None:
    clip_dir = Path(clip_dir)
    if not clip_dir.is_dir() or not clip_dir.name.startswith("clip_"):
        return None
    render_config_path = clip_dir / "render_config.json"
    k3d_path = clip_dir / "keypoints_3d.jsonl"
    if not render_config_path.is_file() or not k3d_path.is_file():
        return None

    render_config = _load_json(render_config_path)
    trajectory_csv = str(render_config.get("trajectory_csv", "") or "")
    trajectory_sha256 = str(render_config.get("trajectory_sha256", "") or "")
    if trajectory_csv and not trajectory_sha256:
        csv_path = Path(trajectory_csv)
        if csv_path.is_file():
            trajectory_sha256 = trajectory_file_sha256(csv_path)

    n_frames = int(render_config.get("frame_end", 0)) - int(render_config.get("frame_start", 1)) + 1
    if n_frames <= 0:
        with k3d_path.open(encoding="utf-8") as handle:
            n_frames = sum(1 for line in handle if line.strip())

    has_dynamics = False
    with k3d_path.open(encoding="utf-8") as handle:
        first = handle.readline().strip()
        if first:
            has_dynamics = isinstance(json.loads(first).get("dynamics_gt"), dict)

    return ClipRecord(
        clip_id=clip_dir.name,
        clip_dir=clip_dir,
        scene_id=str(render_config.get("scene_id", "")),
        camera_seed=str(render_config.get("camera_seed", "")),
        trajectory_csv=trajectory_csv,
        trajectory_sha256=trajectory_sha256,
        n_frames=n_frames,
        has_keypoints_3d=True,
        has_dynamics_gt=has_dynamics,
    )


def scan_raw_clips(raw_root: Path) -> list[ClipRecord]:
    raw_root = Path(raw_root).expanduser().resolve()
    if not raw_root.is_dir():
        return []
    clips = [scan_raw_clip(path) for path in sorted(raw_root.iterdir()) if path.is_dir()]
    return [clip for clip in clips if clip is not None]


def scene_clip_counts(raw_root: Path, *, scene_ids: set[str] | None = None) -> dict[str, int]:
    """Count rendered clips per scene_id under ``raw_root`` (from render_config.json)."""
    counts: dict[str, int] = {}
    for clip in scan_raw_clips(raw_root):
        if scene_ids is not None and clip.scene_id not in scene_ids:
            continue
        counts[clip.scene_id] = counts.get(clip.scene_id, 0) + 1
    return counts


def trajectory_status_for_clip(clip: ClipRecord) -> str:
    """ok | missing_csv | sha_mismatch | unrecorded"""
    if not clip.trajectory_csv:
        return "unrecorded"
    csv_path = Path(clip.trajectory_csv)
    if not csv_path.is_file():
        return "missing_csv"
    if not clip.trajectory_sha256:
        return "unrecorded"
    current = trajectory_file_sha256(csv_path)
    if current != clip.trajectory_sha256:
        return "sha_mismatch"
    return "ok"


def clips_bound_to_trajectory(raw_root: Path, trajectory_csv: Path) -> list[ClipRecord]:
    target = Path(trajectory_csv).expanduser().resolve()
    bound: list[ClipRecord] = []
    for clip in scan_raw_clips(raw_root):
        if not clip.trajectory_csv:
            continue
        if Path(clip.trajectory_csv).expanduser().resolve() == target:
            bound.append(clip)
    return bound


def select_unrendered_trajectories(
    *,
    trajectories: list[dict[str, str]],
    raw_root: Path | None,
    limit: int,
) -> list[dict[str, str]]:
    """Return up to ``limit`` trajectory rows that have no rendered clip on disk.

    Used by ``batch_render`` so that "add N new clips" picks the first N
    trajectories without any matching clip directory, regardless of where
    those trajectories sit in the manifest. Order is preserved.
    """
    if limit <= 0:
        return []
    if raw_root is None or not Path(raw_root).is_dir():
        return list(trajectories[:limit])
    out: list[dict[str, str]] = []
    for row in trajectories:
        csv_raw = row.get("trajectory_csv", "")
        if not csv_raw:
            continue
        csv_path = Path(csv_raw)
        if not csv_path.is_file():
            continue
        if clips_bound_to_trajectory(raw_root, csv_path):
            continue
        out.append(row)
        if len(out) >= limit:
            break
    return out


def load_csv_manifest(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def clip_to_manifest_row(clip: ClipRecord, *, status: str = "on_disk") -> dict[str, str]:
    blend = ""
    blend_path = clip.clip_dir / "render_config.json"
    if blend_path.is_file():
        blend = str(_load_json(blend_path).get("blend_file", ""))
    traj_status = trajectory_status_for_clip(clip)
    if traj_status == "sha_mismatch":
        status = "trajectory_mismatch"
    elif traj_status == "missing_csv":
        status = "trajectory_csv_missing"
    return {
        "clip_id": clip.clip_id,
        "scene_id": clip.scene_id,
        "blend": blend,
        "trajectory_id": clip.trajectory_id,
        "trajectory_csv": clip.trajectory_csv,
        "trajectory_sha256": clip.trajectory_sha256,
        "camera_seed": clip.camera_seed,
        "n_frames": str(clip.n_frames),
        "status": status,
    }


def merge_raw_manifest(
    raw_root: Path,
    planned_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Keep on-disk clips; overlay this run's planned rows by clip_id."""
    merged: dict[str, dict[str, str]] = {}
    for clip in scan_raw_clips(raw_root):
        merged[clip.clip_id] = clip_to_manifest_row(clip)
    for row in planned_rows:
        clip_id = row.get("clip_id", "")
        if not clip_id:
            continue
        if clip_id in merged and row.get("status", "").startswith("skipped"):
            # Preserve richer on-disk metadata unless this run re-rendered.
            prior = merged[clip_id]
            row = {**prior, **row}
            if prior.get("trajectory_sha256") and not row.get("trajectory_sha256"):
                row["trajectory_sha256"] = prior["trajectory_sha256"]
        merged[clip_id] = row
    return [merged[key] for key in sorted(merged)]


def trajectory_csv_path(
    trajectory_root: Path,
    trajectory_id: str,
    display_hz: int,
) -> Path:
    return Path(trajectory_root) / f"{trajectory_id}_{display_hz}hz.csv"


def _trajectory_idx_from_id(trajectory_id: str) -> int | None:
    if not trajectory_id:
        return None
    suffix = trajectory_id.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _max_existing_trajectory_idx(
    *, pattern: str, manifest_entries: dict[str, dict[str, str]], trajectory_root: Path
) -> int:
    """Return the highest trajectory index for ``pattern`` across manifest + disk.

    Returns -1 if no entries exist (so the next index to allocate is 0).
    Considers both manifest rows and any orphan ``mujoco_<pattern>_NNNNN_*.csv``
    files on disk so we never collide with a CSV that lives outside the
    manifest (e.g. if a previous run was interrupted before the manifest got
    written).
    """
    indices: list[int] = []
    prefix = f"mujoco_{pattern}_"
    for tid in manifest_entries:
        if not tid.startswith(prefix):
            continue
        idx = _trajectory_idx_from_id(tid)
        if idx is not None:
            indices.append(idx)
    if Path(trajectory_root).is_dir():
        for path in Path(trajectory_root).glob(f"{prefix}*.csv"):
            stem = path.stem
            for token in stem.split("_"):
                if token.isdigit() and len(token) == 5:
                    indices.append(int(token))
                    break
    return max(indices) if indices else -1


def merge_trajectory_manifest(
    *,
    trajectory_root: Path,
    manifest_path: Path,
    pattern: str,
    num_trajectories: int,
    frames: int,
    display_hz: int,
    seed_base: int,
    always_regenerate: bool,
    overwrite: bool,
    raw_root: Path | None = None,
    append: bool = True,
) -> tuple[list[dict[str, str]], list[str]]:
    """Ensure the trajectory manifest contains the desired trajectories.

    Modes:
      - ``append=True`` (default): generate ``num_trajectories`` NEW trajectories
        starting at ``max_existing_idx + 1`` for the given pattern. Existing
        entries are preserved untouched (they will appear in the returned rows
        with ``should_generate=0``). This is what most users want: each run
        adds new clips on top of the existing dataset.
      - ``append=False``: legacy contract — ensure indices
        ``00000..num_trajectories-1`` exist for the given pattern; reuse
        existing CSVs when possible.

    Returns (rows, log_lines). Does not regenerate CSVs — caller runs MuJoCo.
    """
    trajectory_root = Path(trajectory_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    existing_by_id: dict[str, dict[str, str]] = {}
    for row in load_csv_manifest(manifest_path):
        tid = row.get("trajectory_id", "")
        if tid:
            existing_by_id[tid] = row

    logs: list[str] = []
    rows: list[dict[str, str]] = []

    if append:
        max_idx = _max_existing_trajectory_idx(
            pattern=pattern,
            manifest_entries=existing_by_id,
            trajectory_root=trajectory_root,
        )
        start_idx = max_idx + 1
        index_range = range(start_idx, start_idx + max(0, int(num_trajectories)))
        if num_trajectories > 0:
            logs.append(
                f"[trajectory] append mode: requesting {num_trajectories} new trajectory(ies) "
                f"starting at index {start_idx:05d} (max existing = {max_idx if max_idx >= 0 else 'none'})"
            )
        else:
            logs.append("[trajectory] append mode: num_trajectories=0, no new trajectories will be created")
    else:
        index_range = range(num_trajectories)
        logs.append(
            f"[trajectory] fixed mode: ensuring trajectories 00000..{max(0, num_trajectories - 1):05d}"
        )

    for trajectory_idx in index_range:
        trajectory_id = f"mujoco_{pattern}_{trajectory_idx:05d}"
        csv_path = trajectory_csv_path(trajectory_root, trajectory_id, display_hz)
        prev = existing_by_id.get(trajectory_id, {})
        prev_csv = prev.get("trajectory_csv", "")
        if prev_csv:
            csv_path = Path(prev_csv).expanduser().resolve()

        sha = trajectory_file_sha256(csv_path) if csv_path.is_file() else ""
        bound_clips = clips_bound_to_trajectory(raw_root, csv_path) if raw_root and csv_path.is_file() else []

        should_generate = False
        if overwrite:
            should_generate = True
            logs.append(f"[trajectory] {trajectory_id}: overwrite requested")
        elif not csv_path.is_file():
            should_generate = True
            logs.append(f"[trajectory] {trajectory_id}: missing CSV -> will generate")
        elif always_regenerate:
            if bound_clips:
                mismatched = [c for c in bound_clips if trajectory_status_for_clip(c) != "ok"]
                if mismatched:
                    should_generate = False
                    logs.append(
                        f"[trajectory] {trajectory_id}: skip regenerate — {len(bound_clips)} clip(s) "
                        f"still bound to this CSV (use OVERWRITE=1 to force)"
                    )
                else:
                    should_generate = False
                    logs.append(
                        f"[trajectory] {trajectory_id}: skip regenerate — CSV matches "
                        f"{len(bound_clips)} on-disk clip(s)"
                    )
            else:
                should_generate = True
                logs.append(f"[trajectory] {trajectory_id}: regenerate (no clips bound)")
        else:
            should_generate = False
            logs.append(f"[trajectory] {trajectory_id}: reuse {csv_path.name}")

        used_seed = str(prev.get("seed", seed_base + trajectory_idx))
        row = {
            "trajectory_id": trajectory_id,
            "trajectory_csv": str(csv_path),
            "pattern": pattern,
            "seed": used_seed,
            "frames": str(frames),
            "fps": str(display_hz),
            "sha256": sha,
            "should_generate": "1" if should_generate else "0",
        }
        rows.append(row)

    # Preserve manifest rows outside 0..num_trajectories-1 (historical trajectories).
    for tid, prev in existing_by_id.items():
        if tid not in {r["trajectory_id"] for r in rows}:
            csv_raw = prev.get("trajectory_csv", "")
            if csv_raw:
                p = Path(csv_raw)
                if p.is_file() and not prev.get("sha256"):
                    prev = {**prev, "sha256": trajectory_file_sha256(p)}
            rows.append(prev)
            logs.append(f"[trajectory] preserve historical entry {tid}")

    rows.sort(key=lambda r: r.get("trajectory_id", ""))
    write_manifest = [{k: v for k, v in r.items() if k != "should_generate"} for r in rows]
    write_csv_manifest(manifest_path, TRAJECTORY_MANIFEST_FIELDS, write_manifest)
    return rows, logs


def backfill_render_config(
    clip: ClipRecord,
    *,
    trajectory_csv: Path | None = None,
) -> tuple[bool, str]:
    """
    Write trajectory_csv + trajectory_sha256 into render_config.json.
    Returns (updated, message). Only safe if the CSV on disk is the one used at render time.
    """
    render_config_path = clip.clip_dir / "render_config.json"
    if not render_config_path.is_file():
        return False, "missing render_config.json"

    cfg = _load_json(render_config_path)
    csv_path = Path(trajectory_csv or cfg.get("trajectory_csv") or clip.trajectory_csv or "")
    if not csv_path:
        return False, "no trajectory_csv path"
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        return False, f"trajectory CSV not found: {csv_path}"

    sha = trajectory_file_sha256(csv_path)
    prev_sha = str(cfg.get("trajectory_sha256", "") or "")
    cfg["trajectory_csv"] = str(csv_path)
    cfg["trajectory_sha256"] = sha
    if "dynamics_fields" not in cfg:
        cfg["dynamics_fields"] = ["steer_deg", "roll_deg"]

    render_config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    if prev_sha and prev_sha != sha:
        return True, f"updated (sha changed {prev_sha[:12]}... -> {sha[:12]}...)"
    if prev_sha == sha:
        return True, "already up to date"
    return True, f"set sha256={sha[:12]}..."


def audit_raw_root(raw_root: Path) -> dict[str, Any]:
    clips = scan_raw_clips(raw_root)
    mismatches = [c.clip_id for c in clips if trajectory_status_for_clip(c) == "sha_mismatch"]
    missing_csv = [c.clip_id for c in clips if trajectory_status_for_clip(c) == "missing_csv"]
    missing_dynamics = [c.clip_id for c in clips if not c.has_dynamics_gt]
    return {
        "clip_count": len(clips),
        "trajectory_mismatch_clip_ids": mismatches,
        "trajectory_csv_missing_clip_ids": missing_csv,
        "missing_dynamics_gt_clip_ids": missing_dynamics,
    }

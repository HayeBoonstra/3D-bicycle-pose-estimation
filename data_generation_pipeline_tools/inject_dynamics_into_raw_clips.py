#!/usr/bin/env python3
"""Add dynamics_gt to existing keypoints_3d.jsonl using manifest trajectory CSV paths.

Use when raw keypoints are still valid but jsonl predates dynamics export. Requires the
trajectory CSV on disk to be the same file used when the clip was originally rendered.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.mujoco_trajectory_dynamics import (
    assert_trajectory_frame_count,
    dynamics_gt_payload,
    load_trajectory_dynamics_csv,
)
from data_generation_pipeline_tools.pipeline_clip_registry import (
    scan_raw_clip,
    trajectory_status_for_clip,
)

_TRAJECTORY_ID_RE = re.compile(r"(mujoco_[a-z]+_\d{5})")


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clip_id = (row.get("clip_id") or "").strip()
            trajectory_csv = (row.get("trajectory_csv") or row.get("csv") or "").strip()
            if clip_id and trajectory_csv:
                mapping[clip_id] = trajectory_csv
    return mapping


def _trajectory_csv_from_clip_id(clip_id: str, trajectory_root: Path) -> Path | None:
    match = _TRAJECTORY_ID_RE.search(clip_id)
    if not match:
        return None
    trajectory_id = match.group(1)
    candidates = sorted(trajectory_root.glob(f"{trajectory_id}_*hz.csv"))
    return candidates[0] if candidates else None


def _resolve_trajectory_csv(
    clip_dir: Path,
    manifest: dict[str, str],
    trajectory_root: Path,
) -> Path:
    clip_id = clip_dir.name
    if clip_id in manifest:
        return Path(manifest[clip_id]).expanduser().resolve()
    render_config_path = clip_dir / "render_config.json"
    if render_config_path.is_file():
        render_config = json.loads(render_config_path.read_text(encoding="utf-8"))
        configured = render_config.get("trajectory_csv")
        if configured:
            return Path(configured).expanduser().resolve()
    inferred = _trajectory_csv_from_clip_id(clip_id, trajectory_root)
    if inferred is not None:
        return inferred.resolve()
    raise FileNotFoundError(
        f"Could not resolve trajectory CSV for {clip_id}. "
        "Add a manifest row or render_config.trajectory_csv."
    )


def inject_clip(clip_dir: Path, trajectory_csv: Path, *, dry_run: bool = False) -> dict:
    jsonl_path = clip_dir / "keypoints_3d.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Missing {jsonl_path}")

    clip_record = scan_raw_clip(clip_dir)
    if clip_record is not None:
        traj_status = trajectory_status_for_clip(clip_record)
        if traj_status == "sha_mismatch":
            raise ValueError(
                f"{clip_dir.name}: trajectory CSV on disk does not match render_config "
                f"trajectory_sha256 ({clip_record.trajectory_sha256[:12]}...). "
                "Use the original CSV or re-export annotations with --annotations-only."
            )

    lines = [line.strip() for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    trajectory_rows = assert_trajectory_frame_count(trajectory_csv, len(rows))

    updated: list[str] = []
    for frame_index, row in enumerate(rows):
        row_frame_index = int(row.get("frame_index", frame_index))
        if row_frame_index != frame_index:
            raise ValueError(
                f"{clip_dir.name}: frame_index {row_frame_index} != jsonl line index {frame_index}"
            )
        row["dynamics_gt"] = dynamics_gt_payload(
            trajectory_csv,
            frame_index,
            trajectory_rows=trajectory_rows,
        )
        updated.append(json.dumps(row, separators=(",", ":")))

    if not dry_run:
        jsonl_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        render_config_path = clip_dir / "render_config.json"
        render_config: dict = {}
        if render_config_path.is_file():
            render_config = json.loads(render_config_path.read_text(encoding="utf-8"))
        render_config["trajectory_csv"] = str(Path(trajectory_csv).resolve())
        render_config["dynamics_fields"] = ["steer_deg", "roll_deg"]
        render_config_path.write_text(json.dumps(render_config, indent=2) + "\n", encoding="utf-8")

    return {
        "clip_id": clip_dir.name,
        "frames": len(rows),
        "trajectory_csv": str(trajectory_csv),
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject dynamics_gt into raw keypoints_3d.jsonl clips.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="batch_render manifest.csv with clip_id,trajectory_csv columns",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=Path("/mnt/SmallSSD/3D-bicycle-pose-estimation/mujoco_blender_trajectories"),
    )
    parser.add_argument("--clip-id", default=None, help="Process a single clip directory name")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    manifest = _load_manifest(args.manifest) if args.manifest else {}

    clip_dirs = []
    if args.clip_id:
        clip_dirs = [raw_root / args.clip_id]
    else:
        clip_dirs = [
            path
            for path in sorted(raw_root.iterdir())
            if path.is_dir() and (path / "keypoints_3d.jsonl").is_file()
        ]

    results = []
    for clip_dir in clip_dirs:
        trajectory_csv = _resolve_trajectory_csv(clip_dir, manifest, args.trajectory_root)
        results.append(inject_clip(clip_dir, trajectory_csv, dry_run=args.dry_run))
        print(
            f"[inject-dynamics] {clip_dir.name}: {results[-1]['frames']} frames "
            f"<- {trajectory_csv.name}{' (dry-run)' if args.dry_run else ''}"
        )


if __name__ == "__main__":
    main()

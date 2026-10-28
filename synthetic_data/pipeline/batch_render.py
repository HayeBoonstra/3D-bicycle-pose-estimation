"""Batch-render registered Blender scenes with camera-only randomization."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

from scene_registry import (
    DEFAULT_REGISTRY_PATH,
    load_scene_registry,
    sample_scenes,
    sample_scenes_balanced,
    scene_counts_for_registry,
)

from pipeline_clip_registry import (
    RAW_MANIFEST_FIELDS,
    clip_to_manifest_row,
    merge_raw_manifest,
    scan_raw_clip,
    scene_clip_counts,
    select_unrendered_trajectories,
    trajectory_status_for_clip,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_RENDERS_DIR = REPO_ROOT / "raw_renders"
RENDER_CLIP_SCRIPT = Path(__file__).resolve().parent / "render_clip.py"
RENDER_CLIP_BATCH_SCRIPT = Path(__file__).resolve().parent / "render_clip_batch.py"
_OMP_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _resolve_blender_threads(parallel_jobs: int, explicit: int | None) -> int:
    """Blender -t value: 0 = use all logical CPUs; otherwise an explicit cap."""
    if explicit is not None:
        return max(0, explicit)
    if parallel_jobs <= 1:
        return 0
    return max(1, _cpu_count() // parallel_jobs)


def _subprocess_env_for_threads(blender_threads: int) -> dict[str, str]:
    """Align OpenMP/BLAS libraries with Blender's thread budget."""
    env = os.environ.copy()
    thread_budget = _cpu_count() if blender_threads == 0 else blender_threads
    thread_str = str(thread_budget)
    for key in _OMP_THREAD_VARS:
        env[key] = thread_str
    return env


def _describe_thread_plan(parallel_jobs: int, blender_threads: int) -> str:
    cpus = _cpu_count()
    if blender_threads == 0:
        per_proc = f"all {cpus} CPUs"
    else:
        per_proc = f"{blender_threads} thread(s)"
    if parallel_jobs <= 1:
        return f"sequential clips, Blender {per_proc}"
    total = blender_threads * parallel_jobs if blender_threads > 0 else cpus
    return f"{parallel_jobs} parallel Blender process(es), {per_proc} each (~{total} CPUs requested)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render many raw clips from scene registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_RAW_RENDERS_DIR)
    parser.add_argument("--num-clips", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blender", default="blender", help="Path to Blender executable.")
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        help="Optional CSV manifest with trajectory_id and trajectory_csv columns from bicycle_test exports.",
    )
    parser.add_argument(
        "--cameras-per-trajectory",
        type=int,
        default=1,
        help="When --trajectory-manifest is set, render this many camera seeds per loaded scene/trajectory.",
    )
    parser.add_argument(
        "--respect-scene-frame-range-with-trajectories",
        action="store_true",
        help=(
            "When using --trajectory-manifest, keep scene frame_range from scenes.yaml. "
            "Default behavior ignores scene frame_range so imported trajectory length is preserved."
        ),
    )
    parser.add_argument(
        "--sync-window-size",
        type=int,
        default=80,
        help="Rendered frame window centered on sampled camera frame.",
    )
    parser.add_argument(
        "--no-sync-camera-window",
        action="store_true",
        help="Disable camera-sampled frame window synchronization.",
    )
    parser.add_argument("--encode-video", action="store_true")
    parser.add_argument(
        "--verbose-render-progress",
        action="store_true",
        help="Print per-frame render progress from Blender clips.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Maximum number of clips to render in parallel (default: 1). "
            "With --jobs 1, each Blender uses all CPU cores; with --jobs N>1, cores are split across processes."
        ),
    )
    parser.add_argument(
        "--blender-threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Blender -t thread count per process (0 = all logical CPUs). "
            "Default: 0 when --jobs 1, else max(1, cpu_count // jobs)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-incremental-manifest",
        action="store_true",
        help="Only write clips from this run (do not merge on-disk history into manifest).",
    )
    parser.add_argument("--render-format", choices=("PNG", "JPEG"), default=os.environ.get("RENDER_FORMAT", "PNG"))
    parser.add_argument(
        "--resolution-percentage",
        type=int,
        default=int(os.environ.get("BLENDER_RESOLUTION_PERCENTAGE", "100")),
    )
    parser.add_argument(
        "--cycles-samples",
        type=int,
        default=int(os.environ.get("BLENDER_CYCLES_SAMPLES", "0")),
    )
    parser.add_argument(
        "--camera-min-distance",
        type=float,
        default=float(os.environ.get("CAMERA_MIN_DISTANCE", "4.0")),
    )
    parser.add_argument(
        "--camera-max-distance",
        type=float,
        default=float(os.environ.get("CAMERA_MAX_DISTANCE", "12.0")),
    )
    parser.add_argument(
        "--camera-min-bbox-area-frac",
        type=float,
        default=float(os.environ.get("CAMERA_MIN_BBOX_AREA_FRAC", "0.04")),
    )
    parser.add_argument(
        "--camera-max-bbox-area-frac",
        type=float,
        default=float(os.environ.get("CAMERA_MAX_BBOX_AREA_FRAC", "0.80")),
    )
    parser.add_argument(
        "--camera-min-visible-keypoints",
        type=int,
        default=int(os.environ.get("CAMERA_MIN_VISIBLE_KEYPOINTS", "14")),
    )
    parser.add_argument(
        "--camera-min-visible-frame-ratio",
        type=float,
        default=float(os.environ.get("CAMERA_MIN_VISIBLE_FRAME_RATIO", "0.9")),
    )
    parser.add_argument(
        "--camera-fit-margin",
        type=float,
        default=float(os.environ.get("CAMERA_FIT_MARGIN", "1.25")),
    )
    parser.add_argument(
        "--camera-mode",
        choices=("track", "fixed"),
        default=os.environ.get("CAMERA_MODE", "track"),
    )
    parser.add_argument(
        "--balance-scenes",
        action="store_true",
        help=(
            "When planning new clips, read existing clips under --out and assign scenes "
            "to keep per-scene counts as even as possible (registry scenes only)."
        ),
    )
    return parser.parse_args()


def _clip_id(scene_id: str, camera_seed: int) -> str:
    return f"clip_{scene_id}_{camera_seed:08d}"


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")


def _trajectory_clip_prefix(scene_id: str, trajectory_id: str) -> str:
    return f"clip_{scene_id}_{_safe_id(trajectory_id)}"


def _read_trajectory_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            csv_raw = row.get("trajectory_csv") or row.get("csv") or row.get("path")
            if not csv_raw:
                raise ValueError(f"Trajectory manifest row {idx} is missing trajectory_csv/csv/path")
            csv_path = Path(csv_raw).expanduser()
            if not csv_path.is_absolute():
                csv_path = (path.parent / csv_path).resolve()
            trajectory_id = row.get("trajectory_id") or csv_path.stem
            rows.append({"trajectory_id": _safe_id(trajectory_id), "trajectory_csv": str(csv_path)})
    if not rows:
        raise ValueError(f"No trajectories found in manifest: {path}")
    return rows


def _render_command(args: argparse.Namespace, entry, camera_seed: int, clip_dir: Path) -> list[str]:
    command = [
        args.blender,
        "--background",
        str(entry.blend_path),
        "-t",
        str(args.blender_threads),
        "--python",
        str(RENDER_CLIP_SCRIPT),
        "--",
        "--clip-id",
        clip_dir.name,
        "--scene-id",
        entry.id,
        "--camera-seed",
        str(camera_seed),
        "--camera-target",
        entry.camera_target,
        "--sync-window-size",
        str(args.sync_window_size),
        "--out",
        str(clip_dir),
        "--bike",
        entry.bike,
        "--rider",
        entry.rider,
    ]
    if entry.frame_range is not None:
        command.extend(
            [
                "--frame-start",
                str(entry.frame_range[0]),
                "--frame-end",
                str(entry.frame_range[1]),
            ]
        )
    if args.encode_video:
        command.append("--encode-video")
    if args.verbose_render_progress:
        command.append("--no-quiet-mode")
    if args.no_sync_camera_window:
        command.append("--no-sync-camera-window")
    command.extend(
        [
            "--camera-min-distance",
            str(args.camera_min_distance),
            "--camera-max-distance",
            str(args.camera_max_distance),
            "--camera-min-bbox-area-frac",
            str(args.camera_min_bbox_area_frac),
            "--camera-max-bbox-area-frac",
            str(args.camera_max_bbox_area_frac),
            "--camera-min-visible-keypoints",
            str(args.camera_min_visible_keypoints),
            "--camera-min-visible-frame-ratio",
            str(args.camera_min_visible_frame_ratio),
            "--camera-fit-margin",
            str(args.camera_fit_margin),
            "--camera-mode",
            str(args.camera_mode),
            "--render-format",
            str(args.render_format),
            "--resolution-percentage",
            str(args.resolution_percentage),
            "--cycles-samples",
            str(args.cycles_samples),
        ]
    )
    return command


def _render_batch_command(
    args: argparse.Namespace,
    entry,
    trajectory: dict[str, str],
    camera_seeds: list[int],
) -> list[str]:
    clip_prefix = _trajectory_clip_prefix(entry.id, trajectory["trajectory_id"])
    command = [
        args.blender,
        "--background",
        str(entry.blend_path),
        "-t",
        str(args.blender_threads),
        "--python",
        str(RENDER_CLIP_BATCH_SCRIPT),
        "--",
        "--scene-id",
        entry.id,
        "--out-root",
        str(args.out),
        "--clip-prefix",
        clip_prefix,
        "--camera-seeds",
        ",".join(str(seed) for seed in camera_seeds),
        "--camera-target",
        entry.camera_target,
        "--trajectory-csv",
        trajectory["trajectory_csv"],
        "--sync-window-size",
        str(args.sync_window_size),
        "--bike",
        entry.bike,
        "--rider",
        entry.rider,
        "--camera-min-distance",
        str(args.camera_min_distance),
        "--camera-max-distance",
        str(args.camera_max_distance),
        "--camera-min-bbox-area-frac",
        str(args.camera_min_bbox_area_frac),
        "--camera-max-bbox-area-frac",
        str(args.camera_max_bbox_area_frac),
        "--camera-min-visible-keypoints",
        str(args.camera_min_visible_keypoints),
        "--camera-min-visible-frame-ratio",
        str(args.camera_min_visible_frame_ratio),
        "--camera-fit-margin",
        str(args.camera_fit_margin),
        "--camera-mode",
        str(args.camera_mode),
        "--render-format",
        str(args.render_format),
        "--resolution-percentage",
        str(args.resolution_percentage),
        "--cycles-samples",
        str(args.cycles_samples),
    ]
    if args.encode_video:
        command.append("--encode-video")
    if args.verbose_render_progress:
        command.append("--no-quiet-mode")
    if args.no_sync_camera_window:
        command.append("--no-sync-camera-window")
    if args.respect_scene_frame_range_with_trajectories and entry.frame_range is not None:
        command.extend(["--frame-start", str(entry.frame_range[0]), "--frame-end", str(entry.frame_range[1])])
    return command


def _n_frames(entry) -> str:
    if entry.frame_range is None:
        return ""
    return str(entry.frame_range[1] - entry.frame_range[0] + 1)


def _validate_clip_outputs(clip_dir: Path) -> None:
    annotations_dir = clip_dir / "per_frame_annotations"
    if not annotations_dir.exists():
        raise RuntimeError(f"Missing per-frame annotations directory: {annotations_dir}")
    if not any(annotations_dir.glob("*.json")):
        raise RuntimeError(f"No annotation JSON files found in: {annotations_dir}")
    for required_file in ["render_config.json", "camera.json", "keypoints_3d.jsonl"]:
        required_path = clip_dir / required_file
        if not required_path.exists():
            raise RuntimeError(f"Missing required export file: {required_path}")


def _render_one(
    args: argparse.Namespace,
    entry,
    camera_seed: int,
    clip_dir: Path,
) -> None:
    command = _render_command(args, entry, camera_seed, clip_dir)
    print(f"[batch_render] rendering {clip_dir.name} from scene {entry.id}")
    subprocess.run(command, check=True, env=_subprocess_env_for_threads(args.blender_threads))
    _validate_clip_outputs(clip_dir)


def _render_batch(
    args: argparse.Namespace,
    entry,
    trajectory: dict[str, str],
    camera_seeds: list[int],
    clip_ids: list[str],
) -> None:
    command = _render_batch_command(args, entry, trajectory, camera_seeds)
    print(
        "[batch_render] rendering "
        f"{len(camera_seeds)} camera(s) from scene {entry.id} trajectory {trajectory['trajectory_id']}"
    )
    subprocess.run(command, check=True, env=_subprocess_env_for_threads(args.blender_threads))
    for clip_id in clip_ids:
        _validate_clip_outputs(args.out / clip_id)


def _timed_render_one(
    args: argparse.Namespace,
    entry,
    camera_seed: int,
    clip_dir: Path,
) -> float:
    started = time.perf_counter()
    _render_one(args, entry, camera_seed, clip_dir)
    return time.perf_counter() - started


def _timed_render_batch(
    args: argparse.Namespace,
    entry,
    trajectory: dict[str, str],
    camera_seeds: list[int],
    clip_ids: list[str],
) -> float:
    started = time.perf_counter()
    _render_batch(args, entry, trajectory, camera_seeds, clip_ids)
    return time.perf_counter() - started


def _prior_scene_counts(args: argparse.Namespace, entries) -> dict[str, int]:
    registry_ids = {entry.id for entry in entries}
    on_disk = scene_clip_counts(args.out, scene_ids=registry_ids)
    return scene_counts_for_registry(entries, on_disk)


def _log_scene_balance(
    *,
    label: str,
    entries,
    prior: dict[str, int],
    planned: list[tuple],
) -> None:
    projected = dict(prior)
    for entry, _camera_seed in planned:
        projected[entry.id] = projected.get(entry.id, 0) + 1
    print(f"[batch_render] scene balance ({label}):")
    for entry in entries:
        before = prior.get(entry.id, 0)
        after = projected.get(entry.id, 0)
        delta = after - before
        delta_s = f"+{delta}" if delta else "±0"
        print(f"  - {entry.id}: {before} -> {after} ({delta_s} this run)")


def _plan_scene_samples(
    args: argparse.Namespace,
    entries,
    count: int,
) -> list[tuple]:
    if count < 1:
        return []
    if args.balance_scenes:
        prior = _prior_scene_counts(args, entries)
        planned = sample_scenes_balanced(entries, count, args.seed, prior_counts=prior)
        _log_scene_balance(label="after plan", entries=entries, prior=prior, planned=planned)
        return planned
    return sample_scenes(entries, count, args.seed)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def main() -> None:
    args = _parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"

    entries = load_scene_registry(args.registry)
    rng = __import__("random").Random(args.seed)
    rows_by_clip_id: dict[str, dict[str, str]] = {}
    render_tasks: list[tuple] = []

    if args.trajectory_manifest is not None:
        if args.cameras_per_trajectory < 1:
            raise ValueError("--cameras-per-trajectory must be >= 1")
        trajectories = _read_trajectory_manifest(args.trajectory_manifest)
        # Pick the first ``num_clips`` trajectories that have no rendered clip
        # bound to them yet. This means the user can re-run with
        # NUM_CLIPS=1 and reliably render against the newly appended
        # trajectory rather than recycling trajectory 00000. With
        # ``--overwrite`` set, fall back to the first ``num_clips`` rows so
        # the old behavior is still reachable.
        if args.overwrite:
            selected_trajectories = trajectories[: args.num_clips]
            print(
                f"[batch_render] batch_seed={args.seed} overwrite=1 planned trajectories: "
                f"{len(selected_trajectories)} x {args.cameras_per_trajectory} camera(s)"
            )
        else:
            selected_trajectories = select_unrendered_trajectories(
                trajectories=trajectories,
                raw_root=args.out,
                limit=args.num_clips,
            )
            print(
                f"[batch_render] batch_seed={args.seed} planned trajectories: "
                f"{len(selected_trajectories)} x {args.cameras_per_trajectory} camera(s) "
                f"(picked from {len(trajectories)} manifest rows; unrendered-first)"
            )
            if not selected_trajectories and args.num_clips > 0:
                print(
                    "[batch_render] warning: every manifest trajectory already has a "
                    "rendered clip on disk. Use --overwrite to re-render, or rerun "
                    "sync_pipeline_state / ensure_mujoco_trajectories with a larger "
                    "--num-trajectories to add more trajectories."
                )
        scene_plan = _plan_scene_samples(args, entries, len(selected_trajectories))
        for trajectory, (entry, _planned_camera_seed) in zip(selected_trajectories, scene_plan):
            camera_seeds = [rng.randrange(0, 2**31) for _ in range(args.cameras_per_trajectory)]
            clip_prefix = _trajectory_clip_prefix(entry.id, trajectory["trajectory_id"])
            clip_ids = [f"{clip_prefix}_{seed:08d}" for seed in camera_seeds]
            batch_ready = True
            for camera_seed, clip_id in zip(camera_seeds, clip_ids):
                clip_dir = args.out / clip_id
                rows_by_clip_id[clip_id] = {
                    "clip_id": clip_id,
                    "scene_id": entry.id,
                    "blend": str(entry.blend_path),
                    "trajectory_id": trajectory["trajectory_id"],
                    "trajectory_csv": trajectory["trajectory_csv"],
                    "camera_seed": str(camera_seed),
                    "n_frames": "",
                    "status": "pending",
                }
                row = rows_by_clip_id[clip_id]
                if clip_dir.exists() and not args.overwrite:
                    existing = scan_raw_clip(clip_dir)
                    try:
                        _validate_clip_outputs(clip_dir)
                        if existing is not None:
                            traj_status = trajectory_status_for_clip(existing)
                            expected_csv = str(Path(trajectory["trajectory_csv"]).resolve())
                            recorded_csv = str(Path(existing.trajectory_csv).resolve()) if existing.trajectory_csv else ""
                            if traj_status == "sha_mismatch":
                                print(
                                    f"[batch_render] WARNING trajectory mismatch for {clip_id}: "
                                    f"annotations were exported with sha {existing.trajectory_sha256[:12]}... "
                                    f"but {Path(expected_csv).name} on disk differs. "
                                    "Skipping re-render; fix CSV or set OVERWRITE=1."
                                )
                                row["status"] = "trajectory_mismatch"
                                row["trajectory_sha256"] = existing.trajectory_sha256
                                continue
                            if recorded_csv and recorded_csv != expected_csv:
                                print(
                                    f"[batch_render] WARNING {clip_id} recorded trajectory "
                                    f"{recorded_csv} != planned {expected_csv}; skipping."
                                )
                                row["status"] = "trajectory_path_mismatch"
                                row["trajectory_sha256"] = existing.trajectory_sha256
                                continue
                        print(f"[batch_render] skipping existing clip: {clip_dir}")
                        row["status"] = "skipped_existing"
                        if existing is not None:
                            row["trajectory_sha256"] = existing.trajectory_sha256
                            row["n_frames"] = str(existing.n_frames)
                        continue
                    except RuntimeError:
                        print("[batch_render] found incomplete existing clip, re-rendering: " f"{clip_dir}")
                        shutil.rmtree(clip_dir)
                batch_ready = False
                clip_dir.mkdir(parents=True, exist_ok=True)
            if not batch_ready:
                render_tasks.append(("batch", entry, trajectory, camera_seeds, clip_ids))
    else:
        sampled = _plan_scene_samples(args, entries, args.num_clips)
        print(f"[batch_render] batch_seed={args.seed} planned clips:")
        for entry, camera_seed in sampled:
            print(f"  - scene={entry.id} camera_seed={camera_seed} -> clip_{entry.id}_{camera_seed:08d}")
        for entry, camera_seed in sampled:
            clip_id = _clip_id(entry.id, camera_seed)
            clip_dir = args.out / clip_id
            rows_by_clip_id[clip_id] = {
                "clip_id": clip_id,
                "scene_id": entry.id,
                "blend": str(entry.blend_path),
                "trajectory_id": "",
                "trajectory_csv": "",
                "camera_seed": str(camera_seed),
                "n_frames": _n_frames(entry),
                "status": "pending",
            }
            row = rows_by_clip_id[clip_id]

            if clip_dir.exists() and not args.overwrite:
                existing = scan_raw_clip(clip_dir)
                try:
                    _validate_clip_outputs(clip_dir)
                    if existing is not None and trajectory_status_for_clip(existing) == "sha_mismatch":
                        print(
                            f"[batch_render] WARNING trajectory mismatch for {clip_id}; "
                            "skipping re-render (OVERWRITE=1 to replace)."
                        )
                        row["status"] = "trajectory_mismatch"
                        row["trajectory_sha256"] = existing.trajectory_sha256
                        continue
                    print(f"[batch_render] skipping existing clip: {clip_dir}")
                    row["status"] = "skipped_existing"
                    if existing is not None:
                        row["trajectory_sha256"] = existing.trajectory_sha256
                        row["n_frames"] = str(existing.n_frames)
                    continue
                except RuntimeError:
                    print(
                        "[batch_render] found incomplete existing clip, "
                        f"re-rendering: {clip_dir}"
                    )
                    shutil.rmtree(clip_dir)

            clip_dir.mkdir(parents=True, exist_ok=True)
            render_tasks.append(("single", entry, camera_seed, clip_dir, clip_id))

    _write_manifest(manifest_path, list(rows_by_clip_id.values()))

    max_workers = min(args.jobs, len(render_tasks))
    args.blender_threads = _resolve_blender_threads(max_workers, args.blender_threads)
    print(f"[batch_render] {_describe_thread_plan(max_workers, args.blender_threads)}")
    if max_workers > 0:
        total_to_render = len(render_tasks)
        completed = 0
        rendered = 0
        failed = 0
        clip_durations_sec: list[float] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_clip_ids = {}
            for task in render_tasks:
                if task[0] == "batch":
                    _kind, entry, trajectory, camera_seeds, clip_ids = task
                    future = executor.submit(_timed_render_batch, args, entry, trajectory, camera_seeds, clip_ids)
                    future_to_clip_ids[future] = clip_ids
                else:
                    _kind, entry, camera_seed, clip_dir, clip_id = task
                    future = executor.submit(_timed_render_one, args, entry, camera_seed, clip_dir)
                    future_to_clip_ids[future] = [clip_id]
            for future in concurrent.futures.as_completed(future_to_clip_ids):
                clip_ids = future_to_clip_ids[future]
                completed += 1
                try:
                    clip_duration = future.result()
                    for clip_id in clip_ids:
                        rows_by_clip_id[clip_id]["status"] = "rendered"
                    rendered += len(clip_ids)
                    clip_durations_sec.append(clip_duration)
                except subprocess.CalledProcessError as exc:
                    for clip_id in clip_ids:
                        rows_by_clip_id[clip_id]["status"] = f"failed:{exc.returncode}"
                    failed += len(clip_ids)
                    clip_duration = 0.0
                except RuntimeError:
                    for clip_id in clip_ids:
                        rows_by_clip_id[clip_id]["status"] = "failed:missing_outputs"
                    failed += len(clip_ids)
                    clip_duration = 0.0

                remaining = total_to_render - completed
                avg_clip_time = (
                    sum(clip_durations_sec) / len(clip_durations_sec) if clip_durations_sec else 0.0
                )
                eta_seconds = (avg_clip_time * remaining) / max_workers if remaining > 0 else 0.0
                print(
                    "[batch_render] progress "
                    f"{completed}/{total_to_render} complete "
                    f"(rendered={rendered}, failed={failed}, remaining={remaining}) "
                    f"clip_time={_format_duration(clip_duration)} "
                    f"eta={_format_duration(eta_seconds)}"
                )
                _write_manifest(manifest_path, list(rows_by_clip_id.values()))

    planned = list(rows_by_clip_id.values())
    if not args.no_incremental_manifest:
        planned = merge_raw_manifest(args.out, planned)
    _write_manifest(manifest_path, planned)
    failed = [row for row in planned if str(row.get("status", "")).startswith("failed:")]
    on_disk = sum(1 for row in planned if row.get("status") in {"on_disk", "skipped_existing"})
    mismatched = [row["clip_id"] for row in planned if "mismatch" in str(row.get("status", ""))]
    print(f"[batch_render] wrote manifest: {manifest_path} ({len(planned)} clips, {on_disk} on disk/skipped)")
    if mismatched:
        print(f"[batch_render] trajectory mismatch clips ({len(mismatched)}): {', '.join(mismatched[:5])}" + (
            " ..." if len(mismatched) > 5 else ""
        ))
    if failed:
        raise RuntimeError(f"{len(failed)} clip(s) failed; see manifest for details.")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = RAW_MANIFEST_FIELDS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

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

from scene_registry import DEFAULT_REGISTRY_PATH, load_scene_registry, sample_scenes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_RENDERS_DIR = REPO_ROOT / "raw_renders"
RENDER_CLIP_SCRIPT = Path(__file__).resolve().parent / "render_clip.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render many raw clips from scene registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_RAW_RENDERS_DIR)
    parser.add_argument("--num-clips", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blender", default="blender", help="Path to Blender executable.")
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
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Maximum number of clips to render in parallel.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _clip_id(scene_id: str, camera_seed: int) -> str:
    return f"clip_{scene_id}_{camera_seed:08d}"


def _render_command(args: argparse.Namespace, entry, camera_seed: int, clip_dir: Path) -> list[str]:
    command = [
        args.blender,
        "--background",
        str(entry.blend_path),
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
    if args.no_sync_camera_window:
        command.append("--no-sync-camera-window")
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
    subprocess.run(command, check=True)
    _validate_clip_outputs(clip_dir)


def _timed_render_one(
    args: argparse.Namespace,
    entry,
    camera_seed: int,
    clip_dir: Path,
) -> float:
    started = time.perf_counter()
    _render_one(args, entry, camera_seed, clip_dir)
    return time.perf_counter() - started


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
    sampled = sample_scenes(entries, args.num_clips, args.seed)
    rows_by_clip_id: dict[str, dict[str, str]] = {}
    render_tasks: list[tuple] = []

    for entry, camera_seed in sampled:
        clip_id = _clip_id(entry.id, camera_seed)
        clip_dir = args.out / clip_id
        rows_by_clip_id[clip_id] = {
            "clip_id": clip_id,
            "scene_id": entry.id,
            "blend": str(entry.blend_path),
            "camera_seed": str(camera_seed),
            "n_frames": _n_frames(entry),
            "status": "pending",
        }
        row = rows_by_clip_id[clip_id]

        if clip_dir.exists() and not args.overwrite:
            try:
                _validate_clip_outputs(clip_dir)
                print(f"[batch_render] skipping existing clip: {clip_dir}")
                row["status"] = "skipped_existing"
                continue
            except RuntimeError:
                print(
                    "[batch_render] found incomplete existing clip, "
                    f"re-rendering: {clip_dir}"
                )
                shutil.rmtree(clip_dir)

        clip_dir.mkdir(parents=True, exist_ok=True)
        render_tasks.append((entry, camera_seed, clip_dir, clip_id))

    _write_manifest(manifest_path, list(rows_by_clip_id.values()))

    max_workers = min(args.jobs, len(render_tasks))
    if max_workers > 0:
        total_to_render = len(render_tasks)
        completed = 0
        rendered = 0
        failed = 0
        clip_durations_sec: list[float] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_clip_id = {
                executor.submit(_timed_render_one, args, entry, camera_seed, clip_dir): clip_id
                for entry, camera_seed, clip_dir, clip_id in render_tasks
            }
            for future in concurrent.futures.as_completed(future_to_clip_id):
                clip_id = future_to_clip_id[future]
                row = rows_by_clip_id[clip_id]
                completed += 1
                try:
                    clip_duration = future.result()
                    row["status"] = "rendered"
                    rendered += 1
                    clip_durations_sec.append(clip_duration)
                except subprocess.CalledProcessError as exc:
                    row["status"] = f"failed:{exc.returncode}"
                    failed += 1
                    clip_duration = 0.0
                except RuntimeError:
                    row["status"] = "failed:missing_outputs"
                    failed += 1
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

    _write_manifest(manifest_path, list(rows_by_clip_id.values()))
    failed = [row for row in rows_by_clip_id.values() if row["status"].startswith("failed:")]
    print(f"[batch_render] wrote manifest: {manifest_path}")
    if failed:
        raise RuntimeError(f"{len(failed)} clip(s) failed; see manifest for details.")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["clip_id", "scene_id", "blend", "camera_seed", "n_frames", "status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

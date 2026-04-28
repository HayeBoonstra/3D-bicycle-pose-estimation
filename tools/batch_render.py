"""Batch-render registered Blender scenes with camera-only randomization."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from scene_registry import DEFAULT_REGISTRY_PATH, load_scene_registry, sample_scenes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_RENDERS_DIR = REPO_ROOT / "raw_renders"
RENDER_CLIP_SCRIPT = REPO_ROOT / "tools" / "render_clip.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render many raw clips from scene registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_RAW_RENDERS_DIR)
    parser.add_argument("--num-clips", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blender", default="blender", help="Path to Blender executable.")
    parser.add_argument("--encode-video", action="store_true")
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
    return command


def _n_frames(entry) -> str:
    if entry.frame_range is None:
        return ""
    return str(entry.frame_range[1] - entry.frame_range[0] + 1)


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"

    entries = load_scene_registry(args.registry)
    sampled = sample_scenes(entries, args.num_clips, args.seed)
    rows: list[dict[str, str]] = []

    for entry, camera_seed in sampled:
        clip_id = _clip_id(entry.id, camera_seed)
        clip_dir = args.out / clip_id
        row = {
            "clip_id": clip_id,
            "scene_id": entry.id,
            "blend": str(entry.blend_path),
            "camera_seed": str(camera_seed),
            "n_frames": _n_frames(entry),
            "status": "pending",
        }

        if clip_dir.exists() and not args.overwrite:
            print(f"[batch_render] skipping existing clip: {clip_dir}")
            row["status"] = "skipped_existing"
            rows.append(row)
            continue

        clip_dir.mkdir(parents=True, exist_ok=True)
        command = _render_command(args, entry, camera_seed, clip_dir)
        print(f"[batch_render] rendering {clip_id} from scene {entry.id}")
        try:
            subprocess.run(command, check=True)
            row["status"] = "rendered"
        except subprocess.CalledProcessError as exc:
            row["status"] = f"failed:{exc.returncode}"
            rows.append(row)
            _write_manifest(manifest_path, rows)
            raise
        rows.append(row)
        _write_manifest(manifest_path, rows)

    _write_manifest(manifest_path, rows)
    print(f"[batch_render] wrote manifest: {manifest_path}")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["clip_id", "scene_id", "blend", "camera_seed", "n_frames", "status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Render one registered Blender scene into a raw clip dataset folder.

Run through Blender, for example:

blender --background "Blender files/Scenes/mountain bike.blend" \
  --python tools/render_clip.py -- \
  --clip-id clip_mountain_bike_00000017 \
  --scene-id mountain_bike \
  --camera-seed 17 \
  --out raw_renders/clip_mountain_bike_00000017
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[1]
BLENDER_FILES_DIR = REPO_ROOT / "Blender files"
RANDOMIZE_CAMERA_SCRIPT = BLENDER_FILES_DIR / "randomize_camera.py"
EXPORT_SCRIPT = BLENDER_FILES_DIR / "extract_2D_annotation.py"


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one Blender clip and export labels.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--camera-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bike", default="")
    parser.add_argument("--rider", default="")
    parser.add_argument("--camera-target", default="k_handlebar_middle")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--encode-video", action="store_true")
    parser.add_argument("--fps", type=int)
    return parser.parse_args(_argv_after_double_dash())


def _set_env(args: argparse.Namespace, output_dir: Path) -> None:
    os.environ["CLIP_ID"] = args.clip_id
    os.environ["SCENE_ID"] = args.scene_id
    os.environ["CAMERA_SEED"] = str(args.camera_seed)
    os.environ["CAMERA_TARGET"] = args.camera_target
    os.environ["CLIP_OUTPUT_DIR"] = str(output_dir)
    os.environ["RAW_RENDERS_DIR"] = str(output_dir)
    os.environ["BIKE_TAG"] = args.bike
    os.environ["RIDER_TAG"] = args.rider


def _configure_scene(args: argparse.Namespace, output_dir: Path) -> None:
    scene = bpy.context.scene
    if args.frame_start is not None:
        scene.frame_start = args.frame_start
    if args.frame_end is not None:
        scene.frame_end = args.frame_end
    if args.fps is not None:
        scene.render.fps = args.fps

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(frames_dir / "frame_")


def _run_blender_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[render_clip] running {path}")
    runpy.run_path(str(path), run_name="__main__")


def _encode_mp4(output_dir: Path, clip_id: str, fps: int, frame_start: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")

    output_video = output_dir / f"{clip_id}.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(frame_start),
        "-i",
        str(output_dir / "frames" / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)
    print(f"[render_clip] wrote {output_video}")


def main() -> None:
    args = _parse_args()
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_env(args, output_dir)
    _configure_scene(args, output_dir)

    scene = bpy.context.scene
    print(
        f"[render_clip] clip={args.clip_id} scene={args.scene_id} "
        f"frames={scene.frame_start}-{scene.frame_end} camera_seed={args.camera_seed}"
    )

    _run_blender_script(RANDOMIZE_CAMERA_SCRIPT)
    bpy.ops.render.render(animation=True)
    _run_blender_script(EXPORT_SCRIPT)

    if args.encode_video:
        _encode_mp4(output_dir, args.clip_id, int(scene.render.fps), int(scene.frame_start))


if __name__ == "__main__":
    main()

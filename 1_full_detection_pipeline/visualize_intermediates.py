#!/usr/bin/env python3
"""Render intermediate pipeline outputs: bboxes, 2D keypoints, and 3D lift."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import (
    BICYCLE_SKELETON_NAMES,
    KEYPOINT_INDEX,
)
from keypoint_detector_pipeline.io_utils import iter_jsonl

from pipeline_io import DETECTIONS_NAME, KEYPOINTS_2D_NAME, KEYPOINTS_3D_NAME, REPO_ROOT

VIS_DETECTIONS_DIR = "vis/detections"
VIS_KEYPOINTS_2D_DIR = "vis/keypoints_2d"
VIS_KEYPOINTS_3D_DIR = "vis/keypoints_3d"

CONF_THRESHOLD = 0.01


def _skeleton_edges() -> list[tuple[int, int]]:
    return [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]) for a, b in BICYCLE_SKELETON_NAMES]


def _frame_path(vis_dir: Path, frame_id: int) -> Path:
    return vis_dir / f"frame_{frame_id:06d}.jpg"


def draw_detection_frame(row: dict):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(row["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = row.get("bbox_xyxy")
    score = float(row.get("score", 0.0))
    label = row.get("class_name", "bicycle")

    if bbox is None:
        draw.text((12, 12), "no bicycle detected", fill=(255, 80, 80))
        return image

    x1, y1, x2, y2 = [float(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 80), width=3)
    caption = f"{label} {score:.2f}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text_y = max(0, y1 - 18)
    draw.rectangle([x1, text_y, x1 + 8 * len(caption), text_y + 16], fill=(0, 180, 60))
    draw.text((x1 + 2, text_y + 1), caption, fill=(0, 0, 0), font=font)
    return image


def draw_keypoints_2d_frame(row: dict):
    from PIL import Image, ImageDraw

    image = Image.open(row["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [float(v) for v in row["bbox_xyxy"]]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 100), width=2)

    kps = row["keypoints_2d"]
    conf = row.get("confidence", [1.0] * len(kps))
    points = [(float(p[0]), float(p[1])) for p in kps]

    for i, j in _skeleton_edges():
        if i >= len(points) or j >= len(points):
            continue
        ci = float(conf[i]) if i < len(conf) else 0.0
        cj = float(conf[j]) if j < len(conf) else 0.0
        if ci < CONF_THRESHOLD or cj < CONF_THRESHOLD:
            continue
        draw.line([*points[i], *points[j]], fill=(0, 200, 255), width=2)

    for idx, (x, y) in enumerate(points):
        c = float(conf[idx]) if idx < len(conf) else 0.0
        if c < CONF_THRESHOLD:
            continue
        r = 4
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 90, 0), outline=(255, 255, 255))

    det_score = float(row.get("det_score", 0.0))
    draw.text((12, 12), f"det {det_score:.2f}", fill=(240, 240, 240))
    return image


def write_frame_sequence(
    rows: Iterable[dict],
    vis_dir: Path,
    draw_fn,
    *,
    resume: bool,
) -> int:
    vis_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        frame_id = int(row["frame_id"])
        out_path = _frame_path(vis_dir, frame_id)
        if resume and out_path.is_file():
            count += 1
            continue
        image = draw_fn(row)
        image.save(out_path, quality=92)
        count += 1
    return count


def _assemble_mp4_ffmpeg(frames_dir: Path, output_mp4: Path, fps: int) -> Path | None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return None

    pattern = str(frames_dir / "frame_%06d.jpg")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)
    return output_mp4


def _assemble_mp4_imageio(frames_dir: Path, output_mp4: Path, fps: int) -> Path | None:
    import imageio.v2 as imageio

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_mp4, fps=fps)
    for path in frame_files:
        writer.append_data(imageio.imread(path))
    writer.close()
    return output_mp4


def assemble_mp4(frames_dir: Path, output_mp4: Path, fps: int) -> Path | None:
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        print(f"[vis] No frames in {frames_dir}, skipping video.")
        return None

    try:
        return _assemble_mp4_ffmpeg(frames_dir, output_mp4, fps)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[vis] ffmpeg failed ({exc}); trying imageio.")

    try:
        return _assemble_mp4_imageio(frames_dir, output_mp4, fps)
    except ImportError:
        pass

    print("[vis] Install imageio or ffmpeg to build MP4 videos.")
    return None


def visualize_detections(
    output_dir: Path,
    *,
    resume: bool = False,
    write_video: bool = True,
    fps: int = 30,
) -> dict:
    detections_path = output_dir / DETECTIONS_NAME
    if not detections_path.is_file():
        raise FileNotFoundError(f"Missing {detections_path}")

    vis_root = output_dir / VIS_DETECTIONS_DIR
    frames_dir = vis_root / "frames"
    rows = list(iter_jsonl(detections_path))
    n = write_frame_sequence(rows, frames_dir, draw_detection_frame, resume=resume)
    print(f"[vis:detections] Wrote {n} frames to {frames_dir}")

    video_path = None
    if write_video:
        video_path = assemble_mp4(frames_dir, vis_root / "detections.mp4", fps)
        if video_path:
            print(f"[vis:detections] Wrote {video_path}")
    return {"frames_dir": str(frames_dir), "video": str(video_path) if video_path else None}


def visualize_keypoints_2d(
    output_dir: Path,
    *,
    resume: bool = False,
    write_video: bool = True,
    fps: int = 30,
) -> dict:
    keypoints_path = output_dir / KEYPOINTS_2D_NAME
    if not keypoints_path.is_file():
        raise FileNotFoundError(f"Missing {keypoints_path}")

    vis_root = output_dir / VIS_KEYPOINTS_2D_DIR
    frames_dir = vis_root / "frames"
    rows = list(iter_jsonl(keypoints_path))
    n = write_frame_sequence(rows, frames_dir, draw_keypoints_2d_frame, resume=resume)
    print(f"[vis:2d] Wrote {n} frames to {frames_dir}")

    video_path = None
    if write_video:
        video_path = assemble_mp4(frames_dir, vis_root / "keypoints_2d.mp4", fps)
        if video_path:
            print(f"[vis:2d] Wrote {video_path}")
    return {"frames_dir": str(frames_dir), "video": str(video_path) if video_path else None}


def visualize_keypoints_3d(
    output_dir: Path,
    *,
    write_video: bool = True,
    fps: int = 30,
    elev: float = 20.0,
    azim: float = -70.0,
    python_exe: str | None = None,
) -> dict:
    npz_path = output_dir / KEYPOINTS_3D_NAME
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")

    vis_root = output_dir / VIS_KEYPOINTS_3D_DIR
    vis_root.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "data_generation_pipeline_tools" / "visualize_bicycle_pose3d.py"
    exe = python_exe or sys.executable

    cmd = [
        exe,
        str(script),
        "--pred",
        str(npz_path.resolve()),
        "--npz-key",
        "pred",
        "--out",
        str(vis_root.resolve()),
        "--layout",
        "overlay",
        "--reorient",
        "camera_up",
        "--fps",
        str(fps),
        "--elev",
        str(elev),
        "--azim",
        str(azim),
    ]
    if write_video:
        cmd.append("--video")

    print(f"[vis:3d] Running {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    video_path = vis_root.with_suffix(".mp4")
    return {
        "frames_dir": str(vis_root),
        "video": str(video_path) if video_path.is_file() else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize full pipeline intermediates.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("detections", "2d", "3d", "all"),
        default=["all"],
        help="Which visualizations to run (default: all).",
    )
    parser.add_argument("--no-video", action="store_true", help="Write frame PNGs only.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resume", action="store_true", help="Skip existing detection/2d frames.")
    parser.add_argument(
        "--python-3d",
        default=None,
        help="Python executable for 3D viz (default: current; use posemamba env python).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    stages = set(args.stages)
    if "all" in stages:
        stages = {"detections", "2d", "3d"}

    write_video = not args.no_video

    if "detections" in stages:
        visualize_detections(output_dir, resume=args.resume, write_video=write_video, fps=args.fps)
    if "2d" in stages:
        visualize_keypoints_2d(output_dir, resume=args.resume, write_video=write_video, fps=args.fps)
    if "3d" in stages:
        visualize_keypoints_3d(
            output_dir,
            write_video=write_video,
            fps=args.fps,
            python_exe=args.python_3d,
        )

    print(f"[vis] Done. Outputs under {output_dir / 'vis'}")


if __name__ == "__main__":
    main()

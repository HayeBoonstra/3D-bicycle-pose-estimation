import os
import json
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw


def overlay_keypoints(
    image_path: Path,
    json_path: Path,
    output_path: Path,
    dot_radius: int = 4,
) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for kp in data.get("keypoints", []):
        if not kp.get("visible_in_frame", True):
            continue

        x = float(kp["x"])
        y = float(kp["y"])
        draw.ellipse(
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            fill=(255, 0, 0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _frame_index_from_name(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"No frame index found in filename: {path.name}")
    return int(match.group(1))


def _overlay_frame_task(task: tuple[str, str, str, int]) -> int:
    image_path_str, json_path_str, output_path_str, dot_radius = task
    image_path = Path(image_path_str)
    json_path = Path(json_path_str)
    output_path = Path(output_path_str)
    overlay_keypoints(
        image_path=image_path,
        json_path=json_path,
        output_path=output_path,
        dot_radius=dot_radius,
    )
    return 1


def build_overlay_video(
    raw_images_dir: Path,
    annotations_dir: Path,
    overlay_frames_dir: Path,
    output_video_path: Path,
    dot_radius: int = 4,
    fps: int = 60,
    max_workers: int | None = None,
) -> None:
    raw_images = sorted(raw_images_dir.glob("*.png"), key=_frame_index_from_name)
    if not raw_images:
        raise RuntimeError(f"No PNG files found in raw images folder: {raw_images_dir}")

    overlay_frames_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, str, int]] = []
    for image_path in raw_images:
        frame_idx = _frame_index_from_name(image_path)
        json_path = annotations_dir / f"keypoints_2d_frame_{frame_idx:04d}.json"

        if not json_path.exists():
            print(f"Skipping frame {frame_idx:04d}: missing annotation {json_path.name}")
            continue

        overlay_image_path = overlay_frames_dir / f"frame{frame_idx:04d}.png"
        tasks.append(
            (
                str(image_path),
                str(json_path),
                str(overlay_image_path),
                dot_radius,
            )
        )

    if not tasks:
        raise RuntimeError("No overlay frames were generated. Check annotation/image filenames.")

    worker_count = max_workers or (os.cpu_count() or 1)
    worker_count = max(1, worker_count)
    print(f"Rendering {len(tasks)} overlay frames using {worker_count} workers...")
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        processed = sum(executor.map(_overlay_frame_task, tasks, chunksize=8))

    print(f"Finished overlay rendering for {processed} frames.")

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        ffmpeg_exe,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(overlay_frames_dir / "frame%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video_path),
    ]

    print("Encoding MP4 with ffmpeg...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Saved overlay video: {output_video_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    raw_images_dir = repo_root / "2D annotated videos" / "raw images"
    annotations_dir = repo_root / "2D annotated videos" / "2D annotations"
    overlay_frames_dir = repo_root / "2D annotated videos" / "overlay frames"
    output_video_path = repo_root / "2D annotated videos" / "keypoints_overlay.mp4"

    build_overlay_video(
        raw_images_dir=raw_images_dir,
        annotations_dir=annotations_dir,
        overlay_frames_dir=overlay_frames_dir,
        output_video_path=output_video_path,
        dot_radius=4,
        fps=60,
        max_workers=None,
    )

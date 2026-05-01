"""Visualize COCO-format bicycle keypoint annotations as overlay frames or MP4."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "bicycle_pose_dataset"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw keypoint overlays from COCO annotations.")
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATASET_DIR / "overlays")
    parser.add_argument("--clip-id", help="Optional clip id to visualize.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--encode-video", action="store_true")
    parser.add_argument("--dot-radius", type=int, default=4)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _category(coco: dict[str, Any]) -> dict[str, Any]:
    if not coco.get("categories"):
        raise ValueError("COCO file does not contain category metadata.")
    return coco["categories"][0]


def _draw_annotation(
    image_path: Path,
    out_path: Path,
    annotation: dict[str, Any],
    skeleton: list[list[int]],
    *,
    dot_radius: int,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    keypoints = annotation["keypoints"]

    points: dict[int, tuple[float, float]] = {}
    for idx in range(0, len(keypoints), 3):
        kp_idx = idx // 3 + 1
        x, y, visibility = keypoints[idx], keypoints[idx + 1], int(keypoints[idx + 2])
        if visibility > 0:
            points[kp_idx] = (float(x), float(y))

    for start, end in skeleton:
        if start in points and end in points:
            draw.line([points[start], points[end]], fill=(0, 255, 0), width=2)

    for x, y in points.values():
        draw.ellipse(
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            fill=(255, 0, 0),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _encode_clip(frames_dir: Path, output_video: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = _parse_args()
    coco = _load_json(args.coco)
    category = _category(coco)
    skeleton = category.get("skeleton", [])
    images = {image["id"]: image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    images_to_draw = [
        image
        for image in images.values()
        if args.clip_id is None or image.get("clip_id") == args.clip_id
    ]
    images_to_draw.sort(key=lambda item: (item.get("clip_id", ""), item.get("frame_index", 0)))

    for image in images_to_draw:
        image_path = args.dataset_dir / image["file_name"]
        clip_id = image.get("clip_id", "unknown_clip")
        frame_index = int(image.get("frame_index", image["id"]))
        out_path = args.out / clip_id / f"frame_{frame_index:04d}.png"
        anns = annotations_by_image.get(int(image["id"]), [])
        if not anns:
            continue
        _draw_annotation(
            image_path,
            out_path,
            anns[0],
            skeleton,
            dot_radius=args.dot_radius,
        )

    if args.encode_video:
        clip_ids = sorted({image.get("clip_id", "unknown_clip") for image in images_to_draw})
        for clip_id in clip_ids:
            _encode_clip(args.out / clip_id, args.out / f"{clip_id}.mp4", args.fps)

    print(f"Wrote overlays to {args.out}")


if __name__ == "__main__":
    main()

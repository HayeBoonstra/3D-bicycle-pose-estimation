"""Convert raw Blender keypoint exports into COCO keypoints datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bicycle_keypoint_schema import (
    BICYCLE_KEYPOINT_NAMES,
    CATEGORY_ID,
    coco_category,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_RENDERS_DIR = REPO_ROOT / "raw_renders"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "bicycle_pose_dataset"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw renders to COCO keypoints JSON.")
    parser.add_argument("--raw-renders", type=Path, default=DEFAULT_RAW_RENDERS_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_DATASET_DIR / "splits.json")
    parser.add_argument(
        "--outside-visibility",
        choices=["occluded", "unlabeled"],
        default="occluded",
        help="COCO flag for points in front of the camera but outside the image.",
    )
    parser.add_argument("--link", action="store_true", help="Hardlink images instead of copying.")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_or_link(src: Path, dst: Path, *, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _source_frame_path(clip_dir: Path, annotation: dict[str, Any]) -> Path:
    image_file = Path(annotation["image_file"])
    candidates = [
        clip_dir / image_file,
        clip_dir / "frames" / f"frame_{int(annotation['frame']):04d}.png",
        clip_dir / "frames" / f"frame{int(annotation['frame']):04d}.png",
        clip_dir / "frames" / f"{int(annotation['frame']):04d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find rendered frame for annotation {annotation}")


def _keypoint_visibility(kp: dict[str, Any], outside_policy: str) -> int:
    if kp.get("missing"):
        return 0
    if kp.get("visible_in_frame"):
        return 2
    if kp.get("in_front_of_camera", float(kp.get("z_cam", 0.0)) > 0.0):
        return 1 if outside_policy == "occluded" else 0
    return 0


def _flatten_keypoints(annotation: dict[str, Any], outside_policy: str) -> tuple[list[float], int]:
    by_name = {kp["name"]: kp for kp in annotation.get("keypoints", [])}
    flattened: list[float] = []
    num_keypoints = 0
    for name in BICYCLE_KEYPOINT_NAMES:
        kp = by_name.get(name)
        if kp is None:
            flattened.extend([0.0, 0.0, 0])
            continue
        visibility = _keypoint_visibility(kp, outside_policy)
        x = float(kp["x"]) if visibility else 0.0
        y = float(kp["y"]) if visibility else 0.0
        flattened.extend([x, y, visibility])
        if visibility > 0:
            num_keypoints += 1
    return flattened, num_keypoints


def _bbox_from_keypoints(flattened: list[float], width: int, height: int) -> list[float]:
    visible_xy: list[tuple[float, float]] = []
    labeled_xy: list[tuple[float, float]] = []
    for idx in range(0, len(flattened), 3):
        x, y, visibility = flattened[idx], flattened[idx + 1], int(flattened[idx + 2])
        if visibility > 0:
            labeled_xy.append((x, y))
        if visibility == 2:
            visible_xy.append((x, y))

    xy = visible_xy or labeled_xy
    if not xy:
        return [0.0, 0.0, 1.0, 1.0]

    xs = [min(max(x, 0.0), float(width - 1)) for x, _ in xy]
    ys = [min(max(y, 0.0), float(height - 1)) for _, y in xy]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox_w = max(1.0, x_max - x_min)
    bbox_h = max(1.0, y_max - y_min)
    margin_x = bbox_w * 0.1
    margin_y = bbox_h * 0.1
    x_min = max(0.0, x_min - margin_x)
    y_min = max(0.0, y_min - margin_y)
    x_max = min(float(width - 1), x_max + margin_x)
    y_max = min(float(height - 1), y_max + margin_y)
    return [x_min, y_min, max(1.0, x_max - x_min), max(1.0, y_max - y_min)]


def _copy_metadata(clip_dir: Path, dataset_dir: Path, clip_id: str) -> None:
    metadata_dir = dataset_dir / "metadata" / clip_id
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["camera.json", "keypoints_3d.jsonl", "render_config.json"]:
        src = clip_dir / filename
        if src.exists():
            shutil.copy2(src, metadata_dir / filename)

    for video in clip_dir.glob("*.mp4"):
        dst = dataset_dir / "videos" / video.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, dst)


def _build_split_coco(
    split_name: str,
    clip_ids: list[str],
    *,
    raw_renders_dir: Path,
    dataset_dir: Path,
    outside_policy: str,
    link: bool,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    annotation_id = 1

    for clip_id in clip_ids:
        clip_dir = raw_renders_dir / clip_id
        render_config = _load_json(clip_dir / "render_config.json")
        annotation_files = sorted((clip_dir / "per_frame_annotations").glob("*.json"))
        if not annotation_files:
            raise RuntimeError(f"No per-frame annotations found for clip {clip_id}")
        _copy_metadata(clip_dir, dataset_dir, clip_id)

        for annotation_file in annotation_files:
            frame_annotation = _load_json(annotation_file)
            frame_index = int(frame_annotation.get("frame_index", frame_annotation["frame"]))
            width = int(frame_annotation["image_width"])
            height = int(frame_annotation["image_height"])
            src_frame = _source_frame_path(clip_dir, frame_annotation)
            dst_rel = Path("images") / split_name / clip_id / f"frame_{frame_index:04d}.png"
            dst_frame = dataset_dir / dst_rel
            _copy_or_link(src_frame, dst_frame, link=link)

            keypoints, num_keypoints = _flatten_keypoints(frame_annotation, outside_policy)
            bbox = _bbox_from_keypoints(keypoints, width, height)
            images.append(
                {
                    "id": image_id,
                    "file_name": str(dst_rel).replace(os.sep, "/"),
                    "width": width,
                    "height": height,
                    "clip_id": clip_id,
                    "scene_id": render_config.get("scene_id", ""),
                    "frame_index": frame_index,
                    "frame": int(frame_annotation["frame"]),
                    "fps": int(render_config.get("fps", 0)),
                }
            )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID,
                    "keypoints": keypoints,
                    "num_keypoints": num_keypoints,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            image_id += 1
            annotation_id += 1

    return {
        "info": {
            "description": "Synthetic bicycle keypoint dataset",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [coco_category()],
    }


def main() -> None:
    args = _parse_args()
    splits = _load_json(args.splits)
    annotations_dir = args.dataset_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    split_copy_path = args.dataset_dir / "splits.json"
    if args.splits.resolve() != split_copy_path.resolve():
        shutil.copy2(args.splits, split_copy_path)

    for split_name in ["train", "val", "test"]:
        coco = _build_split_coco(
            split_name,
            splits.get(split_name, []),
            raw_renders_dir=args.raw_renders,
            dataset_dir=args.dataset_dir,
            outside_policy=args.outside_visibility,
            link=args.link,
        )
        out_path = annotations_dir / f"{split_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
        print(
            f"Wrote {out_path}: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations"
        )


if __name__ == "__main__":
    main()

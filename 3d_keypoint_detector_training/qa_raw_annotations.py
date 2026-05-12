"""QA checks for raw bicycle annotation folders.

Works for both Blender-exported clips and MuJoCo-direct clips that follow the
same raw annotation contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES
from keypoint_detector_pipeline.world_transform import camera_from_json, project_to_image, reprojection_rmse


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clip_dirs(raw_root: Path) -> list[Path]:
    return [path for path in sorted(raw_root.iterdir()) if path.is_dir() and (path / "keypoints_3d.jsonl").exists()]


def _points_from_2d(row: dict) -> np.ndarray:
    by_name = {kp["name"]: kp for kp in row["keypoints"]}
    points = np.zeros((len(BICYCLE_KEYPOINT_NAMES), 2), dtype=np.float32)
    missing = sorted(set(BICYCLE_KEYPOINT_NAMES) - set(by_name))
    if missing:
        raise ValueError(f"2D annotation missing keypoints: {missing}")
    for idx, name in enumerate(BICYCLE_KEYPOINT_NAMES):
        points[idx] = [float(by_name[name]["x"]), float(by_name[name]["y"])]
    return points


def _bbox_contains_visible(row: dict, tolerance: float = 1e-4) -> bool:
    bbox = row.get("gt_bbox_xywh")
    if bbox is None:
        return False
    x, y, w, h = [float(v) for v in bbox]
    visible = [kp for kp in row["keypoints"] if int(kp.get("v", 0)) > 0]
    if not visible:
        return True
    for kp in visible:
        px = float(kp["x"])
        py = float(kp["y"])
        if px < x - tolerance or py < y - tolerance or px > x + w + tolerance or py > y + h + tolerance:
            return False
    return True


def qa_clip(clip_dir: Path, max_reprojection_rmse: float) -> dict:
    camera_path = clip_dir / "camera.json"
    k3d_path = clip_dir / "keypoints_3d.jsonl"
    annotation_dir = clip_dir / "per_frame_annotations"
    if not camera_path.exists() or not k3d_path.exists() or not annotation_dir.exists():
        raise FileNotFoundError(f"Clip is missing required raw annotation files: {clip_dir}")

    camera_payload = _load_json(camera_path)
    camera = camera_from_json(camera_payload)
    rows3d = _load_jsonl(k3d_path)
    rows2d = sorted(annotation_dir.glob("keypoints_2d_frame_*.json"))
    if len(rows3d) != len(rows2d):
        raise ValueError(f"{clip_dir.name}: 2D/3D frame count mismatch ({len(rows2d)} vs {len(rows3d)})")

    rmses: list[float] = []
    min_depth = float("inf")
    bbox_failures = 0
    for row3d, ann_path in zip(rows3d, rows2d):
        if list(row3d.get("joint_names", [])) != BICYCLE_KEYPOINT_NAMES:
            raise ValueError(f"{clip_dir.name}: joint_names do not match canonical bicycle schema")
        points_cam = np.asarray(row3d["kps_camera"], dtype=np.float32)
        points_2d = _points_from_2d(_load_json(ann_path))
        projected = project_to_image(points_cam, camera)
        rmses.append(reprojection_rmse(points_cam, points_2d, camera))
        min_depth = min(min_depth, float(np.min(points_cam[:, 2])))
        if not _bbox_contains_visible(_load_json(ann_path)):
            bbox_failures += 1

        max_abs = float(np.max(np.abs(projected - points_2d)))
        if max_abs > max(1e-3, max_reprojection_rmse * 3.0):
            raise ValueError(f"{clip_dir.name}: projection mismatch at {ann_path.name}: max_abs={max_abs:.6f}")

    max_rmse = float(max(rmses) if rmses else 0.0)
    if max_rmse > max_reprojection_rmse:
        raise ValueError(f"{clip_dir.name}: reprojection RMSE {max_rmse:.6f} exceeds {max_reprojection_rmse}")
    if bbox_failures:
        raise ValueError(f"{clip_dir.name}: {bbox_failures} frames have bboxes that do not enclose visible keypoints")

    return {
        "clip_id": clip_dir.name,
        "frames": len(rows3d),
        "max_reprojection_rmse": max_rmse,
        "min_camera_depth": min_depth,
    }


def run_qa(raw_root: Path, max_reprojection_rmse: float) -> dict:
    clips = _clip_dirs(raw_root)
    if not clips:
        raise RuntimeError(f"No raw annotation clips found in {raw_root}")
    results = [qa_clip(path, max_reprojection_rmse) for path in clips]
    return {
        "raw_root": str(raw_root),
        "num_clips": len(results),
        "total_frames": int(sum(item["frames"] for item in results)),
        "max_reprojection_rmse": float(max(item["max_reprojection_rmse"] for item in results)),
        "clips": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw bicycle 2D/3D annotation clips.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--max-reprojection-rmse", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_qa(args.raw_root, args.max_reprojection_rmse), indent=2))


if __name__ == "__main__":
    main()

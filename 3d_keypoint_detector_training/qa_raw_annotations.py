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
from keypoint_detector_pipeline.world_transform import (
    camera_3d_consistency_rmse,
    camera_from_json,
    reprojection_rmse,
)


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


# Mesh-derived gt_bbox vs keypoint empties can disagree by sub-pixel amounts; 1 px is enough.
_BBOX_CONTAINMENT_TOLERANCE_PX = 1.0


def _bbox_contains_visible(row: dict, tolerance: float = _BBOX_CONTAINMENT_TOLERANCE_PX) -> bool:
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


def _bbox_area_fraction(row: dict) -> float | None:
    bbox = row.get("gt_bbox_xywh")
    if bbox is None:
        return None
    width = float(row.get("image_width", 0) or 0)
    height = float(row.get("image_height", 0) or 0)
    if width <= 0 or height <= 0:
        return None
    _x, _y, w, h = [float(v) for v in bbox]
    return max(0.0, w * h) / max(1.0, width * height)


def _bbox_framing_stats(bbox_area_fracs: list[float], min_gt_bbox_area_frac: float) -> tuple[float, float, float]:
    if not bbox_area_fracs:
        return 0.0, 0.0, 0.0
    arr = np.asarray(bbox_area_fracs, dtype=np.float64)
    low_frac = float(np.mean(arr < min_gt_bbox_area_frac)) if min_gt_bbox_area_frac > 0.0 else 0.0
    return float(np.min(arr)), float(np.mean(arr)), low_frac


def qa_clip(
    clip_dir: Path,
    max_camera_3d_rmse: float,
    max_reprojection_rmse: float,
    *,
    min_gt_bbox_area_frac: float,
    min_gt_bbox_hard_floor: float,
    max_low_bbox_frame_frac: float,
) -> dict:
    camera_path = clip_dir / "camera.json"
    k3d_path = clip_dir / "keypoints_3d.jsonl"
    annotation_dir = clip_dir / "per_frame_annotations"
    if not camera_path.exists() or not k3d_path.exists() or not annotation_dir.exists():
        raise FileNotFoundError(f"Clip is missing required raw annotation files: {clip_dir}")

    camera_payload = _load_json(camera_path)
    fallback_camera = camera_from_json(camera_payload)
    rows3d = _load_jsonl(k3d_path)
    rows2d = sorted(annotation_dir.glob("keypoints_2d_frame_*.json"))
    if len(rows3d) != len(rows2d):
        raise ValueError(f"{clip_dir.name}: 2D/3D frame count mismatch ({len(rows2d)} vs {len(rows3d)})")

    camera_3d_rmses: list[float] = []
    pinhole_2d_rmses: list[float] = []
    min_depth = float("inf")
    bbox_failures = 0
    bbox_area_fracs: list[float] = []
    missing_dynamics = 0
    for row3d, ann_path in zip(rows3d, rows2d):
        if list(row3d.get("joint_names", [])) != BICYCLE_KEYPOINT_NAMES:
            raise ValueError(f"{clip_dir.name}: joint_names do not match canonical bicycle schema")
        dynamics = row3d.get("dynamics_gt")
        if not isinstance(dynamics, dict):
            missing_dynamics += 1
        else:
            for key in ("steer_deg", "roll_deg"):
                if key not in dynamics:
                    raise ValueError(f"{clip_dir.name}: dynamics_gt missing {key}")
        if "K" in row3d and "R" in row3d and "t" in row3d:
            camera = camera_from_json(row3d)
        else:
            camera = fallback_camera
        points_world = np.asarray(row3d["kps_world"], dtype=np.float32)
        points_cam = np.asarray(row3d["kps_camera"], dtype=np.float32)
        points_2d = _points_from_2d(_load_json(ann_path))
        camera_3d_rmses.append(camera_3d_consistency_rmse(points_world, points_cam, camera))
        pinhole_2d_rmses.append(reprojection_rmse(points_cam, points_2d, camera))
        min_depth = min(min_depth, float(np.min(points_cam[:, 2])))
        ann_row = _load_json(ann_path)
        if not _bbox_contains_visible(ann_row):
            bbox_failures += 1
        area_frac = _bbox_area_fraction(ann_row)
        if area_frac is not None:
            bbox_area_fracs.append(area_frac)
    if missing_dynamics:
        raise ValueError(
            f"{clip_dir.name}: {missing_dynamics}/{len(rows3d)} frames lack dynamics_gt in keypoints_3d.jsonl"
        )

    max_camera_3d = float(max(camera_3d_rmses) if camera_3d_rmses else 0.0)
    if max_camera_3d > max_camera_3d_rmse:
        raise ValueError(
            f"{clip_dir.name}: camera 3D consistency RMSE {max_camera_3d:.6f} exceeds {max_camera_3d_rmse}"
        )

    max_pinhole_2d = float(max(pinhole_2d_rmses) if pinhole_2d_rmses else 0.0)
    # MuJoCo exports pinhole-consistent 2D (RMSE ~0). Blender uses world_to_camera_view (hundreds of px).
    pinhole_consistent = max_pinhole_2d < 1.0
    if pinhole_consistent and max_reprojection_rmse > 0.0 and max_pinhole_2d > max_reprojection_rmse:
        raise ValueError(
            f"{clip_dir.name}: pinhole 2D reprojection RMSE {max_pinhole_2d:.6f} exceeds {max_reprojection_rmse}"
        )
    if bbox_failures:
        raise ValueError(f"{clip_dir.name}: {bbox_failures} frames have bboxes that do not enclose visible keypoints")

    min_bbox_frac, mean_bbox_frac, low_bbox_frame_frac = _bbox_framing_stats(
        bbox_area_fracs, min_gt_bbox_area_frac
    )
    if min_gt_bbox_hard_floor > 0.0 and min_bbox_frac < min_gt_bbox_hard_floor:
        raise ValueError(
            f"{clip_dir.name}: gt_bbox area below hard floor for RTMPose "
            f"(min={min_bbox_frac:.4f}, floor={min_gt_bbox_hard_floor:.4f}). "
            "Re-render with tighter CAMERA_MAX_DISTANCE or higher CAMERA_MIN_BBOX_AREA_FRAC."
        )
    if (
        min_gt_bbox_area_frac > 0.0
        and max_low_bbox_frame_frac >= 0.0
        and low_bbox_frame_frac > max_low_bbox_frame_frac
    ):
        raise ValueError(
            f"{clip_dir.name}: too many small gt_bbox frames for RTMPose "
            f"({low_bbox_frame_frac:.1%} below {min_gt_bbox_area_frac:.4f}, "
            f"max allowed {max_low_bbox_frame_frac:.1%}). "
            "Re-render with tighter CAMERA_MAX_DISTANCE or enable multi-frame bbox checks."
        )

    return {
        "clip_id": clip_dir.name,
        "frames": len(rows3d),
        "max_camera_3d_rmse": max_camera_3d,
        "max_pinhole_2d_rmse": max_pinhole_2d,
        "pinhole_2d_consistent": pinhole_consistent,
        "min_camera_depth": min_depth,
        "min_gt_bbox_area_frac": min_bbox_frac,
        "mean_gt_bbox_area_frac": mean_bbox_frac,
        "low_bbox_frame_frac": low_bbox_frame_frac,
    }


def _is_bbox_framing_error(exc: BaseException) -> bool:
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc)
    return "gt_bbox" in msg or "bboxes that do not enclose" in msg


def run_qa(
    raw_root: Path,
    max_camera_3d_rmse: float,
    max_reprojection_rmse: float,
    min_gt_bbox_area_frac: float,
    min_gt_bbox_hard_floor: float,
    max_low_bbox_frame_frac: float,
    *,
    allow_bbox_framing_failures: bool = False,
) -> dict:
    clips = _clip_dirs(raw_root)
    if not clips:
        raise RuntimeError(f"No raw annotation clips found in {raw_root}")
    results: list[dict] = []
    bbox_framing_failures: list[dict[str, str]] = []
    for path in clips:
        try:
            results.append(
                qa_clip(
                    path,
                    max_camera_3d_rmse,
                    max_reprojection_rmse,
                    min_gt_bbox_area_frac=min_gt_bbox_area_frac,
                    min_gt_bbox_hard_floor=min_gt_bbox_hard_floor,
                    max_low_bbox_frame_frac=max_low_bbox_frame_frac,
                )
            )
        except ValueError as exc:
            if allow_bbox_framing_failures and _is_bbox_framing_error(exc):
                bbox_framing_failures.append({"clip_id": path.name, "error": str(exc)})
                continue
            raise
    if not results and bbox_framing_failures:
        raise RuntimeError(f"All {len(bbox_framing_failures)} clips failed bbox framing QA in {raw_root}")
    return {
        "raw_root": str(raw_root),
        "num_clips": len(results),
        "num_bbox_framing_failures": len(bbox_framing_failures),
        "total_frames": int(sum(item["frames"] for item in results)),
        "max_camera_3d_rmse": float(max(item["max_camera_3d_rmse"] for item in results)) if results else 0.0,
        "max_pinhole_2d_rmse": float(max(item["max_pinhole_2d_rmse"] for item in results)) if results else 0.0,
        "clips": results,
        "bbox_framing_failures": bbox_framing_failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw bicycle 2D/3D annotation clips.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--max-camera-3d-rmse",
        type=float,
        default=1e-3,
        help="Max RMSE between kps_camera and R,t @ kps_world (OpenCV Z auto-detected).",
    )
    parser.add_argument(
        "--max-reprojection-rmse",
        type=float,
        default=1e-3,
        help="Max pinhole 2D RMSE; only enforced when export is pinhole-consistent (MuJoCo).",
    )
    parser.add_argument(
        "--min-gt-bbox-area-frac",
        type=float,
        default=0.0,
        help="Target gt_bbox area fraction (RTMPose framing QA).",
    )
    parser.add_argument(
        "--min-gt-bbox-hard-floor",
        type=float,
        default=0.025,
        help="Always fail when any frame gt_bbox area is below this fraction.",
    )
    parser.add_argument(
        "--max-low-bbox-frame-frac",
        type=float,
        default=0.15,
        help="Max fraction of frames allowed below --min-gt-bbox-area-frac (track camera dips).",
    )
    parser.add_argument(
        "--allow-bbox-framing-failures",
        action="store_true",
        help="Warn and skip clips that fail bbox framing QA instead of aborting the batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_qa(
        args.raw_root,
        args.max_camera_3d_rmse,
        args.max_reprojection_rmse,
        args.min_gt_bbox_area_frac,
        args.min_gt_bbox_hard_floor,
        args.max_low_bbox_frame_frac,
        allow_bbox_framing_failures=args.allow_bbox_framing_failures,
    )
    if report.get("bbox_framing_failures"):
        print(
            f"[qa_raw_annotations] skipped {len(report['bbox_framing_failures'])} clip(s) with bbox framing issues",
            file=sys.stderr,
        )
        for item in report["bbox_framing_failures"]:
            print(f"  - {item['clip_id']}: {item['error']}", file=sys.stderr)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

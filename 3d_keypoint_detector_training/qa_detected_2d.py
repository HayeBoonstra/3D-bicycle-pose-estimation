#!/usr/bin/env python3
"""QA RTMPose detected 2D sidecars against GT 2D and optional 3D reprojection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES, KEYPOINT_INDEX
from keypoint_detector_pipeline.world_transform import camera_from_json, project_to_image, reprojection_rmse

DETECTED_PREFIX = "keypoints_2d_detected_frame_"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clip_dirs(raw_root: Path) -> list[Path]:
    return [p for p in sorted(raw_root.iterdir()) if p.is_dir() and (p / "keypoints_3d.jsonl").exists()]


def _points_from_row(row: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((len(BICYCLE_KEYPOINT_NAMES), 2), dtype=np.float32)
    conf = np.zeros((len(BICYCLE_KEYPOINT_NAMES),), dtype=np.float32)
    for kp in row["keypoints"]:
        j = KEYPOINT_INDEX[kp["name"]]
        points[j] = [float(kp["x"]), float(kp["y"])]
        if "det_score" in kp:
            conf[j] = float(kp["det_score"])
        else:
            conf[j] = float(kp.get("v", 0)) / 2.0
    return points, conf


def qa_clip(clip_dir: Path) -> dict:
    annotation_dir = clip_dir / "per_frame_annotations"
    camera = camera_from_json(_load_json(clip_dir / "camera.json"))
    gt_paths = sorted(annotation_dir.glob("keypoints_2d_frame_*.json"))
    det_paths = sorted(annotation_dir.glob(f"{DETECTED_PREFIX}*.json"))
    if not det_paths:
        return {
            "clip_id": clip_dir.name,
            "status": "missing_detected",
            "detected_frames": 0,
        }

    pixel_rmses: list[float] = []
    reproj_rmses: list[float] = []
    mean_conf: list[float] = []

    det_by_idx = {int(_load_json(p)["frame_index"]): p for p in det_paths}
    rows3d = {}
    with (clip_dir / "keypoints_3d.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows3d[int(row["frame_index"])] = row

    for gt_path in gt_paths:
        gt_row = _load_json(gt_path)
        idx = int(gt_row["frame_index"])
        det_path = det_by_idx.get(idx)
        if det_path is None:
            continue
        det_row = _load_json(det_path)
        gt_pts, _ = _points_from_row(gt_row)
        det_pts, det_conf = _points_from_row(det_row)
        diff = det_pts - gt_pts
        pixel_rmses.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1)))))
        mean_conf.append(float(np.mean(det_conf)))

        row3d = rows3d.get(idx)
        if row3d is not None:
            kps_cam = row3d.get("kps_camera")
            if isinstance(kps_cam, list) and len(kps_cam) == len(BICYCLE_KEYPOINT_NAMES):
                world_pts = []
                for j, cam_pt in enumerate(kps_cam):
                    if cam_pt is None:
                        world_pts.append(None)
                    else:
                        world_pts.append(cam_pt)
                try:
                    reproj_rmses.append(
                        float(
                            reprojection_rmse(
                                world_pts,
                                det_pts,
                                camera,
                                image_size=(int(gt_row["image_width"]), int(gt_row["image_height"])),
                            )
                        )
                    )
                except Exception:
                    pass

    return {
        "clip_id": clip_dir.name,
        "status": "ok",
        "detected_frames": len(det_paths),
        "gt_frames": len(gt_paths),
        "coverage": len(det_by_idx) / max(1, len(gt_paths)),
        "mean_pixel_rmse": float(np.mean(pixel_rmses)) if pixel_rmses else 0.0,
        "mean_reproj_rmse": float(np.mean(reproj_rmses)) if reproj_rmses else 0.0,
        "mean_det_conf": float(np.mean(mean_conf)) if mean_conf else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA detected 2D sidecars in raw Blender/MuJoCo clips.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--max-mean-pixel-rmse", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_root = args.raw_root.resolve()
    clip_dirs = _clip_dirs(raw_root)
    if args.limit_clips is not None:
        clip_dirs = clip_dirs[: args.limit_clips]
    if not clip_dirs:
        print(f"[qa_detected_2d] No clips in {raw_root}")
        return

    summary = [qa_clip(path) for path in clip_dirs]
    out_path = raw_root / "detected_2d_qa_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ok = [row for row in summary if row.get("status") == "ok"]
    missing = [row for row in summary if row.get("status") == "missing_detected"]
    high_rmse = [row for row in ok if row.get("mean_pixel_rmse", 0.0) > args.max_mean_pixel_rmse]

    print(f"[qa_detected_2d] clips={len(summary)} ok={len(ok)} missing_detected={len(missing)}")
    if ok:
        print(
            f"  mean pixel RMSE (GT vs det): "
            f"{np.mean([r['mean_pixel_rmse'] for r in ok]):.2f}px"
        )
        print(
            f"  mean det conf: {np.mean([r['mean_det_conf'] for r in ok]):.3f}"
        )
    if high_rmse:
        print(f"[qa_detected_2d] WARNING: {len(high_rmse)} clips exceed pixel RMSE threshold")
    print(f"[qa_detected_2d] Wrote {out_path}")


if __name__ == "__main__":
    main()

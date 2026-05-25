#!/usr/bin/env python3
"""Run RF-DETR bbox + RTMPose top-down on Blender frames; write detected 2D sidecars."""

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
from keypoint_detector_pipeline.io_utils import iter_jsonl
from keypoint_detector_pipeline.pose2d_mmpose import MMPose2DInferencer

DEFAULT_MMPOSE_CONFIG = REPO_ROOT / "2d_keypoint_detector_training" / "rtmpose_bicycle_full.py"
DEFAULT_MMPOSE_CHECKPOINT = REPO_ROOT / "training_outputs" / "mmpose_bicycle_rtmpose_l_gpu" / "epoch_340.pth"
DETECTED_PREFIX = "keypoints_2d_detected_frame_"
DETECTIONS_NAME = "detections.jsonl"
BICYCLE_INPUT_SIZE = (256, 320)
CONF_VISIBLE = 0.3
SOURCE_BY_MODE = {
    "detection_bbox": "rtmpose_detection_bbox",
    "full_image": "rtmpose_full_image",
    "auto": "rtmpose_detection_bbox",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clip_dirs(raw_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(raw_root.iterdir())
        if path.is_dir() and (path / "keypoints_3d.jsonl").exists()
    ]


def _gt_frame_paths(annotation_dir: Path) -> list[Path]:
    return sorted(annotation_dir.glob("keypoints_2d_frame_*.json"))


def _frame_image_path(clip_dir: Path, gt_row: dict) -> Path:
    image_file = gt_row.get("image_file", "")
    if image_file:
        candidate = clip_dir / str(image_file)
        if candidate.is_file():
            return candidate
    frame_index = int(gt_row["frame_index"])
    frames_dir = clip_dir / "frames"
    candidate = frames_dir / f"frame_{frame_index:04d}.png"
    if candidate.is_file():
        return candidate
    matches = sorted(frames_dir.glob("*.png"))
    if frame_index < len(matches):
        return matches[frame_index]
    raise FileNotFoundError(f"No image for frame_index={frame_index} in {clip_dir}")


def _load_detections_by_frame(clip_dir: Path) -> dict[int, dict]:
    detections_path = clip_dir / DETECTIONS_NAME
    if not detections_path.is_file():
        raise FileNotFoundError(
            f"Missing {detections_path}. Run export_clip_detections.py in the rfdetr conda env first."
        )
    by_frame: dict[int, dict] = {}
    for row in iter_jsonl(detections_path):
        frame_index = int(row.get("frame_index", row.get("frame_id", -1)))
        by_frame[frame_index] = row
    return by_frame


def _visibility_from_conf(conf: float) -> int:
    if conf >= CONF_VISIBLE:
        return 2
    if conf > 0.05:
        return 1
    return 0


def _build_detected_row(
    gt_row: dict,
    kps: np.ndarray,
    conf: np.ndarray,
    *,
    det_bbox_xyxy: list[float] | None,
    det_score: float,
    bbox_xyxy: list[float],
    pose_mode: str,
    source: str,
) -> dict:
    keypoints = []
    for name in BICYCLE_KEYPOINT_NAMES:
        j = KEYPOINT_INDEX[name]
        keypoints.append(
            {
                "name": name,
                "x": float(kps[j, 0]),
                "y": float(kps[j, 1]),
                "v": _visibility_from_conf(float(conf[j])),
                "det_score": float(conf[j]),
            }
        )
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return {
        "clip_id": gt_row.get("clip_id"),
        "scene_id": gt_row.get("scene_id"),
        "frame": gt_row.get("frame"),
        "frame_index": int(gt_row["frame_index"]),
        "image_width": gt_row.get("image_width"),
        "image_height": gt_row.get("image_height"),
        "camera": gt_row.get("camera"),
        "image_file": gt_row.get("image_file"),
        "gt_bbox_xywh": gt_row.get("gt_bbox_xywh"),
        "det_bbox_xyxy": det_bbox_xyxy,
        "det_score": det_score,
        "bbox_xyxy": bbox_xyxy,
        "bbox_xywh": [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)],
        "pose_mode": pose_mode,
        "source": source,
        "keypoints": keypoints,
    }


def _skeleton_span(kps: np.ndarray, conf: np.ndarray) -> tuple[float, float]:
    mask = conf >= CONF_VISIBLE
    if not np.any(mask):
        mask = conf > 0.05
    if not np.any(mask):
        return 0.0, 0.0
    pts = kps[mask]
    return float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))


def export_clip(
    clip_dir: Path,
    infer2d: MMPose2DInferencer,
    detections_by_frame: dict[int, dict],
    *,
    pose_mode: str,
    min_det_bbox_area_frac: float,
    resume: bool,
    limit_frames: int | None,
    expected_source: str,
) -> dict:
    annotation_dir = clip_dir / "per_frame_annotations"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations: {annotation_dir}")

    gt_paths = _gt_frame_paths(annotation_dir)
    if limit_frames is not None:
        gt_paths = gt_paths[:limit_frames]

    processed = 0
    skipped = 0
    missing_det = 0
    conf_vals: list[float] = []
    spans_x: list[float] = []
    spans_y: list[float] = []

    for gt_path in gt_paths:
        gt_row = _load_json(gt_path)
        frame_index = int(gt_row["frame_index"])
        out_path = annotation_dir / f"{DETECTED_PREFIX}{frame_index:04d}.json"
        if resume and out_path.is_file():
            existing = _load_json(out_path)
            if existing.get("source") == expected_source:
                skipped += 1
                continue

        image_path = _frame_image_path(clip_dir, gt_row)
        det_row = detections_by_frame.get(frame_index)
        if det_row is None:
            raise ValueError(f"Missing detection for frame_index={frame_index} in {clip_dir.name}")

        det_bbox = det_row.get("bbox_xyxy")
        if det_bbox is None:
            missing_det += 1

        kps, conf, bbox, mode_used = infer2d.predict_frame(
            image_path,
            det_bbox,
            mode=pose_mode,
            min_det_bbox_area_frac=min_det_bbox_area_frac,
        )
        detected_row = _build_detected_row(
            gt_row,
            kps,
            conf,
            det_bbox_xyxy=det_bbox,
            det_score=float(det_row.get("score", 0.0)),
            bbox_xyxy=bbox,
            pose_mode=mode_used,
            source=expected_source if mode_used != "full_image" else SOURCE_BY_MODE["full_image"],
        )
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(detected_row, f, indent=2)

        conf_vals.append(float(np.mean(conf)))
        sx, sy = _skeleton_span(kps, conf)
        spans_x.append(sx)
        spans_y.append(sy)
        processed += 1

    return {
        "clip_id": clip_dir.name,
        "processed": processed,
        "skipped": skipped,
        "missing_detections": missing_det,
        "mean_conf": float(np.mean(conf_vals)) if conf_vals else 0.0,
        "mean_span_x": float(np.mean(spans_x)) if spans_x else 0.0,
        "mean_span_y": float(np.mean(spans_y)) if spans_y else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RTMPose 2D detections for Blender raw clips.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mmpose-config", type=Path, default=DEFAULT_MMPOSE_CONFIG)
    parser.add_argument("--mmpose-checkpoint", type=Path, default=DEFAULT_MMPOSE_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pose-mode",
        choices=("auto", "full_image", "detection_bbox"),
        default="detection_bbox",
        help="Match 1_full_detection_pipeline stage2 (default: detection_bbox).",
    )
    parser.add_argument(
        "--min-det-bbox-area-frac",
        type=float,
        default=0.01,
        help="In auto mode, minimum detection bbox area fraction for top-down warp.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None, help="Per-clip frame cap for debugging.")
    parser.add_argument("--clip-id", default=None, help="Process a single clip directory name.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_root = args.raw_root.resolve()
    if not args.mmpose_config.is_file():
        raise FileNotFoundError(f"MMPose config not found: {args.mmpose_config}")
    if not args.mmpose_checkpoint.is_file():
        raise FileNotFoundError(f"MMPose checkpoint not found: {args.mmpose_checkpoint}")

    expected_source = SOURCE_BY_MODE.get(args.pose_mode, "rtmpose_detection_bbox")

    infer2d = MMPose2DInferencer(
        pose2d_model=str(args.mmpose_config.resolve()),
        pose2d_weights=str(args.mmpose_checkpoint.resolve()),
        input_size=BICYCLE_INPUT_SIZE,
        device=args.device,
    )
    if infer2d._backend != "mmpose":
        raise RuntimeError("MMPose backend failed to load; activate the mmpose conda env.")

    clip_dirs = _clip_dirs(raw_root)
    if args.clip_id:
        clip_dirs = [raw_root / args.clip_id]
    if args.limit_clips is not None:
        clip_dirs = clip_dirs[: args.limit_clips]
    if not clip_dirs:
        raise RuntimeError(f"No clips under {raw_root}")

    summary = []
    for clip_dir in clip_dirs:
        print(f"[export_detected_2d] {clip_dir.name}")
        detections_by_frame = _load_detections_by_frame(clip_dir)
        row = export_clip(
            clip_dir,
            infer2d,
            detections_by_frame,
            pose_mode=args.pose_mode,
            min_det_bbox_area_frac=args.min_det_bbox_area_frac,
            resume=args.resume,
            limit_frames=args.limit_frames,
            expected_source=expected_source,
        )
        summary.append(row)
        print(
            f"  processed={row['processed']} skipped={row['skipped']} "
            f"missing_det={row['missing_detections']} mean_conf={row['mean_conf']:.3f} "
            f"span_xy=({row['mean_span_x']:.0f},{row['mean_span_y']:.0f})"
        )

    summary_path = raw_root / "detected_2d_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[export_detected_2d] Wrote {summary_path}")


if __name__ == "__main__":
    main()

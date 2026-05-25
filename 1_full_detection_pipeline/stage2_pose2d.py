#!/usr/bin/env python3
"""Stage 2: bicycle RTMPose 2D keypoints (conda env: mmpose)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_detector_pipeline.io_utils import iter_jsonl
from keypoint_detector_pipeline.pose2d_mmpose import MMPose2DInferencer
from keypoint_detector_pipeline.preprocess_roi import bbox_area_fraction

from pipeline_io import (
    DETECTIONS_NAME,
    KEYPOINTS_2D_NAME,
    count_frames,
    require_clip_length,
    should_skip_output,
    write_jsonl,
)

DEFAULT_MMPOSE_CONFIG = REPO_ROOT / "2d_keypoint_detector_training" / "rtmpose_bicycle_full.py"
DEFAULT_MMPOSE_CHECKPOINT = (
    REPO_ROOT / "training_outputs" / "mmpose_bicycle_rtmpose_l_gpu" / "best_coco_AP_epoch_160.pth"
)
BICYCLE_INPUT_SIZE = (256, 320)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: bicycle RTMPose 2D keypoints.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mmpose-config",
        type=Path,
        default=DEFAULT_MMPOSE_CONFIG,
    )
    parser.add_argument(
        "--mmpose-checkpoint",
        type=Path,
        default=DEFAULT_MMPOSE_CHECKPOINT,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pose-mode",
        choices=("auto", "full_image", "detection_bbox"),
        default="detection_bbox",
        help=(
            "detection_bbox (default): RTMPose on full frame with RF-DETR bbox via inference_topdown "
            "(matches training pipeline; bbox for stage 3 from detection). "
            "full_image: MMPose on full frame without detection bbox; stage-3 bbox from keypoints. "
            "auto: detection_bbox only if bbox area >= min-det-bbox-area-frac."
        ),
    )
    parser.add_argument(
        "--min-det-bbox-area-frac",
        type=float,
        default=0.01,
        help="In auto mode, minimum detection bbox area fraction to use detection_bbox cropping.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip if keypoints_2d.jsonl exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frames_dir = args.frames_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = count_frames(frames_dir)
    require_clip_length(num_frames)

    detections_path = output_dir / DETECTIONS_NAME
    if not detections_path.is_file():
        raise FileNotFoundError(f"Missing stage 1 output: {detections_path}")

    output_path = output_dir / KEYPOINTS_2D_NAME
    if should_skip_output(output_path, args.resume):
        print(f"[stage2] Skipping (exists): {output_path}")
        return

    if not args.mmpose_config.is_file():
        raise FileNotFoundError(f"MMPose config not found: {args.mmpose_config}")
    if not args.mmpose_checkpoint.is_file():
        raise FileNotFoundError(f"MMPose checkpoint not found: {args.mmpose_checkpoint}")

    infer2d = MMPose2DInferencer(
        pose2d_model=str(args.mmpose_config.resolve()),
        pose2d_weights=str(args.mmpose_checkpoint.resolve()),
        input_size=BICYCLE_INPUT_SIZE,
        device=args.device,
    )
    if infer2d._backend != "mmpose":
        raise RuntimeError("MMPose backend failed to load; activate the mmpose conda env.")

    mode_counts: dict[str, int] = {}
    rows_2d = []
    for det in iter_jsonl(detections_path):
        image_path = Path(det["image_path"])
        det_bbox = det.get("bbox_xyxy")
        kps, conf, bbox, mode_used = infer2d.predict_frame(
            image_path,
            det_bbox,
            mode=args.pose_mode,
            min_det_bbox_area_frac=args.min_det_bbox_area_frac,
        )
        mode_counts[mode_used] = mode_counts.get(mode_used, 0) + 1

        det_bbox_frac = None
        if det_bbox is not None:
            from PIL import Image

            det_bbox_frac = bbox_area_fraction(det_bbox, Image.open(image_path).size)

        rows_2d.append(
            {
                "frame_id": int(det["frame_id"]),
                "image_path": det["image_path"],
                "bbox_xyxy": bbox,
                "det_bbox_xyxy": det_bbox,
                "det_bbox_area_frac": det_bbox_frac,
                "pose_mode": mode_used,
                "det_score": float(det.get("score", 0.0)),
                "keypoints_2d": kps.tolist(),
                "confidence": conf.tolist(),
            }
        )

    if len(rows_2d) != num_frames:
        raise RuntimeError(f"Expected {num_frames} 2D rows, got {len(rows_2d)}")

    write_jsonl(output_path, rows_2d)
    print(f"[stage2] Wrote {output_path} ({len(rows_2d)} frames)")
    print(f"[stage2] pose modes: {mode_counts}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 1: RF-DETR bicycle detection (conda env: rfdetr)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_detector_pipeline.detect_rfdetr import run_detection

from pipeline_io import (
    DETECTIONS_NAME,
    count_frames,
    require_clip_length,
    should_skip_output,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: RF-DETR bicycle detection.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rfdetr-model", default="rfdetr-2xlarge")
    parser.add_argument("--det-confidence", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Skip if detections.jsonl exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frames_dir = args.frames_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = count_frames(frames_dir)
    if num_frames == 0:
        raise FileNotFoundError(f"No images found in {frames_dir}")
    require_clip_length(num_frames)

    output_path = output_dir / DETECTIONS_NAME
    if should_skip_output(output_path, args.resume):
        print(f"[stage1] Skipping (exists): {output_path}")
        return

    print(f"[stage1] Detecting bicycles in {num_frames} frames...")
    run_detection(
        image_dir=frames_dir,
        output_path=output_path,
        model_id=args.rfdetr_model,
        confidence=args.det_confidence,
    )
    print(f"[stage1] Wrote {output_path}")


if __name__ == "__main__":
    main()

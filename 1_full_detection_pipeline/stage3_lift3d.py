#!/usr/bin/env python3
"""Stage 3: PoseMamba 2D -> 3D lifting (conda env: posemamba)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
TRAINING_DIR = REPO_ROOT / "3d_keypoint_detector_training"

for path in (REPO_ROOT, TRAINING_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lift_from_2d_array import lift_2d_to_3d, load_posemamba_lifter, squeeze_batch
from posemamba_bicycle_io import DEFAULT_CHECKPOINT

from pipeline_io import (
    CLIP_LEN,
    KEYPOINTS_2D_NAME,
    KEYPOINTS_3D_NAME,
    build_normalized_2d_sequence,
    count_frames,
    load_keypoints_2d_rows,
    require_clip_length,
    save_keypoints_3d_npz,
    should_skip_output,
)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: PoseMamba 3D lifting.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lifter-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--lifter-config",
        type=Path,
        default=TRAINING_DIR / "PoseMamba_train_bicycle.generated.yaml",
    )
    parser.add_argument("--resume", action="store_true", help="Skip if keypoints_3d.npz exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frames_dir = args.frames_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = count_frames(frames_dir)
    require_clip_length(num_frames)

    keypoints_2d_path = output_dir / KEYPOINTS_2D_NAME
    if not keypoints_2d_path.is_file():
        raise FileNotFoundError(f"Missing stage 2 output: {keypoints_2d_path}")

    output_path = output_dir / KEYPOINTS_3D_NAME
    if should_skip_output(output_path, args.resume):
        print(f"[stage3] Skipping (exists): {output_path}")
        return

    if not args.lifter_checkpoint.is_file():
        raise FileNotFoundError(f"Lifter checkpoint not found: {args.lifter_checkpoint}")

    rows = load_keypoints_2d_rows(keypoints_2d_path)
    if len(rows) != CLIP_LEN:
        raise RuntimeError(f"Expected {CLIP_LEN} rows in {keypoints_2d_path}, got {len(rows)}")

    normalized, frame_ids, _raw = build_normalized_2d_sequence(rows)

    print("[stage3] Loading PoseMamba checkpoint...")
    model, cfg, device = load_posemamba_lifter(
        args.lifter_checkpoint,
        fallback_config=args.lifter_config,
    )
    clip_len = int(getattr(cfg, "clip_len", getattr(cfg, "maxlen", CLIP_LEN)))
    if normalized.shape[0] != clip_len:
        raise ValueError(f"Normalized sequence length {normalized.shape[0]} != model clip_len {clip_len}")

    print(f"[stage3] Lifting shape (1, {clip_len}, 18, 2) on {device}...")
    pred_batch = lift_2d_to_3d(model, cfg, device, normalized)
    pred = squeeze_batch(pred_batch)

    if pred.shape != (clip_len, 18, 3):
        raise RuntimeError(f"Unexpected pred shape {pred.shape}, expected ({clip_len}, 18, 3)")

    save_keypoints_3d_npz(
        output_path,
        pred,
        frame_ids=frame_ids,
        data_input=normalized,
    )
    print(f"[stage3] Wrote {output_path} pred.shape={pred.shape}")


if __name__ == "__main__":
    main()

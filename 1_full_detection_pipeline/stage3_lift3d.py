#!/usr/bin/env python3
"""Stage 3: PoseMamba 2D -> 3D lifting (conda env: posemamba)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
TRAINING_DIR = REPO_ROOT / "3d_keypoint_detector_training"

for path in (REPO_ROOT, TRAINING_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lift_from_2d_array import (  # noqa: E402
    checkpoint_train_maxlen,
    lift_2d_to_3d_sequence,
    load_posemamba_lifter,
)
from pipeline_io import DEFAULT_LIFTER_CHECKPOINT

from pipeline_io import (  # noqa: E402
    KEYPOINTS_2D_NAME,
    KEYPOINTS_3D_NAME,
    build_normalized_2d_sequence,
    count_frames,
    load_keypoints_2d_rows,
    save_keypoints_3d_npz,
    should_skip_output,
    validate_clip_length,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: PoseMamba 3D lifting.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lifter-checkpoint",
        type=Path,
        default=DEFAULT_LIFTER_CHECKPOINT,
    )
    parser.add_argument(
        "--lifter-config",
        type=Path,
        default=PIPELINE_DIR / "PoseMamba_train_bicycle_X.generated.yaml",
    )
    parser.add_argument("--resume", action="store_true", help="Skip if keypoints_3d.npz exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frames_dir = args.frames_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = count_frames(frames_dir)

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
    if len(rows) != num_frames:
        raise RuntimeError(
            f"Frame count mismatch: {num_frames} images in {frames_dir}, "
            f"{len(rows)} rows in {keypoints_2d_path}"
        )

    normalized, frame_ids, _raw = build_normalized_2d_sequence(rows)
    validate_clip_length(normalized.shape[0], context="stage3")

    print("[stage3] Loading PoseMamba checkpoint...")
    model, cfg, device = load_posemamba_lifter(
        args.lifter_checkpoint,
        fallback_config=args.lifter_config,
    )
    train_len = checkpoint_train_maxlen(cfg)

    print(
        f"[stage3] Lifting full measurement ({normalized.shape[0]}, 18, 2) on {device} "
        f"(trained PE={train_len}, single forward)..."
    )
    pred = lift_2d_to_3d_sequence(model, cfg, device, normalized)

    if pred.shape != (normalized.shape[0], 18, 3):
        raise RuntimeError(
            f"Unexpected pred shape {pred.shape}, expected ({normalized.shape[0]}, 18, 3)"
        )

    save_keypoints_3d_npz(
        output_path,
        pred,
        frame_ids=frame_ids,
        data_input=normalized,
    )
    print(f"[stage3] Wrote {output_path} pred.shape={pred.shape}")


if __name__ == "__main__":
    main()

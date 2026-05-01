"""Train state-space 2D->3D lifter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.lift3d_ssm import TemporalSSMLifter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal lifter.")
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = TemporalSSMLifter(num_keypoints=18)
    # Training loop intentionally minimal scaffold.
    if getattr(model, "_mode", "numpy") == "torch":
        import torch

        args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model._model.state_dict(), args.out_checkpoint)
    else:
        args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out_checkpoint.with_suffix(".npy"), np.array([0.0], dtype=np.float32))
    print(f"Saved checkpoint scaffold to {args.out_checkpoint}")


if __name__ == "__main__":
    main()


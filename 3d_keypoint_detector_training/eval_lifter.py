"""Evaluate a trained PoseMamba bicycle lifter checkpoint."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PoseMamba bicycle lifter.")
    parser.add_argument("--conda-env", default="posemamba")
    parser.add_argument("--posemamba-root", type=Path, default=Path("PoseMamba"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("3d_keypoint_detector_training/PoseMamba_train_bicycle.generated.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "train.py",
        "--config",
        str(args.config),
        "--evaluate",
        str(args.checkpoint),
    ]
    subprocess.run(cmd, check=True, cwd=args.posemamba_root)


if __name__ == "__main__":
    main()


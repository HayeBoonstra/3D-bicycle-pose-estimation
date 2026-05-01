"""Launch MMPose training for bicycle 2D keypoints."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D keypoint model with MMPose.")
    parser.add_argument("--config", type=Path, required=True, help="MMPose full config path")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--launcher", default="none")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cmd = [
        "python3",
        "-m",
        "mmpose.tools.train",
        str(args.config),
        "--work-dir",
        str(args.work_dir),
        "--launcher",
        args.launcher,
    ]
    if args.resume_from:
        cmd.extend(["--resume", str(args.resume_from)])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


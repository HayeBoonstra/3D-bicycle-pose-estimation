"""Launch MMPose training for bicycle 2D keypoints."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _cpu_count() -> int:
    return os.cpu_count() or 8


def _train_workers_per_gpu(cpus: int, gpus: int) -> int:
    if gpus >= 2:
        w = cpus // gpus // 10
        return max(4, min(12, w))
    w = cpus // gpus // 6
    return max(8, min(32, w))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D keypoint model with MMPose.")
    parser.add_argument("--config", type=Path, required=True, help="MMPose full config path")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint .pth to resume from",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in --work-dir (last_checkpoint file)",
    )
    parser.add_argument(
        "--launcher",
        default="pytorch",
        choices=["none", "pytorch", "slurm"],
        help="Job launcher (use pytorch for multi-GPU on one machine)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (requires --launcher pytorch or slurm)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Per-GPU train batch size (default: 64; 128/GPU often OOMs on 24GB with this model)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runner = shutil.which("mim")
    if runner is None:
        raise RuntimeError(
            "OpenMMLab runner not found. Install it with: pip install -U openmim"
        )
    train_bs = args.train_batch_size
    if train_bs is None:
        train_bs = 64

    train_workers = _train_workers_per_gpu(_cpu_count(), args.gpus)
    val_workers = max(4, train_workers // 2)

    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(key, "2")

    cmd = [
        runner,
        "train",
        "mmpose",
        str(args.config),
        "--work-dir",
        str(args.work_dir),
        "--launcher",
        args.launcher,
        "--gpus",
        str(args.gpus),
        "--cfg-options",
        f"train_dataloader.batch_size={train_bs}",
        f"val_dataloader.batch_size={train_bs}",
        f"test_dataloader.batch_size={train_bs}",
        f"train_dataloader.num_workers={train_workers}",
        f"val_dataloader.num_workers={val_workers}",
        f"test_dataloader.num_workers={val_workers}",
    ]
    if args.resume_from:
        cmd.extend(["--resume", str(args.resume_from)])
    elif args.resume:
        last_ckpt = args.work_dir / "last_checkpoint"
        if not last_ckpt.is_file():
            raise FileNotFoundError(
                f"No checkpoint to resume: {last_ckpt} (train once or pass --resume-from)"
            )
        cmd.extend(["--resume", "auto"])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()


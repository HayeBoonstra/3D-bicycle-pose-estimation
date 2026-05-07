"""Train PoseMamba on bicycle synthetic sequence data."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml


def _write_config(
    output_path: Path,
    data_root: Path,
    subset_name: str,
    clip_len: int,
    num_joints: int,
    batch_size: int,
    gt_2d: bool,
    epochs: int,
    max_batches: int,
    no_eval: bool,
    max_eval_batches: int,
) -> None:
    cfg = {
        "train_2d": False,
        "no_eval": no_eval,
        "finetune": False,
        "partial_train": None,
        "epochs": epochs,
        "checkpoint_frequency": 10,
        "batch_size": batch_size,
        "dropout": 0.0,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "lr_decay": 0.99,
        "backbone": "PoseMamba",
        "maxlen": clip_len,
        "dim_feat": 64,
        "mlp_ratio": 2,
        "depth": 8,
        "att_fuse": True,
        "data_root": str(data_root),
        "subset_list": [subset_name],
        "dt_file": "unused_for_bicycle.pkl",
        "clip_len": clip_len,
        "data_stride": max(1, clip_len // 3),
        "rootrel": True,
        "sample_stride": 1,
        "num_joints": num_joints,
        "no_conf": True,
        "gt_2d": gt_2d,
        "lambda_3d_velocity": 20.0,
        "lambda_scale": 0.5,
        "lambda_lv": 0.0,
        "lambda_lg": 0.0,
        "lambda_a": 0.0,
        "lambda_av": 0.0,
        "lambda_3dw": 0.0,
        "lambda_3d": 1.0,
        "lambda_diff": 0.5,
        "synthetic": True,
        "flip": False,
        "mask_ratio": 0.0,
        "mask_T_ratio": 0.0,
        "noise": False,
        "max_batches": max_batches,
        "max_eval_batches": max_eval_batches,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PoseMamba bicycle lifter.")
    parser.add_argument("--conda-env", default="posemamba")
    parser.add_argument("--posemamba-root", type=Path, default=Path("PoseMamba"))
    parser.add_argument("--sequence-root", type=Path, default=Path("data/posemamba_sequences"))
    parser.add_argument("--window-size", type=int, default=27)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--subset-name", default="BICYCLE")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/posemamba_bicycle"))
    parser.add_argument(
        "--generated-config",
        type=Path,
        default=Path("3d_keypoint_detector_training/PoseMamba_train_bicycle.generated.yaml"),
    )
    parser.add_argument("--num-joints", type=int, default=18)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=80,
        help="Training/eval batch size written into PoseMamba config (increase to use more VRAM).",
    )
    parser.add_argument("--gt-2d", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    posemamba_data_root = (args.sequence_root / f"PoseMamba_f{args.window_size}s{args.stride}").resolve()
    config_path = args.generated_config.resolve()
    _write_config(
        output_path=config_path,
        data_root=posemamba_data_root,
        subset_name=args.subset_name,
        clip_len=args.window_size,
        num_joints=args.num_joints,
        batch_size=args.batch_size,
        gt_2d=args.gt_2d,
        epochs=args.epochs,
        max_batches=args.max_batches,
        no_eval=args.no_eval,
        max_eval_batches=args.max_eval_batches,
    )

    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    in_target_env = os.environ.get("CONDA_DEFAULT_ENV") == args.conda_env
    cmd = []
    if not in_target_env:
        cmd.extend(["conda", "run", "-n", args.conda_env])
    cmd.extend(
        [
            "python",
            "train.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_dir),
        ]
    )
    env = os.environ.copy()
    for name in ("_PYTHON_SYSCONFIGDATA_NAME", "CC", "CXX", "CUDAHOSTCXX"):
        env.pop(name, None)
    subprocess.run(cmd, check=True, cwd=args.posemamba_root, env=env)


if __name__ == "__main__":
    main()


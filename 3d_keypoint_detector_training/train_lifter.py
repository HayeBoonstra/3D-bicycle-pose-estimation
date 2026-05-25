"""Train PoseMamba on bicycle synthetic sequence data (bbox-normalized image 2D)."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml

GENERATED_CONFIG_NAME = "PoseMamba_train_bicycle.generated.yaml"


def _write_config(
    output_path: Path,
    data_root: Path,
    subset_name: str,
    clip_len: int,
    offline_stride: int,
    num_joints: int,
    batch_size: int,
    bicycle_2d_noise_sigma: float,
    epochs: int,
    max_batches: int,
    no_eval: bool,
    max_eval_batches: int,
    dim_feat: int,
    flip: bool,
    checkpoint_frequency: int,
) -> None:
    cfg = {
        "train_2d": False,
        "no_eval": no_eval,
        "finetune": False,
        "partial_train": None,
        "epochs": epochs,
        "checkpoint_frequency": checkpoint_frequency,
        "batch_size": batch_size,
        "dropout": 0.0,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "lr_decay": 0.99,
        "backbone": "PoseMamba",
        "maxlen": clip_len,
        "dim_feat": dim_feat,
        "mlp_ratio": 2,
        "depth": 10,
        "att_fuse": True,
        "data_root": str(data_root),
        "subset_list": [subset_name],
        "dt_file": "unused_for_bicycle.pkl",
        "clip_len": clip_len,
        "data_stride": offline_stride,
        "rootrel": True,
        "sample_stride": 1,
        "num_joints": num_joints,
        "no_conf": True,
        "gt_2d": False,  # ignored for BICYCLE: dataset always uses data_input
        "eval_snap_xy_to_input": False,
        "synthetic": False,  # do not use oracle 2D from GT 3D
        "bicycle_2d_noise_sigma": bicycle_2d_noise_sigma,
        "lambda_3d_velocity": 20.0,
        "lambda_scale": 0.5,
        "lambda_lv": 0.0,
        "lambda_lg": 0.0,
        "lambda_a": 0.0,
        "lambda_av": 0.0,
        "lambda_3dw": 0.0,
        "lambda_3d": 1.0,
        "lambda_diff": 0.5,
        "flip": flip,
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
    parser = argparse.ArgumentParser(
        description="Train PoseMamba bicycle lifter on bbox-normalized image 2D (data_input).",
    )
    parser.add_argument("--conda-env", default="posemamba")
    parser.add_argument("--posemamba-root", type=Path, default=Path("PoseMamba"))
    parser.add_argument("--sequence-root", type=Path, default=Path("data/posemamba_training_sequences"))
    parser.add_argument(
        "--noise-2d",
        action="store_true",
        help="Apply synthetic detector noise to data_input during training.",
    )
    parser.add_argument("--window-size", type=int, default=243)
    parser.add_argument("--stride", type=int, default=81)
    parser.add_argument(
        "--dataset-tag",
        default=None,
        help="Match build_sequences --dataset-tag (e.g. detected2d).",
    )
    parser.add_argument("--subset-name", default="BICYCLE")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/posemamba_bicycle"),
    )
    parser.add_argument(
        "--generated-config",
        type=Path,
        default=Path("3d_keypoint_detector_training") / GENERATED_CONFIG_NAME,
    )
    parser.add_argument("--num-joints", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dim-feat", type=int, default=64)
    parser.add_argument("--checkpoint-frequency", type=int, default=30)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.generated_config.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    noise_sigma = 0.02 if args.noise_2d else 0.0

    subdir = f"PoseMamba_f{args.window_size}s{args.stride}"
    if args.dataset_tag:
        subdir += f"_{args.dataset_tag}"
    posemamba_data_root = (args.sequence_root / subdir).resolve()
    if not posemamba_data_root.is_dir():
        raise SystemExit(f"error: PoseMamba data_root not found: {posemamba_data_root}")
    _write_config(
        output_path=config_path,
        data_root=posemamba_data_root,
        subset_name=args.subset_name,
        clip_len=args.window_size,
        offline_stride=args.stride,
        num_joints=args.num_joints,
        batch_size=args.batch_size,
        bicycle_2d_noise_sigma=noise_sigma,
        epochs=args.epochs,
        max_batches=args.max_batches,
        no_eval=args.no_eval,
        max_eval_batches=args.max_eval_batches,
        dim_feat=args.dim_feat,
        flip=not args.no_flip,
        checkpoint_frequency=args.checkpoint_frequency,
    )

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
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise SystemExit(f"error: --resume is not a file: {resume_path}")
        if resume_path.suffix.lower() != ".bin":
            raise SystemExit(
                f"error: --resume must be a PoseMamba checkpoint (.bin), got: {resume_path}\n"
                "  Example: ./3d_keypoint_detector_training/start_training.sh "
                "checkpoints/posemamba_bicycle/<run>/latest_epoch.bin"
            )
        cmd.extend(["-r", str(resume_path)])
    env = os.environ.copy()
    for name in ("_PYTHON_SYSCONFIGDATA_NAME", "CC", "CXX", "CUDAHOSTCXX"):
        env.pop(name, None)
    subprocess.run(cmd, check=True, cwd=args.posemamba_root, env=env)


if __name__ == "__main__":
    main()

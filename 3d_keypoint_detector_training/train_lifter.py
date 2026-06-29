"""Train PoseMamba on bicycle synthetic sequence data (bbox-normalized image 2D)."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import yaml

GENERATED_CONFIG_NAME = "PoseMamba_train_bicycle.generated.yaml"
DATA_ROOT_PLACEHOLDER = "__DATA_ROOT__/PoseMamba_f243s81_detected2d"
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")
_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sanitize_experiment_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "_")
    if not cleaned or not _EXPERIMENT_NAME_RE.match(cleaned):
        raise SystemExit(
            "error: experiment name must start with a letter or digit and contain only "
            "letters, digits, underscore, dot, or hyphen"
        )
    return cleaned


def allocate_run_checkpoint_dir(checkpoint_base: Path) -> Path:
    """Next numbered run under checkpoint_base (run_001, run_002, ...)."""
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for child in checkpoint_base.iterdir():
        if not child.is_dir():
            continue
        match = _RUN_DIR_RE.match(child.name)
        if match:
            max_n = max(max_n, int(match.group(1)))
    return checkpoint_base / f"run_{max_n + 1:03d}"


def resolve_run_checkpoint_dir(
    checkpoint_base: Path,
    experiment_name: str | None,
) -> Path:
    if experiment_name:
        run_dir = checkpoint_base / sanitize_experiment_name(experiment_name)
        if run_dir.exists():
            raise SystemExit(
                f"error: experiment directory already exists: {run_dir}\n"
                "  Resume with: ./3d_keypoint_detector_training/start_training.sh "
                f"{run_dir}/latest_epoch.bin"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    return allocate_run_checkpoint_dir(checkpoint_base)


def build_bicycle_config(
    *,
    data_root: str | Path,
    subset_name: str = "BICYCLE",
    clip_len: int = 243,
    offline_stride: int = 81,
    num_joints: int = 18,
    batch_size: int = 4,
    bicycle_2d_noise_sigma: float = 0.0,
    epochs: int = 120,
    max_batches: int = 0,
    no_eval: bool = False,
    max_eval_batches: int = 0,
    dim_feat: int = 64,
    depth: int = 10,
    flip: bool = True,
    checkpoint_frequency: int = 30,
    lambda_steer: float = 0.0,
    lambda_steer_velocity: float = 0.0,
    lambda_roll: float = 0.0,
    lambda_roll_velocity: float = 0.0,
    lambda_3d: float = 1.0,
    lambda_scale: float = 0.5,
    lambda_3d_velocity: float = 20.0,
    lambda_diff: float = 0.5,
    lambda_lv: float = 0.0,
    lambda_lg: float = 0.0,
    lambda_a: float = 0.0,
    lambda_av: float = 0.0,
    lambda_3dw: float = 0.0,
) -> dict:
    """Return a complete PoseMamba bicycle training config dict."""
    return {
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
        "depth": depth,
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
        "lambda_3d_velocity": lambda_3d_velocity,
        "lambda_scale": lambda_scale,
        "lambda_lv": lambda_lv,
        "lambda_lg": lambda_lg,
        "lambda_a": lambda_a,
        "lambda_av": lambda_av,
        "lambda_3dw": lambda_3dw,
        "lambda_3d": lambda_3d,
        "lambda_diff": lambda_diff,
        "lambda_steer": lambda_steer,
        "lambda_steer_velocity": lambda_steer_velocity,
        "lambda_roll": lambda_roll,
        "lambda_roll_velocity": lambda_roll_velocity,
        "flip": flip,
        "mask_ratio": 0.0,
        "mask_T_ratio": 0.0,
        "noise": False,
        "max_batches": max_batches,
        "max_eval_batches": max_eval_batches,
    }


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
    depth: int,
    flip: bool,
    checkpoint_frequency: int,
    lambda_steer: float,
    lambda_steer_velocity: float,
    lambda_roll: float,
    lambda_roll_velocity: float,
) -> None:
    cfg = build_bicycle_config(
        data_root=data_root,
        subset_name=subset_name,
        clip_len=clip_len,
        offline_stride=offline_stride,
        num_joints=num_joints,
        batch_size=batch_size,
        bicycle_2d_noise_sigma=bicycle_2d_noise_sigma,
        epochs=epochs,
        max_batches=max_batches,
        no_eval=no_eval,
        max_eval_batches=max_eval_batches,
        dim_feat=dim_feat,
        depth=depth,
        flip=flip,
        checkpoint_frequency=checkpoint_frequency,
        lambda_steer=lambda_steer,
        lambda_steer_velocity=lambda_steer_velocity,
        lambda_roll=lambda_roll,
        lambda_roll_velocity=lambda_roll_velocity,
    )
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
        default=Path("posemamba_weights"),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Named folder under --checkpoint-dir (e.g. posemamba_s_roll_loss). "
            "Default: auto-increment run_001, run_002, ..."
        ),
    )
    parser.add_argument(
        "--generated-config",
        type=Path,
        default=Path("3d_keypoint_detector_training") / GENERATED_CONFIG_NAME,
    )
    parser.add_argument("--num-joints", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dim-feat", type=int, default=64)
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help=(
            "Number of STE/TTE block pairs in PoseMamba (each pair is one spatial + one "
            "temporal BiSTSSM). Paper layer count N = 2 * depth (e.g. depth 10 -> N=20 for S/B)."
        ),
    )
    parser.add_argument("--checkpoint-frequency", type=int, default=30)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--lambda-steer",
        type=float,
        default=0.0,
        help="Weight on L1 steer-angle error (radians) between pred and GT 3D keypoints.",
    )
    parser.add_argument(
        "--lambda-steer-velocity",
        type=float,
        default=0.0,
        help="Weight on L1 steer-angle velocity error (rad/frame).",
    )
    parser.add_argument(
        "--lambda-roll",
        type=float,
        default=0.0,
        help="Weight on L1 roll-angle error (radians) in camera frame between pred and GT 3D.",
    )
    parser.add_argument(
        "--lambda-roll-velocity",
        type=float,
        default=0.0,
        help="Weight on L1 roll-angle velocity error (rad/frame).",
    )
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
        depth=args.depth,
        flip=not args.no_flip,
        checkpoint_frequency=args.checkpoint_frequency,
        lambda_steer=args.lambda_steer,
        lambda_steer_velocity=args.lambda_steer_velocity,
        lambda_roll=args.lambda_roll,
        lambda_roll_velocity=args.lambda_roll_velocity,
    )

    if args.resume is not None:
        run_checkpoint_dir = args.resume.expanduser().resolve().parent
    else:
        run_checkpoint_dir = resolve_run_checkpoint_dir(
            checkpoint_dir, args.experiment_name
        )
    print(f"[train_lifter] checkpoint run dir: {run_checkpoint_dir}", flush=True)

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
            str(run_checkpoint_dir),
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
                "posemamba_weights/<run>/latest_epoch.bin"
            )
        cmd.extend(["-r", str(resume_path)])
    env = os.environ.copy()
    env["POSEMAMBA_CHECKPOINT_RUN_DIR"] = str(run_checkpoint_dir)
    for name in ("_PYTHON_SYSCONFIGDATA_NAME", "CC", "CXX", "CUDAHOSTCXX"):
        env.pop(name, None)
    subprocess.run(cmd, check=True, cwd=args.posemamba_root, env=env)


if __name__ == "__main__":
    main()

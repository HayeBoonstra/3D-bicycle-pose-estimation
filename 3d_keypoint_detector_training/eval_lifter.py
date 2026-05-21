"""Evaluate a trained PoseMamba bicycle lifter checkpoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "3d_keypoint_detector_training") not in sys.path:
    sys.path.insert(0, str(_REPO / "3d_keypoint_detector_training"))
from posemamba_bicycle_io import export_plain_config, load_training_config  # noqa: E402


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
    ckpt = args.checkpoint.resolve()
    cfg = load_training_config(ckpt, args.config.resolve())
    print("[eval] BICYCLE uses data_input (bbox-normalized 2D) for train and test", flush=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        prefix="posemamba_eval_",
    ) as tmp:
        config_path = export_plain_config(cfg, Path(tmp.name))

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
            "--evaluate",
            str(args.checkpoint.resolve()),
        ]
    )
    env = os.environ.copy()
    for name in ("_PYTHON_SYSCONFIGDATA_NAME", "CC", "CXX", "CUDAHOSTCXX"):
        env.pop(name, None)
    subprocess.run(cmd, check=True, cwd=args.posemamba_root, env=env)


if __name__ == "__main__":
    main()

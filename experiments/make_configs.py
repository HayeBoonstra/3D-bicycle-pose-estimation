#!/usr/bin/env python3
"""Generate uploadable PoseMamba training YAMLs for the headline ablation matrix."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "3d_keypoint_detector_training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from train_lifter import DATA_ROOT_PLACEHOLDER, build_bicycle_config  # noqa: E402

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
CLIP_LEN = 243
STRIDE = 81
BACKBONE_B = {"dim_feat": 128, "depth": 10}


def _base(**overrides) -> dict:
    cfg = build_bicycle_config(
        data_root=DATA_ROOT_PLACEHOLDER,
        clip_len=CLIP_LEN,
        offline_stride=STRIDE,
        dim_feat=BACKBONE_B["dim_feat"],
        depth=BACKBONE_B["depth"],
        checkpoint_frequency=10,
    )
    cfg.update(overrides)
    return cfg


def experiment_matrix() -> list[dict]:
    """Headline ablation matrix (~13 runs), detected-2D input only."""
    experiments: list[dict] = []

    def add(name: str, category: str, description: str, **overrides) -> None:
        cfg = _base(**overrides)
        experiments.append(
            {
                "name": name,
                "category": category,
                "description": description,
                "config": cfg,
            }
        )

    # Capacity ablations (4)
    add(
        "capacity_s",
        "capacity",
        "Small: dim_feat=64, depth=10 (paper N=20)",
        dim_feat=64,
        depth=10,
    )
    add(
        "capacity_b",
        "capacity",
        "Base (main backbone): dim_feat=128, depth=10",
        dim_feat=128,
        depth=10,
    )
    add(
        "capacity_l",
        "capacity",
        "Large: dim_feat=128, depth=20 (paper N=40)",
        dim_feat=128,
        depth=20,
    )
    add(
        "capacity_x",
        "capacity",
        "Extra-large: dim_feat=256, depth=20",
        dim_feat=256,
        depth=20,
    )

    # Loss-term ablations on backbone B (3)
    add(
        "loss_no_velocity",
        "loss",
        "Disable velocity loss (lambda_3d_velocity=0)",
        lambda_3d_velocity=0.0,
    )
    add(
        "loss_no_nmpjpe",
        "loss",
        "Disable normalized MPJPE loss (lambda_scale=0)",
        lambda_scale=0.0,
    )
    add(
        "loss_no_diff",
        "loss",
        "Disable temporal diff loss (lambda_diff=0)",
        lambda_diff=0.0,
    )

    # Bicycle dynamics losses on backbone B (4)
    add(
        "dyn_steer",
        "dynamics_loss",
        "Add steer-angle loss (lambda_steer=1)",
        lambda_steer=1.0,
    )
    add(
        "dyn_roll",
        "dynamics_loss",
        "Add roll-angle loss (lambda_roll=1)",
        lambda_roll=1.0,
    )
    add(
        "dyn_steer_roll",
        "dynamics_loss",
        "Add steer + roll losses",
        lambda_steer=1.0,
        lambda_roll=1.0,
    )
    add(
        "dyn_steer_roll_vel",
        "dynamics_loss",
        "Add steer/roll + velocity losses",
        lambda_steer=1.0,
        lambda_roll=1.0,
        lambda_steer_velocity=1.0,
        lambda_roll_velocity=1.0,
    )

    # Augmentation ablations on backbone B (2)
    add(
        "aug_no_flip",
        "augmentation",
        "Disable test-time flip averaging",
        flip=False,
    )
    add(
        "aug_noise2d",
        "augmentation",
        "Extra 2D noise during training (sigma=0.02)",
        bicycle_2d_noise_sigma=0.02,
    )

    return experiments


def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    experiments = experiment_matrix()
    manifest: dict[str, dict] = {}

    for exp in experiments:
        name = exp["name"]
        cfg = deepcopy(exp["config"])
        yaml_path = CONFIGS_DIR / f"{name}.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        manifest[name] = {
            "category": exp["category"],
            "description": exp["description"],
            "yaml": str(yaml_path.relative_to(REPO_ROOT)),
            "dim_feat": cfg["dim_feat"],
            "depth": cfg["depth"],
            "flip": cfg["flip"],
            "bicycle_2d_noise_sigma": cfg["bicycle_2d_noise_sigma"],
            "lambda_3d": cfg["lambda_3d"],
            "lambda_scale": cfg["lambda_scale"],
            "lambda_3d_velocity": cfg["lambda_3d_velocity"],
            "lambda_diff": cfg["lambda_diff"],
            "lambda_steer": cfg["lambda_steer"],
            "lambda_steer_velocity": cfg["lambda_steer_velocity"],
            "lambda_roll": cfg["lambda_roll"],
            "lambda_roll_velocity": cfg["lambda_roll_velocity"],
        }
        print(f"[make_configs] wrote {yaml_path}")

    manifest_path = CONFIGS_DIR / "experiments.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[make_configs] wrote {manifest_path} ({len(experiments)} experiments)")


if __name__ == "__main__":
    main()

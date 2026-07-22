#!/usr/bin/env python3
"""Generate PoseMamba training YAMLs for GT-2D capacity ablations (S/B/L/X)."""

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
GT_CORPUS = "PoseMamba_f243s81_gt"


def _base(**overrides) -> dict:
    cfg = build_bicycle_config(
        data_root=f"{DATA_ROOT_PLACEHOLDER.rsplit('/', 1)[0]}/{GT_CORPUS}",
        clip_len=CLIP_LEN,
        offline_stride=STRIDE,
        dim_feat=128,
        depth=10,
        checkpoint_frequency=10,
    )
    cfg.update(overrides)
    return cfg


def experiment_matrix() -> list[dict]:
    experiments: list[dict] = []

    def add(name: str, description: str, **overrides) -> None:
        experiments.append(
            {
                "name": name,
                "description": description,
                "config": _base(**overrides),
            }
        )

    add("capacity_s_gt", "Small GT-2D: dim_feat=64, depth=10", dim_feat=64, depth=10)
    add("capacity_b_gt", "Base GT-2D: dim_feat=128, depth=10", dim_feat=128, depth=10)
    add("capacity_l_gt", "Large GT-2D: dim_feat=128, depth=20", dim_feat=128, depth=20)
    add("capacity_x_gt", "Extra-large GT-2D: dim_feat=256, depth=20", dim_feat=256, depth=20)
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
            "category": "capacity_gt",
            "description": exp["description"],
            "yaml": str(yaml_path.relative_to(REPO_ROOT)),
            "corpus_subdir": GT_CORPUS,
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
        print(f"[make_gt_capacity_configs] wrote {yaml_path}")

    manifest_path = CONFIGS_DIR / "gt_capacity_experiments.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[make_gt_capacity_configs] wrote {manifest_path} ({len(experiments)} experiments)")


if __name__ == "__main__":
    main()

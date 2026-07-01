#!/usr/bin/env python3
"""Generate PoseMamba training YAMLs for the temporal-window (T) ablation matrix."""

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

from train_lifter import build_bicycle_config  # noqa: E402

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DATA_ROOT_PLACEHOLDER = "__DATA_ROOT__"
DATASET_TAG = "detected2d"
BACKBONE_B = {"dim_feat": 128, "depth": 10}

# T=243 / stride=81 is the main-analysis baseline (capacity_b); not retrained here.
# Stride scales as T/3 so overlap fraction matches f243s81 (~2/3 between consecutive windows).
STRIDE_DIVISOR = 3
WINDOW_SIZES = (27, 81, 121, 162)


def stride_for_window(clip_len: int, *, divisor: int = STRIDE_DIVISOR) -> int:
    stride = clip_len // divisor
    if stride < 1:
        raise ValueError(f"window T={clip_len} too small for stride divisor {divisor}")
    return stride


def corpus_subdir(clip_len: int, stride: int, dataset_tag: str = DATASET_TAG) -> str:
    name = f"PoseMamba_f{clip_len}s{stride}"
    if dataset_tag:
        name += f"_{dataset_tag}"
    return name


def data_root_for(clip_len: int, stride: int) -> str:
    return f"{DATA_ROOT_PLACEHOLDER}/{corpus_subdir(clip_len, stride)}"


def _base(clip_len: int, stride: int, **overrides) -> dict:
    cfg = build_bicycle_config(
        data_root=data_root_for(clip_len, stride),
        clip_len=clip_len,
        offline_stride=stride,
        dim_feat=BACKBONE_B["dim_feat"],
        depth=BACKBONE_B["depth"],
        checkpoint_frequency=10,
    )
    cfg.update(overrides)
    return cfg


def experiment_matrix() -> list[dict]:
    """Window-size ablations on backbone B, detected-2D input, stride = T/3."""
    experiments: list[dict] = []
    for clip_len in WINDOW_SIZES:
        stride = stride_for_window(clip_len)
        name = f"window_t{clip_len}"
        experiments.append(
            {
                "name": name,
                "category": "window_size",
                "description": (
                    f"Temporal window T={clip_len}, stride={stride} (T/{STRIDE_DIVISOR}), "
                    f"backbone B (dim_feat={BACKBONE_B['dim_feat']}, depth={BACKBONE_B['depth']})"
                ),
                "clip_len": clip_len,
                "data_stride": stride,
                "corpus_subdir": corpus_subdir(clip_len, stride),
                "config": _base(clip_len, stride),
            }
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
            "clip_len": exp["clip_len"],
            "data_stride": exp["data_stride"],
            "corpus_subdir": exp["corpus_subdir"],
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
        print(f"[make_window_configs] wrote {yaml_path}")

    manifest_path = CONFIGS_DIR / "window_experiments.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[make_window_configs] wrote {manifest_path} ({len(experiments)} experiments)")


if __name__ == "__main__":
    main()

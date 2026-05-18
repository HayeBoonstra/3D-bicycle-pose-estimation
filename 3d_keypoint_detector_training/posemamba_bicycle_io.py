"""Shared bicycle PoseMamba I/O: image 2D input, noise, and MPJPE aligned with train.py."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

DEFAULT_NOISE_SIGMA = 0.02
DEFAULT_NOISE_DROPOUT_P = 0.05

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints" / "posemamba_gpu_run_2026_05_18_T_17_22_57"
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "best_epoch.bin"


class Input2DMode(str, Enum):
    """How to build model 2D input from a sequence pickle."""

    IMAGE_2D = "image_2d"
    IMAGE_2D_NOISY = "image_2d_noisy"


@dataclass
class Noise2DConfig:
    sigma: float = DEFAULT_NOISE_SIGMA
    dropout_p: float = DEFAULT_NOISE_DROPOUT_P
    seed: Optional[int] = None


def load_sequence_pkl(path: Union[Path, str]) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected dict pickle, got {type(obj)}")
    return obj


def apply_bicycle_2d_noise(
    motion_2d: np.ndarray,
    sigma: float = DEFAULT_NOISE_SIGMA,
    dropout_p: float = DEFAULT_NOISE_DROPOUT_P,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    out = np.array(motion_2d, dtype=np.float32, copy=True)
    if rng is None:
        rng = np.random.default_rng()
    if sigma > 0:
        out[..., :2] += rng.normal(0.0, sigma, size=out[..., :2].shape).astype(np.float32)
    if dropout_p > 0:
        mask = rng.random(out.shape[:-1]) < dropout_p
        out[mask, :2] = 0.0
    return out


def prepare_2d(
    motion_file: dict[str, Any],
    mode: Union[Input2DMode, str] = Input2DMode.IMAGE_2D,
    *,
    no_conf: bool = True,
    noise_cfg: Optional[Noise2DConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build (T, J, C) 2D from bbox-normalized data_input (matches MotionDataset3D when gt_2d=false)."""
    if isinstance(mode, str):
        mode = Input2DMode(mode)

    data_input = motion_file.get("data_input")
    if data_input is None:
        raise KeyError("prepare_2d requires data_input in pickle")

    motion_2d = np.asarray(data_input, dtype=np.float32)
    if mode == Input2DMode.IMAGE_2D_NOISY:
        cfg = noise_cfg or Noise2DConfig()
        motion_2d = apply_bicycle_2d_noise(
            motion_2d,
            sigma=cfg.sigma,
            dropout_p=cfg.dropout_p,
            rng=rng,
        )

    if no_conf and motion_2d.shape[-1] > 2:
        motion_2d = motion_2d[..., :2]
    return motion_2d.astype(np.float32, copy=False)


def prepare_gt_3d(data_label: np.ndarray, *, rootrel: bool = True) -> np.ndarray:
    gt = np.asarray(data_label, dtype=np.float32)
    if rootrel:
        gt = gt - gt[:, 0:1, :]
    return gt


def to_batch_2d(motion_2d: np.ndarray) -> np.ndarray:
    if motion_2d.ndim == 3:
        return motion_2d[None, ...]
    if motion_2d.ndim == 4:
        return motion_2d
    raise ValueError(f"Expected (T,J,C) or (N,T,J,C), got {motion_2d.shape}")


def mpjpe_per_frame(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    per_frame = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)
    return float(np.mean(per_frame))


def mpjpe_eval(pred: np.ndarray, gt: np.ndarray, cfg: Any) -> dict[str, float]:
    """Root-relative MPJPE (same as train.py evaluate() for bicycle gt_2d=false)."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[None, ...]
        gt = gt[None, ...]

    if getattr(cfg, "rootrel", True):
        pred = pred - pred[:, :, 0:1, :]
        gt = gt - gt[:, :, 0:1, :]

    errors = [mpjpe_per_frame(pred[n], gt[n]) for n in range(pred.shape[0])]
    mean_mpjpe = float(np.mean(errors)) if errors else 0.0
    return {"mpjpe_m": mean_mpjpe, "mpjpe_per_sequence": errors}


def _easydict_checkpoint_config(config_path: Path) -> Any:
    """Load config.yaml written by PoseMamba train.py (yaml.dump on EasyDict)."""
    import yaml

    data = yaml.unsafe_load(config_path.read_text(encoding="utf-8"))
    if hasattr(data, "state"):
        state = data.state
        raw = dict(state) if isinstance(state, dict) else dict(state)
    elif isinstance(data, dict):
        raw = data
    else:
        raw = dict(data)

    import sys

    posemamba_root = Path(__file__).resolve().parents[1] / "PoseMamba"
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))
    from lib.utils.tools import edict

    cfg = edict(raw)
    cfg.name = config_path.stem
    return cfg


def load_training_config(checkpoint_path: Path, fallback_config: Path) -> Any:
    """Load model config from checkpoint dir, or plain YAML fallback."""
    import sys

    posemamba_root = Path(__file__).resolve().parents[1] / "PoseMamba"
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))
    from lib.utils.tools import get_config

    ckpt_cfg = checkpoint_path.resolve().parent / "config.yaml"
    if ckpt_cfg.is_file():
        text = ckpt_cfg.read_text(encoding="utf-8")
        if "python/object" in text or "easydict.EasyDict" in text:
            return _easydict_checkpoint_config(ckpt_cfg)
        try:
            return get_config(str(ckpt_cfg))
        except Exception:
            pass

    return get_config(str(fallback_config.resolve()))

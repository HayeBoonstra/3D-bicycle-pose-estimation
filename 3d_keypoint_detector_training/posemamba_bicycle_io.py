"""Shared bicycle PoseMamba I/O: image 2D input, noise, and MPJPE aligned with train.py."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

DEFAULT_NOISE_SIGMA = 0.02
DEFAULT_NOISE_DROPOUT_P = 0.05

_REPO_ROOT = Path(__file__).resolve().parents[1]
BICYCLE_GENERATED_CONFIG = (
    _REPO_ROOT / "3d_keypoint_detector_training" / "PoseMamba_train_bicycle.generated.yaml"
)
# Default: latest posemamba_weights/run_NNN/best_epoch.bin
CHECKPOINT_BASE = _REPO_ROOT / "posemamba_weights"


def latest_run_checkpoint_dir(checkpoint_base: Path = CHECKPOINT_BASE) -> Path | None:
    """Highest-numbered run_* directory under checkpoint_base, if any."""
    if not checkpoint_base.is_dir():
        return None
    best_n = -1
    best_dir: Path | None = None
    for child in checkpoint_base.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^run_(\d+)$", child.name)
        if match and int(match.group(1)) > best_n:
            best_n = int(match.group(1))
            best_dir = child
    return best_dir


def resolve_default_checkpoint() -> Path:
    run_dir = latest_run_checkpoint_dir()
    if run_dir is not None:
        ckpt = run_dir / "best_epoch.bin"
        if ckpt.is_file():
            return ckpt
    raise FileNotFoundError(
        f"No bicycle checkpoint under {CHECKPOINT_BASE}/run_NNN/best_epoch.bin. "
        "Train with 3d_keypoint_detector_training/start_training.sh first."
    )


DEFAULT_CHECKPOINT_DIR = CHECKPOINT_BASE
try:
    DEFAULT_CHECKPOINT = resolve_default_checkpoint()
except FileNotFoundError:
    DEFAULT_CHECKPOINT = CHECKPOINT_BASE / "run_001" / "best_epoch.bin"


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
    """Build (T, J, C) 2D from bbox-normalized ``data_input`` (matches BICYCLE training)."""
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
    """Root-relative MPJPE (same as train.py evaluate() for bicycle, datareader=None)."""
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


def export_plain_config(cfg: Any, output_path: Path) -> Path:
    """Write a SafeLoader-friendly YAML from an EasyDict (for PoseMamba train.py)."""
    import yaml

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(cfg) if isinstance(cfg, dict) else {k: cfg[k] for k in cfg.keys()}
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return output_path


def _load_config_yaml(path: Path) -> Any | None:
    """Load a YAML config via PoseMamba get_config or pickle EasyDict."""
    import sys

    posemamba_root = Path(__file__).resolve().parents[1] / "PoseMamba"
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))
    from lib.utils.tools import get_config

    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if "python/object" in text or "easydict.EasyDict" in text:
        return _easydict_checkpoint_config(path)
    try:
        return get_config(str(path))
    except Exception:
        return None


def resolve_training_config_path(
    checkpoint_path: Path,
    fallback_config: Path,
    *,
    repo_root: Path | None = None,
    experiment_name: str | None = None,
) -> Path:
    """Pick the YAML that matches checkpoint architecture (ablation / capacity runs)."""
    ckpt = checkpoint_path.resolve()
    fallback = fallback_config.resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    # Sibling configs written by training (ablations use train_config.yaml).
    for name in ("config.yaml", "train_config.yaml"):
        sibling = ckpt.parent / name
        if sibling.is_file():
            return sibling

    candidates: list[str] = []
    if experiment_name:
        candidates.append(experiment_name)
    if ckpt.name == "best_epoch.bin":
        candidates.append(ckpt.parent.name)

    stem = ckpt.stem.lower()
    if "posemamba_x" in stem or stem.startswith("posemamba_x"):
        candidates.append("capacity_x")

    # Dedupe while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(name)

    exp_dir = repo_root / "experiments" / "configs"
    for name in ordered:
        exp_cfg = exp_dir / f"{name}.yaml"
        if exp_cfg.is_file():
            return exp_cfg

    return fallback


def load_training_config(checkpoint_path: Path, fallback_config: Path) -> Any:
    """Load model config from checkpoint dir, or plain YAML fallback."""
    ckpt = checkpoint_path.resolve()
    for name in ("config.yaml", "train_config.yaml"):
        cfg = _load_config_yaml(ckpt.parent / name)
        if cfg is not None:
            _apply_bicycle_config_overrides(cfg)
            return cfg

    cfg = _load_config_yaml(fallback_config.resolve())
    if cfg is None:
        raise FileNotFoundError(f"Could not load training config: {fallback_config}")
    _apply_bicycle_config_overrides(cfg)
    return cfg


def _apply_bicycle_config_overrides(cfg: Any) -> None:
    if "BICYCLE" in getattr(cfg, "subset_list", []):
        cfg.gt_2d = False
        cfg.synthetic = False
        cfg.eval_snap_xy_to_input = False

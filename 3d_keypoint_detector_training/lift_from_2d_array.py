"""Load PoseMamba and run 2D -> 3D lifting on bbox-normalized sequences."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from posemamba_bicycle_io import load_training_config, to_batch_2d


def load_posemamba_lifter(
    checkpoint_path: Path,
    *,
    fallback_config: Path | None = None,
    posemamba_root: Path | None = None,
    depth_override: int | None = None,
    dim_feat_override: int | None = None,
    maxlen_override: int | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load PoseMamba model, config, and device from a bicycle checkpoint."""
    ckpt_path = checkpoint_path.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if fallback_config is None:
        fallback_config = _SCRIPT_DIR / "PoseMamba_train_bicycle.generated.yaml"
    fallback_cfg = fallback_config.resolve()

    if posemamba_root is None:
        posemamba_root = _REPO_ROOT / "PoseMamba"
    posemamba_root = posemamba_root.resolve()
    if not posemamba_root.is_dir():
        raise FileNotFoundError(f"PoseMamba root not found: {posemamba_root}")

    os.chdir(posemamba_root)
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))

    from lib.utils.learning import load_backbone

    cfg = load_training_config(ckpt_path, fallback_cfg)
    if depth_override is not None:
        cfg.depth = depth_override
    if dim_feat_override is not None:
        cfg.dim_feat = dim_feat_override
    if maxlen_override is not None:
        cfg.maxlen = maxlen_override
        cfg.clip_len = maxlen_override

    model = load_backbone(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    else:
        model = model.to(device)

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state = checkpoint["model_pos"]
    if not torch.cuda.is_available():
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, device


def lift_2d_to_3d(
    model: Any,
    cfg: Any,
    device: torch.device,
    motion_2d: np.ndarray,
) -> np.ndarray:
    """Run lifting on (T, J, C) or (N, T, J, C) normalized 2D; returns same batch rank."""
    input_2d = to_batch_2d(np.asarray(motion_2d, dtype=np.float32))
    tensor_in = torch.from_numpy(input_2d).to(device)
    run_in = tensor_in[:, :, :, :2] if getattr(cfg, "no_conf", True) else tensor_in

    with torch.no_grad():
        pred = model(run_in)
        if getattr(cfg, "rootrel", True):
            pred[:, :, 0, :] = 0
    return pred.detach().cpu().numpy()


def squeeze_batch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr

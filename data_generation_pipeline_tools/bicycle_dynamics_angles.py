"""Bicycle dynamics angles — canonical numpy API.

Steer and roll match ``PoseMamba.lib.model.loss`` (used for training losses and
``test_bicycle_dynamics_losses.py``). Steer is rake-independent: fork-rigid
directions are projected into the plane perpendicular to the head-tube axis,
measured with signed ``atan2`` angles, then fused with a circular mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_POSEMAMBA_ROOT = _REPO_ROOT / "PoseMamba"
for _p in (_REPO_ROOT, _POSEMAMBA_ROOT):
    s = str(_p.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _steer_roll_torch(kpts: np.ndarray):
    import torch

    t = torch.from_numpy(np.asarray(kpts, dtype=np.float64))
    if t.ndim == 2:
        return t.unsqueeze(0), True
    if t.ndim == 3:
        return t, False
    raise ValueError(f"Expected (T, J, 3) or (J, 3), got {kpts.shape}")


def bicycle_steer_angle(kpts: np.ndarray) -> np.ndarray:
    """Signed steering angle (rad), shape (T,) or scalar for single pose."""
    from PoseMamba.lib.model.loss import bicycle_steer_angle as _steer

    t, squeezed = _steer_roll_torch(kpts)
    out = _steer(t).numpy()
    if squeezed:
        return np.asarray(out, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def bicycle_steer_angle_hub(kpts: np.ndarray) -> np.ndarray:
    """Hub-only signed steering angle (rad), retained for diagnostics."""
    from PoseMamba.lib.model.loss import bicycle_steer_angle_hub as _steer

    t, squeezed = _steer_roll_torch(kpts)
    out = _steer(t).numpy()
    if squeezed:
        return np.asarray(out, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def bicycle_roll_angle(kpts: np.ndarray) -> np.ndarray:
    """Signed roll angle (rad) in the camera frame, shape (T,) or scalar."""
    from PoseMamba.lib.model.loss import bicycle_roll_angle as _roll

    t, squeezed = _steer_roll_torch(kpts)
    out = _roll(t).numpy()
    if squeezed:
        return np.asarray(out, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)

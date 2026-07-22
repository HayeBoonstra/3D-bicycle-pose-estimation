"""Load PoseMamba and run 2D -> 3D lifting on bbox-normalized sequences."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from posemamba_bicycle_io import (
    load_training_config,
    resolve_training_config_path,
    to_batch_2d,
)

# Upper bound for a single deployment measurement (~30 s at 60 Hz).
DEFAULT_MAX_MEASUREMENT_FRAMES = int(os.environ.get("POSEMAMBA_MAX_MEASUREMENT_FRAMES", "1800"))


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def checkpoint_train_maxlen(cfg: Any) -> int:
    """Temporal positional-embedding length the checkpoint was trained with."""
    return int(getattr(cfg, "clip_len", None) or getattr(cfg, "maxlen", 243))


def checkpoint_maxlen(cfg: Any) -> int:
    """Backward-compatible alias for training window size (not an inference cap)."""
    return checkpoint_train_maxlen(cfg)


def _interpolate_temporal_pos_embed(pe: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample learned temporal positional embeddings along the time axis."""
    if target_len <= pe.shape[1]:
        return pe[:, :target_len, :]
    return torch.nn.functional.interpolate(
        pe.permute(0, 2, 1),
        size=target_len,
        mode="linear",
        align_corners=True,
    ).permute(0, 2, 1)


@contextmanager
def extended_temporal_context(model: Any, num_frames: int) -> Iterator[int]:
    """Temporarily extend Temporal_pos_embed so one forward can cover num_frames."""
    core = _unwrap_model(model)
    original = core.Temporal_pos_embed
    train_len = int(original.shape[1])
    if num_frames > train_len:
        extended = _interpolate_temporal_pos_embed(original, num_frames)
        core.Temporal_pos_embed = nn.Parameter(
            extended.to(device=original.device, dtype=original.dtype),
            requires_grad=False,
        )
    try:
        yield train_len
    finally:
        core.Temporal_pos_embed = original


def validate_measurement_length(
    num_frames: int,
    *,
    max_frames: int = DEFAULT_MAX_MEASUREMENT_FRAMES,
    context: str = "inference",
) -> None:
    if num_frames <= 0:
        raise ValueError(f"{context}: sequence length must be positive, got {num_frames}")
    if num_frames > max_frames:
        raise ValueError(
            f"{context}: sequence has {num_frames} frames, above limit {max_frames}. "
            f"Increase POSEMAMBA_MAX_MEASUREMENT_FRAMES if needed."
        )


def load_posemamba_lifter(
    checkpoint_path: Path,
    *,
    fallback_config: Path | None = None,
    experiment_name: str | None = None,
    posemamba_root: Path | None = None,
    depth_override: int | None = None,
    dim_feat_override: int | None = None,
    maxlen_override: int | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load PoseMamba model, config, and device from a bicycle checkpoint."""
    ckpt_path = checkpoint_path.expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (_REPO_ROOT / ckpt_path).resolve()
    else:
        ckpt_path = ckpt_path.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if fallback_config is None:
        fallback_config = _SCRIPT_DIR / "PoseMamba_train_bicycle.generated.yaml"
    fallback_cfg = resolve_training_config_path(
        ckpt_path,
        fallback_config.resolve(),
        repo_root=_REPO_ROOT,
        experiment_name=experiment_name,
    )
    print(f"[posemamba] config={fallback_cfg} checkpoint={ckpt_path.name}")

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
    *,
    extend_temporal: bool = True,
) -> np.ndarray:
    """Run lifting on (T, J, C) or (N, T, J, C) normalized 2D; returns same batch rank."""
    input_2d = to_batch_2d(np.asarray(motion_2d, dtype=np.float32))
    batch_size = int(input_2d.shape[0]) if input_2d.ndim == 4 else 1
    num_frames = int(input_2d.shape[1]) if input_2d.ndim == 4 else int(input_2d.shape[0])
    validate_measurement_length(num_frames, context="lift_2d_to_3d")

    tensor_in = torch.from_numpy(input_2d).to(device)
    run_in = tensor_in[:, :, :, :2] if getattr(cfg, "no_conf", True) else tensor_in

    with torch.no_grad():
        if extend_temporal:
            with extended_temporal_context(model, num_frames) as train_len:
                if num_frames > train_len:
                    print(
                        f"[posemamba] extending temporal PE {train_len} -> {num_frames} "
                        f"(single forward, no windowing)",
                        flush=True,
                    )
                pred = model(run_in)
        else:
            train_len = checkpoint_train_maxlen(cfg)
            if num_frames > train_len:
                raise ValueError(
                    f"lift_2d_to_3d: T={num_frames} exceeds trained PE length {train_len}. "
                    f"Use extend_temporal=True (default) for full measurements."
                )
            pred = model(run_in)
        if getattr(cfg, "rootrel", True):
            pred[:, :, 0, :] = 0
    return pred.detach().cpu().numpy()


def lift_2d_to_3d_sequence(
    model: Any,
    cfg: Any,
    device: torch.device,
    motion_2d: np.ndarray,
    *,
    extend_temporal: bool = True,
) -> np.ndarray:
    """Lift one contiguous measurement with a single model forward pass (no maxlen chunking)."""
    seq = np.asarray(motion_2d, dtype=np.float32)
    if seq.ndim != 3:
        raise ValueError(f"lift_2d_to_3d_sequence expects (T, J, C), got {seq.shape}")
    num_frames = int(seq.shape[0])
    if num_frames == 0:
        return np.zeros((0, seq.shape[1], 3), dtype=np.float32)

    try:
        pred = lift_2d_to_3d(model, cfg, device, seq, extend_temporal=extend_temporal)
    except torch.cuda.OutOfMemoryError as exc:
        raise torch.cuda.OutOfMemoryError(
            f"GPU OOM lifting T={num_frames} in one forward pass. "
            f"Try a shorter clip, a GPU with more memory, or POSEMAMBA_MAX_MEASUREMENT_FRAMES "
            f"below {num_frames}."
        ) from exc
    return squeeze_batch(pred) if pred.ndim == 4 else pred


def squeeze_batch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr

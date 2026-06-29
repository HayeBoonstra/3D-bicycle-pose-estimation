"""Model efficiency metrics: params, FLOPs, throughput, GPU memory."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "3d_keypoint_detector_training"
POSEMAMBA_ROOT = REPO_ROOT / "PoseMamba"

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(POSEMAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(POSEMAMBA_ROOT))

from train_lifter import build_bicycle_config, DATA_ROOT_PLACEHOLDER  # noqa: E402
from lib.utils.learning import load_backbone  # noqa: E402
from lib.utils.tools import edict  # noqa: E402


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _estimate_flops(model: nn.Module, sample: torch.Tensor) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis

        return float(FlopCountAnalysis(model, sample).total()) / 1e9
    except Exception:
        pass
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(sample,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        return None


def compute_efficiency_metrics(
    dim_feat: int = 128,
    depth: int = 10,
    clip_len: int = 243,
    num_joints: int = 18,
    batch_size: int = 1,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    cfg_dict = build_bicycle_config(
        data_root=DATA_ROOT_PLACEHOLDER,
        dim_feat=dim_feat,
        depth=depth,
        clip_len=clip_len,
        num_joints=num_joints,
    )
    cfg = edict(cfg_dict)
    model = load_backbone(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    params_m = _count_params(model) / 1e6
    flops_g = None
    seq_per_s = None
    peak_mem_mb = None
    note = None

    try:
        if device.type != "cuda":
            raise RuntimeError("CUDA not available")

        sample = torch.randn(batch_size, clip_len, num_joints, 2, device=device)
        flops_g = _estimate_flops(model, sample)

        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(warmup):
                model(sample)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(sample)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        seq_per_s = repeats * batch_size / max(elapsed, 1e-9)
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    except Exception as exc:
        note = f"FLOPs/throughput skipped: {exc}"

    return {
        "dim_feat": dim_feat,
        "depth": depth,
        "paper_layers_N": depth * 2,
        "params_M": float(params_m),
        "flops_G": flops_g,
        "throughput_seq_per_s": float(seq_per_s) if seq_per_s is not None else None,
        "latency_ms_per_seq": float(1000.0 / max(seq_per_s / batch_size, 1e-9)) if seq_per_s else None,
        "peak_gpu_mem_MB": peak_mem_mb,
        "device": str(device),
        "note": note,
    }


def compute_capacity_frontier() -> dict[str, Any]:
    """Efficiency metrics for S/B/L/X capacity presets."""
    presets = [
        ("S", 64, 10),
        ("B", 128, 10),
        ("L", 128, 20),
        ("X", 256, 20),
    ]
    results = {}
    for name, dim, depth in presets:
        results[name] = compute_efficiency_metrics(dim_feat=dim, depth=depth)
    return results

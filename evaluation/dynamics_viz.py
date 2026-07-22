"""Align dynamics time-series figures across window ablations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SCRIPT_3D = REPO_ROOT / "3d_keypoint_detector_training"
if str(_SCRIPT_3D) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_3D))

from build_sequences import _read_clip  # noqa: E402
from evaluation.common import default_raw_root  # noqa: E402
from evaluation.metrics.dynamics import _build_time_series_payload  # noqa: E402
from lift_from_2d_array import lift_2d_to_3d_sequence, load_posemamba_lifter  # noqa: E402
from posemamba_bicycle_io import prepare_gt_3d  # noqa: E402

REFERENCE_DYNAMICS_EXPERIMENT = "capacity_b"
WINDOW_ABLATION_PREFIX = "window_t"

_MODEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}


def _require_posemamba_inference_env() -> None:
    """Window figure alignment re-runs PoseMamba; needs CUDA Mamba kernels."""
    try:
        import selective_scan_cuda_core  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Window ablation dynamics figures re-run PoseMamba inference. "
            "Use the posemamba conda env (selective_scan_cuda_core), not rfdetr:\n"
            "  conda activate posemamba && python evaluation/make_figures.py"
        ) from exc
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Window ablation dynamics figures require CUDA. "
            "Run make_figures with the posemamba env on a GPU machine."
        )


def reference_dynamics_span(
    results_dir: Path,
    *,
    reference_experiment: str = REFERENCE_DYNAMICS_EXPERIMENT,
) -> dict[str, Any]:
    """Return clip_id and frame_idx used by the standard (243-frame) dynamics figure."""
    metrics_path = results_dir / reference_experiment / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Reference metrics not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ts = metrics.get("dynamics", {}).get("time_series", {})
    clip_id = ts.get("clip_id")
    frame_idx = ts.get("frame_idx")
    if not clip_id or not frame_idx:
        raise ValueError(f"No dynamics time_series in {metrics_path}")
    frame_idx_arr = np.asarray(frame_idx, dtype=np.int32)
    if frame_idx_arr.size == 0:
        raise ValueError(f"Empty frame_idx in {metrics_path}")
    if np.any(np.diff(frame_idx_arr) != 1):
        raise ValueError(
            f"Reference frame_idx in {metrics_path} is not contiguous; "
            "regenerate capacity_b metrics first."
        )
    return {
        "clip_id": str(clip_id),
        "frame_idx": frame_idx_arr,
        "reference_experiment": reference_experiment,
    }


def resolve_experiment_checkpoint(exp_dir: Path) -> Path:
    summary_path = exp_dir / "extract_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ckpt = summary.get("checkpoint")
        if ckpt:
            ckpt_path = Path(str(ckpt))
            if ckpt_path.is_file():
                return ckpt_path.resolve()
    default = REPO_ROOT / "posemamba_weights" / exp_dir.name / "best_epoch.bin"
    if default.is_file():
        return default.resolve()
    raise FileNotFoundError(f"No checkpoint found for experiment {exp_dir.name}")


def _load_model_cached(checkpoint: Path, *, experiment_name: str | None) -> tuple[Any, Any, Any]:
    key = str(checkpoint.resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_posemamba_lifter(checkpoint, experiment_name=experiment_name)
    return _MODEL_CACHE[key]


def build_time_series_for_frame_span(
    checkpoint: Path,
    clip_id: str,
    frame_indices: np.ndarray | list[int],
    *,
    raw_root: Path | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Lift and build steer/roll time series for an explicit contiguous frame span."""
    _require_posemamba_inference_env()
    raw_root = (raw_root or default_raw_root()).resolve()
    clip_dir = raw_root / clip_id
    if not clip_dir.is_dir():
        raise FileNotFoundError(f"Raw clip not found: {clip_dir}")

    frame_indices = np.asarray(frame_indices, dtype=np.int32)
    clip = _read_clip(clip_dir, input_2d="detected", bbox_source="detection")
    idx_map = {int(f): i for i, f in enumerate(clip.frame_idx)}
    rows = np.asarray([idx_map[int(f)] for f in frame_indices], dtype=np.int64)

    motion_2d = clip.points_2d[rows]
    gt = prepare_gt_3d(clip.points_3d_cam[rows], rootrel=True)
    mujoco_steer = clip.steer_deg[rows]
    gt_roll_mujoco = clip.roll_deg[rows]
    frame_idx_out = clip.frame_idx[rows]

    model, cfg, device = _load_model_cached(checkpoint, experiment_name=experiment_name)
    pred = lift_2d_to_3d_sequence(model, cfg, device, motion_2d)
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(
            f"pred/gt length mismatch for {clip_id}: {pred.shape[0]} vs {gt.shape[0]}"
        )

    return _build_time_series_payload(
        clip_id=clip_id,
        pred_viz=pred,
        gt_viz=gt,
        mujoco_steer_viz=mujoco_steer,
        gt_roll_mujoco_viz=gt_roll_mujoco,
        frame_idx_viz=frame_idx_out,
    )


def aligned_window_ablation_time_series(
    exp_dir: Path,
    ref_span: dict[str, Any],
    *,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    checkpoint = resolve_experiment_checkpoint(exp_dir)
    return build_time_series_for_frame_span(
        checkpoint,
        ref_span["clip_id"],
        ref_span["frame_idx"],
        raw_root=raw_root,
        experiment_name=exp_dir.name,
    )

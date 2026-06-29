"""3D keypoint metrics for PoseMamba bicycle lifter."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.common import (
    DEFAULT_MAX_CLIP_MPJPE_MM,
    JOINT_GROUPS,
    frame_mask_for_clips,
    group_mean,
    mm_from_m,
)


def n_mpjpe_numpy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Scale-normalized MPJPE (numpy), matching loss n_mpjpe logic."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim == 3:
        pred = pred[None, ...]
        gt = gt[None, ...]
    norm_pred = np.mean(np.sum(pred**2, axis=3, keepdims=True), axis=2, keepdims=True)
    norm_target = np.mean(np.sum(gt * pred, axis=3, keepdims=True), axis=2, keepdims=True)
    scale = norm_target / np.maximum(norm_pred, 1e-12)
    scaled = scale * pred
    err = np.linalg.norm(scaled - gt, axis=-1).mean(axis=-1)
    return float(np.mean(err))


def mpjpe_per_joint(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-joint MPJPE in meters, shape (J,)."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=0)


def mpjve(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint velocity error (m/frame)."""
    if pred.shape[0] <= 1:
        return 0.0
    v_pred = pred[1:] - pred[:-1]
    v_gt = gt[1:] - gt[:-1]
    return float(np.linalg.norm(v_pred - v_gt, axis=-1).mean())


def mpjae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint acceleration error (m/frame^2)."""
    if pred.shape[0] <= 2:
        return 0.0
    a_pred = pred[2:] - 2 * pred[1:-1] + pred[:-2]
    a_gt = gt[2:] - 2 * gt[1:-1] + gt[:-2]
    return float(np.linalg.norm(a_pred - a_gt, axis=-1).mean())


def _load_mpjpe_fns():
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    posemamba_root = repo / "PoseMamba"
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))
    from lib.model.loss import mpjpe, p_mpjpe

    return mpjpe, p_mpjpe


def _aggregate_pose3d(
    pred_rr: np.ndarray,
    gt_rr: np.ndarray,
    mpjpe_fn,
    p_mpjpe_fn,
) -> dict[str, Any]:
    mpjpe_m = float(np.mean(mpjpe_fn(pred_rr[None, ...], gt_rr[None, ...])))
    pmpjpe_m = float(np.mean(p_mpjpe_fn(pred_rr, gt_rr)))
    nmpjpe_m = n_mpjpe_numpy(pred_rr, gt_rr)
    per_joint = mpjpe_per_joint(pred_rr, gt_rr)
    return {
        "mpjpe_m": mpjpe_m,
        "mpjpe_mm": mm_from_m(mpjpe_m),
        "p_mpjpe_m": pmpjpe_m,
        "p_mpjpe_mm": mm_from_m(pmpjpe_m),
        "n_mpjpe_m": nmpjpe_m,
        "n_mpjpe_mm": mm_from_m(nmpjpe_m),
        "mpjve_m_per_frame": mpjve(pred_rr, gt_rr),
        "mpjae_m_per_frame2": mpjae(pred_rr, gt_rr),
        "passes_40mm_target": mpjpe_m < 0.04,
        "per_joint_mpjpe_m": {str(j): float(per_joint[j]) for j in range(len(per_joint))},
        "per_joint_mpjpe_mm": {str(j): mm_from_m(float(per_joint[j])) for j in range(len(per_joint))},
        "group_mpjpe_mm": {g: mm_from_m(group_mean(per_joint, g)) for g in JOINT_GROUPS},
        "num_frames": int(pred_rr.shape[0]),
    }


def compute_pose3d_metrics(
    preds_npz_path: str | Any,
    *,
    max_clip_mpjpe_mm: float = DEFAULT_MAX_CLIP_MPJPE_MM,
) -> dict[str, Any]:
    """Compute 3D metrics from extract.py preds_3d.npz.

    Clips with per-clip MPJPE above ``max_clip_mpjpe_mm`` are excluded from
    headline aggregates (treated as likely out-of-distribution / erroneous).
    """
    data = np.load(preds_npz_path, allow_pickle=True)
    pred = np.asarray(data["pred"], dtype=np.float32)
    gt = np.asarray(data["gt"], dtype=np.float32)
    mpjpe_fn, p_mpjpe_fn = _load_mpjpe_fns()

    pred_rr = pred - pred[:, 0:1, :]
    gt_rr = gt - gt[:, 0:1, :]

    per_clip: list[dict[str, Any]] = []
    clip_ids = data.get("clip_ids")
    accepted_clip_ids: set[str] = set()
    rejected_clip_ids: list[str] = []

    if clip_ids is not None:
        clip_ids = list(clip_ids)
        for cid in sorted(set(str(c) for c in clip_ids)):
            mask = np.array([str(c) == cid for c in clip_ids])
            pc = float(np.mean(mpjpe_fn(pred_rr[mask][None, ...], gt_rr[mask][None, ...])))
            pc_mm = mm_from_m(pc)
            accepted = pc_mm <= max_clip_mpjpe_mm
            per_clip.append(
                {
                    "clip_id": cid,
                    "mpjpe_m": pc,
                    "mpjpe_mm": pc_mm,
                    "accepted": accepted,
                }
            )
            if accepted:
                accepted_clip_ids.add(cid)
            else:
                rejected_clip_ids.append(cid)

    if clip_ids is not None and accepted_clip_ids:
        frame_mask = frame_mask_for_clips(clip_ids, accepted_clip_ids)
        pred_use = pred_rr[frame_mask]
        gt_use = gt_rr[frame_mask]
    else:
        pred_use = pred_rr
        gt_use = gt_rr

    metrics = _aggregate_pose3d(pred_use, gt_use, mpjpe_fn, p_mpjpe_fn)
    all_metrics = _aggregate_pose3d(pred_rr, gt_rr, mpjpe_fn, p_mpjpe_fn)

    accepted_per_clip_mpjpe = [c["mpjpe_m"] for c in per_clip if c.get("accepted")]

    metrics["macro_mpjpe_m"] = float(np.mean(accepted_per_clip_mpjpe)) if accepted_per_clip_mpjpe else None
    metrics["macro_mpjpe_mm"] = (
        mm_from_m(float(np.mean(accepted_per_clip_mpjpe))) if accepted_per_clip_mpjpe else None
    )
    metrics["per_clip"] = per_clip
    metrics["median_mpjpe_mm"] = (
        mm_from_m(float(np.median(accepted_per_clip_mpjpe))) if accepted_per_clip_mpjpe else None
    )
    metrics["iqr_mpjpe_mm"] = (
        mm_from_m(float(np.percentile(accepted_per_clip_mpjpe, 75) - np.percentile(accepted_per_clip_mpjpe, 25)))
        if accepted_per_clip_mpjpe
        else None
    )
    metrics["clip_filter"] = {
        "max_clip_mpjpe_mm": max_clip_mpjpe_mm,
        "num_clips_total": len(per_clip),
        "num_clips_accepted": len(accepted_clip_ids),
        "num_clips_rejected": len(rejected_clip_ids),
        "num_frames_total": int(pred_rr.shape[0]),
        "num_frames_accepted": int(pred_use.shape[0]),
        "rejected_clip_ids": rejected_clip_ids,
        "accepted_clip_ids": sorted(accepted_clip_ids),
    }
    metrics["all_clips"] = {
        "mpjpe_m": all_metrics["mpjpe_m"],
        "mpjpe_mm": all_metrics["mpjpe_mm"],
        "macro_mpjpe_mm": mm_from_m(float(np.mean([c["mpjpe_m"] for c in per_clip]))) if per_clip else None,
        "median_mpjpe_mm": mm_from_m(float(np.median([c["mpjpe_m"] for c in per_clip]))) if per_clip else None,
    }

    n = pred_use.shape[0]
    if n > 0:
        pos = np.linspace(0, 1, n)
        frame_err = np.linalg.norm(pred_use - gt_use, axis=-1).mean(axis=-1)
        bins = np.linspace(0, 1, 5, endpoint=False)
        bin_means = []
        for b0, b1 in zip(bins, list(bins[1:]) + [1.0]):
            m = (pos >= b0) & (pos < b1 if b1 < 1 else pos <= b1)
            if np.any(m):
                bin_means.append(float(np.mean(frame_err[m])))
            else:
                bin_means.append(None)
        metrics["mpjpe_vs_clip_position"] = bin_means

    return metrics

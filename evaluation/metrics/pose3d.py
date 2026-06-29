"""3D keypoint metrics for PoseMamba bicycle lifter."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.common import JOINT_GROUPS, group_mean, mm_from_m


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


def compute_pose3d_metrics(preds_npz_path: str | Any) -> dict[str, Any]:
    """Compute 3D metrics from extract.py preds_3d.npz."""
    data = np.load(preds_npz_path, allow_pickle=True)
    pred = np.asarray(data["pred"], dtype=np.float32)
    gt = np.asarray(data["gt"], dtype=np.float32)

    # Import PoseMamba metrics
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    posemamba_root = repo / "PoseMamba"
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))
    from lib.model.loss import mpjpe, p_mpjpe

    pred_rr = pred - pred[:, 0:1, :]
    gt_rr = gt - gt[:, 0:1, :]

    mpjpe_m = float(np.mean(mpjpe(pred_rr[None, ...], gt_rr[None, ...])))
    pmpjpe_m = float(np.mean(p_mpjpe(pred_rr, gt_rr)))
    nmpjpe_m = n_mpjpe_numpy(pred_rr, gt_rr)

    per_joint = mpjpe_per_joint(pred_rr, gt_rr)
    per_clip = []
    clip_ids = data.get("clip_ids")
    if clip_ids is not None:
        clip_ids = list(clip_ids)
        unique_clips = sorted(set(clip_ids))
        for cid in unique_clips:
            mask = np.array([c == cid for c in clip_ids])
            pc = float(np.mean(mpjpe(pred_rr[mask][None, ...], gt_rr[mask][None, ...])))
            per_clip.append({"clip_id": str(cid), "mpjpe_m": pc, "mpjpe_mm": mm_from_m(pc)})

    per_clip_mpjpe = [c["mpjpe_m"] for c in per_clip]
    metrics: dict[str, Any] = {
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
        "per_clip": per_clip,
        "median_mpjpe_mm": mm_from_m(float(np.median(per_clip_mpjpe))) if per_clip_mpjpe else None,
        "iqr_mpjpe_mm": mm_from_m(float(np.percentile(per_clip_mpjpe, 75) - np.percentile(per_clip_mpjpe, 25)))
        if per_clip_mpjpe
        else None,
    }

    # MPJPE vs clip position (normalized timeline)
    n = pred_rr.shape[0]
    if n > 0:
        pos = np.linspace(0, 1, n)
        frame_err = np.linalg.norm(pred_rr - gt_rr, axis=-1).mean(axis=-1)
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

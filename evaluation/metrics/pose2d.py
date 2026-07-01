"""2D keypoint metrics for RTMPose on bicycle test set."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.common import JOINT_GROUPS, group_mean


def _pck_at_threshold(errors_norm: np.ndarray, visible: np.ndarray, thr: float) -> float:
    if not np.any(visible):
        return float("nan")
    return float(np.mean(errors_norm[visible] <= thr))


def _pck_curve(errors_norm: np.ndarray, visible: np.ndarray, n_points: int = 20) -> tuple[list[float], list[float], float]:
    thresholds = np.linspace(0.0, 0.5, n_points)
    values = [_pck_at_threshold(errors_norm, visible, float(t)) for t in thresholds]
    auc = float(np.trapz(values, thresholds) / 0.5) if thresholds.size > 1 else 0.0
    return thresholds.tolist(), values, auc


def compute_pose2d_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute 2D metrics from stage12 records with GT and detected keypoints."""
    all_err_px: list[float] = []
    all_err_norm: list[float] = []
    visible_mask: list[bool] = []
    occluded_mask: list[bool] = []
    per_joint_sum = np.zeros(18, dtype=np.float64)
    per_joint_cnt = np.zeros(18, dtype=np.float64)
    conf_sum = np.zeros(18, dtype=np.float64)
    conf_cnt = np.zeros(18, dtype=np.float64)

    gt_bbox_err_px: list[float] = []
    det_bbox_err_px: list[float] = []

    for rec in records:
        gt_kp = rec.get("gt_keypoints_2d")
        det_kp = rec.get("det_keypoints_2d")
        if gt_kp is None or det_kp is None:
            continue

        gt_pts = np.asarray(gt_kp["points"], dtype=np.float32)
        det_pts = np.asarray(det_kp["points"], dtype=np.float32)
        gt_vis = np.asarray(gt_kp.get("visible", np.ones(18, dtype=bool)), dtype=bool)
        gt_occ = np.asarray(gt_kp.get("occluded", np.zeros(18, dtype=bool)), dtype=bool)
        conf = np.asarray(det_kp.get("confidence", np.zeros(18)), dtype=np.float32)

        gt_bbox = rec.get("gt_bbox_xywh")
        if gt_bbox is not None:
            bw = max(float(gt_bbox[2]), 1.0)
            bh = max(float(gt_bbox[3]), 1.0)
            norm_scale = max(bw, bh)
        else:
            norm_scale = max(float(rec.get("image_width", 1)), float(rec.get("image_height", 1)))

        for j in range(min(len(gt_pts), 18)):
            if not gt_vis[j]:
                continue
            err = float(np.linalg.norm(det_pts[j] - gt_pts[j]))
            err_n = err / norm_scale
            all_err_px.append(err)
            all_err_norm.append(err_n)
            visible_mask.append(not gt_occ[j])
            occluded_mask.append(bool(gt_occ[j]))
            per_joint_sum[j] += err
            per_joint_cnt[j] += 1.0
            conf_sum[j] += conf[j]
            conf_cnt[j] += 1.0

        # Crop sensitivity: compare det on det bbox vs gt on gt bbox if both present
        gt_on_gt = rec.get("gt_on_gt_bbox_keypoints_2d")
        if gt_on_gt is not None:
            gog = np.asarray(gt_on_gt["points"], dtype=np.float32)
            for j in range(18):
                if gt_vis[j]:
                    gt_bbox_err_px.append(float(np.linalg.norm(gog[j] - gt_pts[j])))
                    det_bbox_err_px.append(float(np.linalg.norm(det_pts[j] - gt_pts[j])))

    err_norm_arr = np.asarray(all_err_norm, dtype=np.float64)
    vis_arr = np.asarray(visible_mask, dtype=bool)
    occ_arr = np.asarray(occluded_mask, dtype=bool)

    per_joint = {
        str(j): float(per_joint_sum[j] / per_joint_cnt[j]) if per_joint_cnt[j] > 0 else None
        for j in range(18)
    }
    per_joint_arr = np.array(
        [per_joint_sum[j] / per_joint_cnt[j] if per_joint_cnt[j] > 0 else np.nan for j in range(18)],
        dtype=np.float64,
    )

    thr_x, pck_vals, pck_auc = _pck_curve(err_norm_arr, vis_arr) if err_norm_arr.size else ([], [], float("nan"))

    nme_hist_counts: list[int] = []
    nme_hist_edges: list[float] = []
    if err_norm_arr.size and vis_arr.any():
        counts, edges = np.histogram(err_norm_arr[vis_arr], bins=10, range=(0.0, 0.5))
        nme_hist_counts = counts.astype(int).tolist()
        nme_hist_edges = edges.astype(float).tolist()

    return {
        "num_keypoints_evaluated": int(len(all_err_px)),
        "mean_pixel_error": float(np.mean(all_err_px)) if all_err_px else None,
        "mean_nme": float(np.mean(all_err_norm)) if all_err_norm else None,
        "pck_at_0_1": _pck_at_threshold(err_norm_arr, vis_arr, 0.1) if err_norm_arr.size else None,
        "pck_at_0_2": _pck_at_threshold(err_norm_arr, vis_arr, 0.2) if err_norm_arr.size else None,
        "pck_auc": pck_auc,
        "pck_thresholds": thr_x,
        "pck_values": pck_vals,
        "nme_histogram_counts": nme_hist_counts,
        "nme_histogram_edges": nme_hist_edges,
        "mean_confidence": float(np.mean(conf_sum / np.maximum(conf_cnt, 1.0))),
        "per_joint_mean_px_error": per_joint,
        "group_mean_px_error": {
            g: group_mean(per_joint_arr, g) if np.any(np.isfinite(per_joint_arr[JOINT_GROUPS[g]])) else None
            for g in JOINT_GROUPS
        },
        "visible_mean_px_error": float(np.mean(np.asarray(all_err_px)[vis_arr])) if vis_arr.any() else None,
        "occluded_mean_px_error": float(np.mean(np.asarray(all_err_px)[occ_arr])) if occ_arr.any() else None,
        "crop_sensitivity_mean_px_gt_bbox": float(np.mean(gt_bbox_err_px)) if gt_bbox_err_px else None,
        "crop_sensitivity_mean_px_det_bbox": float(np.mean(det_bbox_err_px)) if det_bbox_err_px else None,
    }

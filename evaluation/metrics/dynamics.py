"""Dynamics metrics: roll, steer, crank from 3D keypoints vs MuJoCo GT."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.common import (
    DEFAULT_MAX_CLIP_MPJPE_MM,
    DEFAULT_SEQUENCE_FPS,
    first_clip_mask,
    frame_mask_for_clips,
    longest_contiguous_slice,
    pearson_r,
    r_squared,
    unique_clip_ids_ordered,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_dynamics_angles import (  # noqa: E402
    bicycle_roll_angle,
    bicycle_steer_angle,
    bicycle_steer_angle_hub,
)
from data_generation_pipeline_tools.visualize_bicycle_pose3d import (  # noqa: E402
    _wrap_pi,
    bicycle_crank_angle,
)


def _rmse_mae(pred_deg: np.ndarray, gt_deg: np.ndarray) -> dict[str, float]:
    diff = pred_deg - gt_deg
    return {
        "rmse_deg": float(np.sqrt(np.mean(diff**2))),
        "mae_deg": float(np.mean(np.abs(diff))),
    }


def _circular_rmse_mae(pred_deg: np.ndarray, gt_deg: np.ndarray) -> dict[str, float]:
    diff = np.rad2deg(np.abs(_wrap_pi(np.deg2rad(pred_deg - gt_deg))))
    return {
        "rmse_deg": float(np.sqrt(np.mean(diff**2))),
        "mae_deg": float(np.mean(diff)),
    }


def _steer_sign_to_reference(pred_deg: np.ndarray, ref_deg: np.ndarray) -> int:
    n = min(len(pred_deg), len(ref_deg))
    if n == 0:
        return 1
    p = pred_deg[:n]
    r = ref_deg[:n]
    pos_err = float(np.mean(np.abs(p - r)))
    neg_err = float(np.mean(np.abs(p + r)))
    return 1 if pos_err <= neg_err else -1


def _velocity_mae(pred_rad: np.ndarray, gt_rad: np.ndarray) -> float:
    if pred_rad.size <= 1:
        return 0.0
    vp = np.diff(pred_rad)
    vg = np.diff(gt_rad)
    return float(np.mean(np.abs(np.rad2deg(vp - vg))))


def _circular_diff_deg(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
    return np.rad2deg(_wrap_pi(np.deg2rad(np.asarray(pred_deg, dtype=np.float64) - gt_deg)))


def _signed_bias_deg(pred_deg: np.ndarray, gt_deg: np.ndarray, *, circular: bool) -> float:
    if circular:
        return float(np.mean(_circular_diff_deg(pred_deg, gt_deg)))
    diff = np.asarray(pred_deg, dtype=np.float64) - np.asarray(gt_deg, dtype=np.float64)
    return float(np.mean(diff))


def _abs_error_deg(pred_deg: np.ndarray, gt_deg: np.ndarray, *, circular: bool) -> np.ndarray:
    if circular:
        return np.abs(_circular_diff_deg(pred_deg, gt_deg))
    return np.abs(np.asarray(pred_deg, dtype=np.float64) - np.asarray(gt_deg, dtype=np.float64))


def _angular_velocity_deg_per_s(
    angle_deg: np.ndarray,
    *,
    fps: float = DEFAULT_SEQUENCE_FPS,
    circular: bool = False,
) -> np.ndarray:
    """Frame-to-frame angular rate; circular angles use shortest arc per step."""
    rad = np.deg2rad(np.asarray(angle_deg, dtype=np.float64))
    if rad.size <= 1:
        return np.array([], dtype=np.float64)
    delta = rad[1:] - rad[:-1]
    if circular:
        delta = np.asarray(_wrap_pi(delta), dtype=np.float64)
    return np.rad2deg(delta) * fps


def _velocity_rmse_deg_per_s(
    pred_rad: np.ndarray,
    gt_rad: np.ndarray,
    *,
    fps: float = DEFAULT_SEQUENCE_FPS,
    circular: bool = False,
) -> float:
    if pred_rad.size <= 1:
        return 0.0
    pred_deg = np.rad2deg(pred_rad)
    gt_deg = np.rad2deg(gt_rad)
    vp = _angular_velocity_deg_per_s(pred_deg, fps=fps, circular=circular)
    vg = _angular_velocity_deg_per_s(gt_deg, fps=fps, circular=circular)
    if vp.size == 0:
        return 0.0
    err = vp - vg
    return float(np.sqrt(np.mean(err**2)))


def _unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(angle_deg, dtype=np.float64))))


def _iter_contiguous_keypoint_segments(
    preds_npz_path: str | Path,
    *,
    accepted_clip_ids: set[str] | None = None,
    max_clip_mpjpe_mm: float = DEFAULT_MAX_CLIP_MPJPE_MM,
):
    """Yield (pred, gt) keypoint arrays for each contiguous clip segment."""
    data = np.load(preds_npz_path, allow_pickle=True)
    pred_full = np.asarray(data["pred"], dtype=np.float32)
    gt_full = np.asarray(data["gt"], dtype=np.float32)
    clip_ids = data.get("clip_ids")
    frame_idx_all = (
        np.asarray(data["frame_idx"], dtype=np.int32) if data.get("frame_idx") is not None else None
    )

    if accepted_clip_ids is None and clip_ids is not None:
        from evaluation.metrics.pose3d import compute_pose3d_metrics

        pose3d = compute_pose3d_metrics(preds_npz_path, max_clip_mpjpe_mm=max_clip_mpjpe_mm)
        accepted_clip_ids = set(pose3d.get("clip_filter", {}).get("accepted_clip_ids", []))

    if clip_ids is None:
        yield pred_full, gt_full
        return

    accepted = {str(c) for c in (accepted_clip_ids or [])}
    for cid in unique_clip_ids_ordered(clip_ids):
        if accepted and cid not in accepted:
            continue
        mask = np.array([str(c) == cid for c in clip_ids])
        pred_seg = pred_full[mask]
        gt_seg = gt_full[mask]
        if frame_idx_all is not None:
            frame_idx = frame_idx_all[mask]
            seg = longest_contiguous_slice(frame_idx)
            pred_seg = pred_seg[seg]
            gt_seg = gt_seg[seg]
        if pred_seg.size:
            yield pred_seg, gt_seg


def _pearson_r_from_angle_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    angle: str,
    circular: bool,
) -> float:
    pred_parts: list[np.ndarray] = []
    gt_parts: list[np.ndarray] = []
    for pred_kpt, gt_kpt in segments:
        pred_deg, gt_deg, _ = _angle_arrays_from_keypoints(pred_kpt, gt_kpt)[angle]
        if circular:
            pred_parts.append(_unwrap_deg(pred_deg))
            gt_parts.append(_unwrap_deg(gt_deg))
        else:
            pred_parts.append(np.asarray(pred_deg, dtype=np.float64))
            gt_parts.append(np.asarray(gt_deg, dtype=np.float64))
    if not pred_parts:
        return float("nan")
    return pearson_r(np.concatenate(pred_parts), np.concatenate(gt_parts))


def _rate_rmse_from_angle_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    angle: str,
    fps: float,
    circular: bool,
) -> float:
    vp_parts: list[np.ndarray] = []
    vg_parts: list[np.ndarray] = []
    for pred_kpt, gt_kpt in segments:
        pred_deg, gt_deg, is_circ = _angle_arrays_from_keypoints(pred_kpt, gt_kpt)[angle]
        use_circular = circular or is_circ
        vp = _angular_velocity_deg_per_s(pred_deg, fps=fps, circular=use_circular)
        vg = _angular_velocity_deg_per_s(gt_deg, fps=fps, circular=use_circular)
        if vp.size:
            vp_parts.append(vp)
            vg_parts.append(vg)
    if not vp_parts:
        return 0.0
    vp_all = np.concatenate(vp_parts)
    vg_all = np.concatenate(vg_parts)
    return float(np.sqrt(np.mean((vp_all - vg_all) ** 2)))


def _angle_arrays_from_keypoints(
    pred: np.ndarray,
    gt: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "roll": (
            np.rad2deg(bicycle_roll_angle(pred)),
            np.rad2deg(bicycle_roll_angle(gt)),
            False,
        ),
        "steer": (
            np.rad2deg(bicycle_steer_angle(pred)),
            np.rad2deg(bicycle_steer_angle(gt)),
            True,
        ),
        "crank": (
            np.rad2deg(bicycle_crank_angle(pred)),
            np.rad2deg(bicycle_crank_angle(gt)),
            True,
        ),
    }


def _angle_stats(pred_deg: np.ndarray, gt_deg: np.ndarray, *, circular: bool) -> dict[str, float]:
    if circular:
        rmse_mae = _circular_rmse_mae(pred_deg, gt_deg)
    else:
        rmse_mae = _rmse_mae(pred_deg, gt_deg)
    out = {
        **rmse_mae,
        "bias_deg": _signed_bias_deg(pred_deg, gt_deg, circular=circular),
        "pearson_r": pearson_r(pred_deg, gt_deg),
    }
    if circular:
        out["r2"] = r_squared(pred_deg, gt_deg)
    return out


def _load_npz_angles(
    preds_npz_path: str | Path,
    *,
    accepted_clip_ids: set[str] | None = None,
    max_clip_mpjpe_mm: float = DEFAULT_MAX_CLIP_MPJPE_MM,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, bool]], np.ndarray | None, set[str]]:
    data = np.load(preds_npz_path, allow_pickle=True)
    pred_full = np.asarray(data["pred"], dtype=np.float32)
    gt_full = np.asarray(data["gt"], dtype=np.float32)
    clip_ids = data.get("clip_ids")

    if accepted_clip_ids is None and clip_ids is not None:
        from evaluation.metrics.pose3d import compute_pose3d_metrics

        pose3d = compute_pose3d_metrics(preds_npz_path, max_clip_mpjpe_mm=max_clip_mpjpe_mm)
        accepted_clip_ids = set(pose3d.get("clip_filter", {}).get("accepted_clip_ids", []))

    if clip_ids is not None and accepted_clip_ids:
        frame_mask = frame_mask_for_clips(clip_ids, accepted_clip_ids)
        pred = pred_full[frame_mask]
        gt = gt_full[frame_mask]
        clip_ids_filtered = np.asarray(clip_ids)[frame_mask]
    else:
        pred = pred_full
        gt = gt_full
        clip_ids_filtered = np.asarray(clip_ids) if clip_ids is not None else None
        accepted_clip_ids = accepted_clip_ids or set()

    angles = _angle_arrays_from_keypoints(pred, gt)
    return angles, clip_ids_filtered, accepted_clip_ids


def compute_kinematics_scorecard_metrics(
    preds_npz_path: str | Path,
    *,
    accepted_clip_ids: set[str] | None = None,
    max_clip_mpjpe_mm: float = DEFAULT_MAX_CLIP_MPJPE_MM,
    fps: float = DEFAULT_SEQUENCE_FPS,
) -> dict[str, float]:
    """Aggregate kinematics metrics for thesis scorecard rows (pred vs GT keypoints)."""
    segments = list(
        _iter_contiguous_keypoint_segments(
            preds_npz_path,
            accepted_clip_ids=accepted_clip_ids,
            max_clip_mpjpe_mm=max_clip_mpjpe_mm,
        )
    )
    angles, _, _ = _load_npz_angles(
        preds_npz_path,
        accepted_clip_ids=accepted_clip_ids,
        max_clip_mpjpe_mm=max_clip_mpjpe_mm,
    )
    out: dict[str, float] = {}
    for name, (pred_deg, gt_deg, circular) in angles.items():
        stats = _angle_stats(pred_deg, gt_deg, circular=circular)
        out[f"{name}_rmse_deg"] = stats["rmse_deg"]
        out[f"{name}_bias_deg"] = stats["bias_deg"]
        out[f"{name}_pearson_r"] = _pearson_r_from_angle_segments(
            segments, angle=name, circular=circular
        )
        out[f"{name}_rate_rmse_deg_per_s"] = _rate_rmse_from_angle_segments(
            segments, angle=name, fps=fps, circular=circular
        )
    return out


def framewise_abs_error_curves(
    preds_npz_path: str | Path,
    *,
    angle: str = "roll",
    accepted_clip_ids: set[str] | None = None,
    max_clip_mpjpe_mm: float = DEFAULT_MAX_CLIP_MPJPE_MM,
    n_grid: int = 100,
) -> dict[str, Any]:
    """Per-clip abs-error curves on normalized time [0,1]; returns mean/std for plotting."""
    data = np.load(preds_npz_path, allow_pickle=True)
    pred_full = np.asarray(data["pred"], dtype=np.float32)
    gt_full = np.asarray(data["gt"], dtype=np.float32)
    clip_ids = data.get("clip_ids")
    frame_idx_all = (
        np.asarray(data["frame_idx"], dtype=np.int32) if data.get("frame_idx") is not None else None
    )

    if accepted_clip_ids is None and clip_ids is not None:
        from evaluation.metrics.pose3d import compute_pose3d_metrics

        pose3d = compute_pose3d_metrics(preds_npz_path, max_clip_mpjpe_mm=max_clip_mpjpe_mm)
        accepted_clip_ids = set(pose3d.get("clip_filter", {}).get("accepted_clip_ids", []))

    if clip_ids is None:
        raise ValueError(f"{preds_npz_path} has no clip_ids")

    accepted = {str(c) for c in (accepted_clip_ids or [])}
    grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float64)
    curves: list[np.ndarray] = []

    for cid in unique_clip_ids_ordered(clip_ids):
        if accepted and cid not in accepted:
            continue
        mask = np.array([str(c) == cid for c in clip_ids])
        pred_viz = pred_full[mask]
        gt_viz = gt_full[mask]
        frame_idx_viz = frame_idx_all[mask] if frame_idx_all is not None else None

        if frame_idx_viz is not None:
            seg = longest_contiguous_slice(frame_idx_viz)
            pred_viz = pred_viz[seg]
            gt_viz = gt_viz[seg]

        if pred_viz.size == 0:
            continue

        angles = _angle_arrays_from_keypoints(pred_viz, gt_viz)
        if angle not in angles:
            raise ValueError(f"unknown angle: {angle}")
        pred_deg, gt_deg, circular = angles[angle]
        err = _abs_error_deg(pred_deg, gt_deg, circular=circular)
        if err.size < 2:
            t_src = np.array([0.0, 1.0], dtype=np.float64)
            err_src = np.array([float(err[0]), float(err[0])], dtype=np.float64)
        else:
            t_src = np.linspace(0.0, 1.0, err.size, dtype=np.float64)
            err_src = err.astype(np.float64)
        curves.append(np.interp(grid, t_src, err_src))

    if not curves:
        return {"grid": grid.tolist(), "curves": [], "mean": [], "std": []}

    arr = np.stack(curves, axis=0)
    return {
        "grid": grid.tolist(),
        "curves": arr,
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
    }


def mean_coherence_across_clips(
    preds_npz_path: str | Path,
    *,
    angle: str = "roll",
    accepted_clip_ids: set[str] | None = None,
    max_clip_mpjpe_mm: float = DEFAULT_MAX_CLIP_MPJPE_MM,
    fps: float = DEFAULT_SEQUENCE_FPS,
    nperseg: int = 256,
) -> dict[str, Any]:
    """Mean magnitude-squared coherence C_xy(f) across accepted clips."""
    from scipy.signal import coherence

    data = np.load(preds_npz_path, allow_pickle=True)
    pred_full = np.asarray(data["pred"], dtype=np.float32)
    gt_full = np.asarray(data["gt"], dtype=np.float32)
    clip_ids = data.get("clip_ids")
    frame_idx_all = (
        np.asarray(data["frame_idx"], dtype=np.int32) if data.get("frame_idx") is not None else None
    )

    if accepted_clip_ids is None and clip_ids is not None:
        from evaluation.metrics.pose3d import compute_pose3d_metrics

        pose3d = compute_pose3d_metrics(preds_npz_path, max_clip_mpjpe_mm=max_clip_mpjpe_mm)
        accepted_clip_ids = set(pose3d.get("clip_filter", {}).get("accepted_clip_ids", []))

    if clip_ids is None:
        raise ValueError(f"{preds_npz_path} has no clip_ids")

    accepted = {str(c) for c in (accepted_clip_ids or [])}
    freq_ref: np.ndarray | None = None
    coh_list: list[np.ndarray] = []

    for cid in unique_clip_ids_ordered(clip_ids):
        if accepted and cid not in accepted:
            continue
        mask = np.array([str(c) == cid for c in clip_ids])
        pred_viz = pred_full[mask]
        gt_viz = gt_full[mask]
        frame_idx_viz = frame_idx_all[mask] if frame_idx_all is not None else None

        if frame_idx_viz is not None:
            seg = longest_contiguous_slice(frame_idx_viz)
            pred_viz = pred_viz[seg]
            gt_viz = gt_viz[seg]

        if pred_viz.size < 8:
            continue

        angles = _angle_arrays_from_keypoints(pred_viz, gt_viz)
        if angle not in angles:
            raise ValueError(f"unknown angle: {angle}")
        pred_deg, gt_deg, _ = angles[angle]
        seg_len = min(len(pred_deg), nperseg)
        if seg_len < 8:
            continue
        f, cxy = coherence(
            pred_deg.astype(np.float64),
            gt_deg.astype(np.float64),
            fs=fps,
            nperseg=seg_len,
        )
        if freq_ref is None:
            freq_ref = f
            coh_list.append(cxy)
        elif len(f) == len(freq_ref) and np.allclose(f, freq_ref):
            coh_list.append(cxy)
        else:
            coh_interp = np.interp(freq_ref, f, cxy, left=0.0, right=0.0)
            coh_list.append(coh_interp)

    if not coh_list or freq_ref is None:
        return {"freq_hz": [], "mean": [], "std": []}

    arr = np.stack(coh_list, axis=0)
    return {
        "freq_hz": freq_ref.tolist(),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
    }


def pooled_bland_altman_arrays(
    preds_npz_path: str | Path,
    *,
    angle: str = "roll",
    accepted_clip_ids: set[str] | None = None,
    max_clip_mpjpe_mm: float = DEFAULT_MAX_CLIP_MPJPE_MM,
) -> dict[str, np.ndarray]:
    """Pooled pred/GT/diff arrays for Bland-Altman (all accepted frames)."""
    angles, _, _ = _load_npz_angles(
        preds_npz_path,
        accepted_clip_ids=accepted_clip_ids,
        max_clip_mpjpe_mm=max_clip_mpjpe_mm,
    )
    if angle not in angles:
        raise ValueError(f"unknown angle: {angle}")
    pred_deg, gt_deg, circular = angles[angle]
    diff = _circular_diff_deg(pred_deg, gt_deg) if circular else (pred_deg - gt_deg)
    return {"x": gt_deg.astype(np.float64), "diff": diff.astype(np.float64)}


def _build_time_series_payload(
    *,
    clip_id: str,
    pred_viz: np.ndarray,
    gt_viz: np.ndarray,
    mujoco_steer_viz: np.ndarray | None,
    gt_roll_mujoco_viz: np.ndarray | None,
    frame_idx_viz: np.ndarray | None,
    segment_id_viz: np.ndarray | None = None,
) -> dict[str, Any]:
    pred_steer_viz = np.rad2deg(bicycle_steer_angle(pred_viz))
    pred_roll_viz = np.rad2deg(bicycle_roll_angle(pred_viz))
    pred_crank_viz = np.rad2deg(bicycle_crank_angle(pred_viz))
    gt_roll_kpt_viz = np.rad2deg(bicycle_roll_angle(gt_viz))
    gt_crank_kpt_viz = np.rad2deg(bicycle_crank_angle(gt_viz))
    if mujoco_steer_viz is not None:
        viz_steer_sign = _steer_sign_to_reference(pred_steer_viz, mujoco_steer_viz)
        gt_steer_mujoco_viz = viz_steer_sign * mujoco_steer_viz
    else:
        gt_steer_mujoco_viz = np.rad2deg(bicycle_steer_angle(gt_viz))

    payload: dict[str, Any] = {
        "clip_id": clip_id,
        "pred_steer_deg": pred_steer_viz.tolist(),
        "gt_steer_deg": gt_steer_mujoco_viz.tolist(),
        "pred_roll_deg": pred_roll_viz.tolist(),
        "gt_roll_deg": gt_roll_kpt_viz.tolist(),
        "pred_crank_deg": pred_crank_viz.tolist(),
        "gt_crank_deg": gt_crank_kpt_viz.tolist(),
        "gt_roll_mujoco_deg": (
            gt_roll_mujoco_viz.tolist() if gt_roll_mujoco_viz is not None else None
        ),
    }
    if frame_idx_viz is not None:
        payload["frame_idx"] = np.asarray(frame_idx_viz, dtype=np.int32).tolist()
    if segment_id_viz is not None:
        payload["segment_id"] = np.asarray(segment_id_viz, dtype=np.int32).tolist()
    return payload


def build_time_series_for_clip(
    preds_npz_path: str | Path,
    clip_id: str,
    *,
    contiguous_only: bool = True,
) -> dict[str, Any]:
    """Build steer/roll time series for a single clip (for thesis figures)."""
    data = np.load(preds_npz_path, allow_pickle=True)
    clip_ids = data.get("clip_ids")
    if clip_ids is None:
        raise ValueError(f"{preds_npz_path} has no clip_ids")
    mask = np.array([str(c) == clip_id for c in clip_ids])
    if not np.any(mask):
        raise ValueError(f"clip_id not found in {preds_npz_path}: {clip_id}")

    pred_viz = np.asarray(data["pred"], dtype=np.float32)[mask]
    gt_viz = np.asarray(data["gt"], dtype=np.float32)[mask]
    frame_idx_viz = (
        np.asarray(data["frame_idx"], dtype=np.int32)[mask]
        if data.get("frame_idx") is not None
        else None
    )
    mujoco_steer_viz = (
        np.asarray(data["steer_deg"], dtype=np.float32)[mask]
        if data.get("steer_deg") is not None
        else None
    )
    gt_roll_mujoco_viz = (
        np.asarray(data["roll_deg"], dtype=np.float32)[mask]
        if data.get("roll_deg") is not None
        else None
    )

    segment_id_viz = (
        np.asarray(data["segment_id"], dtype=np.int32)[mask]
        if data.get("segment_id") is not None
        else None
    )

    if contiguous_only and frame_idx_viz is not None:
        seg = longest_contiguous_slice(frame_idx_viz)
        pred_viz = pred_viz[seg]
        gt_viz = gt_viz[seg]
        frame_idx_viz = frame_idx_viz[seg]
        if mujoco_steer_viz is not None:
            mujoco_steer_viz = mujoco_steer_viz[seg]
        if gt_roll_mujoco_viz is not None:
            gt_roll_mujoco_viz = gt_roll_mujoco_viz[seg]
        if segment_id_viz is not None:
            segment_id_viz = segment_id_viz[seg]

    return _build_time_series_payload(
        clip_id=clip_id,
        pred_viz=pred_viz,
        gt_viz=gt_viz,
        mujoco_steer_viz=mujoco_steer_viz,
        gt_roll_mujoco_viz=gt_roll_mujoco_viz,
        frame_idx_viz=frame_idx_viz,
        segment_id_viz=segment_id_viz,
    )


def compute_dynamics_metrics(
    preds_npz_path: str | Path,
    *,
    accepted_clip_ids: set[str] | None = None,
    max_clip_mpjpe_mm: float = DEFAULT_MAX_CLIP_MPJPE_MM,
) -> dict[str, Any]:
    data = np.load(preds_npz_path, allow_pickle=True)
    pred_full = np.asarray(data["pred"], dtype=np.float32)
    gt_full = np.asarray(data["gt"], dtype=np.float32)
    clip_ids = data.get("clip_ids")

    if accepted_clip_ids is None and clip_ids is not None:
        from evaluation.metrics.pose3d import compute_pose3d_metrics

        pose3d = compute_pose3d_metrics(preds_npz_path, max_clip_mpjpe_mm=max_clip_mpjpe_mm)
        accepted_clip_ids = set(pose3d.get("clip_filter", {}).get("accepted_clip_ids", []))

    if clip_ids is not None and accepted_clip_ids:
        frame_mask = frame_mask_for_clips(clip_ids, accepted_clip_ids)
        pred = pred_full[frame_mask]
        gt = gt_full[frame_mask]
        data_steer = (
            np.asarray(data["steer_deg"], dtype=np.float32)[frame_mask]
            if data.get("steer_deg") is not None
            else None
        )
        data_roll = (
            np.asarray(data["roll_deg"], dtype=np.float32)[frame_mask]
            if data.get("roll_deg") is not None
            else None
        )
    else:
        pred = pred_full
        gt = gt_full
        data_steer = np.asarray(data["steer_deg"], dtype=np.float32) if data.get("steer_deg") is not None else None
        data_roll = np.asarray(data["roll_deg"], dtype=np.float32) if data.get("roll_deg") is not None else None

    pred_steer = np.rad2deg(bicycle_steer_angle(pred))
    pred_steer_hub = np.rad2deg(bicycle_steer_angle_hub(pred))
    pred_roll = np.rad2deg(bicycle_roll_angle(pred))
    pred_crank = np.rad2deg(bicycle_crank_angle(pred))
    gt_steer_kpt = np.rad2deg(bicycle_steer_angle(gt))
    gt_steer_hub = np.rad2deg(bicycle_steer_angle_hub(gt))
    gt_roll_kpt = np.rad2deg(bicycle_roll_angle(gt))
    gt_crank_kpt = np.rad2deg(bicycle_crank_angle(gt))

    steer_keypoint = _circular_rmse_mae(pred_steer, gt_steer_kpt)
    steer_keypoint["bias_deg"] = _signed_bias_deg(pred_steer, gt_steer_kpt, circular=True)
    steer_keypoint["pearson_r"] = pearson_r(pred_steer, gt_steer_kpt)
    steer_hub_keypoint = _circular_rmse_mae(pred_steer_hub, gt_steer_hub)
    roll_keypoint = _rmse_mae(pred_roll, gt_roll_kpt)
    roll_keypoint["bias_deg"] = _signed_bias_deg(pred_roll, gt_roll_kpt, circular=False)
    roll_keypoint["pearson_r"] = pearson_r(pred_roll, gt_roll_kpt)
    crank_keypoint = _circular_rmse_mae(pred_crank, gt_crank_kpt)
    crank_keypoint["bias_deg"] = _signed_bias_deg(pred_crank, gt_crank_kpt, circular=True)
    crank_keypoint["pearson_r"] = pearson_r(pred_crank, gt_crank_kpt)
    crank_keypoint["r2"] = r_squared(pred_crank, gt_crank_kpt)

    # Primary headline metrics: kinematic angles from predicted vs GT keypoints.
    steer_stats = steer_keypoint
    roll_stats = roll_keypoint
    mujoco_steer = data_steer
    mujoco_roll = data_roll
    mujoco_metrics: dict[str, Any] = {}
    steer_sign = 1
    steer_hub_diagnostic: dict[str, Any] = {
        "keypoint_self_consistent": steer_hub_keypoint,
    }

    if mujoco_steer is not None and mujoco_roll is not None:
        ms = np.asarray(mujoco_steer, dtype=np.float32)
        mr = np.asarray(mujoco_roll, dtype=np.float32)
        n = min(len(ms), len(pred_steer))
        steer_sign = _steer_sign_to_reference(pred_steer[:n], ms[:n])
        ms_aligned = steer_sign * ms[:n]
        hub_sign = _steer_sign_to_reference(pred_steer_hub[:n], ms[:n])
        ms_hub_aligned = hub_sign * ms[:n]
        steer_vs_mujoco = _circular_rmse_mae(pred_steer[:n], ms_aligned)
        roll_vs_mujoco = _rmse_mae(pred_roll[:n], mr[:n])
        roll_gt_kpt_vs_mujoco = _rmse_mae(gt_roll_kpt[:n], mr[:n])
        steer_hub_diagnostic.update(
            {
                "steer_sign_to_mujoco": hub_sign,
                "steer_vs_mujoco": _circular_rmse_mae(pred_steer_hub[:n], ms_hub_aligned),
                "steer_pearson_r": pearson_r(pred_steer_hub[:n], ms_hub_aligned),
                "steer_r2": r_squared(pred_steer_hub[:n], ms_hub_aligned),
            }
        )
        mujoco_metrics = {
            "steer_sign_to_mujoco": steer_sign,
            "steer_vs_mujoco": steer_vs_mujoco,
            "roll_vs_mujoco": roll_vs_mujoco,
            "roll_gt_kpt_vs_mujoco": roll_gt_kpt_vs_mujoco,
            "steer_pearson_r": pearson_r(pred_steer[:n], ms_aligned),
            "steer_r2": r_squared(pred_steer[:n], ms_aligned),
            "roll_pearson_r": pearson_r(pred_roll[:n], mr[:n]),
            "roll_r2": r_squared(pred_roll[:n], mr[:n]),
            "steer_keypoint_self_consistent": steer_keypoint,
            "roll_keypoint_self_consistent": roll_keypoint,
        }

    viz_clip_id, viz_mask = first_clip_mask(clip_ids, accepted_clip_ids)
    if viz_mask is not None:
        pred_viz = pred_full[viz_mask]
        gt_viz = gt_full[viz_mask]
        frame_idx_viz = (
            np.asarray(data["frame_idx"], dtype=np.int32)[viz_mask]
            if data.get("frame_idx") is not None
            else None
        )
        mujoco_steer_viz = (
            np.asarray(data["steer_deg"], dtype=np.float32)[viz_mask]
            if data.get("steer_deg") is not None
            else None
        )
        gt_roll_mujoco_viz = (
            np.asarray(data["roll_deg"], dtype=np.float32)[viz_mask]
            if data.get("roll_deg") is not None
            else None
        )
        segment_id_viz = (
            np.asarray(data["segment_id"], dtype=np.int32)[viz_mask]
            if data.get("segment_id") is not None
            else None
        )
        if frame_idx_viz is not None:
            seg = longest_contiguous_slice(frame_idx_viz)
            pred_viz = pred_viz[seg]
            gt_viz = gt_viz[seg]
            frame_idx_viz = frame_idx_viz[seg]
            if mujoco_steer_viz is not None:
                mujoco_steer_viz = mujoco_steer_viz[seg]
            if gt_roll_mujoco_viz is not None:
                gt_roll_mujoco_viz = gt_roll_mujoco_viz[seg]
            if segment_id_viz is not None:
                segment_id_viz = segment_id_viz[seg]
        time_series = _build_time_series_payload(
            clip_id=str(viz_clip_id),
            pred_viz=pred_viz,
            gt_viz=gt_viz,
            mujoco_steer_viz=mujoco_steer_viz,
            gt_roll_mujoco_viz=gt_roll_mujoco_viz,
            frame_idx_viz=frame_idx_viz,
            segment_id_viz=segment_id_viz,
        )
    else:
        time_series = _build_time_series_payload(
            clip_id="all",
            pred_viz=pred,
            gt_viz=gt,
            mujoco_steer_viz=mujoco_steer if mujoco_steer is not None else None,
            gt_roll_mujoco_viz=(
                np.asarray(data["roll_deg"], dtype=np.float32) if data.get("roll_deg") is not None else None
            ),
            frame_idx_viz=None,
        )

    # Error vs magnitude bins
    def _error_vs_magnitude(pred_deg: np.ndarray, gt_deg: np.ndarray) -> list[dict[str, float]]:
        mag = np.abs(gt_deg)
        bins = np.percentile(mag, [0, 25, 50, 75, 100])
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (mag >= lo) & (mag <= hi)
            if np.any(m):
                out.append(
                    {
                        "bin_lo": float(lo),
                        "bin_hi": float(hi),
                        "mae_deg": float(np.mean(np.abs(pred_deg[m] - gt_deg[m]))),
                    }
                )
        return out

    per_joint_mpjpe = np.linalg.norm(pred - gt, axis=-1).mean(axis=0)
    frame_steer_err = np.rad2deg(np.abs(_wrap_pi(np.deg2rad(pred_steer - gt_steer_kpt))))
    frame_roll_err = np.abs(pred_roll - gt_roll_kpt)
    frame_crank_err = np.rad2deg(np.abs(_wrap_pi(np.deg2rad(pred_crank - gt_crank_kpt))))
    joint_dyn_corr = {
        "steer_err_mean_deg": float(np.mean(frame_steer_err)),
        "roll_err_mean_deg": float(np.mean(frame_roll_err)),
        "crank_err_mean_deg": float(np.mean(frame_crank_err)),
    }

    return {
        "steer": steer_stats,
        "roll": roll_stats,
        "crank": crank_keypoint,
        "steer_velocity_mae_deg_per_frame": _velocity_mae(
            np.deg2rad(pred_steer), np.deg2rad(gt_steer_kpt)
        ),
        "roll_velocity_mae_deg_per_frame": _velocity_mae(
            np.deg2rad(pred_roll), np.deg2rad(gt_roll_kpt)
        ),
        "crank_velocity_mae_deg_per_frame": _velocity_mae(
            np.deg2rad(pred_crank), np.deg2rad(gt_crank_kpt)
        ),
        "steer_velocity_rmse_deg_per_s": _velocity_rmse_deg_per_s(
            np.deg2rad(pred_steer), np.deg2rad(gt_steer_kpt), circular=True
        ),
        "roll_velocity_rmse_deg_per_s": _velocity_rmse_deg_per_s(
            np.deg2rad(pred_roll), np.deg2rad(gt_roll_kpt)
        ),
        "crank_velocity_rmse_deg_per_s": _velocity_rmse_deg_per_s(
            np.deg2rad(pred_crank), np.deg2rad(gt_crank_kpt), circular=True
        ),
        "mujoco_gt": mujoco_metrics,
        "steer_hub_only": steer_hub_diagnostic,
        "steer_error_vs_magnitude": _error_vs_magnitude(pred_steer, gt_steer_kpt),
        "roll_error_vs_magnitude": _error_vs_magnitude(pred_roll, gt_roll_kpt),
        "joint_dynamics_summary": joint_dyn_corr,
        "per_joint_mpjpe_m": {str(j): float(per_joint_mpjpe[j]) for j in range(len(per_joint_mpjpe))},
        "time_series": time_series,
    }

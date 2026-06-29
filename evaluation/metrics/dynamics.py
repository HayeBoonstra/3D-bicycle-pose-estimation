"""Dynamics metrics: roll, steer, crank from 3D keypoints vs MuJoCo GT."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.common import pearson_r, r_squared

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.visualize_bicycle_pose3d import (  # noqa: E402
    bicycle_crank_angle,
    bicycle_roll_angle,
    bicycle_steer_angle,
)


def _rmse_mae(pred_deg: np.ndarray, gt_deg: np.ndarray) -> dict[str, float]:
    diff = pred_deg - gt_deg
    return {
        "rmse_deg": float(np.sqrt(np.mean(diff**2))),
        "mae_deg": float(np.mean(np.abs(diff))),
    }


def _velocity_mae(pred_rad: np.ndarray, gt_rad: np.ndarray) -> float:
    if pred_rad.size <= 1:
        return 0.0
    vp = np.diff(pred_rad)
    vg = np.diff(gt_rad)
    return float(np.mean(np.abs(np.rad2deg(vp - vg))))


def compute_dynamics_metrics(preds_npz_path: str | Path) -> dict[str, Any]:
    data = np.load(preds_npz_path, allow_pickle=True)
    pred = np.asarray(data["pred"], dtype=np.float32)
    gt = np.asarray(data["gt"], dtype=np.float32)

    pred_steer = np.rad2deg(bicycle_steer_angle(pred))
    pred_roll = np.rad2deg(bicycle_roll_angle(pred))
    pred_crank = np.rad2deg(bicycle_crank_angle(pred))
    gt_steer = np.rad2deg(bicycle_steer_angle(gt))
    gt_roll = np.rad2deg(bicycle_roll_angle(gt))
    gt_crank = np.rad2deg(bicycle_crank_angle(gt))

    steer_stats = _rmse_mae(pred_steer, gt_steer)
    roll_stats = _rmse_mae(pred_roll, gt_roll)
    crank_stats = _rmse_mae(pred_crank, gt_crank)

    # Compare to MuJoCo dynamics_gt if present
    mujoco_steer = data.get("steer_deg")
    mujoco_roll = data.get("roll_deg")
    mujoco_metrics: dict[str, Any] = {}
    if mujoco_steer is not None and mujoco_roll is not None:
        ms = np.asarray(mujoco_steer, dtype=np.float32)
        mr = np.asarray(mujoco_roll, dtype=np.float32)
        n = min(len(ms), len(pred_steer))
        mujoco_metrics = {
            "steer_vs_mujoco": _rmse_mae(pred_steer[:n], ms[:n]),
            "roll_vs_mujoco": _rmse_mae(pred_roll[:n], mr[:n]),
            "steer_pearson_r": pearson_r(pred_steer[:n], ms[:n]),
            "steer_r2": r_squared(pred_steer[:n], ms[:n]),
            "roll_pearson_r": pearson_r(pred_roll[:n], mr[:n]),
            "roll_r2": r_squared(pred_roll[:n], mr[:n]),
        }

    # Error vs magnitude bins
    def _error_vs_magnitude(pred_deg: np.ndarray, gt_deg: np.ndarray) -> list[dict[str, float]]:
        mag = np.abs(gt_deg)
        bins = np.percentile(mag, [0, 25, 50, 75, 100])
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (mag >= lo) & (mag <= hi)
            if np.any(m):
                out.append({"bin_lo": float(lo), "bin_hi": float(hi), "mae_deg": float(np.mean(np.abs(pred_deg[m] - gt_deg[m])))})
        return out

    # Per-joint MPJPE correlation with dynamics error
    per_joint_mpjpe = np.linalg.norm(pred - gt, axis=-1).mean(axis=0)
    frame_steer_err = np.abs(pred_steer - gt_steer)
    frame_roll_err = np.abs(pred_roll - gt_roll)
    joint_dyn_corr = {
        "steer_err_mean_deg": float(np.mean(frame_steer_err)),
        "roll_err_mean_deg": float(np.mean(frame_roll_err)),
    }

    return {
        "steer": steer_stats,
        "roll": roll_stats,
        "crank": crank_stats,
        "steer_velocity_mae_deg_per_frame": _velocity_mae(np.deg2rad(pred_steer), np.deg2rad(gt_steer)),
        "roll_velocity_mae_deg_per_frame": _velocity_mae(np.deg2rad(pred_roll), np.deg2rad(gt_roll)),
        "mujoco_gt": mujoco_metrics,
        "steer_error_vs_magnitude": _error_vs_magnitude(pred_steer, gt_steer),
        "roll_error_vs_magnitude": _error_vs_magnitude(pred_roll, gt_roll),
        "joint_dynamics_summary": joint_dyn_corr,
        "per_joint_mpjpe_m": {str(j): float(per_joint_mpjpe[j]) for j in range(len(per_joint_mpjpe))},
        "time_series": {
            "pred_steer_deg": pred_steer.tolist(),
            "gt_steer_deg": gt_steer.tolist(),
            "pred_roll_deg": pred_roll.tolist(),
            "gt_roll_deg": gt_roll.tolist(),
        },
    }

#!/usr/bin/env python3
"""Generate thesis result figures from computed metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES  # noqa: E402
from evaluation.common import (
    CAPACITY_EXPERIMENTS,
    CAPACITY_GT_TRAINING_EXPERIMENTS,
    HEADLINE_EXPERIMENT,
    WINDOW_ABLATION_PLOT_EXPERIMENTS,
    WINDOW_BASELINE_EXPERIMENT,
    ensure_dir,
    EXCLUDED_ABLATION_EXPERIMENTS,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def plot_per_joint_mpjpe(metrics: dict[str, Any], out_path: Path) -> None:
    per_joint = metrics.get("pose3d", {}).get("per_joint_mpjpe_mm", {})
    if not per_joint:
        return
    joints = [BICYCLE_KEYPOINT_NAMES[int(k)] for k in sorted(per_joint, key=int)]
    vals = [per_joint[str(k)] for k in sorted(per_joint, key=int)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(vals)), vals, color="steelblue")
    ax.set_xticks(range(len(joints)))
    ax.set_xticklabels([j.replace("k_", "") for j in joints], rotation=45, ha="right")
    ax.set_ylabel("MPJPE (mm)")
    ax.axhline(40, color="red", linestyle="--", label="40 mm target")
    ax.legend()
    ax.set_title("Per-joint MPJPE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _normalized_hist(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    return arr / total if total > 0 else arr


def plot_pck_histogram(metrics: dict[str, Any], out_path: Path, *, title_suffix: str = "") -> None:
    """Normalized histogram of per-keypoint NME (visible joints only)."""
    p2 = metrics.get("pose2d", {})
    counts = p2.get("nme_histogram_counts", [])
    edges = p2.get("nme_histogram_edges", [])
    if not counts or not edges or len(edges) < 2:
        return
    fractions = _normalized_hist(counts)
    centers = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    width = float(edges[1] - edges[0]) * 0.9
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(centers, fractions, width=width, color="steelblue")
    ax.set_xlabel("Normalized distance (NME)")
    ax.set_ylabel("Fraction of keypoints")
    ax.set_ylim(0.0, max(0.05, float(fractions.max()) * 1.15))
    suffix = f" ({title_suffix})" if title_suffix else ""
    ax.set_title(f"PCK error distribution{suffix}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_iou_histogram(metrics: dict[str, Any], out_path: Path, *, title_suffix: str = "") -> None:
    det = metrics.get("detection", {})
    fractions = det.get("iou_histogram_fraction")
    if not fractions:
        counts = det.get("iou_histogram", [])
        if not counts:
            return
        fractions = _normalized_hist(counts).tolist()
    centers = np.linspace(0.05, 0.95, len(fractions))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(centers, fractions, width=0.08, color="teal")
    ax.set_xlabel("IoU bin center")
    ax.set_ylabel("Fraction of detections")
    suffix = f" ({title_suffix})" if title_suffix else ""
    ax.set_title(f"Detection IoU distribution{suffix}")
    ax.set_ylim(0.0, max(0.05, float(np.max(fractions)) * 1.15))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_capacity_frontier(frontier: dict[str, Any], summary_rows: list[dict], out_path: Path) -> None:
    mpjpe_by_exp = {r["experiment"]: r.get("mpjpe_mm") for r in summary_rows if r.get("mpjpe_mm") is not None}
    xs, ys, labels = [], [], []
    for name, eff in frontier.items():
        exp_key = f"capacity_{name.lower()}"
        if exp_key in mpjpe_by_exp:
            xs.append(eff.get("params_M", 0))
            ys.append(mpjpe_by_exp[exp_key])
            labels.append(name)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, s=80, c="darkorange")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Capacity vs accuracy frontier")
    ax.axhline(40, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_capacity_detected_vs_gt2d(summary_rows: list[dict], out_path: Path) -> None:
    """Grouped bar chart: detected-2D vs GT-2D MPJPE for capacity S/B/L/X."""
    mpjpe_by_exp = {
        r["experiment"]: r.get("mpjpe_mm")
        for r in summary_rows
        if r.get("mpjpe_mm") is not None
    }
    labels: list[str] = []
    detected_vals: list[float] = []
    gt2d_vals: list[float] = []
    for exp_name in CAPACITY_EXPERIMENTS:
        det_mpjpe = mpjpe_by_exp.get(exp_name)
        gt_mpjpe = mpjpe_by_exp.get(f"{exp_name}_gt2d")
        if det_mpjpe is None or gt_mpjpe is None:
            continue
        labels.append(exp_name.replace("capacity_", "").upper())
        detected_vals.append(float(det_mpjpe))
        gt2d_vals.append(float(gt_mpjpe))
    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, detected_vals, width, label="Detected 2D", color="steelblue")
    ax.bar(x + width / 2, gt2d_vals, width, label="GT-2D", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Capacity: detected vs GT-2D input")
    ax.axhline(40, color="red", linestyle="--", alpha=0.5, label="40 mm target")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_capacity_gt_training(summary_rows: list[dict], out_path: Path) -> None:
    """Grouped MPJPE/NMPJPE and MPJVE bars for capacity S/B/L/X (GT-trained)."""
    by_exp = {r["experiment"]: r for r in summary_rows}
    labels: list[str] = []
    mpjpe_vals: list[float] = []
    nmpjpe_vals: list[float] = []
    mpjve_vals: list[float] = []
    for exp_name in CAPACITY_GT_TRAINING_EXPERIMENTS:
        row = by_exp.get(exp_name)
        if row is None or row.get("mpjpe_mm") is None:
            continue
        labels.append(exp_name.replace("capacity_", "").replace("_gt", "").upper())
        mpjpe_vals.append(float(row["mpjpe_mm"]))
        nmpjpe_vals.append(float(row.get("n_mpjpe_mm") or 0.0))
        mpjve_vals.append(float(row.get("mpjve_mm_per_s") or 0.0))
    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

    axes[0].bar(x - width / 2, mpjpe_vals, width, label="MPJPE (mm)", color="darkorange")
    axes[0].bar(x + width / 2, nmpjpe_vals, width, label="NMPJPE (mm)", color="sandybrown")
    axes[0].set_ylabel("Error (mm)")
    axes[0].set_title("Capacity ablation (GT-trained, GT test corpus)")
    axes[0].axhline(40, color="red", linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, mpjve_vals, width, label="MPJVE (mm/s)", color="peru")
    axes[1].set_ylabel("Error (mm/s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_capacity_detected_vs_gt_training(summary_rows: list[dict], out_path: Path) -> None:
    """Grouped bar chart: detected-trained vs GT-trained MPJPE for capacity S/B/L/X."""
    mpjpe_by_exp = {
        r["experiment"]: r.get("mpjpe_mm")
        for r in summary_rows
        if r.get("mpjpe_mm") is not None
    }
    labels: list[str] = []
    detected_vals: list[float] = []
    gt_vals: list[float] = []
    for exp_name in CAPACITY_EXPERIMENTS:
        det_mpjpe = mpjpe_by_exp.get(exp_name)
        gt_mpjpe = mpjpe_by_exp.get(f"{exp_name}_gt")
        if det_mpjpe is None or gt_mpjpe is None:
            continue
        labels.append(exp_name.replace("capacity_", "").upper())
        detected_vals.append(float(det_mpjpe))
        gt_vals.append(float(gt_mpjpe))
    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, detected_vals, width, label="Detected-2D train", color="steelblue")
    ax.bar(x + width / 2, gt_vals, width, label="GT-2D train", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Capacity: detected vs GT training (matching test corpora)")
    ax.axhline(40, color="red", linestyle="--", alpha=0.5, label="40 mm target")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_capacity_detected(summary_rows: list[dict], out_path: Path) -> None:
    """Grouped MPJPE/NMPJPE and MPJVE bars for capacity S/B/L/X (detected-2D)."""
    by_exp = {r["experiment"]: r for r in summary_rows}
    labels: list[str] = []
    mpjpe_vals: list[float] = []
    nmpjpe_vals: list[float] = []
    mpjve_vals: list[float] = []
    for exp_name in CAPACITY_EXPERIMENTS:
        row = by_exp.get(exp_name)
        if row is None or row.get("mpjpe_mm") is None:
            continue
        labels.append(exp_name.replace("capacity_", "").upper())
        mpjpe_vals.append(float(row["mpjpe_mm"]))
        nmpjpe_vals.append(float(row.get("n_mpjpe_mm") or 0.0))
        mpjve_vals.append(float(row.get("mpjve_mm_per_s") or 0.0))
    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

    axes[0].bar(x - width / 2, mpjpe_vals, width, label="MPJPE (mm)", color="steelblue")
    axes[0].bar(x + width / 2, nmpjpe_vals, width, label="NMPJPE (mm)", color="cornflowerblue")
    axes[0].set_ylabel("Error (mm)")
    axes[0].set_title("Capacity ablation (detected-2D input)")
    axes[0].axhline(40, color="red", linestyle="--", alpha=0.5, label="40 mm target")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim(0, max(max(mpjpe_vals), max(nmpjpe_vals), 40) * 1.12)

    axes[1].bar(x, mpjve_vals, width=0.55, label="MPJVE (mm/s)", color="teal")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Error (mm/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, max(mpjve_vals) * 1.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _window_label(exp_name: str) -> str:
    if exp_name.startswith("window_t"):
        return f"T={exp_name.split('_t', 1)[1]}"
    if exp_name == WINDOW_BASELINE_EXPERIMENT:
        return "T=243"
    return exp_name


def plot_window_ablation(summary_rows: list[dict], out_path: Path) -> None:
    """Grouped MPJPE/NMPJPE and MPJVE for temporal window ablations."""
    by_exp = {r["experiment"]: r for r in summary_rows}
    labels: list[str] = []
    mpjpe_vals: list[float] = []
    nmpjpe_vals: list[float] = []
    mpjve_vals: list[float] = []
    for exp_name in WINDOW_ABLATION_PLOT_EXPERIMENTS:
        row = by_exp.get(exp_name)
        if row is None or row.get("mpjpe_mm") is None:
            continue
        labels.append(_window_label(exp_name))
        mpjpe_vals.append(float(row["mpjpe_mm"]))
        nmpjpe_vals.append(float(row.get("n_mpjpe_mm") or 0.0))
        mpjve_vals.append(float(row.get("mpjve_mm_per_s") or 0.0))
    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    axes[0].bar(x - width / 2, mpjpe_vals, width, label="MPJPE (mm)", color="steelblue")
    axes[0].bar(x + width / 2, nmpjpe_vals, width, label="NMPJPE (mm)", color="cornflowerblue")
    axes[0].set_ylabel("Error (mm)")
    axes[0].set_title("Temporal window ablation (detected-2D input)")
    axes[0].axhline(40, color="red", linestyle="--", alpha=0.5, label="40 mm target")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim(0, max(max(mpjpe_vals), max(nmpjpe_vals), 40) * 1.12)

    axes[1].bar(x, mpjve_vals, width=0.55, label="MPJVE (mm/s)", color="teal")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Error (mm/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, max(mpjve_vals) * 1.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_window_dynamics_bars(summary_rows: list[dict], out_path: Path) -> None:
    """Roll and steer RMSE for temporal window ablations (aggregate metrics)."""
    by_exp = {r["experiment"]: r for r in summary_rows}
    labels: list[str] = []
    roll_vals: list[float] = []
    steer_vals: list[float] = []
    for exp_name in WINDOW_ABLATION_PLOT_EXPERIMENTS:
        row = by_exp.get(exp_name)
        if row is None:
            continue
        labels.append(_window_label(exp_name))
        roll_vals.append(float(row.get("roll_rmse_deg") or 0.0))
        steer_vals.append(float(row.get("steer_rmse_deg") or 0.0))
    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, roll_vals, width, label="Roll RMSE (deg)", color="darkorange")
    ax.bar(x + width / 2, steer_vals, width, label="Steer RMSE (deg)", color="goldenrod")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (deg)")
    ax.set_title("Dynamics accuracy vs training window length")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(roll_vals), max(steer_vals)) * 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_window_dynamics_multiples(
    results_dir: Path,
    ref_span: dict[str, Any],
    out_path: Path,
    *,
    max_frames: int = 500,
) -> None:
    """Small-multiples steer/roll: one panel per window (pred vs GT only, aligned span)."""
    from evaluation.dynamics_viz import aligned_window_ablation_time_series

    out_path = out_path.resolve()
    results_dir = results_dir.resolve()

    series_by_exp: list[tuple[str, dict[str, Any]]] = []
    for exp_name in WINDOW_ABLATION_PLOT_EXPERIMENTS:
        exp_dir = results_dir / exp_name
        if not exp_dir.is_dir():
            continue
        ts = aligned_window_ablation_time_series(exp_dir, ref_span)
        series_by_exp.append((_window_label(exp_name), ts))
    if not series_by_exp:
        return

    n_cols = len(series_by_exp)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 5.5), sharex=True, sharey="row")
    if n_cols == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    clip_id = ref_span.get("clip_id", "")
    fig.suptitle(f"Aligned dynamics comparison ({clip_id})", y=0.98, fontsize=11)

    for col, (label, ts) in enumerate(series_by_exp):
        n = min(max_frames, len(ts.get("pred_steer_deg", [])))
        frame_idx = ts.get("frame_idx")
        if frame_idx:
            t = np.asarray(frame_idx[:n], dtype=np.int64) - int(frame_idx[0])
        else:
            t = np.arange(n)

        pred_steer = np.asarray(ts["pred_steer_deg"][:n], dtype=np.float64)
        gt_steer = np.asarray(ts["gt_steer_deg"][:n], dtype=np.float64)
        pred_roll = np.asarray(ts["pred_roll_deg"][:n], dtype=np.float64)
        gt_roll = np.asarray(ts["gt_roll_deg"][:n], dtype=np.float64)

        steer_rmse = float(np.sqrt(np.mean((pred_steer - gt_steer) ** 2)))
        roll_rmse = float(np.sqrt(np.mean((pred_roll - gt_roll) ** 2)))

        ax_steer = axes[0, col]
        ax_steer.plot(t, gt_steer, color="black", linestyle="--", linewidth=1.2, label="GT", alpha=0.85)
        ax_steer.plot(t, pred_steer, color="steelblue", linewidth=1.0, label="Pred", alpha=0.9)
        ax_steer.set_title(f"{label}\nsteer RMSE {steer_rmse:.2f}°", fontsize=9)
        ax_steer.grid(True, alpha=0.3)
        if col == 0:
            ax_steer.set_ylabel("Steer (deg)")

        ax_roll = axes[1, col]
        ax_roll.plot(t, gt_roll, color="black", linestyle="--", linewidth=1.2, alpha=0.85)
        ax_roll.plot(t, pred_roll, color="steelblue", linewidth=1.0, alpha=0.9)
        ax_roll.set_title(f"roll RMSE {roll_rmse:.2f}°", fontsize=9)
        ax_roll.set_xlabel("Frame")
        ax_roll.grid(True, alpha=0.3)
        if col == 0:
            ax_roll.set_ylabel("Roll (deg)")

        if col == n_cols - 1:
            ax_steer.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _circular_diff_deg(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
    from data_generation_pipeline_tools.visualize_bicycle_pose3d import _wrap_pi

    return np.rad2deg(_wrap_pi(np.deg2rad(np.asarray(pred_deg, dtype=np.float64) - gt_deg)))


def plot_crank_accuracy(metrics: dict[str, Any], out_path: Path, *, max_frames: int | None = None) -> None:
    """Thesis figure: Bland-Altman of crank pred vs GT (error vs crank position)."""
    ts = metrics.get("dynamics", {}).get("time_series", {})
    crank_stats = metrics.get("dynamics", {}).get("crank", {})
    if not ts.get("pred_crank_deg") or not ts.get("gt_crank_deg"):
        return

    n_total = len(ts["pred_crank_deg"])
    n = n_total if max_frames is None else min(max_frames, n_total)

    pred = np.asarray(ts["pred_crank_deg"][:n], dtype=np.float64)
    gt = np.asarray(ts["gt_crank_deg"][:n], dtype=np.float64)
    diff = _circular_diff_deg(pred, gt)

    rmse = float(crank_stats.get("rmse_deg") or np.sqrt(np.mean(diff**2)))
    mae = float(crank_stats.get("mae_deg") or np.mean(np.abs(diff)))
    pearson = crank_stats.get("pearson_r")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle(
        f"Crank angle estimation (circular RMSE {rmse:.2f}°, MAE {mae:.2f}°"
        + (f", r={float(pearson):.2f}" if pearson is not None else "")
        + ")",
        y=1.02,
        fontsize=11,
    )

    ax.scatter(gt, diff, s=10, alpha=0.35, color="#9467bd")
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    ax.axhline(md, color="red", linestyle="-", label=f"bias={md:.2f}°")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--", alpha=0.7, label="±1.96 SD")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Crank angle (deg)")
    ax.set_ylabel("Pred − GT (deg)")
    ax.set_title("Bland-Altman: crank position vs error")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_crank_capacity_bars(summary_rows: list[dict], out_path: Path) -> None:
    """Crank RMSE and Pearson r across capacity presets (detected-2D)."""
    by_exp = {r["experiment"]: r for r in summary_rows}
    labels: list[str] = []
    rmse_vals: list[float] = []
    r_vals: list[float] = []
    for exp_name in CAPACITY_EXPERIMENTS:
        row = by_exp.get(exp_name)
        if row is None or row.get("crank_rmse_deg") in (None, ""):
            continue
        labels.append(exp_name.replace("capacity_", "").upper())
        rmse_vals.append(float(row["crank_rmse_deg"]))
        r_vals.append(float(row.get("crank_pearson_r") or 0.0))
    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(x, rmse_vals, width=0.55, color="#9467bd", label="Crank RMSE (deg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Circular RMSE (deg)")
    ax1.set_xlabel("Model capacity")
    ax1.set_title("Crank angle accuracy vs model capacity (detected-2D)")
    ax1.grid(True, alpha=0.3, axis="y")
    ymax = max(rmse_vals) * 1.2 if rmse_vals else 1.0
    ax1.set_ylim(0, ymax)

    ax2 = ax1.twinx()
    ax2.plot(x, r_vals, color="black", marker="o", linewidth=1.2, label="Pearson r", zorder=3)
    ax2.set_ylabel("Pearson r")
    ax2.set_ylim(0.0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dynamics_timeseries(metrics: dict[str, Any], out_path: Path, max_frames: int = 500) -> None:
    ts = metrics.get("dynamics", {}).get("time_series", {})
    if not ts:
        return
    n = min(max_frames, len(ts.get("pred_steer_deg", [])))
    frame_idx = ts.get("frame_idx")
    if frame_idx:
        t = np.asarray(frame_idx[:n], dtype=np.int64)
        t = t - int(t[0])
        xlabel = "Frame (contiguous segment)"
    else:
        t = np.arange(n)
        xlabel = "Frame"
    clip_label = ts.get("clip_id")
    clip_suffix = f" ({clip_label})" if clip_label else ""

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)

    axes[0].plot(t, ts["pred_steer_deg"][:n], label="pred", alpha=0.8)
    axes[0].plot(t, ts["gt_steer_deg"][:n], label="gt MuJoCo", alpha=0.8)
    axes[0].set_ylabel("Steer (deg)")
    axes[0].set_title(f"Steer vs MuJoCo GT{clip_suffix}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, ts["pred_roll_deg"][:n], label="pred", alpha=0.8)
    axes[1].plot(t, ts["gt_roll_deg"][:n], label="gt kinematic", alpha=0.8)
    mujoco_roll = ts.get("gt_roll_mujoco_deg")
    if mujoco_roll:
        axes[1].plot(t, mujoco_roll[:n], label="gt MuJoCo", alpha=0.55, linestyle="--")
    axes[1].set_ylabel("Roll (deg)")
    axes[1].set_title(f"Roll (kinematic pred vs kinematic GT){clip_suffix}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    pred_crank = ts.get("pred_crank_deg")
    gt_crank = ts.get("gt_crank_deg")
    if pred_crank and gt_crank:
        pred_crank_arr = np.asarray(pred_crank[:n], dtype=np.float64)
        gt_crank_arr = np.asarray(gt_crank[:n], dtype=np.float64)
        axes[2].plot(t, gt_crank_arr, label="gt kinematic", color="black", linestyle="--", alpha=0.85)
        axes[2].plot(t, pred_crank_arr, label="pred", color="#9467bd", alpha=0.9)
        crank_stats = metrics.get("dynamics", {}).get("crank", {})
        rmse = crank_stats.get("rmse_deg")
        mae = crank_stats.get("mae_deg")
        pearson = crank_stats.get("pearson_r")
        stats_bits = []
        if rmse is not None:
            stats_bits.append(f"RMSE {float(rmse):.2f}°")
        if mae is not None:
            stats_bits.append(f"MAE {float(mae):.2f}°")
        if pearson is not None:
            stats_bits.append(f"r={float(pearson):.2f}")
        stats_suffix = f" ({', '.join(stats_bits)})" if stats_bits else ""
        axes[2].set_title(f"Crank angle{stats_suffix}{clip_suffix}")
    else:
        axes[2].set_title(f"Crank angle{clip_suffix}")
    axes[2].set_ylabel("Crank (deg)")
    axes[2].set_xlabel(xlabel)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bland_altman(metrics: dict[str, Any], out_path: Path, angle: str = "roll") -> None:
    ts = metrics.get("dynamics", {}).get("time_series", {})
    pred_key = f"pred_{angle}_deg"
    gt_key = f"gt_{angle}_deg"
    if pred_key not in ts:
        return
    pred = np.asarray(ts[pred_key], dtype=np.float64)
    gt = np.asarray(ts[gt_key], dtype=np.float64)
    if angle == "crank":
        diff = _circular_diff_deg(pred, gt)
        x_vals = gt
    else:
        x_vals = 0.5 * (pred + gt)
        diff = pred - gt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_vals, diff, s=8, alpha=0.4)
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    ax.axhline(md, color="red", linestyle="-", label=f"bias={md:.2f}")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--")
    if angle == "crank":
        ax.set_xlabel("Crank angle (deg)")
        ax.set_ylabel("Pred − GT (deg)")
        ax.set_title("Bland-Altman: crank position vs error")
    else:
        ax.set_xlabel(f"Mean {angle} (deg)")
        ax.set_ylabel(f"Pred - GT {angle} (deg)")
        ax.set_title(f"Bland-Altman: {angle}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation_bars(summary_rows: list[dict], out_path: Path) -> None:
    ablation_rows = [
        r
        for r in summary_rows
        if r["experiment"].startswith(("loss_", "dyn_", "aug_"))
        and r["experiment"] not in EXCLUDED_ABLATION_EXPERIMENTS
    ]
    if not ablation_rows:
        return
    names = [r["experiment"] for r in ablation_rows]
    x = np.arange(len(names))

    def _val(row: dict, key: str) -> float:
        v = row.get(key)
        return float(v) if v not in (None, "") else 0.0

    mpjpe = [_val(r, "mpjpe_mm") for r in ablation_rows]
    nmpjpe = [_val(r, "n_mpjpe_mm") for r in ablation_rows]
    mpjve = [_val(r, "mpjve_mm_per_s") for r in ablation_rows]
    roll = [_val(r, "roll_rmse_deg") for r in ablation_rows]
    steer = [_val(r, "steer_rmse_deg") if r.get("steer_rmse_deg") not in (None, "") else _val(r, "rmse_deg") for r in ablation_rows]
    crank = [_val(r, "crank_rmse_deg") for r in ablation_rows]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    width2 = 0.35
    axes[0].bar(x - width2 / 2, mpjpe, width2, label="MPJPE (mm)", color="steelblue")
    axes[0].bar(x + width2 / 2, nmpjpe, width2, label="NMPJPE (mm)", color="cornflowerblue")
    axes[0].set_ylabel("Error (mm)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim(0, max(max(mpjpe), max(nmpjpe)) * 1.15)

    axes[1].bar(x, mpjve, width2, label="MPJVE (mm/s)", color="teal")
    axes[1].set_ylabel("Error (mm/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, max(mpjve) * 1.15)

    width2d = 0.35
    axes[2].bar(x - width2d / 2, roll, width2d, label="Roll RMSE (deg)", color="darkorange")
    axes[2].bar(x + width2d / 2, steer, width2d, label="Steer RMSE (deg)", color="goldenrod")
    axes[2].set_ylabel("Error (deg)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].set_ylim(0, max(max(roll), max(steer)) * 1.2)

    axes[3].bar(x, crank, width2d, label="Crank RMSE (deg)", color="saddlebrown")
    axes[3].set_ylabel("Error (deg)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3, axis="y")
    axes[3].set_ylim(0, max(crank) * 1.12)

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(names, rotation=45, ha="right")
    fig.suptitle("Ablation: 3D, temporal, and dynamics error", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis figures.")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Only regenerate figures for this experiment subfolder.",
    )
    p.add_argument(
        "--dynamics-clip",
        type=str,
        default=None,
        help="Override dynamics time-series example clip (requires --experiment).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    cross_fig_dir = ensure_dir(results_dir / "figures")

    lifterinput = _load_json(results_dir / "stage12_lifterinput_metrics.json")
    if not lifterinput:
        lifterinput = _load_json(results_dir / "stage12_metrics.json")
    static = _load_json(results_dir / "stage12_static_metrics.json")
    frontier = _load_json(results_dir / "capacity_frontier.json")

    summary_rows = []
    summary_csv = results_dir / "summary.csv"
    if summary_csv.is_file():
        import csv

        with summary_csv.open(newline="", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))
        for r in summary_rows:
            for k in (
                "mpjpe_mm",
                "n_mpjpe_mm",
                "mpjve_mm_per_s",
                "roll_rmse_deg",
                "steer_rmse_deg",
                "crank_rmse_deg",
                "roll_mae_deg",
                "params_M",
                "flops_G",
            ):
                if k in r and r[k]:
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass

    if lifterinput:
        plot_pck_histogram(lifterinput, cross_fig_dir / "pck_histogram_lifterinput.png", title_suffix="test clips")
        plot_iou_histogram(lifterinput, cross_fig_dir / "iou_histogram_lifterinput.png", title_suffix="test clips")
    if static:
        plot_pck_histogram(static, cross_fig_dir / "pck_histogram_static.png", title_suffix="static-frame test set")
        plot_iou_histogram(static, cross_fig_dir / "iou_histogram_static.png", title_suffix="static-frame test set")

    if frontier and summary_rows:
        plot_capacity_frontier(frontier, summary_rows, cross_fig_dir / "capacity_frontier.png")
    if summary_rows:
        plot_capacity_detected(summary_rows, cross_fig_dir / "capacity_detected.png")
        plot_capacity_gt_training(summary_rows, cross_fig_dir / "capacity_gt.png")
        plot_capacity_detected_vs_gt_training(
            summary_rows, cross_fig_dir / "capacity_detected_vs_gt.png"
        )
        plot_capacity_detected_vs_gt2d(summary_rows, cross_fig_dir / "capacity_detected_vs_gt2d.png")
        plot_ablation_bars(summary_rows, cross_fig_dir / "ablation_bars.png")
        plot_window_ablation(summary_rows, cross_fig_dir / "window_ablation.png")
        plot_window_dynamics_bars(summary_rows, cross_fig_dir / "window_dynamics_bars.png")
        plot_crank_capacity_bars(summary_rows, cross_fig_dir / "crank_capacity.png")

    headline_metrics_path = results_dir / HEADLINE_EXPERIMENT / "metrics.json"
    if headline_metrics_path.is_file():
        plot_crank_accuracy(
            _load_json(headline_metrics_path),
            cross_fig_dir / "crank_accuracy.png",
        )

    from evaluation.dynamics_viz import (
        WINDOW_ABLATION_PREFIX,
        aligned_window_ablation_time_series,
        reference_dynamics_span,
    )
    from evaluation.metrics.dynamics import build_time_series_for_clip

    ref_span = None
    try:
        ref_span = reference_dynamics_span(results_dir)
    except (FileNotFoundError, ValueError):
        ref_span = None

    if ref_span:
        try:
            plot_window_dynamics_multiples(
                results_dir,
                ref_span,
                cross_fig_dir / "window_dynamics_multiples.png",
            )
        except RuntimeError as exc:
            print(f"[make_figures] warn: window_dynamics_multiples skipped ({exc})")

    exp_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if args.experiment:
        exp_dirs = [results_dir / args.experiment]

    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _load_json(metrics_path)
        if args.dynamics_clip:
            npz_path = exp_dir / "preds_3d.npz"
            if not npz_path.is_file():
                raise SystemExit(f"--dynamics-clip requires {npz_path}")
            metrics.setdefault("dynamics", {})["time_series"] = build_time_series_for_clip(
                npz_path, args.dynamics_clip
            )
        elif ref_span and exp_dir.name.startswith(WINDOW_ABLATION_PREFIX):
            metrics.setdefault("dynamics", {})["time_series"] = aligned_window_ablation_time_series(
                exp_dir, ref_span
            )
        fig_dir = ensure_dir(exp_dir / "figures")
        plot_per_joint_mpjpe(metrics, fig_dir / "per_joint_mpjpe.png")
        plot_dynamics_timeseries(metrics, fig_dir / "dynamics_timeseries.png")
        plot_crank_accuracy(metrics, fig_dir / "crank_accuracy.png")
        plot_bland_altman(metrics, fig_dir / "bland_altman_roll.png", angle="roll")
        plot_bland_altman(metrics, fig_dir / "bland_altman_steer.png", angle="steer")
        plot_bland_altman(metrics, fig_dir / "bland_altman_crank.png", angle="crank")

    print(f"[make_figures] figures written under {results_dir}")


if __name__ == "__main__":
    main()

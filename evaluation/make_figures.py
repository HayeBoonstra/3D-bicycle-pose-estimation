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
from evaluation.common import ensure_dir  # noqa: E402


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


def plot_pck_curve(stage12_metrics: dict[str, Any], out_path: Path) -> None:
    p2 = stage12_metrics.get("pose2d", {})
    thr = p2.get("pck_thresholds", [])
    vals = p2.get("pck_values", [])
    if not thr:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thr, vals, marker="o")
    ax.set_xlabel("Normalized distance threshold")
    ax.set_ylabel("PCK")
    ax.set_title(f"PCK curve (AUC={p2.get('pck_auc', 0):.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_iou_histogram(stage12_metrics: dict[str, Any], out_path: Path) -> None:
    hist = stage12_metrics.get("detection", {}).get("iou_histogram", [])
    if not hist:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(np.linspace(0.05, 0.95, len(hist)), hist, width=0.08, color="teal")
    ax.set_xlabel("IoU bin center")
    ax.set_ylabel("Count")
    ax.set_title("Detection IoU distribution")
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


def plot_dynamics_timeseries(metrics: dict[str, Any], out_path: Path, max_frames: int = 500) -> None:
    ts = metrics.get("dynamics", {}).get("time_series", {})
    if not ts:
        return
    n = min(max_frames, len(ts.get("pred_steer_deg", [])))
    t = np.arange(n)
    clip_label = ts.get("clip_id")
    clip_suffix = f" ({clip_label})" if clip_label else ""

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)

    axes[0].plot(t, ts["pred_steer_deg"][:n], label="pred", alpha=0.8)
    axes[0].plot(t, ts["gt_steer_deg"][:n], label="gt MuJoCo", alpha=0.8)
    axes[0].set_ylabel("Steer (deg)")
    axes[0].set_title(f"Steer vs MuJoCo GT{clip_suffix}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, ts["pred_roll_deg"][:n], label="pred", alpha=0.8)
    axes[1].plot(t, ts["gt_roll_deg"][:n], label="gt", alpha=0.8)
    axes[1].set_ylabel("Roll (deg)")
    axes[1].set_xlabel("Frame")
    axes[1].set_title(f"Roll (kinematic pred vs kinematic GT){clip_suffix}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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
    mean = 0.5 * (pred + gt)
    diff = pred - gt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean, diff, s=8, alpha=0.4)
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    ax.axhline(md, color="red", linestyle="-", label=f"bias={md:.2f}")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--")
    ax.set_xlabel(f"Mean {angle} (deg)")
    ax.set_ylabel(f"Pred - GT {angle} (deg)")
    ax.set_title(f"Bland-Altman: {angle}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation_bars(summary_rows: list[dict], out_path: Path) -> None:
    ablation_rows = [r for r in summary_rows if r["experiment"].startswith(("loss_", "dyn_", "aug_"))]
    if not ablation_rows:
        return
    names = [r["experiment"] for r in ablation_rows]
    mpjpe = [r.get("mpjpe_mm", 0) for r in ablation_rows]
    roll = [r.get("roll_rmse_deg", 0) for r in ablation_rows]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, mpjpe, width, label="MPJPE (mm)")
    ax.bar(x + width / 2, roll, width, label="Roll RMSE (deg)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_title("Ablation: 3D error vs dynamics error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis figures.")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    cross_fig_dir = ensure_dir(results_dir / "figures")

    stage12 = _load_json(results_dir / "stage12_metrics.json")
    frontier = _load_json(results_dir / "capacity_frontier.json")

    summary_rows = []
    summary_csv = results_dir / "summary.csv"
    if summary_csv.is_file():
        import csv

        with summary_csv.open(newline="", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))
        for r in summary_rows:
            for k in ("mpjpe_mm", "roll_rmse_deg", "roll_mae_deg", "params_M", "flops_G"):
                if k in r and r[k]:
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass

    if stage12:
        plot_pck_curve(stage12, cross_fig_dir / "pck_curve.png")
        plot_iou_histogram(stage12, cross_fig_dir / "iou_histogram.png")

    if frontier and summary_rows:
        plot_capacity_frontier(frontier, summary_rows, cross_fig_dir / "capacity_frontier.png")
    if summary_rows:
        plot_ablation_bars(summary_rows, cross_fig_dir / "ablation_bars.png")

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _load_json(metrics_path)
        fig_dir = ensure_dir(exp_dir / "figures")
        plot_per_joint_mpjpe(metrics, fig_dir / "per_joint_mpjpe.png")
        plot_dynamics_timeseries(metrics, fig_dir / "dynamics_timeseries.png")
        plot_bland_altman(metrics, fig_dir / "bland_altman_roll.png", angle="roll")
        plot_bland_altman(metrics, fig_dir / "bland_altman_steer.png", angle="steer")

    print(f"[make_figures] figures written under {results_dir}")


if __name__ == "__main__":
    main()

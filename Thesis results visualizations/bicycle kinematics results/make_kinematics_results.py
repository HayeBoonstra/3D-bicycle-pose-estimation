#!/usr/bin/env python3
"""Generate minimal bicycle kinematics thesis results (scorecard + aggregate figures)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.common import (  # noqa: E402
    CAPACITY_EXPERIMENTS,
    DEFAULT_MAX_CLIP_MPJPE_MM,
    EXCLUDED_ABLATION_EXPERIMENTS,
    WINDOW_ABLATION_PLOT_EXPERIMENTS,
    WINDOW_BASELINE_EXPERIMENT,
    ensure_dir,
)
from evaluation.metrics.dynamics import (  # noqa: E402
    compute_kinematics_scorecard_metrics,
    framewise_abs_error_curves,
    mean_coherence_across_clips,
    pooled_bland_altman_arrays,
)
from evaluation.metrics.pose3d import compute_pose3d_metrics  # noqa: E402

# Primary thesis model (PoseMamba-B on detected-2D); captions name it explicitly.
DEFAULT_HEADLINE = WINDOW_BASELINE_EXPERIMENT  # capacity_b

CAPACITY_LABELS = {
    "capacity_s": "S",
    "capacity_b": "B",
    "capacity_l": "L",
    "capacity_x": "X",
}

# Cap y-axis so sparse wrap/outlier spikes do not flatten the mean curves.
ERROR_YLIM_DEG = {
    "roll": 10.0,
    "steer": 10.0,
    "crank": 45.0,
}

ANGLE_LABELS = {
    "roll": r"Roll $\phi$",
    "steer": r"Steer $\delta$",
    "crank": r"Crank $\theta$",
}

EXPERIMENT_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Capacity", CAPACITY_EXPERIMENTS),
    (
        "Loss",
        ("loss_no_diff", "loss_no_nmpjpe", "loss_no_velocity"),
    ),
    (
        "Augmentation",
        ("aug_no_flip", "aug_noise2d"),
    ),
    ("Window", WINDOW_ABLATION_PLOT_EXPERIMENTS),
    ("Dynamics", ("dyn_roll",)),
]

SCORECARD_COLUMNS: list[tuple[str, str]] = [
    ("experiment", "Experiment"),
    ("group", "Group"),
    ("mpjpe_mm", "MPJPE (mm)"),
    ("roll_rmse_deg", r"$\phi$ RMSE"),
    ("roll_bias_deg", r"$\phi$ bias"),
    ("roll_pearson_r", r"$\phi$ $r$"),
    ("steer_rmse_deg", r"$\delta$ RMSE"),
    ("steer_bias_deg", r"$\delta$ bias"),
    ("steer_pearson_r", r"$\delta$ $r$"),
    ("crank_rmse_deg", r"$\theta$ RMSE"),
    ("crank_bias_deg", r"$\theta$ bias"),
    ("crank_pearson_r", r"$\theta$ $r$"),
    ("roll_rate_rmse_deg_per_s", r"$\dot\phi$ RMSE"),
    ("steer_rate_rmse_deg_per_s", r"$\dot\delta$ RMSE"),
    ("crank_rate_rmse_deg_per_s", r"$\dot\theta$ RMSE"),
]

OUTLIER_KEYS = [
    "roll_rmse_deg",
    "steer_rmse_deg",
    "crank_rmse_deg",
    "roll_rate_rmse_deg_per_s",
    "steer_rate_rmse_deg_per_s",
    "crank_rate_rmse_deg_per_s",
    "mpjpe_mm",
]


def _discover_experiments(results_dir: Path) -> list[str]:
    names: list[str] = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if not (exp_dir / "preds_3d.npz").is_file():
            continue
        name = exp_dir.name
        if name.endswith("_gt") or name.endswith("_gt2d"):
            continue
        if name in EXCLUDED_ABLATION_EXPERIMENTS:
            continue
        names.append(name)
    return names


def _group_for_experiment(name: str) -> str:
    for group_name, exps in EXPERIMENT_GROUPS:
        if name in exps:
            return group_name
    return "Other"


def _ordered_scorecard_experiments(available: set[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for _, exps in EXPERIMENT_GROUPS:
        for exp in exps:
            if exp in available and exp not in seen:
                ordered.append(exp)
                seen.add(exp)
    for exp in sorted(available):
        if exp not in seen:
            ordered.append(exp)
    return ordered


def _accepted_clip_ids(results_dir: Path, exp_name: str) -> set[str]:
    metrics_path = results_dir / exp_name / "metrics.json"
    if metrics_path.is_file():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        ids = data.get("pose3d", {}).get("clip_filter", {}).get("accepted_clip_ids", [])
        if ids:
            return {str(c) for c in ids}
    npz = results_dir / exp_name / "preds_3d.npz"
    pose3d = compute_pose3d_metrics(npz, max_clip_mpjpe_mm=DEFAULT_MAX_CLIP_MPJPE_MM)
    return {str(c) for c in pose3d.get("clip_filter", {}).get("accepted_clip_ids", [])}


def build_scorecard_rows(results_dir: Path) -> list[dict[str, Any]]:
    available = set(_discover_experiments(results_dir))
    rows: list[dict[str, Any]] = []
    for exp_name in _ordered_scorecard_experiments(available):
        npz = results_dir / exp_name / "preds_3d.npz"
        accepted = _accepted_clip_ids(results_dir, exp_name)
        kin = compute_kinematics_scorecard_metrics(
            npz,
            accepted_clip_ids=accepted,
            max_clip_mpjpe_mm=DEFAULT_MAX_CLIP_MPJPE_MM,
        )
        pose3d = compute_pose3d_metrics(npz, max_clip_mpjpe_mm=DEFAULT_MAX_CLIP_MPJPE_MM)
        row: dict[str, Any] = {
            "experiment": exp_name,
            "group": _group_for_experiment(exp_name),
            "mpjpe_mm": pose3d.get("mpjpe_mm"),
            **kin,
        }
        rows.append(row)
    return rows


def _fmt_cell(key: str, value: Any) -> str:
    if value is None or value == "":
        return "---"
    if isinstance(value, float):
        if key.endswith("_pearson_r"):
            return f"{value:.3f}"
        return f"{value:.2f}"
    return str(value)


def _is_outlier(key: str, value: float | None, baseline: dict[str, float]) -> bool:
    if value is None or key not in baseline:
        return False
    base = baseline.get(key)
    if base is None or base <= 0:
        return False
    if key.endswith("_pearson_r"):
        return value < max(0.0, base - 0.15)
    return float(value) > 2.0 * float(base)


def write_scorecard_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [k for k, _ in SCORECARD_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_scorecard_json(rows: list[dict[str, Any]], path: Path, baseline_exp: str) -> None:
    baseline = next((r for r in rows if r["experiment"] == baseline_exp), None)
    payload = {
        "baseline_experiment": baseline_exp,
        "rows": rows,
        "baseline": baseline,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_scorecard_tex(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    baseline_exp: str,
) -> None:
    baseline = next((r for r in rows if r["experiment"] == baseline_exp), {})
    baseline_vals = {k: baseline.get(k) for k in OUTLIER_KEYS if k in baseline}

    col_keys = [k for k, _ in SCORECARD_COLUMNS if k not in ("experiment", "group")]
    headers = [h for k, h in SCORECARD_COLUMNS if k not in ("group",)]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Bicycle kinematics scorecard (pred vs GT keypoints; rates in deg/s). "
        "Circular $\\delta$ and $\\theta$ use wrapped position error; rates use shortest-arc "
        "frame differences per contiguous clip segment; $r$ uses unwrapped angles. "
        + f"Bold values exceed $2\\times$ the {baseline_exp.replace('_', ' ')} baseline."
        + "}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    current_group = None
    for row in rows:
        group = row.get("group", "")
        if group != current_group:
            if current_group is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{len(headers)}}}{{l}}{{\\textit{{{group}}}}} \\\\")
            current_group = group

        exp_label = row["experiment"].replace("_", "\\_")
        cells = [exp_label]
        for key in col_keys:
            val = row.get(key)
            text = _fmt_cell(key, val)
            if isinstance(val, (int, float)) and _is_outlier(key, float(val), baseline_vals):
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_error_over_time(
    results_dir: Path,
    *,
    experiment: str,
    out_path: Path,
) -> None:
    accepted = _accepted_clip_ids(results_dir, experiment)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    colors = {"roll": "darkorange", "steer": "steelblue", "crank": "#9467bd"}

    for ax, angle in zip(axes, ("roll", "steer", "crank")):
        payload = framewise_abs_error_curves(
            results_dir / experiment / "preds_3d.npz",
            angle=angle,
            accepted_clip_ids=accepted,
        )
        grid = np.asarray(payload["grid"], dtype=np.float64)
        curves = payload["curves"]
        mean = np.asarray(payload["mean"], dtype=np.float64)
        std = np.asarray(payload["std"], dtype=np.float64)

        if isinstance(curves, np.ndarray) and curves.size:
            for curve in curves:
                ax.plot(grid, curve, color=colors[angle], alpha=0.12, linewidth=0.8)
        ax.fill_between(grid, mean - std, mean + std, color=colors[angle], alpha=0.2, label=r"$\pm 1\sigma$")
        ax.plot(grid, mean, color=colors[angle], linewidth=2.5, label="Mean")
        ax.set_ylabel("|error| (deg)")
        ax.set_title(ANGLE_LABELS[angle])
        ylim = ERROR_YLIM_DEG.get(angle)
        if ylim is not None:
            ax.set_ylim(0.0, ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Normalized clip progress")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_error_over_time_all_capacity(results_dir: Path, out_dir: Path, *, headline: str) -> None:
    """One error-over-time figure per capacity model (S/B/L/X)."""
    available = set(_discover_experiments(results_dir))

    for exp_name in CAPACITY_EXPERIMENTS:
        if exp_name not in available:
            continue
        if not (results_dir / exp_name / "preds_3d.npz").is_file():
            continue
        plot_error_over_time(
            results_dir,
            experiment=exp_name,
            out_path=out_dir / f"error_over_time_{exp_name}",
        )
        if exp_name == headline:
            plot_error_over_time(
                results_dir,
                experiment=exp_name,
                out_path=out_dir / "error_over_time",
            )


def plot_coherence_capacity(results_dir: Path, out_path: Path) -> None:
    """Mean coherence across clips for capacity S/B/L/X only."""
    available = set(_discover_experiments(results_dir))
    exps = [e for e in CAPACITY_EXPERIMENTS if e in available]
    if not exps:
        return

    angles = ("roll", "steer", "crank")
    colors = {
        "capacity_s": "#1f77b4",
        "capacity_b": "#ff7f0e",
        "capacity_l": "#2ca02c",
        "capacity_x": "#d62728",
    }
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for ax, angle in zip(axes, angles):
        for exp_name in exps:
            accepted = _accepted_clip_ids(results_dir, exp_name)
            coh = mean_coherence_across_clips(
                results_dir / exp_name / "preds_3d.npz",
                angle=angle,
                accepted_clip_ids=accepted,
            )
            freq = np.asarray(coh.get("freq_hz") or [], dtype=np.float64)
            mean_arr = coh.get("mean")
            mean = np.asarray([] if mean_arr is None else mean_arr, dtype=np.float64)
            if freq.size == 0:
                continue
            label = CAPACITY_LABELS.get(exp_name, exp_name)
            ax.plot(
                freq,
                mean,
                color=colors.get(exp_name, "gray"),
                linewidth=2.0,
                label=label,
            )
        ax.set_ylabel(f"{ANGLE_LABELS[angle]}\n$C_{{xy}}$")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3)
        if angle == "roll":
            ax.legend(title="Capacity", loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Coherence on the validation set", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_bland_altman_pooled(
    results_dir: Path,
    *,
    headline: str,
    out_path: Path,
) -> None:
    accepted = _accepted_clip_ids(results_dir, headline)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, angle in zip(axes, ("roll", "steer", "crank")):
        ba = pooled_bland_altman_arrays(
            results_dir / headline / "preds_3d.npz",
            angle=angle,
            accepted_clip_ids=accepted,
        )
        x = ba["x"]
        diff = ba["diff"]
        md = float(np.mean(diff))
        sd = float(np.std(diff))
        ax.scatter(x, diff, s=4, alpha=0.15, color="steelblue")
        ax.axhline(md, color="red", linestyle="-", linewidth=1.2, label=f"bias={md:.2f}°")
        ax.axhline(md + 1.96 * sd, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(md - 1.96 * sd, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Pred − GT (deg)")
        ax.set_title(ANGLE_LABELS[angle])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate bicycle kinematics thesis results.")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--headline", type=str, default=DEFAULT_HEADLINE)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out)
    results_dir = args.results_dir.resolve()

    if not (results_dir / args.headline / "preds_3d.npz").is_file():
        raise SystemExit(f"Missing headline preds: {results_dir / args.headline / 'preds_3d.npz'}")

    rows = build_scorecard_rows(results_dir)
    write_scorecard_csv(rows, out_dir / "kinematics_scorecard.csv")
    write_scorecard_json(rows, out_dir / "kinematics_scorecard.json", args.headline)
    write_scorecard_tex(rows, out_dir / "kinematics_scorecard.tex", baseline_exp=args.headline)

    plot_error_over_time_all_capacity(results_dir, out_dir, headline=args.headline)
    plot_coherence_capacity(results_dir, out_dir / "coherence_capacity")
    plot_bland_altman_pooled(results_dir, headline=args.headline, out_path=out_dir / "bland_altman")

    print(f"[kinematics_results] wrote scorecard + figures -> {out_dir}")


if __name__ == "__main__":
    main()

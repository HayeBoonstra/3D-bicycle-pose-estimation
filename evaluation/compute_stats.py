#!/usr/bin/env python3
"""Compute all thesis metrics and write summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.common import ensure_dir  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    compute_capacity_frontier,
    compute_detection_metrics,
    compute_dynamics_metrics,
    compute_pose2d_metrics,
    compute_pose3d_metrics,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _flatten(prefix: str, d: dict[str, Any], out: dict[str, Any]) -> None:
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(key, v, out)
        elif isinstance(v, (list, tuple)) and k not in ("pck_thresholds", "pck_values", "time_series"):
            out[key] = json.dumps(v)
        else:
            out[key] = v


def _write_tex_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], path: Path, caption: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(h for _, h in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = []
        for col_key, _ in columns:
            v = row.get(col_key, "---")
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute thesis evaluation metrics.")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--stage12", type=Path, default=None, help="Path to stage12_records.jsonl")
    p.add_argument("--experiments", type=Path, default=REPO_ROOT / "experiments/configs/experiments.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    stage12_path = args.stage12 or (results_dir / "stage12_records.jsonl")

    stage12_records = _load_jsonl(stage12_path)
    stage12_metrics = {}
    if stage12_records:
        stage12_metrics = {
            "detection": compute_detection_metrics(stage12_records),
            "pose2d": compute_pose2d_metrics(stage12_records),
        }
        (results_dir / "stage12_metrics.json").write_text(
            json.dumps(stage12_metrics, indent=2), encoding="utf-8"
        )

    experiments_manifest = {}
    if args.experiments.is_file():
        experiments_manifest = json.loads(args.experiments.read_text(encoding="utf-8"))

    summary_rows: list[dict[str, Any]] = []
    capacity_frontier = compute_capacity_frontier()
    (results_dir / "capacity_frontier.json").write_text(
        json.dumps(capacity_frontier, indent=2), encoding="utf-8"
    )

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        npz = exp_dir / "preds_3d.npz"
        if not npz.is_file():
            continue
        exp_name = exp_dir.name
        metrics: dict[str, Any] = {"experiment": exp_name}
        if exp_name in experiments_manifest:
            metrics["config"] = experiments_manifest[exp_name]

        metrics["pose3d"] = compute_pose3d_metrics(npz)
        metrics["dynamics"] = compute_dynamics_metrics(npz)

        exp_preset = exp_name.replace("capacity_", "").upper()
        if exp_preset in capacity_frontier:
            metrics["efficiency"] = capacity_frontier[exp_preset]
        elif exp_name == "capacity_b":
            metrics["efficiency"] = capacity_frontier.get("B", {})

        (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        row: dict[str, Any] = {"experiment": exp_name}
        _flatten("", metrics.get("pose3d", {}), row)
        _flatten("", metrics.get("dynamics", {}).get("steer", {}), row)
        dyn_roll = metrics.get("dynamics", {}).get("roll", {})
        row["roll_rmse_deg"] = dyn_roll.get("rmse_deg")
        row["roll_mae_deg"] = dyn_roll.get("mae_deg")
        if "efficiency" in metrics:
            row["params_M"] = metrics["efficiency"].get("params_M")
            row["flops_G"] = metrics["efficiency"].get("flops_G")
        summary_rows.append(row)

    summary_csv = results_dir / "summary.csv"
    if summary_rows:
        fieldnames = sorted({k for r in summary_rows for k in r.keys()})
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    # LaTeX tables
    tables_dir = ensure_dir(results_dir / "tables")
    if stage12_metrics.get("detection"):
        det = stage12_metrics["detection"]
        _write_tex_table(
            [{"metric": "Detection rate", "value": det.get("detection_rate")},
             {"metric": "Mean IoU", "value": det.get("mean_iou")},
             {"metric": "AP@0.5 (proxy)", "value": det.get("ap50_proxy")}],
            [("metric", "Metric"), ("value", "Value")],
            tables_dir / "detection_headline.tex",
            "RF-DETR bicycle detection on synthetic test set.",
        )
    if stage12_metrics.get("pose2d"):
        p2 = stage12_metrics["pose2d"]
        _write_tex_table(
            [{"metric": "Mean pixel error", "value": p2.get("mean_pixel_error")},
             {"metric": "PCK@0.1", "value": p2.get("pck_at_0_1")},
             {"metric": "Mean NME", "value": p2.get("mean_nme")}],
            [("metric", "Metric"), ("value", "Value")],
            tables_dir / "pose2d_headline.tex",
            "RTMPose-L 2D keypoint accuracy on synthetic test set.",
        )
    if summary_rows:
        _write_tex_table(
            [{"experiment": r["experiment"], "mpjpe_mm": r.get("mpjpe_mm"), "roll_rmse_deg": r.get("roll_rmse_deg")}
             for r in summary_rows],
            [("experiment", "Experiment"), ("mpjpe_mm", "MPJPE (mm)"), ("roll_rmse_deg", "Roll RMSE (deg)")],
            tables_dir / "pose3d_ablations.tex",
            "PoseMamba 3D lifting ablation results (detected-2D input).",
        )

    print(f"[compute_stats] wrote metrics for {len(summary_rows)} experiments -> {summary_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute all thesis metrics and write summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.common import (  # noqa: E402
    CAPACITY_EXPERIMENTS,
    CAPACITY_GT_TRAINING_EXPERIMENTS,
    DEFAULT_MAX_CLIP_MPJPE_MM,
    EXCLUDED_ABLATION_EXPERIMENTS,
    ensure_dir,
)
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
        elif isinstance(v, (list, tuple)) and k not in (
            "pck_thresholds",
            "pck_values",
            "nme_histogram_counts",
            "nme_histogram_edges",
            "iou_histogram",
            "iou_histogram_fraction",
            "time_series",
        ):
            out[key] = json.dumps(v)
        else:
            out[key] = v


def _latex_text(text: str) -> str:
    """Make plain text safe for LaTeX captions and table cells (underscores break text mode)."""
    return text.replace("_", " ")


def _latex_experiment(name: str) -> str:
    """Readable experiment label for LaTeX tables."""
    return _latex_text(name.replace("_gt2d", " GT-2D"))


def _write_tex_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], path: Path, caption: str) -> None:
    safe_caption = _latex_text(caption)
    safe_headers = [_latex_text(h) for _, h in columns]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{safe_caption}}}",
        "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(safe_headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = []
        for col_key, _ in columns:
            v = row.get(col_key, "---")
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(_latex_text(str(v)))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def _load_experiment_metrics(results_dir: Path, exp_name: str) -> dict[str, Any] | None:
    path = results_dir / exp_name / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _capacity_gt2d_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_name in CAPACITY_EXPERIMENTS:
        gt2d_name = f"{exp_name}_gt2d"
        metrics = _load_experiment_metrics(results_dir, gt2d_name)
        if metrics is None:
            continue
        pose3d = metrics.get("pose3d", {})
        dynamics = metrics.get("dynamics", {})
        rows.append(
            {
                "model": exp_name.replace("capacity_", "").upper(),
                "mpjpe_mm": pose3d.get("mpjpe_mm"),
                "n_mpjpe_mm": pose3d.get("n_mpjpe_mm"),
                "mpjve_mm_per_s": pose3d.get("mpjve_mm_per_s"),
                "mpjae_mm_per_s2": pose3d.get("mpjae_mm_per_s2"),
                "roll_rmse_deg": dynamics.get("roll", {}).get("rmse_deg"),
                "steer_rmse_deg": dynamics.get("steer", {}).get("rmse_deg"),
                "crank_rmse_deg": dynamics.get("crank", {}).get("rmse_deg"),
            }
        )
    return rows


def _capacity_gt_training_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_name in CAPACITY_GT_TRAINING_EXPERIMENTS:
        metrics = _load_experiment_metrics(results_dir, exp_name)
        if metrics is None:
            continue
        pose3d = metrics.get("pose3d", {})
        dynamics = metrics.get("dynamics", {})
        rows.append(
            {
                "model": exp_name.replace("capacity_", "").replace("_gt", "").upper(),
                "mpjpe_mm": pose3d.get("mpjpe_mm"),
                "n_mpjpe_mm": pose3d.get("n_mpjpe_mm"),
                "mpjve_mm_per_s": pose3d.get("mpjve_mm_per_s"),
                "mpjae_mm_per_s2": pose3d.get("mpjae_mm_per_s2"),
                "roll_rmse_deg": dynamics.get("roll", {}).get("rmse_deg"),
                "steer_rmse_deg": dynamics.get("steer", {}).get("rmse_deg"),
                "crank_rmse_deg": dynamics.get("crank", {}).get("rmse_deg"),
            }
        )
    return rows


def _capacity_detected_vs_gt_training_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_name in CAPACITY_EXPERIMENTS:
        gt_name = f"{exp_name}_gt"
        det_metrics = _load_experiment_metrics(results_dir, exp_name)
        gt_metrics = _load_experiment_metrics(results_dir, gt_name)
        if det_metrics is None or gt_metrics is None:
            continue
        det_mpjpe = det_metrics.get("pose3d", {}).get("mpjpe_mm")
        gt_mpjpe = gt_metrics.get("pose3d", {}).get("mpjpe_mm")
        if det_mpjpe is None or gt_mpjpe is None:
            continue
        rows.append(
            {
                "model": exp_name.replace("capacity_", "").upper(),
                "mpjpe_detected_mm": det_mpjpe,
                "mpjpe_gt_training_mm": gt_mpjpe,
                "input_noise_gap_mm": float(det_mpjpe) - float(gt_mpjpe),
            }
        )
    return rows


def _capacity_detected_vs_gt2d_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_name in CAPACITY_EXPERIMENTS:
        det_metrics = _load_experiment_metrics(results_dir, exp_name)
        gt_metrics = _load_experiment_metrics(results_dir, f"{exp_name}_gt2d")
        if det_metrics is None or gt_metrics is None:
            continue
        det_mpjpe = det_metrics.get("pose3d", {}).get("mpjpe_mm")
        gt_mpjpe = gt_metrics.get("pose3d", {}).get("mpjpe_mm")
        if det_mpjpe is None or gt_mpjpe is None:
            continue
        rows.append(
            {
                "model": exp_name.replace("capacity_", "").upper(),
                "mpjpe_detected_mm": det_mpjpe,
                "mpjpe_gt2d_mm": gt_mpjpe,
                "frontend_gap_mm": float(det_mpjpe) - float(gt_mpjpe),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute thesis evaluation metrics.")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument(
        "--stage12-lifterinput",
        type=Path,
        default=None,
        help="Path to stage12_lifterinput_records.jsonl (3D-lifter input clips)",
    )
    p.add_argument(
        "--stage12-static",
        type=Path,
        default=None,
        help="Path to stage12_static_records.jsonl (bicycle_pose_dataset static frames)",
    )
    p.add_argument("--experiments", type=Path, default=REPO_ROOT / "experiments/configs/experiments.json")
    p.add_argument(
        "--max-clip-mpjpe-mm",
        type=float,
        default=DEFAULT_MAX_CLIP_MPJPE_MM,
        help="Exclude clips with per-clip MPJPE above this threshold from aggregates (default: 70 mm).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    lifterinput_path = args.stage12_lifterinput or _first_existing(
        results_dir / "stage12_lifterinput_records.jsonl",
        results_dir / "stage12_records.jsonl",
    )
    static_path = args.stage12_static or _first_existing(
        results_dir / "stage12_static_records.jsonl",
        results_dir / "bicycle_pose_stage12_records.jsonl",
    )

    lifterinput_records = _load_jsonl(lifterinput_path) if lifterinput_path else []
    lifterinput_metrics: dict[str, Any] = {}
    if lifterinput_records:
        lifterinput_metrics = {
            "corpus": "lifter_input",
            "detection": compute_detection_metrics(lifterinput_records),
            "pose2d": compute_pose2d_metrics(lifterinput_records),
        }
        (results_dir / "stage12_lifterinput_metrics.json").write_text(
            json.dumps(lifterinput_metrics, indent=2), encoding="utf-8"
        )

    static_records = _load_jsonl(static_path) if static_path else []
    static_metrics: dict[str, Any] = {}
    if static_records:
        static_metrics = {
            "corpus": "static_frames",
            "dataset": "bicycle_pose_dataset",
            "split": static_records[0].get("split", "unknown"),
            "detection": compute_detection_metrics(static_records),
            "pose2d": compute_pose2d_metrics(static_records),
        }
        (results_dir / "stage12_static_metrics.json").write_text(
            json.dumps(static_metrics, indent=2), encoding="utf-8"
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

        metrics["pose3d"] = compute_pose3d_metrics(npz, max_clip_mpjpe_mm=args.max_clip_mpjpe_mm)
        accepted = set(metrics["pose3d"].get("clip_filter", {}).get("accepted_clip_ids", []))
        metrics["dynamics"] = compute_dynamics_metrics(
            npz,
            accepted_clip_ids=accepted,
            max_clip_mpjpe_mm=args.max_clip_mpjpe_mm,
        )

        exp_preset = exp_name.replace("capacity_", "").upper()
        if exp_preset in capacity_frontier:
            metrics["efficiency"] = capacity_frontier[exp_preset]
        elif exp_name == "capacity_b":
            metrics["efficiency"] = capacity_frontier.get("B", {})

        (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        row: dict[str, Any] = {"experiment": exp_name}
        _flatten("", metrics.get("pose3d", {}), row)
        dyn_steer = metrics.get("dynamics", {}).get("steer", {})
        dyn_roll = metrics.get("dynamics", {}).get("roll", {})
        dyn_crank = metrics.get("dynamics", {}).get("crank", {})
        row["steer_rmse_deg"] = dyn_steer.get("rmse_deg")
        row["crank_rmse_deg"] = dyn_crank.get("rmse_deg")
        row["crank_mae_deg"] = dyn_crank.get("mae_deg")
        row["crank_pearson_r"] = dyn_crank.get("pearson_r")
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
    if lifterinput_metrics.get("detection"):
        det = lifterinput_metrics["detection"]
        _write_tex_table(
            [{"metric": "Detection rate", "value": det.get("detection_rate")},
             {"metric": "Mean IoU", "value": det.get("mean_iou")},
             {"metric": "AP@0.5 (proxy)", "value": det.get("ap50_proxy")}],
            [("metric", "Metric"), ("value", "Value")],
            tables_dir / "detection_lifterinput_headline.tex",
            "RF-DETR bicycle detection on 3D-lifter input test clips.",
        )
    if lifterinput_metrics.get("pose2d"):
        p2 = lifterinput_metrics["pose2d"]
        _write_tex_table(
            [{"metric": "Mean pixel error", "value": p2.get("mean_pixel_error")},
             {"metric": "PCK@0.1", "value": p2.get("pck_at_0_1")},
             {"metric": "Mean NME", "value": p2.get("mean_nme")}],
            [("metric", "Metric"), ("value", "Value")],
            tables_dir / "pose2d_lifterinput_headline.tex",
            "RTMPose-L 2D keypoint accuracy on 3D-lifter input test clips.",
        )
    if static_metrics.get("detection"):
        det = static_metrics["detection"]
        _write_tex_table(
            [{"metric": "Detection rate", "value": det.get("detection_rate")},
             {"metric": "Mean IoU", "value": det.get("mean_iou")},
             {"metric": "AP@0.5 (proxy)", "value": det.get("ap50_proxy")}],
            [("metric", "Metric"), ("value", "Value")],
            tables_dir / "detection_static_headline.tex",
            "RF-DETR bicycle detection on static-frame test set (bicycle pose dataset).",
        )
    if static_metrics.get("pose2d"):
        p2 = static_metrics["pose2d"]
        _write_tex_table(
            [{"metric": "Mean pixel error", "value": p2.get("mean_pixel_error")},
             {"metric": "PCK@0.1", "value": p2.get("pck_at_0_1")},
             {"metric": "Mean NME", "value": p2.get("mean_nme")}],
            [("metric", "Metric"), ("value", "Value")],
            tables_dir / "pose2d_static_headline.tex",
            "RTMPose-L 2D keypoint accuracy on static-frame test set (bicycle pose dataset).",
        )
    if summary_rows:
        _write_tex_table(
            [
                {
                    "experiment": _latex_experiment(r["experiment"]),
                    "mpjpe_mm": r.get("mpjpe_mm"),
                    "n_mpjpe_mm": r.get("n_mpjpe_mm"),
                    "mpjve_mm_per_s": r.get("mpjve_mm_per_s"),
                    "mpjae_mm_per_s2": r.get("mpjae_mm_per_s2"),
                    "roll_rmse_deg": r.get("roll_rmse_deg"),
                    "steer_rmse_deg": r.get("steer_rmse_deg"),
                    "crank_rmse_deg": r.get("crank_rmse_deg"),
                }
                for r in summary_rows
                if r["experiment"] not in EXCLUDED_ABLATION_EXPERIMENTS
            ],
            [
                ("experiment", "Experiment"),
                ("mpjpe_mm", "MPJPE (mm)"),
                ("n_mpjpe_mm", "NMPJPE (mm)"),
                ("mpjve_mm_per_s", "MPJVE (mm/s)"),
                ("mpjae_mm_per_s2", "MPJAE (mm/s$^2$)"),
                ("roll_rmse_deg", "Roll RMSE (deg)"),
                ("steer_rmse_deg", "Steer RMSE (deg)"),
                ("crank_rmse_deg", "Crank RMSE (deg)"),
            ],
            tables_dir / "pose3d_ablations.tex",
            "PoseMamba 3D lifting ablation results (detected-2D input). "
            "MPJVE and MPJAE are mean per-joint velocity and acceleration errors on root-relative 3D. "
            "Roll, steer, and crank RMSE are kinematic pred vs kinematic GT keypoints; "
            "crank error is circular (wrapped to $\\pm 180^\\circ$).",
        )

        gt2d_rows = _capacity_gt2d_rows(results_dir)
        if gt2d_rows:
            _write_tex_table(
                gt2d_rows,
                [
                    ("model", "Model"),
                    ("mpjpe_mm", "MPJPE (mm)"),
                    ("n_mpjpe_mm", "NMPJPE (mm)"),
                    ("mpjve_mm_per_s", "MPJVE (mm/s)"),
                    ("mpjae_mm_per_s2", "MPJAE (mm/s$^2$)"),
                    ("roll_rmse_deg", "Roll RMSE (deg)"),
                    ("steer_rmse_deg", "Steer RMSE (deg)"),
                    ("crank_rmse_deg", "Crank RMSE (deg)"),
                ],
                tables_dir / "pose3d_capacity_gt2d.tex",
                "PoseMamba 3D lifting capacity results with GT-projected 2D keypoints "
                "normalized in the detector-bbox frame.",
            )

        gt_training_rows = _capacity_gt_training_rows(results_dir)
        if gt_training_rows:
            _write_tex_table(
                gt_training_rows,
                [
                    ("model", "Model"),
                    ("mpjpe_mm", "MPJPE (mm)"),
                    ("n_mpjpe_mm", "NMPJPE (mm)"),
                    ("mpjve_mm_per_s", "MPJVE (mm/s)"),
                    ("mpjae_mm_per_s2", "MPJAE (mm/s$^2$)"),
                    ("roll_rmse_deg", "Roll RMSE (deg)"),
                    ("steer_rmse_deg", "Steer RMSE (deg)"),
                    ("crank_rmse_deg", "Crank RMSE (deg)"),
                ],
                tables_dir / "pose3d_capacity_gt.tex",
                "PoseMamba 3D lifting capacity results trained and evaluated on the "
                "GT-projected 2D corpus (gt\\_bbox normalization).",
            )

        gt_training_comparison_rows = _capacity_detected_vs_gt_training_rows(results_dir)
        if gt_training_comparison_rows:
            _write_tex_table(
                gt_training_comparison_rows,
                [
                    ("model", "Model"),
                    ("mpjpe_detected_mm", "MPJPE detected train (mm)"),
                    ("mpjpe_gt_training_mm", "MPJPE GT train (mm)"),
                    ("input_noise_gap_mm", "Input-noise gap (mm)"),
                ],
                tables_dir / "pose3d_capacity_detected_vs_gt.tex",
                "Capacity ablation: detected-2D vs GT-2D training, each evaluated on its "
                "matching test corpus. Positive gaps indicate RTMPose input noise cost.",
            )

        comparison_rows = _capacity_detected_vs_gt2d_rows(results_dir)
        if comparison_rows:
            _write_tex_table(
                comparison_rows,
                [
                    ("model", "Model"),
                    ("mpjpe_detected_mm", "MPJPE detected (mm)"),
                    ("mpjpe_gt2d_mm", "MPJPE GT-2D (mm)"),
                    ("frontend_gap_mm", "Front-end gap (mm)"),
                ],
                tables_dir / "pose3d_capacity_detected_vs_gt2d.tex",
                "Capacity ablation: detected RTMPose 2D vs GT-projected 2D, both normalized "
                "in the detector-bbox frame. Negative gaps indicate detected-2D training distribution effects.",
            )

    print(f"[compute_stats] wrote metrics for {len(summary_rows)} experiments -> {summary_csv}")


if __name__ == "__main__":
    main()

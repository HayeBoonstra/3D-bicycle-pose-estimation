"""Evaluate MMPose 2D model and export quick error histograms."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 2D keypoint model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pred-json", type=Path, default=None)
    parser.add_argument("--gt-json", type=Path, default=None)
    parser.add_argument("--out-metrics", type=Path, required=True)
    return parser.parse_args()


def _compute_basic_histogram(pred_json: Path, gt_json: Path) -> dict:
    pred = json.loads(pred_json.read_text(encoding="utf-8"))
    gt = json.loads(gt_json.read_text(encoding="utf-8"))
    gt_by_id = {ann["image_id"]: ann for ann in gt.get("annotations", [])}
    bins = [0] * 10
    count = 0
    total = 0.0
    for ann in pred.get("annotations", []):
        gt_ann = gt_by_id.get(ann["image_id"])
        if not gt_ann:
            continue
        p = ann.get("keypoints", [])
        g = gt_ann.get("keypoints", [])
        for i in range(0, min(len(p), len(g)), 3):
            if g[i + 2] <= 0:
                continue
            d = math.dist((p[i], p[i + 1]), (g[i], g[i + 1]))
            total += d
            count += 1
            bins[min(9, int(d // 5))] += 1
    return {"mean_pixel_error": (total / count) if count else None, "histogram_bin_5px": bins, "num_points": count}


def main() -> None:
    args = _parse_args()
    runner = shutil.which("mim")
    if runner is None:
        raise RuntimeError(
            "OpenMMLab runner not found. Install it with: pip install -U openmim"
        )
    cmd = [
        runner,
        "test",
        "mmpose",
        str(args.config),
        str(args.checkpoint),
        "--work-dir",
        str(args.out_metrics.parent),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    metrics = {"note": "Primary metrics are written by MMPose runner."}
    if args.pred_json and args.gt_json and args.pred_json.exists() and args.gt_json.exists():
        metrics["pixel_error_histogram"] = _compute_basic_histogram(args.pred_json, args.gt_json)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_metrics}")


if __name__ == "__main__":
    main()


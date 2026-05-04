"""Evaluate 3D lifting quality with MPJPE-style metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D lifter outputs.")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted [N,K,3] npy file")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth [N,K,3] npy file")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def _temporal_jitter(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    vel = np.diff(points, axis=0)
    acc = np.diff(vel, axis=0)
    return float(np.mean(np.linalg.norm(acc, axis=-1)))


def main() -> None:
    args = _parse_args()
    pred = np.load(args.pred)
    gt = np.load(args.gt)
    metrics = {
        "mpjpe": _mpjpe(pred, gt),
        "temporal_jitter": _temporal_jitter(pred),
        "num_frames": int(pred.shape[0]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


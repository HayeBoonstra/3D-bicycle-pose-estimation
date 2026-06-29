"""Shared utilities for thesis results evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Joint groups for per-group error reporting.
JOINT_GROUPS = {
    "wheels": [8, 9, 10, 11, 12, 13, 14, 15],
    "frame": [0, 1, 2, 3, 4],
    "steering": [5, 6, 7, 16, 17],
}


def mm_from_m(value_m: float) -> float:
    return float(value_m) * 1000.0


def bbox_xywh_to_xyxy(bbox: Iterable[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU for xyxy boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def group_mean(values_per_joint: np.ndarray, group_name: str) -> float:
    idx = JOINT_GROUPS[group_name]
    return float(np.mean(values_per_joint[idx]))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    r = pearson_r(x, y)
    return float(r * r) if np.isfinite(r) else float("nan")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

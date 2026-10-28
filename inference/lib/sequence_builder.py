"""Temporal window construction utilities for 3D lifting."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _fill_missing_points(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    out = points.copy()
    for t in range(points.shape[0]):
        mask = confidence[t] <= 0.0
        if not np.any(mask):
            continue
        if t > 0:
            out[t, mask] = out[t - 1, mask]
        else:
            out[t, mask] = 0.0
    return out


def normalize_points(points: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    out = points.copy()
    centers = np.stack([(bboxes[:, 0] + bboxes[:, 2]) * 0.5, (bboxes[:, 1] + bboxes[:, 3]) * 0.5], axis=-1)
    scales = np.maximum(1.0, np.maximum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]))
    out = (out - centers[:, None, :]) / scales[:, None, None]
    return out


def build_temporal_windows(
    points_2d: np.ndarray,
    confidence: np.ndarray,
    bboxes_xyxy: np.ndarray,
    window_size: int = 27,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if points_2d.shape[0] == 0:
        return np.zeros((0, window_size, points_2d.shape[1], 2)), np.zeros((0, window_size, points_2d.shape[1]))

    points_filled = _fill_missing_points(points_2d, confidence)
    norm_points = normalize_points(points_filled, bboxes_xyxy)
    radius = window_size // 2
    windows = []
    conf_windows = []
    for center in range(points_2d.shape[0]):
        idx = np.clip(np.arange(center - radius, center + radius + 1), 0, points_2d.shape[0] - 1)
        windows.append(norm_points[idx])
        conf_windows.append(confidence[idx])
    return np.asarray(windows, dtype=np.float32), np.asarray(conf_windows, dtype=np.float32)


def rows_to_arrays(rows: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_rows = sorted(rows, key=lambda r: int(r["frame_id"]))
    points = np.asarray([r["keypoints_2d"] for r in sorted_rows], dtype=np.float32)
    confidence = np.asarray([r["confidence"] for r in sorted_rows], dtype=np.float32)
    bboxes = np.asarray([r["bbox_xyxy"] for r in sorted_rows], dtype=np.float32)
    return points, confidence, bboxes


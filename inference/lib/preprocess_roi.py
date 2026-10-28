"""ROI crop/resize utilities with reversible transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class RoiTransform:
    bbox_xyxy: list[float]
    scale_x: float
    scale_y: float

    def roi_to_image(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        out = xy.copy()
        out[:, 0] = (xy[:, 0] / self.scale_x) + self.bbox_xyxy[0]
        out[:, 1] = (xy[:, 1] / self.scale_y) + self.bbox_xyxy[1]
        return out

    def image_to_roi(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        out = xy.copy()
        out[:, 0] = (xy[:, 0] - self.bbox_xyxy[0]) * self.scale_x
        out[:, 1] = (xy[:, 1] - self.bbox_xyxy[1]) * self.scale_y
        return out


def sanitize_bbox(bbox_xyxy: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = max(0.0, min(x1, width - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 <= x1:
        x2 = min(width - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(height - 1.0, y1 + 1.0)
    return [x1, y1, x2, y2]


def bbox_xyxy_from_keypoints(
    keypoints: np.ndarray,
    image_size: tuple[int, int],
    *,
    confidence: np.ndarray | None = None,
    margin_ratio: float = 0.1,
    min_conf: float = 0.05,
) -> list[float]:
    """Tight xyxy bbox from visible keypoints with margin (matches COCO export)."""
    width, height = image_size
    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.ndim != 2 or kps.shape[1] < 2:
        return [0.0, 0.0, float(width - 1), float(height - 1)]

    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        mask = conf >= min_conf
        pts = kps[mask] if np.any(mask) else kps
    else:
        pts = kps

    if pts.size == 0:
        return [0.0, 0.0, float(width - 1), float(height - 1)]

    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    bbox_w = max(1.0, x_max - x_min)
    bbox_h = max(1.0, y_max - y_min)
    margin_x = bbox_w * margin_ratio
    margin_y = bbox_h * margin_ratio
    x1 = max(0.0, x_min - margin_x)
    y1 = max(0.0, y_min - margin_y)
    x2 = min(float(width - 1), x_max + margin_x)
    y2 = min(float(height - 1), y_max + margin_y)
    return sanitize_bbox([x1, y1, x2, y2], width, height)


def bbox_area_fraction(bbox_xyxy: Iterable[float], image_size: tuple[int, int]) -> float:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(width * height))


def crop_and_resize(image: Image.Image, bbox_xyxy: Iterable[float], output_size: tuple[int, int]):
    width, height = image.size
    x1, y1, x2, y2 = sanitize_bbox(bbox_xyxy, width, height)
    crop = image.crop((x1, y1, x2, y2))
    out_w, out_h = output_size
    resized = crop.resize((out_w, out_h), Image.Resampling.BILINEAR)
    scale_x = out_w / max(1e-6, x2 - x1)
    scale_y = out_h / max(1e-6, y2 - y1)
    transform = RoiTransform(bbox_xyxy=[x1, y1, x2, y2], scale_x=scale_x, scale_y=scale_y)
    return resized, transform


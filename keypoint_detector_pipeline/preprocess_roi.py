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


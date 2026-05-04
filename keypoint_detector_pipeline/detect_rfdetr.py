"""RF-DETR bicycle detector stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PIL import Image

from pipeline.io_utils import write_jsonl


def _set_inference_env() -> None:
    os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")


def _iter_image_paths(image_dir: Path) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        for path in sorted(image_dir.glob(ext)):
            yield path


class RFDETRDetector:
    def __init__(
        self,
        model_id: str = "rfdetr-2xlarge",
        confidence: float = 0.5,
        target_class: str = "bicycle",
    ) -> None:
        _set_inference_env()
        from inference import get_model

        self.model = get_model(model_id)
        self.confidence = confidence
        self.target_class = target_class.lower().strip()

    def detect_image(self, image_path: Path, frame_id: int) -> list[dict]:
        image = Image.open(image_path).convert("RGB")
        pred = self.model.infer(image, confidence=self.confidence)[0]
        rows: list[dict] = []
        for item in pred.predictions:
            cls = str(item.class_name).lower()
            if cls != self.target_class:
                continue
            x = float(item.x)
            y = float(item.y)
            w = float(item.width)
            h = float(item.height)
            rows.append(
                {
                    "frame_id": frame_id,
                    "image_path": str(image_path),
                    "class_name": item.class_name,
                    "score": float(item.confidence),
                    "bbox_xyxy": [x - (w / 2.0), y - (h / 2.0), x + (w / 2.0), y + (h / 2.0)],
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows

    def detect_sequence(self, image_dir: Path) -> list[dict]:
        rows: list[dict] = []
        for frame_id, image_path in enumerate(_iter_image_paths(image_dir)):
            dets = self.detect_image(image_path, frame_id=frame_id)
            best = dets[0] if dets else None
            rows.append(
                best
                if best
                else {
                    "frame_id": frame_id,
                    "image_path": str(image_path),
                    "class_name": self.target_class,
                    "score": 0.0,
                    "bbox_xyxy": None,
                }
            )
        return rows


def run_detection(image_dir: Path, output_path: Path, model_id: str, confidence: float) -> Path:
    detector = RFDETRDetector(model_id=model_id, confidence=confidence)
    detections = detector.detect_sequence(image_dir)
    write_jsonl(output_path, detections)
    return output_path


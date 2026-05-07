"""2D keypoint inference stage with MMPose backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from keypoint_detector_pipeline.preprocess_roi import crop_and_resize
from keypoint_detector_pipeline.schema import NUM_KEYPOINTS


class MMPose2DInferencer:
    def __init__(
        self,
        pose2d_model: str = "rtmpose-l_8xb256-420e_coco-256x192",
        pose2d_weights: str | None = None,
        input_size: tuple[int, int] = (256, 192),
    ) -> None:
        self.input_size = input_size
        self._backend = "fallback"
        self._inferencer = None
        try:
            from mmpose.apis import MMPoseInferencer

            self._inferencer = MMPoseInferencer(
                pose2d=pose2d_model,
                pose2d_weights=pose2d_weights,
            )
            self._backend = "mmpose"
        except Exception:
            self._backend = "fallback"

    def _fallback_predict(self, bbox_xyxy: list[float]) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = bbox_xyxy
        grid_x = np.linspace(x1, x2, 6, dtype=np.float32)
        grid_y = np.linspace(y1, y2, 3, dtype=np.float32)
        pts = []
        for yi in range(3):
            for xi in range(6):
                pts.append([float(grid_x[xi]), float(grid_y[yi])])
        pts = np.asarray(pts[:NUM_KEYPOINTS], dtype=np.float32)
        conf = np.full((NUM_KEYPOINTS,), 0.2, dtype=np.float32)
        return pts, conf

    def _mmpose_predict(self, crop: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(crop.convert("RGB"))
        result_iter = self._inferencer(arr, return_vis=False, pred_out_dir="")
        result = next(result_iter)
        predictions = result.get("predictions", [])
        if not predictions or not predictions[0]:
            return np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32), np.zeros((NUM_KEYPOINTS,), dtype=np.float32)
        best = predictions[0][0]
        kps = np.asarray(best.get("keypoints", []), dtype=np.float32)
        kps_score = np.asarray(best.get("keypoint_scores", []), dtype=np.float32)
        if kps.shape[0] < NUM_KEYPOINTS:
            pad = np.zeros((NUM_KEYPOINTS - kps.shape[0], 2), dtype=np.float32)
            kps = np.concatenate([kps, pad], axis=0)
            kps_score = np.concatenate([kps_score, np.zeros((NUM_KEYPOINTS - kps_score.shape[0],), dtype=np.float32)])
        return kps[:NUM_KEYPOINTS], kps_score[:NUM_KEYPOINTS]

    def predict_global(
        self,
        image_path: Path,
        bbox_xyxy: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[float] | None]:
        if bbox_xyxy is None:
            return (
                np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
                np.zeros((NUM_KEYPOINTS,), dtype=np.float32),
                None,
            )

        image = Image.open(image_path).convert("RGB")
        crop, transform = crop_and_resize(image, bbox_xyxy=bbox_xyxy, output_size=self.input_size)

        if self._backend == "mmpose":
            roi_kps, conf = self._mmpose_predict(crop)
            global_kps = transform.roi_to_image(roi_kps)
            return global_kps, conf, transform.bbox_xyxy

        kps, conf = self._fallback_predict(transform.bbox_xyxy)
        return kps, conf, transform.bbox_xyxy


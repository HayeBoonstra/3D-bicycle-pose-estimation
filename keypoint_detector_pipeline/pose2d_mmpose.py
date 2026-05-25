"""2D keypoint inference stage with MMPose backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from keypoint_detector_pipeline.preprocess_roi import (
    bbox_area_fraction,
    bbox_xyxy_from_keypoints,
)
from keypoint_detector_pipeline.schema import NUM_KEYPOINTS


class MMPose2DInferencer:
    def __init__(
        self,
        pose2d_model: str = "rtmpose-l_8xb256-420e_coco-256x192",
        pose2d_weights: str | None = None,
        input_size: tuple[int, int] = (256, 192),
        device: str = "cuda:0",
    ) -> None:
        self.input_size = input_size
        self.device = device
        self._pose2d_model = pose2d_model
        self._pose2d_weights = pose2d_weights
        self._backend = "fallback"
        self._inferencer = None
        self._topdown_model = None
        try:
            from mmpose.apis import MMPoseInferencer

            self._inferencer = MMPoseInferencer(
                pose2d=pose2d_model,
                pose2d_weights=pose2d_weights,
                device=device,
            )
            self._backend = "mmpose"
        except Exception:
            self._backend = "fallback"

    def _get_topdown_model(self):
        """Lazy-load model for inference_topdown (uses val/test pipeline with external bbox)."""
        if self._topdown_model is None:
            from mmpose.apis import init_model

            self._topdown_model = init_model(
                self._pose2d_model,
                self._pose2d_weights,
                device=self.device,
            )
        return self._topdown_model

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

    def _parse_instance(self, instance: dict) -> tuple[np.ndarray, np.ndarray]:
        kps = np.asarray(instance.get("keypoints", []), dtype=np.float32)
        kps_score = np.asarray(instance.get("keypoint_scores", []), dtype=np.float32)
        if kps.shape[0] < NUM_KEYPOINTS:
            pad = np.zeros((NUM_KEYPOINTS - kps.shape[0], 2), dtype=np.float32)
            kps = np.concatenate([kps, pad], axis=0)
            kps_score = np.concatenate(
                [kps_score, np.zeros((NUM_KEYPOINTS - kps_score.shape[0],), dtype=np.float32)]
            )
        return kps[:NUM_KEYPOINTS], kps_score[:NUM_KEYPOINTS]

    def _select_best_instance(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        if not instances:
            return (
                np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
                np.zeros((NUM_KEYPOINTS,), dtype=np.float32),
            )
        best_kps = None
        best_conf = None
        best_score = -1.0
        for inst in instances:
            kps, conf = self._parse_instance(inst)
            score = float(np.mean(conf)) if conf.size else 0.0
            if score > best_score:
                best_score = score
                best_kps = kps
                best_conf = conf
        assert best_kps is not None and best_conf is not None
        return best_kps, best_conf

    def _parse_topdown_result(self, data_sample) -> tuple[np.ndarray, np.ndarray]:
        kps = np.asarray(data_sample.pred_instances.keypoints[0], dtype=np.float32)
        conf = np.asarray(data_sample.pred_instances.keypoint_scores[0], dtype=np.float32)
        if kps.shape[0] < NUM_KEYPOINTS:
            pad = np.zeros((NUM_KEYPOINTS - kps.shape[0], 2), dtype=np.float32)
            kps = np.concatenate([kps, pad], axis=0)
            conf = np.concatenate(
                [conf, np.zeros((NUM_KEYPOINTS - conf.shape[0],), dtype=np.float32)]
            )
        return kps[:NUM_KEYPOINTS], conf[:NUM_KEYPOINTS]

    def predict_full_image(self, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Run MMPose on the full frame (same path as infer_2d.py)."""
        if self._backend != "mmpose":
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            return self._fallback_predict([0.0, 0.0, float(w - 1), float(h - 1)])

        result_iter = self._inferencer(str(image_path), return_vis=False, pred_out_dir="")
        result = next(result_iter)
        predictions = result.get("predictions", [])
        if not predictions or not predictions[0]:
            return np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32), np.zeros((NUM_KEYPOINTS,), dtype=np.float32)
        return self._select_best_instance(predictions[0])

    def predict_global(
        self,
        image_path: Path,
        bbox_xyxy: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[float] | None]:
        """Run top-down pose on the full image using an external detector bbox.

        Uses MMPose ``inference_topdown`` (GetBBoxCenterScale + TopdownAffine), matching
        training/val. Do not pre-crop: MMPoseInferencer on a resized crop double-warps the
        image and ignores external bboxes when no det model is loaded.
        """
        if bbox_xyxy is None:
            return (
                np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
                np.zeros((NUM_KEYPOINTS,), dtype=np.float32),
                None,
            )

        if self._backend == "mmpose":
            from mmpose.apis import inference_topdown

            image = Image.open(image_path).convert("RGB")
            arr = np.asarray(image)
            bbox_arr = np.asarray([bbox_xyxy], dtype=np.float32)
            results = inference_topdown(
                self._get_topdown_model(),
                arr,
                bboxes=bbox_arr,
                bbox_format="xyxy",
            )
            if not results:
                return (
                    np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
                    np.zeros((NUM_KEYPOINTS,), dtype=np.float32),
                    list(bbox_xyxy),
                )
            kps, conf = self._parse_topdown_result(results[0])
            return kps, conf, list(bbox_xyxy)

        kps, conf = self._fallback_predict(bbox_xyxy)
        return kps, conf, list(bbox_xyxy)

    def predict_frame(
        self,
        image_path: Path,
        det_bbox_xyxy: list[float] | None,
        *,
        mode: str = "auto",
        min_det_bbox_area_frac: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, list[float], str]:
        """Predict 2D keypoints and return (kps, conf, bbox_xyxy, mode_used)."""
        image = Image.open(image_path).convert("RGB")
        image_size = image.size

        use_detection_crop = False
        if mode == "detection_bbox":
            use_detection_crop = det_bbox_xyxy is not None
        elif mode == "full_image":
            use_detection_crop = False
        elif mode == "auto":
            if det_bbox_xyxy is not None:
                frac = bbox_area_fraction(det_bbox_xyxy, image_size)
                use_detection_crop = frac >= min_det_bbox_area_frac
        else:
            raise ValueError(f"Unknown pose mode: {mode}")

        if use_detection_crop and det_bbox_xyxy is not None:
            kps, conf, bbox = self.predict_global(image_path, det_bbox_xyxy)
            return kps, conf, bbox if bbox is not None else det_bbox_xyxy, "detection_bbox"

        kps, conf = self.predict_full_image(image_path)
        bbox = bbox_xyxy_from_keypoints(kps, image_size, confidence=conf)
        return kps, conf, bbox, "full_image"


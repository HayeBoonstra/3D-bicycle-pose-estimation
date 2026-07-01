"""Detection metrics for RF-DETR bicycle bounding boxes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from evaluation.common import bbox_iou, bbox_xywh_to_xyxy


def _center_scale_errors(gt_xywh: np.ndarray, det_xyxy: np.ndarray, img_w: float, img_h: float) -> dict[str, float]:
    gt_cx = gt_xywh[0] + 0.5 * gt_xywh[2]
    gt_cy = gt_xywh[1] + 0.5 * gt_xywh[3]
    det_cx = 0.5 * (det_xyxy[0] + det_xyxy[2])
    det_cy = 0.5 * (det_xyxy[1] + det_xyxy[3])
    center_err_px = float(np.hypot(det_cx - gt_cx, det_cy - gt_cy))
    gt_area = float(gt_xywh[2] * gt_xywh[3])
    det_area = float(max(0.0, det_xyxy[2] - det_xyxy[0]) * max(0.0, det_xyxy[3] - det_xyxy[1]))
    gt_ar = float(gt_xywh[2] / max(gt_xywh[3], 1e-6))
    det_ar = float((det_xyxy[2] - det_xyxy[0]) / max(det_xyxy[3] - det_xyxy[1], 1e-6))
    return {
        "center_error_px": center_err_px,
        "center_error_pct_width": center_err_px / max(img_w, 1.0),
        "scale_error_area_ratio": abs(det_area - gt_area) / max(gt_area, 1e-6),
        "aspect_ratio_error": abs(det_ar - gt_ar),
    }


def compute_detection_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute detection metrics from stage12 records."""
    ious: list[float] = []
    detected = 0
    tp = fp = fn = 0
    center_px: list[float] = []
    scale_err: list[float] = []
    aspect_err: list[float] = []
    by_scene: dict[str, list[float]] = defaultdict(list)
    by_pattern: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        gt_bbox = rec.get("gt_bbox_xywh")
        det_bbox = rec.get("det_bbox_xyxy")
        scene = str(rec.get("scene_id", "unknown"))
        pattern = str(rec.get("trajectory_pattern", "unknown"))
        img_w = float(rec.get("image_width", 1))
        img_h = float(rec.get("image_height", 1))

        if gt_bbox is None:
            continue

        if det_bbox is None:
            fn += 1
            continue

        detected += 1
        gt_xyxy = bbox_xywh_to_xyxy(gt_bbox)
        det_xyxy = np.asarray(det_bbox, dtype=np.float32)
        iou = bbox_iou(gt_xyxy, det_xyxy)
        ious.append(iou)
        by_scene[scene].append(iou)
        by_pattern[pattern].append(iou)

        loc = _center_scale_errors(np.asarray(gt_bbox, dtype=np.float32), det_xyxy, img_w, img_h)
        center_px.append(loc["center_error_px"])
        scale_err.append(loc["scale_error_area_ratio"])
        aspect_err.append(loc["aspect_ratio_error"])

        if iou >= 0.5:
            tp += 1
        else:
            fp += 1

    n_gt = len([r for r in records if r.get("gt_bbox_xywh") is not None])
    fn = n_gt - detected
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    iou_arr = np.asarray(ious, dtype=np.float64) if ious else np.array([], dtype=np.float64)
    ap50 = recall  # single-instance proxy when one GT box per frame
    metrics: dict[str, Any] = {
        "num_frames": n_gt,
        "detection_rate": float(detected / max(n_gt, 1)),
        "mean_iou": float(np.mean(iou_arr)) if iou_arr.size else None,
        "median_iou": float(np.median(iou_arr)) if iou_arr.size else None,
        "pct_iou_ge_0_5": float(np.mean(iou_arr >= 0.5)) if iou_arr.size else None,
        "pct_iou_ge_0_75": float(np.mean(iou_arr >= 0.75)) if iou_arr.size else None,
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "ap50_proxy": float(ap50),
        "ar_proxy": float(recall),
        "mean_center_error_px": float(np.mean(center_px)) if center_px else None,
        "mean_scale_error_area_ratio": float(np.mean(scale_err)) if scale_err else None,
        "mean_aspect_ratio_error": float(np.mean(aspect_err)) if aspect_err else None,
        "iou_histogram": np.histogram(iou_arr, bins=10, range=(0.0, 1.0))[0].tolist() if iou_arr.size else [],
        "iou_histogram_fraction": (
            (np.histogram(iou_arr, bins=10, range=(0.0, 1.0))[0] / max(iou_arr.size, 1)).tolist()
            if iou_arr.size
            else []
        ),
        "by_scene_mean_iou": {k: float(np.mean(v)) for k, v in by_scene.items()},
        "by_pattern_mean_iou": {k: float(np.mean(v)) for k, v in by_pattern.items()},
    }

    # Optional pycocotools AP if available
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "bicycle"}]}
        coco_dt = []
        ann_id = 1
        for i, rec in enumerate(records):
            gt_bbox = rec.get("gt_bbox_xywh")
            if gt_bbox is None:
                continue
            img_id = i + 1
            w = float(rec.get("image_width", 1920))
            h = float(rec.get("image_height", 1080))
            coco_gt["images"].append({"id": img_id, "width": int(w), "height": int(h)})
            x, y, bw, bh = [float(v) for v in gt_bbox]
            coco_gt["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            det_bbox = rec.get("det_bbox_xyxy")
            if det_bbox is not None:
                d = [float(v) for v in det_bbox]
                coco_dt.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]],
                        "score": float(rec.get("det_score", 1.0)),
                    }
                )
        if coco_gt["annotations"] and coco_dt:
            coco = COCO()
            coco.dataset = coco_gt
            coco.createIndex()
            coco_eval = COCOeval(coco, iouType="bbox")
            coco_eval.cocoDt = coco.loadRes(coco_dt)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metrics["ap_50_95"] = float(coco_eval.stats[0])
            metrics["ap50"] = float(coco_eval.stats[1])
            metrics["ap75"] = float(coco_eval.stats[2])
            metrics["ar_max_100"] = float(coco_eval.stats[8])
    except Exception:
        pass

    return metrics

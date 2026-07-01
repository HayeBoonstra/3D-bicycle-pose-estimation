#!/usr/bin/env python3
"""Extract stage-1/2 records from bicycle_pose_dataset (static-frame 2D corpus).

Unlike ``extract_stage12_lifterinput.py`` (lifter-input clips with pre-baked sidecars), this
evaluates RF-DETR + RTMPose on single frames where camera distance/framing vary
more widely. Output feeds ``stage12_static_metrics.json`` via ``compute_stats.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.common import ensure_dir  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "bicycle_pose_dataset"
DEFAULT_MMPOSE_CONFIG = REPO_ROOT / "2d_keypoint_detector_training" / "rtmpose_bicycle_full.py"
DEFAULT_MMPOSE_CHECKPOINT = (
    REPO_ROOT / "training_outputs" / "mmpose_bicycle_rtmpose_l_gpu" / "best_coco_AP_epoch_175.pth"
)
BICYCLE_INPUT_SIZE = (256, 320)


def _load_coco(dataset_root: Path, split: str) -> tuple[dict[int, dict], dict[int, dict]]:
    ann_path = dataset_root / "annotations" / f"{split}.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {ann_path}")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(img["id"]): img for img in coco.get("images", [])}
    annotations = {int(ann["image_id"]): ann for ann in coco.get("annotations", [])}
    return images, annotations


def _coco_keypoints_to_arrays(flat_kps: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.zeros((18, 2), dtype=np.float32)
    visible = np.zeros(18, dtype=bool)
    occluded = np.zeros(18, dtype=bool)
    for j in range(18):
        base = j * 3
        if base + 2 >= len(flat_kps):
            break
        x, y, v = float(flat_kps[base]), float(flat_kps[base + 1]), int(flat_kps[base + 2])
        points[j] = [x, y]
        visible[j] = v > 0
        occluded[j] = v == 1
    return points, visible, occluded


def _image_path(dataset_root: Path, image_row: dict) -> Path:
    file_name = str(image_row["file_name"])
    candidate = dataset_root / file_name
    if candidate.is_file():
        return candidate
    # COCO file_name may already include images/<split>/...
    alt = dataset_root / Path(file_name).name
    if alt.is_file():
        return alt
    # Per-clip layout: images/<split>/<clip_id>/frame_XXXX.png
    parts = Path(file_name).parts
    if len(parts) >= 3:
        rel = Path(*parts[-3:]) if parts[0] == "images" else Path(*parts[-2:])
        candidate = dataset_root / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found for {file_name} under {dataset_root}")


def _load_jsonl_by_key(path: Path, key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row[key])] = row
    return out


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _run_detection(
    samples: list[dict[str, Any]],
    cache_path: Path,
    *,
    model_id: str,
    confidence: float,
    resume: bool,
) -> dict[str, dict]:
    from keypoint_detector_pipeline.detect_rfdetr import RFDETRDetector

    cached = _load_jsonl_by_key(cache_path, "image_id") if resume else {}
    if resume and cache_path.is_file() and not cached:
        cached = _load_jsonl_by_key(cache_path, "image_id")

    if not resume and cache_path.is_file():
        cache_path.unlink()

    detector = RFDETRDetector(model_id=model_id, confidence=confidence)
    ensure_dir(cache_path.parent)

    for sample in samples:
        image_id = str(sample["image_id"])
        if image_id in cached:
            continue
        image_path = Path(sample["image_path"])
        dets = detector.detect_image(image_path, frame_id=int(image_id))
        best = dets[0] if dets else None
        row = {
            "image_id": image_id,
            "image_path": str(image_path),
            "bbox_xyxy": best["bbox_xyxy"] if best else None,
            "score": float(best["score"]) if best else 0.0,
        }
        _append_jsonl(cache_path, row)
        cached[image_id] = row
    return cached


def _run_pose(
    samples: list[dict[str, Any]],
    detections: dict[str, dict],
    cache_path: Path,
    *,
    mmpose_config: Path,
    mmpose_checkpoint: Path,
    device: str,
    resume: bool,
) -> dict[str, dict]:
    from keypoint_detector_pipeline.pose2d_mmpose import MMPose2DInferencer

    cached = _load_jsonl_by_key(cache_path, "image_id") if resume else {}
    if not resume and cache_path.is_file():
        cache_path.unlink()

    inferencer = MMPose2DInferencer(
        pose2d_model=str(mmpose_config),
        pose2d_weights=str(mmpose_checkpoint),
        input_size=BICYCLE_INPUT_SIZE,
        device=device,
    )
    ensure_dir(cache_path.parent)

    for sample in samples:
        image_id = str(sample["image_id"])
        if image_id in cached:
            continue
        det = detections.get(image_id, {})
        bbox = det.get("bbox_xyxy")
        image_path = Path(sample["image_path"])
        kps, conf, _ = inferencer.predict_global(image_path, bbox)
        row = {
            "image_id": image_id,
            "points": kps.tolist(),
            "confidence": conf.tolist(),
        }
        _append_jsonl(cache_path, row)
        cached[image_id] = row
    return cached


def _build_records(
    samples: list[dict[str, Any]],
    detections: dict[str, dict],
    poses: dict[str, dict],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        image_id = str(sample["image_id"])
        gt_pts = np.asarray(sample["gt_points"], dtype=np.float32)
        gt_vis = np.asarray(sample["gt_visible"], dtype=bool)
        gt_occ = np.asarray(sample["gt_occluded"], dtype=bool)
        det = detections.get(image_id, {})
        pose = poses.get(image_id)

        rec: dict[str, Any] = {
            "clip_id": sample["clip_id"],
            "frame_index": int(sample["frame_index"]),
            "scene_id": sample["scene_id"],
            "trajectory_pattern": sample.get("trajectory_pattern", "static"),
            "image_width": float(sample["image_width"]),
            "image_height": float(sample["image_height"]),
            "gt_bbox_xywh": sample["gt_bbox_xywh"],
            "det_bbox_xyxy": det.get("bbox_xyxy"),
            "det_score": float(det.get("score", 0.0)),
            "gt_keypoints_2d": {
                "points": gt_pts.tolist(),
                "visible": gt_vis.tolist(),
                "occluded": gt_occ.tolist(),
            },
            "dataset": "bicycle_pose_dataset",
            "split": sample["split"],
            "image_id": image_id,
        }
        if pose is not None:
            rec["det_keypoints_2d"] = {
                "points": pose["points"],
                "confidence": pose["confidence"],
            }
        records.append(rec)
    return records


def _prepare_samples(
    dataset_root: Path,
    split: str,
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    images, annotations = _load_coco(dataset_root, split)
    samples: list[dict[str, Any]] = []
    for image_id in sorted(images):
        image_row = images[image_id]
        ann = annotations.get(image_id)
        if ann is None:
            continue
        image_path = _image_path(dataset_root, image_row)
        gt_pts, gt_vis, gt_occ = _coco_keypoints_to_arrays(ann.get("keypoints", []))
        samples.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "clip_id": str(image_row.get("clip_id", image_path.parent.name)),
                "frame_index": int(image_row.get("frame_index", 0)),
                "scene_id": str(image_row.get("scene_id", "unknown")),
                "trajectory_pattern": "static",
                "image_width": float(image_row.get("width", 1920)),
                "image_height": float(image_row.get("height", 1080)),
                "gt_bbox_xywh": [float(v) for v in ann["bbox"]],
                "gt_points": gt_pts.tolist(),
                "gt_visible": gt_vis.tolist(),
                "gt_occluded": gt_occ.tolist(),
                "split": split,
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract stage-1/2 records from bicycle_pose_dataset.")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/stage12_static_records.jsonl")
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results/stage12_static_cache")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of images.")
    p.add_argument("--run-detection", action="store_true", help="Run RF-DETR and update detection cache.")
    p.add_argument("--run-pose", action="store_true", help="Run RTMPose top-down and update pose cache.")
    p.add_argument("--skip-detection", action="store_true", help="Do not run RF-DETR (load cache only).")
    p.add_argument("--skip-pose", action="store_true", help="Do not run RTMPose (load cache only).")
    p.add_argument("--resume", action="store_true", help="Resume partially written detection/pose caches.")
    p.add_argument("--rfdetr-model", default="rfdetr-2xlarge")
    p.add_argument("--det-confidence", type=float, default=0.5)
    p.add_argument("--mmpose-config", type=Path, default=DEFAULT_MMPOSE_CONFIG)
    p.add_argument("--mmpose-checkpoint", type=Path, default=DEFAULT_MMPOSE_CHECKPOINT)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    samples = _prepare_samples(dataset_root, args.split, limit=args.limit)
    if not samples:
        raise RuntimeError(f"No samples found for split={args.split}")

    cache_dir = ensure_dir(args.cache_dir / args.split)
    det_cache = cache_dir / "detections.jsonl"
    pose_cache = cache_dir / "pose.jsonl"

    run_det = args.run_detection or (not args.skip_detection and not det_cache.is_file())
    run_pose = args.run_pose or (not args.skip_pose and not pose_cache.is_file())

    detections = _load_jsonl_by_key(det_cache, "image_id")
    if run_det:
        detections = _run_detection(
            samples,
            det_cache,
            model_id=args.rfdetr_model,
            confidence=args.det_confidence,
            resume=args.resume or det_cache.is_file(),
        )
    elif not detections:
        raise FileNotFoundError(
            f"No detection cache at {det_cache}. Run with --run-detection in the rfdetr env first."
        )

    if args.skip_pose and not args.run_pose:
        print(f"[extract_stage12_static] detection cache -> {det_cache} ({len(detections)} images)")
        return

    poses = _load_jsonl_by_key(pose_cache, "image_id")
    if run_pose:
        poses = _run_pose(
            samples,
            detections,
            pose_cache,
            mmpose_config=args.mmpose_config,
            mmpose_checkpoint=args.mmpose_checkpoint,
            device=args.device,
            resume=args.resume or pose_cache.is_file(),
        )
    elif not poses:
        raise FileNotFoundError(
            f"No pose cache at {pose_cache}. Run with --run-pose in the mmpose env first."
        )

    records = _build_records(samples, detections, poses)
    out_path = args.out
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_pose = sum(1 for r in records if r.get("det_keypoints_2d"))
    print(f"[extract_stage12_static] dataset={dataset_root}")
    print(f"[extract_stage12_static] split={args.split} samples={len(records)} pose={n_pose}")
    print(f"[extract_stage12_static] wrote {out_path}")


if __name__ == "__main__":
    main()

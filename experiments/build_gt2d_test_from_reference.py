#!/usr/bin/env python3
"""Build an oracle GT-2D PoseMamba test split from a detected-2D reference split.

This is intentionally stricter than ``build_sequences.py --input-2d gt`` for
evaluation: it preserves the exact reference test windows and 3D labels, then
only swaps RTMPose keypoints for projected GT keypoints. By default the GT
keypoints are normalized in the RF-DETR detection bbox frame so the lifter sees
the same coordinate system it was trained/evaluated on for detected-2D input.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import (  # noqa: E402
    BICYCLE_KEYPOINT_NAMES,
    KEYPOINT_INDEX,
)

GT_ROW_CACHE: dict[Path, dict[int, dict[str, Any]]] = {}
DET_ROW_CACHE: dict[Path, dict[int, dict[str, Any]]] = {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_frame_index(annotation_dir: Path, pattern: str, cache: dict[Path, dict[int, dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    if annotation_dir in cache:
        return cache[annotation_dir]
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted(annotation_dir.glob(pattern)):
        row = _load_json(path)
        idx = int(row["frame_index"])
        if idx in rows:
            raise ValueError(f"{annotation_dir}: duplicate frame_index {idx}")
        rows[idx] = row
    cache[annotation_dir] = rows
    return rows


def _normalize_2d(points_2d: np.ndarray, bboxes_xywh: np.ndarray) -> np.ndarray:
    centers = np.stack(
        [
            bboxes_xywh[:, 0] + bboxes_xywh[:, 2] * 0.5,
            bboxes_xywh[:, 1] + bboxes_xywh[:, 3] * 0.5,
        ],
        axis=-1,
    )
    scales = np.maximum(1.0, np.maximum(bboxes_xywh[:, 2], bboxes_xywh[:, 3]))
    return ((points_2d - centers[:, None, :]) / scales[:, None, None]).astype(np.float32)


def _points_from_gt_row(row: dict[str, Any]) -> np.ndarray:
    points = np.zeros((len(BICYCLE_KEYPOINT_NAMES), 2), dtype=np.float32)
    seen = np.zeros((len(BICYCLE_KEYPOINT_NAMES),), dtype=bool)
    for kp in row["keypoints"]:
        j = KEYPOINT_INDEX[kp["name"]]
        points[j] = [float(kp["x"]), float(kp["y"])]
        seen[j] = True
    if not np.all(seen):
        missing = [BICYCLE_KEYPOINT_NAMES[i] for i, ok in enumerate(seen) if not ok]
        raise ValueError(f"GT row missing keypoints: {missing}")
    return points


def _bbox_xywh(row_gt: dict[str, Any], row_det: dict[str, Any] | None, bbox_source: str) -> np.ndarray:
    if bbox_source == "gt":
        bbox = row_gt.get("gt_bbox_xywh")
        if bbox is None:
            raise ValueError(f"GT frame {row_gt.get('frame_index')} missing gt_bbox_xywh")
        x, y, w, h = [float(v) for v in bbox]
        return np.asarray([x, y, max(1.0, w), max(1.0, h)], dtype=np.float32)

    if row_det is None:
        raise ValueError(
            f"Frame {row_gt.get('frame_index')} missing detected sidecar required for detection bbox"
        )
    bbox_xyxy = row_det.get("det_bbox_xyxy") or row_det.get("bbox_xyxy")
    if bbox_xyxy is None:
        raise ValueError(f"Detected frame {row_det.get('frame_index')} missing det_bbox_xyxy/bbox_xyxy")
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return np.asarray([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32)


def _build_data_input(
    raw_root: Path,
    clip_id: str,
    frame_idx: np.ndarray,
    *,
    bbox_source: str,
) -> np.ndarray:
    ann_dir = raw_root / clip_id / "per_frame_annotations"
    gt_rows = _rows_by_frame_index(ann_dir, "keypoints_2d_frame_*.json", GT_ROW_CACHE)
    det_rows = _rows_by_frame_index(ann_dir, "keypoints_2d_detected_frame_*.json", DET_ROW_CACHE)
    points: list[np.ndarray] = []
    bboxes: list[np.ndarray] = []
    for idx in frame_idx.astype(int).tolist():
        if idx not in gt_rows:
            raise ValueError(f"{ann_dir}: missing GT row for frame_index {idx}")
        row_gt = gt_rows[idx]
        row_det = det_rows.get(idx)
        points.append(_points_from_gt_row(row_gt))
        bboxes.append(_bbox_xywh(row_gt, row_det, bbox_source=bbox_source))
    return _normalize_2d(np.stack(points, axis=0), np.stack(bboxes, axis=0))


def _write_manifest(
    out_root: Path,
    *,
    raw_root: Path,
    reference_test_dir: Path,
    bbox_source: str,
    num_windows: int,
    clip_ids: set[str],
) -> None:
    source = "gt_projection_detection_bbox" if bbox_source == "detection" else "gt_projection_gt_bbox"
    manifest = {
        "joint_names": BICYCLE_KEYPOINT_NAMES,
        "input_2d": "gt",
        "input_2d_source": source,
        "bbox_source": bbox_source,
        "reference_test_dir": str(reference_test_dir.resolve()),
        "raw_root": str(raw_root.resolve()),
        "posemamba_subdir": out_root.name,
        "split_sample_counts": {"test": int(num_windows)},
        "split_clip_counts": {"test": int(len(clip_ids))},
        "splits": {"test": sorted(clip_ids)},
        "normalization": "bbox_center_scale",
        "note": (
            "Oracle test split built from detected-2D reference pickles. "
            "Window keys, data_label, and dynamics_gt are copied from the reference; "
            "only data_input is replaced by projected GT 2D."
        ),
    }
    (out_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--reference-test-dir", type=Path, required=True)
    p.add_argument("--out-test-dir", type=Path, required=True)
    p.add_argument(
        "--bbox-source",
        choices=("detection", "gt"),
        default="detection",
        help="Bbox frame used to normalize projected GT 2D (default: detection, matching detected-2D input).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ref_dir = args.reference_test_dir.resolve()
    out_dir = args.out_test_dir.resolve()
    out_root = out_dir.parents[1]

    if not ref_dir.is_dir():
        raise FileNotFoundError(f"reference test dir not found: {ref_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.pkl"):
        stale.unlink()

    clip_ids: set[str] = set()
    num_written = 0
    for ref_path in sorted(ref_dir.glob("*.pkl")):
        with ref_path.open("rb") as f:
            payload = pickle.load(f)
        meta = dict(payload.get("meta", {}))
        clip_id = str(meta["clip_id"])
        clip_ids.add(clip_id)
        frame_idx = np.arange(int(meta["st"]), int(meta["end"]), dtype=np.int32)
        if frame_idx.shape[0] != int(payload["data_label"].shape[0]):
            raise ValueError(f"{ref_path.name}: meta st/end length does not match data_label")

        out_payload = {
            **payload,
            "data_input": _build_data_input(
                args.raw_root.resolve(),
                clip_id,
                frame_idx,
                bbox_source=args.bbox_source,
            ),
            "meta": {
                **meta,
                "input_2d": "gt",
                "input_2d_source": (
                    "gt_projection_detection_bbox"
                    if args.bbox_source == "detection"
                    else "gt_projection_gt_bbox"
                ),
                "bbox_source": args.bbox_source,
                "reference_pickle": str(ref_path.resolve()),
            },
        }
        with (out_dir / ref_path.name).open("wb") as f:
            pickle.dump(out_payload, f)
        num_written += 1

    _write_manifest(
        out_root,
        raw_root=args.raw_root,
        reference_test_dir=ref_dir,
        bbox_source=args.bbox_source,
        num_windows=num_written,
        clip_ids=clip_ids,
    )
    print(
        f"[build_gt2d_test_from_reference] wrote {num_written} windows, "
        f"{len(clip_ids)} clips -> {out_dir}"
    )


if __name__ == "__main__":
    main()

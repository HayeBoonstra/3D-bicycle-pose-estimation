#!/usr/bin/env python3
"""Extract stage-1/2 records from 3D-lifter input clips (pre-baked detection sidecars)."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "3d_keypoint_detector_training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES, KEYPOINT_INDEX  # noqa: E402
from posemamba_bicycle_io import load_sequence_pkl  # noqa: E402
from evaluation.common import default_detected2d_test_dir, default_raw_root, ensure_dir  # noqa: E402

DETECTED_PREFIX = "keypoints_2d_detected_frame_"


def _test_clip_ids(test_dir: Path) -> set[str]:
    ids: set[str] = set()
    for pkl in test_dir.glob("*.pkl"):
        obj = load_sequence_pkl(pkl)
        meta = obj.get("meta", {})
        ids.add(str(meta.get("clip_id", pkl.stem.rsplit("_", 1)[0])))
    return ids


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _points_from_row(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.zeros((18, 2), dtype=np.float32)
    conf = np.zeros(18, dtype=np.float32)
    visible = np.zeros(18, dtype=bool)
    occluded = np.zeros(18, dtype=bool)
    for kp in row.get("keypoints", []):
        j = KEYPOINT_INDEX[kp["name"]]
        points[j] = [float(kp["x"]), float(kp["y"])]
        conf[j] = float(kp.get("det_score", kp.get("v", 0)) / 2.0)
        v = int(kp.get("v", 2))
        visible[j] = v > 0
        occluded[j] = bool(kp.get("occluded_by_prop", False))
    return points, conf, visible, occluded


def _load_manifests(raw_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    scene_by_clip: dict[str, str] = {}
    pattern_by_clip: dict[str, str] = {}

    manifest_csv = raw_root / "manifest.csv"
    if manifest_csv.is_file():
        with manifest_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row.get("clip_id", "")
                scene_by_clip[cid] = row.get("scene_id", "unknown")
                traj_id = row.get("trajectory_id", "")
                pattern_by_clip[cid] = traj_id.split("_")[0] if traj_id else "unknown"

    traj_manifest = raw_root.parent / "mujoco_blender_trajectories" / "manifest.csv"
    traj_pattern: dict[str, str] = {}
    if traj_manifest.is_file():
        with traj_manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = row.get("trajectory_id", "")
                traj_pattern[tid] = row.get("pattern", "unknown")

    for cid in scene_by_clip:
        parts = cid.split("_")
        if len(parts) >= 4:
            traj_id = "_".join(parts[2:-1]) if parts[0] == "clip" else parts[1]
            if traj_id in traj_pattern:
                pattern_by_clip[cid] = traj_pattern[traj_id]

    return scene_by_clip, pattern_by_clip


def _load_detections(clip_dir: Path) -> dict[int, dict]:
    det_path = clip_dir / "detections.jsonl"
    out: dict[int, dict] = {}
    if not det_path.is_file():
        return out
    for line in det_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        idx = int(row.get("frame_index", row.get("frame_id", 0)))
        out[idx] = row
    return out


def extract_clip(
    clip_dir: Path,
    scene_id: str = "unknown",
    trajectory_pattern: str = "unknown",
) -> list[dict[str, Any]]:
    ann_dir = clip_dir / "per_frame_annotations"
    detections = _load_detections(clip_dir)
    records: list[dict[str, Any]] = []

    for gt_path in sorted(ann_dir.glob("keypoints_2d_frame_*.json")):
        gt_row = _load_json(gt_path)
        idx = int(gt_row["frame_index"])
        gt_pts, _, gt_vis, gt_occ = _points_from_row(gt_row)

        det_path = ann_dir / f"{DETECTED_PREFIX}{gt_row['frame']:04d}.json"
        if not det_path.is_file():
            # try alternate naming
            alt = list(ann_dir.glob(f"{DETECTED_PREFIX}*.json"))
            det_path = next((p for p in alt if int(_load_json(p)["frame_index"]) == idx), None)
        det_row = _load_json(det_path) if det_path and det_path.is_file() else None

        det_pts = det_conf = None
        det_vis = det_occ = None
        if det_row:
            det_pts, det_conf, det_vis, det_occ = _points_from_row(det_row)

        det_info = detections.get(idx, {})
        det_bbox = det_row.get("det_bbox_xyxy") if det_row else det_info.get("bbox_xyxy")

        rec: dict[str, Any] = {
            "clip_id": clip_dir.name,
            "frame_index": idx,
            "scene_id": scene_id,
            "trajectory_pattern": trajectory_pattern,
            "image_width": float(gt_row.get("image_width", 1920)),
            "image_height": float(gt_row.get("image_height", 1080)),
            "gt_bbox_xywh": gt_row.get("gt_bbox_xywh"),
            "det_bbox_xyxy": det_bbox,
            "det_score": float(det_info.get("score", det_row.get("det_score", 0.0)) if det_row or det_info else 0.0),
            "gt_keypoints_2d": {
                "points": gt_pts.tolist(),
                "visible": gt_vis.tolist(),
                "occluded": gt_occ.tolist(),
            },
        }
        if det_pts is not None:
            rec["det_keypoints_2d"] = {
                "points": det_pts.tolist(),
                "confidence": det_conf.tolist(),
            }
        records.append(rec)
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract stage-1/2 evaluation records.")
    p.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="BICYCLE test split pickles (default: auto-detect repo or SSD)",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Raw Blender clip root (default: RAW_ROOT env, repo data/, or /mnt/SmallSSD/...)",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/stage12_lifterinput_records.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = (args.raw_root or default_raw_root()).resolve()
    test_dir = (args.test_dir or default_detected2d_test_dir()).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"raw clip root not found: {raw_root}\n"
            "  Set RAW_ROOT=/mnt/SmallSSD/3D-bicycle-pose-estimation/raw_blender_posemamba"
        )
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"test pickle dir not found: {test_dir}\n"
            "  Set DATA_ROOT=/mnt/SmallSSD/3D-bicycle-pose-estimation/posemamba_training_sequences"
        )

    clip_ids = _test_clip_ids(test_dir)
    scene_map, pattern_map = _load_manifests(raw_root)

    all_records: list[dict[str, Any]] = []
    for clip_dir in sorted(raw_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        if clip_dir.name not in clip_ids:
            continue
        scene = scene_map.get(clip_dir.name, "unknown")
        pattern = pattern_map.get(clip_dir.name, "unknown")
        all_records.extend(extract_clip(clip_dir, scene, pattern))

    out_path = args.out
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"[extract_stage12_lifterinput] raw_root={raw_root}")
    print(f"[extract_stage12_lifterinput] test_dir={test_dir}")
    print(f"[extract_stage12_lifterinput] wrote {out_path} ({len(all_records)} frames, {len(clip_ids)} test clips)")


if __name__ == "__main__":
    main()

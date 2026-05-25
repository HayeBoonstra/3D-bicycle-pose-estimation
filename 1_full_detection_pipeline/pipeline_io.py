"""Shared IO and normalization for the full detection pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_detector_pipeline.io_utils import iter_jsonl, write_jsonl
from keypoint_detector_pipeline.schema import BICYCLE_KEYPOINT_NAMES, NUM_KEYPOINTS
from keypoint_detector_pipeline.sequence_builder import normalize_points, rows_to_arrays

CLIP_LEN = 243

DETECTIONS_NAME = "detections.jsonl"
KEYPOINTS_2D_NAME = "keypoints_2d.jsonl"
KEYPOINTS_3D_NAME = "keypoints_3d.npz"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def iter_frame_paths(frames_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(frames_dir.glob(f"*{ext}"))
    return sorted({p.resolve() for p in paths})


def count_frames(frames_dir: Path) -> int:
    return len(iter_frame_paths(frames_dir))


def require_clip_length(num_frames: int, clip_len: int = CLIP_LEN) -> None:
    if num_frames != clip_len:
        raise ValueError(
            f"Pipeline v1 requires exactly {clip_len} frames, got {num_frames}. "
            f"Use a {clip_len}-frame sequence or extend the pipeline for other lengths."
        )


def should_skip_output(path: Path, resume: bool) -> bool:
    return resume and path.is_file()


def load_keypoints_2d_rows(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def build_normalized_2d_sequence(rows: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T, J, 2) bbox-normalized 2D, frame_ids, and raw image-space keypoints."""
    points, confidence, bboxes = rows_to_arrays(rows)
    filled = points.copy()
    for t in range(points.shape[0]):
        mask = confidence[t] <= 0.0
        if not np.any(mask):
            continue
        if t > 0:
            filled[t, mask] = filled[t - 1, mask]
        else:
            filled[t, mask] = 0.0
    normalized = normalize_points(filled, bboxes)
    sorted_rows = sorted(rows, key=lambda r: int(r["frame_id"]))
    frame_ids = np.asarray([int(r["frame_id"]) for r in sorted_rows], dtype=np.int32)
    return normalized.astype(np.float32), frame_ids, points.astype(np.float32)


def save_keypoints_3d_npz(
    path: Path,
    pred: np.ndarray,
    *,
    frame_ids: np.ndarray | None = None,
    data_input: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "pred": pred.astype(np.float32),
        "keypoint_names": np.asarray(BICYCLE_KEYPOINT_NAMES, dtype=object),
    }
    if frame_ids is not None:
        payload["frame_ids"] = frame_ids.astype(np.int32)
    if data_input is not None:
        payload["data_input"] = data_input.astype(np.float32)
    np.savez_compressed(path, **payload)


__all__ = [
    "BICYCLE_KEYPOINT_NAMES",
    "CLIP_LEN",
    "DETECTIONS_NAME",
    "IMAGE_EXTENSIONS",
    "KEYPOINTS_2D_NAME",
    "KEYPOINTS_3D_NAME",
    "NUM_KEYPOINTS",
    "PIPELINE_DIR",
    "REPO_ROOT",
    "build_normalized_2d_sequence",
    "count_frames",
    "iter_frame_paths",
    "iter_jsonl",
    "load_keypoints_2d_rows",
    "require_clip_length",
    "save_keypoints_3d_npz",
    "should_skip_output",
    "write_jsonl",
]

#!/usr/bin/env python3
"""Run RF-DETR on each raw clip's frames and write per-clip detections.jsonl."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_detector_pipeline.detect_rfdetr import RFDETRDetector
from keypoint_detector_pipeline.io_utils import iter_jsonl, write_jsonl

DETECTIONS_NAME = "detections.jsonl"
FRAME_INDEX_RE = re.compile(r"frame_(\d+)$")


def _clip_dirs(raw_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(raw_root.iterdir())
        if path.is_dir() and (path / "keypoints_3d.jsonl").exists()
    ]


def _frame_index(path: Path) -> int:
    match = FRAME_INDEX_RE.match(path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse frame index from {path.name}")


def _iter_frame_paths(clip_dir: Path) -> list[tuple[int, Path]]:
    frames_dir = clip_dir / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Missing frames directory: {frames_dir}")
    paths: list[tuple[int, Path]] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        for path in sorted(frames_dir.glob(ext)):
            paths.append((_frame_index(path), path.resolve()))
    paths.sort(key=lambda item: item[0])
    if not paths:
        raise FileNotFoundError(f"No frame images in {frames_dir}")
    return paths


def export_clip(
    clip_dir: Path,
    detector: RFDETRDetector,
    *,
    resume: bool,
    limit_frames: int | None,
) -> dict:
    output_path = clip_dir / DETECTIONS_NAME
    if resume and output_path.is_file():
        rows = list(iter_jsonl(output_path))
        return {
            "clip_id": clip_dir.name,
            "processed": 0,
            "skipped": len(rows),
            "detected_frames": sum(1 for row in rows if row.get("bbox_xyxy") is not None),
        }

    frame_paths = _iter_frame_paths(clip_dir)
    if limit_frames is not None:
        frame_paths = frame_paths[:limit_frames]

    rows: list[dict] = []
    detected = 0
    for frame_index, image_path in frame_paths:
        dets = detector.detect_image(image_path, frame_id=frame_index)
        best = dets[0] if dets else None
        if best is not None:
            detected += 1
            rows.append(
                {
                    "frame_index": frame_index,
                    "frame_id": frame_index,
                    "image_path": str(image_path),
                    "class_name": best["class_name"],
                    "score": best["score"],
                    "bbox_xyxy": best["bbox_xyxy"],
                }
            )
        else:
            rows.append(
                {
                    "frame_index": frame_index,
                    "frame_id": frame_index,
                    "image_path": str(image_path),
                    "class_name": detector.target_class,
                    "score": 0.0,
                    "bbox_xyxy": None,
                }
            )

    write_jsonl(output_path, rows)
    return {
        "clip_id": clip_dir.name,
        "processed": len(rows),
        "skipped": 0,
        "detected_frames": detected,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RF-DETR bicycle detections for raw clips.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--rfdetr-model", default="rfdetr-2xlarge")
    parser.add_argument("--det-confidence", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None, help="Per-clip frame cap for debugging.")
    parser.add_argument("--clip-id", default=None, help="Process a single clip directory name.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_root = args.raw_root.resolve()
    detector = RFDETRDetector(model_id=args.rfdetr_model, confidence=args.det_confidence)

    clip_dirs = _clip_dirs(raw_root)
    if args.clip_id:
        clip_dirs = [raw_root / args.clip_id]
    if args.limit_clips is not None:
        clip_dirs = clip_dirs[: args.limit_clips]
    if not clip_dirs:
        raise RuntimeError(f"No clips under {raw_root}")

    summary = []
    for clip_dir in clip_dirs:
        print(f"[export_clip_detections] {clip_dir.name}")
        row = export_clip(
            clip_dir,
            detector,
            resume=args.resume,
            limit_frames=args.limit_frames,
        )
        summary.append(row)
        print(
            f"  processed={row['processed']} skipped={row['skipped']} "
            f"detected_frames={row['detected_frames']}"
        )

    summary_path = raw_root / "clip_detections_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[export_clip_detections] Wrote {summary_path}")


if __name__ == "__main__":
    main()

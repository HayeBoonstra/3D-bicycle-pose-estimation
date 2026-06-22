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


def _load_clip_list(path: Path, raw_root: Path) -> list[Path]:
    clip_dirs: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = raw_root / item
        clip_dirs.append(candidate)
    return clip_dirs


def _apply_shard(clip_dirs: list[Path], shard_index: int | None, num_shards: int | None) -> list[Path]:
    if shard_index is None and num_shards is None:
        return clip_dirs
    if shard_index is None or num_shards is None:
        raise ValueError("--shard-index and --num-shards must be provided together")
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return [path for idx, path in enumerate(clip_dirs) if idx % num_shards == shard_index]


def _frame_index(path: Path) -> int:
    match = FRAME_INDEX_RE.match(path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse frame index from {path.name}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_path(clip_dir: Path, row2d: dict) -> Path:
    image_file = row2d.get("image_file", "")
    if image_file:
        candidate = clip_dir / str(image_file)
        if candidate.is_file():
            return candidate.resolve()
    frame = row2d.get("frame")
    if frame is not None:
        candidate = clip_dir / "frames" / f"frame_{int(frame):04d}.png"
        if candidate.is_file():
            return candidate.resolve()
    frame_index = int(row2d.get("frame_index", -1))
    frames_dir = clip_dir / "frames"
    candidate = frames_dir / f"frame_{frame_index:04d}.png"
    if candidate.is_file():
        return candidate.resolve()
    matches = sorted(frames_dir.glob("*.png"))
    if frame_index >= 0 and frame_index < len(matches):
        return matches[frame_index].resolve()
    raise FileNotFoundError(f"No image for frame_index={frame_index} in {clip_dir}")


def _iter_frame_paths(clip_dir: Path) -> list[tuple[int, Path]]:
    annotation_dir = clip_dir / "per_frame_annotations"
    if annotation_dir.is_dir():
        gt_paths = sorted(annotation_dir.glob("keypoints_2d_frame_*.json"))
        if gt_paths:
            rows = [_load_json(path) for path in gt_paths]
            rows.sort(key=lambda item: int(item["frame_index"]))
            return [(int(row["frame_index"]), _resolve_image_path(clip_dir, row)) for row in rows]

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
    parser.add_argument("--clip-list", type=Path, help="Text file of clip directory names/paths to process.")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_root = args.raw_root.resolve()
    detector = RFDETRDetector(model_id=args.rfdetr_model, confidence=args.det_confidence)

    clip_dirs = _load_clip_list(args.clip_list, raw_root) if args.clip_list else _clip_dirs(raw_root)
    if args.clip_id:
        clip_dirs = [raw_root / args.clip_id]
    clip_dirs = _apply_shard(clip_dirs, args.shard_index, args.num_shards)
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

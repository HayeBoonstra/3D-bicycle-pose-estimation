#!/usr/bin/env python3
"""Export PNG assets for a thesis methods-section pipeline diagram.

Reuses the same drawing conventions as:
  - data_generation_pipeline_tools/visualize_raw_annotations.py (Blender GT)
  - 1_full_detection_pipeline/visualize_intermediates.py (bbox + detected 2D)
  - data_generation_pipeline_tools/visualize_bicycle_pose3d.py (lifted 3D)

Outputs (under --output-dir, default methods_diagram_assets/ next to this script):
  blender_render.png
  blender_annotated.png
  bbox_detection.png
  sequence_2d/frame_00.png … frame_02.png
  sequence_3d/frame_00.png … frame_02.png
  manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "1_full_detection_pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from data_generation_pipeline_tools.bicycle_keypoint_schema import (  # noqa: E402
    BICYCLE_KEYPOINT_NAMES,
    BICYCLE_SKELETON_NAMES,
    KEYPOINT_INDEX,
)
import imageio.v2 as imageio  # noqa: E402

from data_generation_pipeline_tools.visualize_bicycle_pose3d import (  # noqa: E402
    _finalize_3d_axes,
    axis_limits_for_poses,
    draw_skeleton,
    reorient_for_display,
)
from data_generation_pipeline_tools.visualize_raw_annotations import (  # noqa: E402
    _annotation_path,
    _image_for_annotation,
    _load_json,
)
from keypoint_detector_pipeline.io_utils import iter_jsonl  # noqa: E402
from pipeline_io import DETECTIONS_NAME, KEYPOINTS_2D_NAME, KEYPOINTS_3D_NAME  # noqa: E402

CONF_THRESHOLD = 0.01
DEFAULT_BLENDER_CLIP = "clip_tree_lined_scene_2010066764"
DEFAULT_SEQUENCE_FRAME_IDS = (50, 120, 190)


def _skeleton_edges() -> list[tuple[int, int]]:
    return [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]) for a, b in BICYCLE_SKELETON_NAMES]


def _rows_by_frame_id(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["frame_id"]): row for row in iter_jsonl(path)}


def _pick_blender_clip(raw_renders_root: Path, clip_name: str | None) -> Path:
    if clip_name:
        clip_dir = raw_renders_root / clip_name
        if not (clip_dir / "keypoints_3d.jsonl").is_file():
            raise FileNotFoundError(f"Blender clip not found or incomplete: {clip_dir}")
        return clip_dir

    candidates = sorted(
        path
        for path in raw_renders_root.iterdir()
        if path.is_dir() and (path / "keypoints_3d.jsonl").is_file() and (path / "frames").is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"No Blender clips under {raw_renders_root}")

    preferred = raw_renders_root / DEFAULT_BLENDER_CLIP
    if preferred in candidates:
        return preferred
    return candidates[0]


def _blender_frame_index(clip_dir: Path) -> int:
    annotations = sorted((clip_dir / "per_frame_annotations").glob("keypoints_2d_frame_*.json"))
    if not annotations:
        raise FileNotFoundError(f"No GT 2D annotations in {clip_dir / 'per_frame_annotations'}")
    stem = annotations[0].stem
    return int(stem.rsplit("_", maxsplit=1)[-1])


def export_blender_render(clip_dir: Path, frame_index: int, out_path: Path) -> dict[str, Any]:
    annotation = _load_json(_annotation_path(clip_dir, frame_index))
    image = _image_for_annotation(
        clip_dir,
        annotation,
        (
            int(annotation.get("image_width", 1920)),
            int(annotation.get("image_height", 1080)),
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=95)
    return {
        "clip": clip_dir.name,
        "frame_index": frame_index,
        "source_image": annotation.get("image_file"),
        "size": [image.width, image.height],
    }


def export_blender_annotated(clip_dir: Path, frame_index: int, out_path: Path) -> None:
    annotation = _load_json(_annotation_path(clip_dir, frame_index))
    image = _image_for_annotation(
        clip_dir,
        annotation,
        (
            int(annotation.get("image_width", 1920)),
            int(annotation.get("image_height", 1080)),
        ),
    )
    draw = ImageDraw.Draw(image)
    by_name = {kp["name"]: kp for kp in annotation.get("keypoints", [])}

    for start, end in BICYCLE_SKELETON_NAMES:
        a = by_name.get(start)
        b = by_name.get(end)
        if a is None or b is None:
            continue
        if int(a.get("v", 0)) <= 0 or int(b.get("v", 0)) <= 0:
            continue
        draw.line(
            [(float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"]))],
            fill=(15, 135, 65),
            width=2,
        )

    radius = 5
    for name in BICYCLE_KEYPOINT_NAMES:
        kp = by_name.get(name)
        if kp is None or int(kp.get("v", 0)) <= 0:
            continue
        x, y = float(kp["x"]), float(kp["y"])
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(220, 40, 40),
            outline=(255, 255, 255),
            width=1,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=95)


def export_bbox_detection(row: dict[str, Any], out_path: Path) -> None:
    image = Image.open(row["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = row.get("bbox_xyxy")
    if bbox is None:
        raise RuntimeError(f"No bbox in detection row for frame {row.get('frame_id')}")

    x1, y1, x2, y2 = [float(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 80), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=95)


def export_detected_2d(row: dict[str, Any], out_path: Path) -> None:
    image = Image.open(row["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [float(v) for v in row["bbox_xyxy"]]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 100), width=2)

    kps = row["keypoints_2d"]
    conf = row.get("confidence", [1.0] * len(kps))
    points = [(float(p[0]), float(p[1])) for p in kps]

    for i, j in _skeleton_edges():
        if i >= len(points) or j >= len(points):
            continue
        ci = float(conf[i]) if i < len(conf) else 0.0
        cj = float(conf[j]) if j < len(conf) else 0.0
        if ci < CONF_THRESHOLD or cj < CONF_THRESHOLD:
            continue
        draw.line([*points[i], *points[j]], fill=(0, 200, 255), width=2)

    for idx, (x, y) in enumerate(points):
        c = float(conf[idx]) if idx < len(conf) else 0.0
        if c < CONF_THRESHOLD:
            continue
        r = 4
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 90, 0), outline=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=95)


def export_lifted_3d(
    pose: np.ndarray,
    out_path: Path,
    *,
    lo: np.ndarray,
    hi: np.ndarray,
    elev: float,
    azim: float,
) -> None:
    import io

    fig = plt.figure(figsize=(7.2, 6.4), dpi=120)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw_skeleton(ax, pose, edgecolor="#00457E", linewidth=2.4, pointcolor="#00457E")
    _finalize_3d_axes(ax, lo, hi, elev=elev, azim=azim, invert_z=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    buf.seek(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(imageio.imread(buf)).save(out_path)


def _resolve_sequence_frame_ids(frame_ids: list[int] | None, clip_len: int) -> list[int]:
    if frame_ids:
        return frame_ids
    if clip_len < 3:
        return list(range(min(3, clip_len)))
    a = clip_len // 4
    b = clip_len // 2
    c = (3 * clip_len) // 4
    return [a, b, c]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "methods_diagram_assets",
    )
    parser.add_argument(
        "--pipeline-output-dir",
        type=Path,
        default=PIPELINE_DIR / "output",
        help="Directory with detections.jsonl, keypoints_2d.jsonl, keypoints_3d.npz",
    )
    parser.add_argument(
        "--raw-renders-root",
        type=Path,
        default=REPO_ROOT / "raw_renders",
    )
    parser.add_argument(
        "--blender-clip",
        default=None,
        help=f"Blender clip folder name under raw_renders (default: {DEFAULT_BLENDER_CLIP} if present)",
    )
    parser.add_argument(
        "--sequence-frame-ids",
        type=int,
        nargs=3,
        default=list(DEFAULT_SEQUENCE_FRAME_IDS),
        metavar=("F0", "F1", "F2"),
        help="Three pipeline frame IDs for the 2D/3D temporal sequence",
    )
    parser.add_argument("--bbox-frame-id", type=int, default=None, help="Detection frame (default: middle sequence frame)")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-70.0)
    parser.add_argument("--reorient", default="camera_up", choices=("none", "camera_up"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    pipeline_dir = args.pipeline_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_dir = _pick_blender_clip(args.raw_renders_root.resolve(), args.blender_clip)
    blender_frame_index = _blender_frame_index(clip_dir)

    detections_path = pipeline_dir / DETECTIONS_NAME
    keypoints_2d_path = pipeline_dir / KEYPOINTS_2D_NAME
    keypoints_3d_path = pipeline_dir / KEYPOINTS_3D_NAME
    for path in (detections_path, keypoints_2d_path, keypoints_3d_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing pipeline artifact: {path}")

    detections = _rows_by_frame_id(detections_path)
    keypoints_2d = _rows_by_frame_id(keypoints_2d_path)
    npz = np.load(keypoints_3d_path)
    pred = npz["pred"]
    frame_ids_npz = npz["frame_ids"] if "frame_ids" in npz else np.arange(len(pred))
    frame_id_to_pred_idx = {int(fid): idx for idx, fid in enumerate(frame_ids_npz)}

    sequence_frame_ids = _resolve_sequence_frame_ids(args.sequence_frame_ids, len(pred))
    bbox_frame_id = args.bbox_frame_id if args.bbox_frame_id is not None else sequence_frame_ids[1]

    manifest: dict[str, Any] = {
        "blender": {},
        "pipeline_output_dir": str(pipeline_dir),
        "bbox_detection": {"frame_id": bbox_frame_id, "path": "bbox_detection.png"},
        "sequence_2d": [],
        "sequence_3d": [],
    }

    blender_meta = export_blender_render(
        clip_dir,
        blender_frame_index,
        output_dir / "blender_render.png",
    )
    manifest["blender"] = blender_meta
    export_blender_annotated(
        clip_dir,
        blender_frame_index,
        output_dir / "blender_annotated.png",
    )
    manifest["blender"]["annotated_path"] = "blender_annotated.png"

    if bbox_frame_id not in detections:
        raise KeyError(f"bbox frame_id {bbox_frame_id} not in {detections_path}")
    export_bbox_detection(detections[bbox_frame_id], output_dir / "bbox_detection.png")

    pred_v = reorient_for_display(pred, args.reorient)
    sequence_pred_indices = [frame_id_to_pred_idx[frame_id] for frame_id in sequence_frame_ids]
    lo, hi = axis_limits_for_poses(pred_v[sequence_pred_indices])

    for seq_idx, frame_id in enumerate(sequence_frame_ids):
        if frame_id not in keypoints_2d:
            raise KeyError(f"sequence frame_id {frame_id} not in {keypoints_2d_path}")
        if frame_id not in frame_id_to_pred_idx:
            raise KeyError(f"sequence frame_id {frame_id} not in {keypoints_3d_path}")

        rel_2d = f"sequence_2d/frame_{seq_idx:02d}.png"
        rel_3d = f"sequence_3d/frame_{seq_idx:02d}.png"
        export_detected_2d(keypoints_2d[frame_id], output_dir / rel_2d)
        export_lifted_3d(
            pred_v[frame_id_to_pred_idx[frame_id]],
            output_dir / rel_3d,
            lo=lo,
            hi=hi,
            elev=args.elev,
            azim=args.azim,
        )
        manifest["sequence_2d"].append({"sequence_index": seq_idx, "frame_id": frame_id, "path": rel_2d})
        manifest["sequence_3d"].append({"sequence_index": seq_idx, "frame_id": frame_id, "path": rel_3d})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote assets under {output_dir}")
    print(f"  blender clip: {clip_dir.name} (frame {blender_frame_index})")
    print(f"  bbox frame_id: {bbox_frame_id}")
    print(f"  sequence frame_ids: {sequence_frame_ids}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()

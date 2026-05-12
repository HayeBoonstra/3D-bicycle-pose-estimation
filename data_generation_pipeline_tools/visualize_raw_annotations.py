"""Visualize raw bicycle 2D/3D annotation clips.

This tool reads the raw annotation contract shared by the Blender exporter and
the MuJoCo-direct exporter:

    clip_dir/
      camera.json
      keypoints_3d.jsonl
      per_frame_annotations/keypoints_2d_frame_XXXX.json

It writes per-frame QA panels with:
  - image-space 2D keypoints/skeleton and bbox (full frame, letterboxed)
  - a bbox-centered square crop (same scaling idea as build_sequences 2D normalization)
  - 3D skeleton projections in bicycle-local, camera, or world coordinates

For MuJoCo-direct clips there are usually no rendered frames, so the 2D panel is
drawn on a blank canvas using the exported image size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import (
    BICYCLE_KEYPOINT_NAMES,
    BICYCLE_SKELETON_NAMES,
    KEYPOINT_INDEX,
)


COLORS = {
    "background": (18, 18, 18),
    "panel": (245, 245, 245),
    "grid": (215, 215, 215),
    "axis_x": (230, 80, 80),
    "axis_y": (70, 150, 230),
    "axis_z": (80, 170, 80),
    "edge": (15, 135, 65),
    "point": (220, 40, 40),
    "hidden_edge": (150, 150, 150),
    "hidden_point": (120, 120, 120),
    "bbox": (245, 210, 45),
    "text": (35, 35, 35),
    "title": (235, 235, 235),
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _clip_dirs(raw_root: Path, clip_id: str | None) -> list[Path]:
    if (raw_root / "keypoints_3d.jsonl").exists():
        clips = [raw_root]
    else:
        clips = [path for path in sorted(raw_root.iterdir()) if path.is_dir() and (path / "keypoints_3d.jsonl").exists()]
    if clip_id is not None:
        clips = [path for path in clips if path.name == clip_id]
    if not clips:
        raise RuntimeError(f"No raw annotation clips found under {raw_root}")
    return clips


def _annotation_path(clip_dir: Path, frame_index: int) -> Path:
    return clip_dir / "per_frame_annotations" / f"keypoints_2d_frame_{frame_index:04d}.json"


def _image_for_annotation(clip_dir: Path, annotation: dict[str, Any], fallback_size: tuple[int, int]) -> Image.Image:
    image_file = annotation.get("image_file")
    if image_file:
        image_path = clip_dir / str(image_file)
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
    return Image.new("RGB", fallback_size, color=(250, 250, 250))


def _resize_with_letterbox(image: Image.Image, target_size: tuple[int, int]) -> tuple[Image.Image, float, tuple[int, int]]:
    target_w, target_h = target_size
    scale = min(target_w / image.width, target_h / image.height)
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", target_size, color=COLORS["panel"])
    offset = ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas, scale, offset


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int] = COLORS["text"]) -> None:
    draw.text(xy, text, font=_font(13), fill=fill)


def _bbox_xywh_from_annotation(annotation: dict[str, Any], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("gt_bbox_xywh")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, w, h = [float(v) for v in bbox]
        if w > 1e-3 and h > 1e-3:
            return x, y, w, h
    xs: list[float] = []
    ys: list[float] = []
    for kp in annotation.get("keypoints", []):
        if int(kp.get("v", 0)) <= 0:
            continue
        xs.append(float(kp["x"]))
        ys.append(float(kp["y"]))
    if len(xs) < 2:
        for kp in annotation.get("keypoints", []):
            xs.append(float(kp["x"]))
            ys.append(float(kp["y"]))
    if not xs:
        return None
    pad = 8.0
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    return x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)


def _square_roi_xyxy(
    bbox_xywh: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    *,
    margin_ratio: float = 0.12,
) -> tuple[float, float, float, float]:
    """Square region centered on bbox, side max(w,h)*(1+2*margin), clamped to image.

    Matches the scalar scale used in build_sequences._normalize_2d (max of w,h),
    so the crop is a reasonable visual proxy for bbox-normalized training coords.
    """

    x, y, w, h = bbox_xywh
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    side = max(w, h, 1.0) * (1.0 + 2.0 * margin_ratio)
    x0 = cx - 0.5 * side
    y0 = cy - 0.5 * side
    x1 = cx + 0.5 * side
    y1 = cy + 0.5 * side
    x0 = max(0.0, min(x0, float(img_w - 1)))
    y0 = max(0.0, min(y0, float(img_h - 1)))
    x1 = max(x0 + 1.0, min(x1, float(img_w)))
    y1 = max(y0 + 1.0, min(y1, float(img_h)))
    return x0, y0, x1, y1


def _draw_2d_panel(
    annotation: dict[str, Any],
    clip_dir: Path,
    *,
    panel_size: tuple[int, int],
    show_names: bool,
) -> Image.Image:
    width = int(annotation.get("image_width", panel_size[0]))
    height = int(annotation.get("image_height", panel_size[1]))
    base = _image_for_annotation(clip_dir, annotation, (width, height))
    panel, scale, offset = _resize_with_letterbox(base, panel_size)
    draw = ImageDraw.Draw(panel)

    by_name = {kp["name"]: kp for kp in annotation.get("keypoints", [])}

    def transform(x: float, y: float) -> tuple[float, float]:
        return x * scale + offset[0], y * scale + offset[1]

    for start, end in BICYCLE_SKELETON_NAMES:
        a = by_name.get(start)
        b = by_name.get(end)
        if a is None or b is None:
            continue
        visible = int(a.get("v", 0)) > 0 and int(b.get("v", 0)) > 0
        color = COLORS["edge"] if visible else COLORS["hidden_edge"]
        draw.line([transform(float(a["x"]), float(a["y"])), transform(float(b["x"]), float(b["y"]))], fill=color, width=2)

    bbox = annotation.get("gt_bbox_xywh")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, w, h = [float(value) for value in bbox]
        x0, y0 = transform(x, y)
        x1, y1 = transform(x + w, y + h)
        draw.rectangle((x0, y0, x1, y1), outline=COLORS["bbox"], width=2)

    radius = 4
    for name in BICYCLE_KEYPOINT_NAMES:
        kp = by_name.get(name)
        if kp is None:
            continue
        x, y = transform(float(kp["x"]), float(kp["y"]))
        visible = int(kp.get("v", 0)) > 0
        color = COLORS["point"] if visible else COLORS["hidden_point"]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
        if show_names and visible:
            _draw_label(draw, (int(x + 5), int(y - 5)), name.replace("k_", ""), fill=COLORS["text"])

    draw.rectangle((0, 0, panel_size[0] - 1, panel_size[1] - 1), outline=(40, 40, 40), width=1)
    _draw_label(draw, (10, 8), "2D full image (letterboxed)")
    return panel


def _draw_2d_cropped_panel(
    annotation: dict[str, Any],
    clip_dir: Path,
    *,
    panel_size: tuple[int, int],
    show_names: bool,
    margin_ratio: float = 0.12,
) -> Image.Image:
    """2D panel cropped to a square ROI around gt_bbox (training-scale preview)."""

    width = int(annotation.get("image_width", panel_size[0]))
    height = int(annotation.get("image_height", panel_size[1]))
    base = _image_for_annotation(clip_dir, annotation, (width, height))
    bbox = _bbox_xywh_from_annotation(annotation, width, height)
    if bbox is None:
        panel = Image.new("RGB", panel_size, color=COLORS["panel"])
        draw = ImageDraw.Draw(panel)
        _draw_label(draw, (10, 8), "2D bbox crop (no bbox / keypoints)")
        draw.rectangle((0, 0, panel_size[0] - 1, panel_size[1] - 1), outline=(40, 40, 40), width=1)
        return panel

    x0, y0, x1, y1 = _square_roi_xyxy(bbox, width, height, margin_ratio=margin_ratio)
    ix0 = int(math.floor(x0))
    iy0 = int(math.floor(y0))
    ix1 = int(math.ceil(x1))
    iy1 = int(math.ceil(y1))
    ix0 = max(0, min(ix0, base.width - 1))
    iy0 = max(0, min(iy0, base.height - 1))
    ix1 = max(ix0 + 1, min(ix1, base.width))
    iy1 = max(iy0 + 1, min(iy1, base.height))
    crop = base.crop((ix0, iy0, ix1, iy1))
    panel, scale, offset = _resize_with_letterbox(crop, panel_size)
    draw = ImageDraw.Draw(panel)
    by_name = {kp["name"]: kp for kp in annotation.get("keypoints", [])}

    def transform(x: float, y: float) -> tuple[float, float]:
        return (x - float(ix0)) * scale + offset[0], (y - float(iy0)) * scale + offset[1]

    for start, end in BICYCLE_SKELETON_NAMES:
        a = by_name.get(start)
        b = by_name.get(end)
        if a is None or b is None:
            continue
        visible = int(a.get("v", 0)) > 0 and int(b.get("v", 0)) > 0
        color = COLORS["edge"] if visible else COLORS["hidden_edge"]
        draw.line([transform(float(a["x"]), float(a["y"])), transform(float(b["x"]), float(b["y"]))], fill=color, width=2)

    bx, by, bw, bh = bbox
    rx0, ry0 = transform(bx, by)
    rx1, ry1 = transform(bx + bw, by + bh)
    draw.rectangle((rx0, ry0, rx1, ry1), outline=COLORS["bbox"], width=2)

    radius = max(3, min(6, panel_size[0] // 120))
    for name in BICYCLE_KEYPOINT_NAMES:
        kp = by_name.get(name)
        if kp is None:
            continue
        x, y = transform(float(kp["x"]), float(kp["y"]))
        visible = int(kp.get("v", 0)) > 0
        color = COLORS["point"] if visible else COLORS["hidden_point"]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
        if show_names and visible:
            _draw_label(draw, (int(x + 5), int(y - 5)), name.replace("k_", ""), fill=COLORS["text"])

    draw.rectangle((0, 0, panel_size[0] - 1, panel_size[1] - 1), outline=(40, 40, 40), width=1)
    _draw_label(draw, (10, 8), "2D bbox-centered crop (training-scale)")
    return panel


def _bounds(points_by_frame: np.ndarray, padding_ratio: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min(points_by_frame.reshape(-1, 3), axis=0)
    hi = np.max(points_by_frame.reshape(-1, 3), axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = span * padding_ratio
    return lo - pad, hi + pad


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise ValueError("Cannot normalize near-zero vector while building bicycle-local frame.")
    return vec / norm


def _bicycle_local_points(points_world: np.ndarray) -> np.ndarray:
    """Convert world keypoints to a per-frame bicycle-local coordinate frame.

    X is forward from rear hub center to front hub center, Y is right across the
    rear hub axle, and Z is up from their cross product. The bottom bracket is
    the origin. This makes side/top/front projections stable even while the bike
    moves through the world or is viewed from different cameras.
    """

    bottom_bracket = points_world[KEYPOINT_INDEX["k_bottom_bracket"]]
    rear_left = points_world[KEYPOINT_INDEX["k_rear_hub_left"]]
    rear_right = points_world[KEYPOINT_INDEX["k_rear_hub_right"]]
    front_left = points_world[KEYPOINT_INDEX["k_front_hub_left"]]
    front_right = points_world[KEYPOINT_INDEX["k_front_hub_right"]]
    rear_center = 0.5 * (rear_left + rear_right)
    front_center = 0.5 * (front_left + front_right)

    x_axis = _unit(front_center - rear_center)
    y_axis = rear_right - rear_left
    y_axis = _unit(y_axis - np.dot(y_axis, x_axis) * x_axis)
    z_axis = _unit(np.cross(x_axis, y_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    return ((points_world - bottom_bracket) @ basis).astype(np.float32)


def _project_axis(
    points: np.ndarray,
    axes: tuple[int, int],
    bounds: tuple[np.ndarray, np.ndarray],
    panel_rect: tuple[int, int, int, int],
) -> np.ndarray:
    lo, hi = bounds
    x0, y0, x1, y1 = panel_rect
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    a0, a1 = axes
    span_x = max(float(hi[a0] - lo[a0]), 1e-6)
    span_y = max(float(hi[a1] - lo[a1]), 1e-6)
    center_x = 0.5 * float(hi[a0] + lo[a0])
    center_y = 0.5 * float(hi[a1] + lo[a1])
    px_per_unit = min(w / span_x, h / span_y)
    panel_center_x = x0 + w * 0.5
    panel_center_y = y0 + h * 0.5
    out = np.zeros((points.shape[0], 2), dtype=np.float32)
    out[:, 0] = panel_center_x + (points[:, a0] - center_x) * px_per_unit
    out[:, 1] = panel_center_y - (points[:, a1] - center_y) * px_per_unit
    return out


def _draw_grid(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    for i in range(1, 4):
        x = int(x0 + (x1 - x0) * i / 4)
        y = int(y0 + (y1 - y0) * i / 4)
        draw.line((x, y0, x, y1), fill=COLORS["grid"], width=1)
        draw.line((x0, y, x1, y), fill=COLORS["grid"], width=1)
    draw.rectangle(rect, outline=(70, 70, 70), width=1)


def _draw_projection(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    axes: tuple[int, int],
    bounds: tuple[np.ndarray, np.ndarray],
    rect: tuple[int, int, int, int],
    label: str,
) -> None:
    _draw_grid(draw, rect)
    projected = _project_axis(points, axes, bounds, rect)
    for start, end in BICYCLE_SKELETON_NAMES:
        a = KEYPOINT_INDEX[start]
        b = KEYPOINT_INDEX[end]
        draw.line([tuple(projected[a]), tuple(projected[b])], fill=COLORS["edge"], width=2)
    radius = 4
    for xy in projected:
        x, y = float(xy[0]), float(xy[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS["point"], outline=(255, 255, 255))
    _draw_label(draw, (rect[0] + 8, rect[1] + 6), label)


def _draw_3d_panel(
    points: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    panel_size: tuple[int, int],
    coord_frame: str,
) -> Image.Image:
    panel = Image.new("RGB", panel_size, color=COLORS["panel"])
    draw = ImageDraw.Draw(panel)
    w, h = panel_size
    margin = 28
    gap = 16
    plot_h = (h - margin * 2 - gap * 2) // 3
    rects = [
        (margin, margin + i * (plot_h + gap), w - margin, margin + i * (plot_h + gap) + plot_h)
        for i in range(3)
    ]
    if coord_frame == "bicycle":
        labels = ["bicycle top: forward/right", "bicycle side: forward/up", "bicycle rear: right/up"]
    else:
        labels = [f"{coord_frame}: X/Y", f"{coord_frame}: X/Z", f"{coord_frame}: Y/Z"]
    axes = [(0, 1), (0, 2), (1, 2)]
    for rect, axis_pair, label in zip(rects, axes, labels):
        _draw_projection(draw, points, axis_pair, bounds, rect, label)
    draw.rectangle((0, 0, w - 1, h - 1), outline=(40, 40, 40), width=1)
    return panel


def _points_from_row(row3d: dict[str, Any], coord_frame: str) -> np.ndarray:
    key = "kps_camera"
    if coord_frame in {"world", "bicycle"}:
        key = "kps_world"
    points = np.asarray(row3d[key], dtype=np.float32)
    if points.shape != (len(BICYCLE_KEYPOINT_NAMES), 3):
        raise ValueError(f"Expected {key} shape {(len(BICYCLE_KEYPOINT_NAMES), 3)}, got {points.shape}")
    if coord_frame == "bicycle":
        return _bicycle_local_points(points)
    return points


def _compose_frame(
    *,
    clip_dir: Path,
    annotation: dict[str, Any],
    row3d: dict[str, Any],
    bounds: tuple[np.ndarray, np.ndarray],
    coord_frame: str,
    output_width: int,
    show_names: bool,
) -> Image.Image:
    panel_gap = 14
    title_h = 42
    panel_w = (output_width - panel_gap) // 2
    panel_h = int(panel_w * 0.75)
    if panel_h % 2:
        panel_h += 1
    stack_gap = 10
    h_crop = (panel_h - stack_gap + 1) // 2
    h_full = panel_h - stack_gap - h_crop
    points = _points_from_row(row3d, coord_frame)
    crop_panel = _draw_2d_cropped_panel(annotation, clip_dir, panel_size=(panel_w, h_crop), show_names=show_names)
    full_panel = _draw_2d_panel(annotation, clip_dir, panel_size=(panel_w, h_full), show_names=show_names)
    image_panel = Image.new("RGB", (panel_w, panel_h), color=COLORS["panel"])
    image_panel.paste(crop_panel, (0, 0))
    image_panel.paste(full_panel, (0, h_crop + stack_gap))
    points_panel = _draw_3d_panel(points, bounds, panel_size=(panel_w, panel_h), coord_frame=coord_frame)

    canvas = Image.new("RGB", (output_width, panel_h + title_h), color=COLORS["background"])
    draw = ImageDraw.Draw(canvas)
    title = f"{clip_dir.name} | frame {int(annotation['frame_index']):04d}"
    draw.text((12, 10), title, font=_font(18), fill=COLORS["title"])
    canvas.paste(image_panel, (0, title_h))
    canvas.paste(points_panel, (panel_w + panel_gap, title_h))
    return canvas


def _selected_indices(count: int, frame_step: int, max_frames: int | None) -> list[int]:
    indices = list(range(0, count, max(1, frame_step)))
    if max_frames is not None:
        indices = indices[: max(0, max_frames)]
    return indices


def visualize_clip(
    clip_dir: Path,
    out_root: Path,
    *,
    coord_frame: str,
    frame_step: int,
    max_frames: int | None,
    output_width: int,
    show_names: bool,
) -> int:
    rows3d = _load_jsonl(clip_dir / "keypoints_3d.jsonl")
    if not rows3d:
        raise RuntimeError(f"No 3D rows found in {clip_dir}")
    all_points = np.asarray([_points_from_row(row, coord_frame) for row in rows3d], dtype=np.float32)
    bounds = _bounds(all_points)
    frame_indices = _selected_indices(len(rows3d), frame_step=frame_step, max_frames=max_frames)

    out_dir = out_root / clip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for output_idx, row_idx in enumerate(frame_indices):
        row3d = rows3d[row_idx]
        annotation = _load_json(_annotation_path(clip_dir, int(row3d["frame_index"])))
        frame = _compose_frame(
            clip_dir=clip_dir,
            annotation=annotation,
            row3d=row3d,
            bounds=bounds,
            coord_frame=coord_frame,
            output_width=output_width,
            show_names=show_names,
        )
        frame.save(out_dir / f"frame_{output_idx:04d}.png")
        written += 1
    return written


def _encode_clip(frames_dir: Path, output_video: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    subprocess.run(command, check=True)


def visualize_raw_annotations(args: argparse.Namespace) -> None:
    clips = _clip_dirs(args.raw_root, args.clip_id)
    if args.max_clips is not None:
        clips = clips[: max(0, args.max_clips)]
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for clip_dir in clips:
        written = visualize_clip(
            clip_dir,
            args.out,
            coord_frame=args.coord_frame,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            output_width=args.output_width,
            show_names=args.show_names,
        )
        summary.append({"clip_id": clip_dir.name, "frames": written, "out_dir": str(args.out / clip_dir.name)})
        if args.encode_video:
            _encode_clip(args.out / clip_dir.name, args.out / f"{clip_dir.name}.mp4", fps=args.fps)
    print(json.dumps({"out": str(args.out), "clips": summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw bicycle 2D/3D annotation clips.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw annotation root or a single clip directory.")
    parser.add_argument("--out", type=Path, help="Output visualization directory.")
    parser.add_argument("--clip-id", help="Optional clip directory name to visualize.")
    parser.add_argument("--max-clips", type=int, help="Visualize at most this many clip directories.")
    parser.add_argument("--coord-frame", choices=["bicycle", "camera", "world"], default="bicycle")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-width", type=int, default=1400)
    parser.add_argument("--show-names", action="store_true")
    parser.add_argument("--encode-video", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if args.out is None:
        args.out = args.raw_root / "visualizations"
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("--max-clips must be >= 1")
    return args


def main() -> None:
    visualize_raw_annotations(parse_args())


if __name__ == "__main__":
    main()

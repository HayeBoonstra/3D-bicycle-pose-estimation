#!/usr/bin/env python3
"""Compose qualitative figure: RGB + detected 2D (left) | 3D pred+GT overlay (right)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "1_full_detection_pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from data_generation_pipeline_tools.bicycle_keypoint_schema import (  # noqa: E402
    BICYCLE_KEYPOINT_NAMES,
    KEYPOINT_INDEX,
)
from data_generation_pipeline_tools.visualize_bicycle_pose3d import (  # noqa: E402
    axis_limits_cube,
    axis_limits_for_poses,
    reorient_for_display,
    render_frame,
    subtract_root,
    view_angles_for_reorient,
)
from keypoint_detector_pipeline.io_utils import iter_jsonl  # noqa: E402
from pipeline_io import KEYPOINTS_2D_NAME, KEYPOINTS_3D_NAME  # noqa: E402
from visualize_intermediates import draw_keypoints_2d_frame  # noqa: E402

_J = len(BICYCLE_KEYPOINT_NAMES)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rows_by_frame_id(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["frame_id"]): row for row in iter_jsonl(path)}


def _discover_clips(raw_root: Path) -> list[Path]:
    clips = sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name.startswith("clip_") and (path / "keypoints_3d.jsonl").is_file()
    )
    if not clips:
        raise FileNotFoundError(f"No qualitative clips under {raw_root}")
    return clips


def _scene_id_from_clip(clip_dir: Path) -> str:
    name = clip_dir.name
    if name.startswith("clip_") and "_qual_" in name:
        return name[len("clip_") : name.index("_qual_")]
    return name


def _gt_camera_frame(rows3d: list[dict[str, Any]], frame_index: int) -> np.ndarray:
    row = next(r for r in rows3d if int(r["frame_index"]) == frame_index)
    kps_cam = row.get("kps_camera")
    if not isinstance(kps_cam, list) or len(kps_cam) != _J:
        raise ValueError(f"Frame {frame_index}: expected kps_camera length {_J}")
    pts = np.zeros((_J, 3), dtype=np.float32)
    for j, pt in enumerate(kps_cam):
        if pt is None:
            raise ValueError(f"Frame {frame_index}: missing camera keypoint at joint {j}")
        pts[j] = np.asarray(pt, dtype=np.float32)
    return subtract_root(pts[np.newaxis, ...], root_index=0)[0]


def _resolve_frame_id(frame_arg: str, num_frames: int, default_frame_id: int | None) -> int:
    if frame_arg == "auto":
        if default_frame_id is not None:
            return default_frame_id
        return num_frames // 2
    return int(frame_arg)


def _match_time_index(frame_ids: np.ndarray, frame_id: int) -> int:
    matches = np.where(frame_ids == frame_id)[0]
    if matches.size == 0:
        raise ValueError(f"frame_id {frame_id} not in pipeline frame_ids {frame_ids.tolist()}")
    return int(matches[0])


def compose_clip_figure(
    *,
    raw_clip_dir: Path,
    pipeline_dir: Path,
    out_path: Path,
    frame_id: int,
    elev: float | None,
    azim: float | None,
    reorient: str,
    panel_height: int,
) -> dict[str, Any]:
    kp2d_path = pipeline_dir / KEYPOINTS_2D_NAME
    kp3d_path = pipeline_dir / KEYPOINTS_3D_NAME
    if not kp2d_path.is_file():
        raise FileNotFoundError(f"Missing {kp2d_path}")
    if not kp3d_path.is_file():
        raise FileNotFoundError(f"Missing {kp3d_path}")

    keypoints_2d = _rows_by_frame_id(kp2d_path)
    if frame_id not in keypoints_2d:
        raise KeyError(f"frame_id {frame_id} not in {kp2d_path}")

    npz = np.load(kp3d_path)
    pred = np.asarray(npz["pred"], dtype=np.float32)
    frame_ids = np.asarray(npz["frame_ids"], dtype=np.int32) if "frame_ids" in npz else np.arange(len(pred))
    t_idx = _match_time_index(frame_ids, frame_id)

    rows3d = _load_jsonl(raw_clip_dir / "keypoints_3d.jsonl")
    gt_frame = _gt_camera_frame(rows3d, frame_id)
    pred_frame = pred[t_idx]

    pred_v = reorient_for_display(pred_frame[np.newaxis, ...], reorient)[0]
    gt_v = reorient_for_display(gt_frame[np.newaxis, ...], reorient)[0]
    if reorient == "camera_view":
        lo, hi = axis_limits_cube(pred_v[np.newaxis, ...], gt_v[np.newaxis, ...])
    else:
        lo, hi = axis_limits_for_poses(pred_v[np.newaxis, ...], gt_v[np.newaxis, ...])
    if elev is None and azim is None:
        elev, azim = view_angles_for_reorient(reorient)

    left = draw_keypoints_2d_frame(keypoints_2d[frame_id])
    right_rgb = render_frame(
        pred_v,
        gt_v,
        layout="overlay",
        lo=lo,
        hi=hi,
        elev=elev,
        azim=azim,
        equal_aspect=reorient == "camera_view",
        title=None,
    )
    right = Image.fromarray(right_rgb)

    target_h = panel_height
    left_scale = target_h / left.height
    right_scale = target_h / right.height
    left_resized = left.resize(
        (max(1, int(round(left.width * left_scale))), target_h),
        Image.Resampling.LANCZOS,
    )
    right_resized = right.resize(
        (max(1, int(round(right.width * right_scale))), target_h),
        Image.Resampling.LANCZOS,
    )

    gap = 12
    canvas = Image.new("RGB", (left_resized.width + gap + right_resized.width, target_h), (255, 255, 255))
    canvas.paste(left_resized, (0, 0))
    canvas.paste(right_resized, (left_resized.width + gap, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)

    return {
        "clip": raw_clip_dir.name,
        "scene_id": _scene_id_from_clip(raw_clip_dir),
        "frame_id": frame_id,
        "output": str(out_path),
        "mpjpe_mm": float(np.mean(np.linalg.norm(pred_frame - gt_frame, axis=-1)) * 1000.0),
    }


def _parse_scene_frames(values: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected SCENE=FRAME, got {item!r}")
        scene, frame_s = item.split("=", maxsplit=1)
        overrides[scene.strip()] = int(frame_s.strip())
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data" / "qualitative_eval" / "raw")
    parser.add_argument("--pipeline-root", type=Path, default=REPO_ROOT / "data" / "qualitative_eval" / "pipeline")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "qualitative_eval" / "figures")
    parser.add_argument(
        "--frame-id",
        default="auto",
        help="Frame index per clip, or 'auto' for mid-clip (default: auto)",
    )
    parser.add_argument(
        "--scene-frame",
        action="append",
        default=[],
        metavar="SCENE=FRAME",
        help="Per-scene frame override, e.g. docks_scene=205",
    )
    parser.add_argument("--elev", type=float, default=20.0, help="Matplotlib elevation (default: 20 oblique)")
    parser.add_argument("--azim", type=float, default=-70.0, help="Matplotlib azimuth (default: -70 oblique)")
    parser.add_argument(
        "--reorient",
        default="camera_up",
        choices=("none", "camera_up", "camera_view"),
        help="Display axis remap (default: camera_up with oblique elev/azim)",
    )
    parser.add_argument("--panel-height", type=int, default=720)
    parser.add_argument("--grid", action="store_true", help="Also write composite_grid.png stacking all scenes")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    pipeline_root = args.pipeline_root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_frames = _parse_scene_frames(args.scene_frame)

    manifest: list[dict[str, Any]] = []
    composite_paths: list[Path] = []

    for raw_clip in _discover_clips(raw_root):
        pipeline_dir = pipeline_root / raw_clip.name
        scene_id = _scene_id_from_clip(raw_clip)
        out_path = out_dir / f"composite_{scene_id}.png"

        kp3d_path = pipeline_dir / KEYPOINTS_3D_NAME
        if not kp3d_path.is_file():
            print(f"[skip] missing pipeline output for {raw_clip.name}")
            continue

        npz = np.load(kp3d_path)
        num_frames = int(np.asarray(npz["pred"]).shape[0])
        if scene_id in scene_frames:
            frame_id = scene_frames[scene_id]
        else:
            frame_id = _resolve_frame_id(str(args.frame_id), num_frames, default_frame_id=None)

        meta = compose_clip_figure(
            raw_clip_dir=raw_clip,
            pipeline_dir=pipeline_dir,
            out_path=out_path,
            frame_id=frame_id,
            elev=args.elev,
            azim=args.azim,
            reorient=args.reorient,
            panel_height=args.panel_height,
        )
        manifest.append(meta)
        composite_paths.append(out_path)
        print(f"[composite] {out_path} frame={frame_id} mpjpe={meta['mpjpe_mm']:.1f} mm")

    if args.grid and composite_paths:
        panels = [Image.open(path).convert("RGB") for path in composite_paths]
        width = max(p.width for p in panels)
        gap = 16
        height = sum(p.height for p in panels) + gap * (len(panels) - 1)
        grid = Image.new("RGB", (width, height), (255, 255, 255))
        y = 0
        for panel in panels:
            grid.paste(panel, ((width - panel.width) // 2, y))
            y += panel.height + gap
        grid_path = out_dir / "composite_grid.png"
        grid.save(grid_path, quality=95)
        print(f"[composite] grid -> {grid_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[composite] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

"""Render several camera seeds from one loaded Blender scene."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_clip import render_one_clip


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multiple follow-camera seeds from one Blender scene load.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--clip-prefix", required=True)
    parser.add_argument("--camera-seeds", required=True, help="Comma-separated camera seeds.")
    parser.add_argument("--camera-target", default="k_handlebar_middle")
    parser.add_argument("--trajectory-csv", type=Path)
    parser.add_argument("--bike", default="")
    parser.add_argument("--rider", default="")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--sync-window-size", type=int, default=80)
    parser.add_argument("--no-sync-camera-window", action="store_true")
    parser.add_argument("--encode-video", action="store_true")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--render-format", choices=("PNG", "JPEG"), default="PNG")
    parser.add_argument("--resolution-percentage", type=int, default=100)
    parser.add_argument("--cycles-samples", type=int, default=0)
    parser.add_argument("--quiet-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-min-distance", type=float, default=4.0)
    parser.add_argument("--camera-max-distance", type=float, default=12.0)
    parser.add_argument("--camera-min-bbox-area-frac", type=float, default=0.04)
    parser.add_argument("--camera-max-bbox-area-frac", type=float, default=0.80)
    parser.add_argument("--camera-min-visible-keypoints", type=int, default=14)
    parser.add_argument("--camera-min-visible-frame-ratio", type=float, default=0.9)
    parser.add_argument("--camera-fit-margin", type=float, default=1.25)
    parser.add_argument("--camera-mode", choices=("track", "fixed"), default="track")
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Skip frame rendering; re-export 2D/3D annotations only.",
    )
    return parser.parse_args(_argv_after_double_dash())


def _camera_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--camera-seeds must contain at least one seed")
    return seeds


def main() -> None:
    args = _parse_args()
    for camera_seed in _camera_seeds(args.camera_seeds):
        clip_id = f"{args.clip_prefix}_{camera_seed:08d}"
        render_one_clip(
            Namespace(
                clip_id=clip_id,
                scene_id=args.scene_id,
                camera_seed=camera_seed,
                out=args.out_root / clip_id,
                bike=args.bike,
                rider=args.rider,
                camera_target=args.camera_target,
                trajectory_csv=args.trajectory_csv,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                sync_window_size=args.sync_window_size,
                no_sync_camera_window=args.no_sync_camera_window,
                encode_video=args.encode_video,
                fps=args.fps,
                render_format=args.render_format,
                resolution_percentage=args.resolution_percentage,
                cycles_samples=args.cycles_samples,
                quiet_mode=args.quiet_mode,
                camera_min_distance=args.camera_min_distance,
                camera_max_distance=args.camera_max_distance,
                camera_min_bbox_area_frac=args.camera_min_bbox_area_frac,
                camera_max_bbox_area_frac=args.camera_max_bbox_area_frac,
                camera_min_visible_keypoints=args.camera_min_visible_keypoints,
                camera_min_visible_frame_ratio=args.camera_min_visible_frame_ratio,
                camera_fit_margin=args.camera_fit_margin,
                camera_mode=args.camera_mode,
                annotations_only=args.annotations_only,
            )
        )


if __name__ == "__main__":
    main()

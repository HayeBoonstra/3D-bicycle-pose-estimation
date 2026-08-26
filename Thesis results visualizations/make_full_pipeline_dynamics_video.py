#!/usr/bin/env python3
"""Greenlight meeting videos: RTMPose 2D only, and 3D skeleton + dynamics (separate MP4s)."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "1_full_detection_pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from data_generation_pipeline_tools.bicycle_dynamics_angles import (  # noqa: E402
    bicycle_roll_angle,
    bicycle_steer_angle,
)
from data_generation_pipeline_tools.visualize_bicycle_pose3d import (  # noqa: E402
    axis_limits_for_poses,
    bicycle_crank_angle,
    reorient_for_display,
    render_frame,
    subtract_root,
)
from evaluation.metrics.dynamics import _steer_sign_to_reference  # noqa: E402
from keypoint_detector_pipeline.io_utils import iter_jsonl  # noqa: E402
from pipeline_io import KEYPOINTS_2D_NAME, KEYPOINTS_3D_NAME  # noqa: E402
from visualize_intermediates import draw_keypoints_2d_frame  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _rows_by_frame_id(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["frame_id"]): row for row in iter_jsonl(path)}


def _gt_poses_from_jsonl(jsonl_path: Path) -> np.ndarray:
    rows = sorted(_load_jsonl(jsonl_path), key=lambda r: int(r["frame_index"]))
    out = np.zeros((len(rows), 18, 3), dtype=np.float32)
    for i, row in enumerate(rows):
        kps = row["kps_camera"]
        pts = np.asarray(kps, dtype=np.float32)
        out[i] = subtract_root(pts[np.newaxis, ...], root_index=0)[0]
    return out


def _dynamics_from_arrays(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    gt_steer_mujoco: np.ndarray | None,
) -> dict[str, np.ndarray]:
    pred_steer = np.rad2deg(bicycle_steer_angle(pred))
    pred_roll = np.rad2deg(bicycle_roll_angle(pred))
    pred_crank = np.rad2deg(bicycle_crank_angle(pred))
    gt_roll_kpt = np.rad2deg(bicycle_roll_angle(gt))
    gt_steer_kpt = np.rad2deg(bicycle_steer_angle(gt))
    gt_crank_kpt = np.rad2deg(bicycle_crank_angle(gt))

    if gt_steer_mujoco is not None and len(gt_steer_mujoco) == len(pred):
        sign = _steer_sign_to_reference(pred_steer, gt_steer_mujoco)
        gt_steer = sign * gt_steer_mujoco
    else:
        gt_steer = gt_steer_kpt

    return {
        "pred_steer": pred_steer,
        "pred_roll": pred_roll,
        "pred_crank": pred_crank,
        "gt_steer": gt_steer,
        "gt_roll": gt_roll_kpt,
        "gt_crank": gt_crank_kpt,
    }


def _mujoco_steer_series(jsonl_path: Path) -> np.ndarray | None:
    rows = sorted(_load_jsonl(jsonl_path), key=lambda r: int(r["frame_index"]))
    steer: list[float] = []
    for row in rows:
        dyn = row.get("dynamics_gt") or {}
        if "steer_deg" in dyn:
            steer.append(float(dyn["steer_deg"]))
    if len(steer) == len(rows):
        return np.asarray(steer, dtype=np.float32)
    return None


def _render_dynamics_panel(
    signals: dict[str, np.ndarray],
    *,
    frame_idx: int,
    width_px: int,
) -> np.ndarray:
    n = len(signals["pred_steer"])
    t = np.arange(n)
    cur = int(np.clip(frame_idx, 0, n - 1))

    fig_w = max(6.0, width_px / 100.0)
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_w * 0.42), sharex=True)
    fig.suptitle("Roll, steer, and crank (pred vs gt)", fontsize=11)

    for ax, key_pred, key_gt, ylab in (
        (axes[0], "pred_steer", "gt_steer", "Steer (deg)"),
        (axes[1], "pred_roll", "gt_roll", "Roll (deg)"),
        (axes[2], "pred_crank", "gt_crank", "Crank (deg, wrapped)"),
    ):
        ax.plot(t, signals[key_pred], color="#1f77b4", linewidth=1.2, label="pred")
        ax.plot(t, signals[key_gt], color="#ff7f0e", linewidth=1.2, label="gt")
        ax.axvline(cur, color="black", linewidth=1.0, alpha=0.5)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=7)

    axes[2].set_xlabel("Frame")
    fig.tight_layout()
    buf = io.BytesIO()
    dpi = max(80, int(round(width_px / fig_w)))
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def _resize_panel(img: np.ndarray, width: int, height: int) -> np.ndarray:
    im = Image.fromarray(img).convert("RGB")
    im = im.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(im)


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    new_h = max(1, int(round(h * scale)))
    return _resize_panel(img, width, new_h)


def _stack_vertical(panels: list[np.ndarray], *, width: int, gap: int = 6) -> np.ndarray:
    fitted = [_fit_width(p, width) for p in panels]
    total_h = sum(p.shape[0] for p in fitted) + gap * (len(fitted) - 1)
    canvas = Image.new("RGB", (width, total_h), (24, 24, 24))
    y = 0
    for panel in fitted:
        canvas.paste(Image.fromarray(panel), (0, y))
        y += panel.shape[0] + gap
    return np.asarray(canvas)


def _write_fixed_size_video(
    out_mp4: Path,
    frames: list[np.ndarray],
    *,
    fps: int,
) -> None:
    writer = imageio.get_writer(out_mp4, fps=fps)
    frame_size: tuple[int, int] | None = None
    for frame in frames:
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])
        elif (frame.shape[1], frame.shape[0]) != frame_size:
            frame = _resize_panel(frame, frame_size[0], frame_size[1])
        writer.append_data(frame)
    writer.close()


def _clip_fps(raw_clip_dir: Path, override: int | None) -> int:
    if override is not None and override > 0:
        return int(override)
    cfg = raw_clip_dir / "render_config.json"
    if cfg.is_file():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        fps = int(data.get("fps", 60))
        if fps > 0:
            return fps
    return 60


def render_greenlight_videos(
    *,
    raw_clip_dir: Path,
    pipeline_dir: Path,
    out_2d_mp4: Path,
    out_3d_dynamics_mp4: Path,
    fps: int | None = None,
    out_width: int = 1280,
    out_3d_mp4: Path | None = None,
    include_dynamics_panel: bool = True,
    show_titles: bool = True,
) -> dict[str, Any]:
    play_fps = _clip_fps(raw_clip_dir, fps)
    keypoints_2d = _rows_by_frame_id(pipeline_dir / KEYPOINTS_2D_NAME)
    npz = np.load(pipeline_dir / KEYPOINTS_3D_NAME)
    pred = np.asarray(npz["pred"], dtype=np.float32)
    frame_ids = np.asarray(npz.get("frame_ids", np.arange(len(pred))), dtype=np.int32)

    gt_jsonl = raw_clip_dir / "keypoints_3d.jsonl"
    gt = _gt_poses_from_jsonl(gt_jsonl)
    if pred.shape[0] != gt.shape[0]:
        n = min(pred.shape[0], gt.shape[0])
        pred, gt = pred[:n], gt[:n]
        frame_ids = frame_ids[:n]

    gt_steer_mj = _mujoco_steer_series(gt_jsonl)
    signals = _dynamics_from_arrays(pred, gt, gt_steer_mujoco=gt_steer_mj)

    pred_v = reorient_for_display(pred, "camera_up")
    gt_v = reorient_for_display(gt, "camera_up")
    lo, hi = axis_limits_for_poses(pred_v, gt_v)

    frames_2d: list[np.ndarray] = []
    frames_3d_dyn: list[np.ndarray] = []
    frames_3d: list[np.ndarray] = []

    for t_idx in tqdm(range(len(pred)), desc="greenlight videos"):
        fid = int(frame_ids[t_idx])
        kp2d_row = keypoints_2d.get(fid)
        if kp2d_row is None:
            raise KeyError(f"Missing pipeline rows for frame_id={fid}")

        row_2d = np.asarray(draw_keypoints_2d_frame(kp2d_row))
        frames_2d.append(_fit_width(row_2d, out_width))

        row_3d = render_frame(
            pred_v[t_idx],
            gt_v[t_idx],
            layout="overlay",
            lo=lo,
            hi=hi,
            elev=20.0,
            azim=-70.0,
            invert_z=True,
            title="3D keypoints (pred vs gt)" if show_titles else None,
            metrics_text=None,
            show_titles=show_titles,
        )
        row_3d_fit = _fit_width(row_3d, out_width)
        frames_3d.append(row_3d_fit)
        if include_dynamics_panel:
            row_dyn = _render_dynamics_panel(signals, frame_idx=t_idx, width_px=out_width)
            frames_3d_dyn.append(_stack_vertical([row_3d_fit, row_dyn], width=out_width))
        else:
            frames_3d_dyn.append(row_3d_fit)

    out_2d_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_3d_dynamics_mp4.parent.mkdir(parents=True, exist_ok=True)
    _write_fixed_size_video(out_2d_mp4, frames_2d, fps=play_fps)
    _write_fixed_size_video(out_3d_dynamics_mp4, frames_3d_dyn, fps=play_fps)
    if out_3d_mp4 is not None:
        out_3d_mp4.parent.mkdir(parents=True, exist_ok=True)
        _write_fixed_size_video(out_3d_mp4, frames_3d, fps=play_fps)

    return {
        "video_2d": str(out_2d_mp4),
        "video_3d_dynamics": str(out_3d_dynamics_mp4),
        "video_3d": str(out_3d_mp4) if out_3d_mp4 is not None else None,
        "num_frames": int(len(pred)),
        "fps": play_fps,
        "out_width_px": out_width,
        "include_dynamics_panel": include_dynamics_panel,
        "show_titles": show_titles,
        "raw_clip": str(raw_clip_dir),
        "pipeline_dir": str(pipeline_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-clip-dir", type=Path, required=True)
    p.add_argument("--pipeline-dir", type=Path, required=True)
    p.add_argument(
        "--out-2d-mp4",
        type=Path,
        required=True,
        help="RTMPose 2D overlay video.",
    )
    p.add_argument(
        "--out-3d-dynamics-mp4",
        type=Path,
        required=True,
        help="PoseMamba 3D overlay + dynamics plots (stacked).",
    )
    p.add_argument(
        "--trajectory-csv",
        type=Path,
        default=None,
        help="Unused; kept for run_greenlight_pipeline_viz.sh compatibility.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback FPS (0 = read from raw clip render_config.json).",
    )
    p.add_argument(
        "--out-width",
        type=int,
        default=1280,
        help="Output width in pixels.",
    )
    p.add_argument(
        "--out-3d-mp4",
        type=Path,
        default=None,
        help="Optional 3D overlay-only MP4 (same frames as the top panel).",
    )
    p.add_argument(
        "--no-dynamics-panel",
        action="store_true",
        help="Omit roll/steer/crank plots from --out-3d-dynamics-mp4.",
    )
    p.add_argument(
        "--no-titles",
        action="store_true",
        help="Hide figure titles and axis labels on 3D panels.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.trajectory_csv is not None:
        pass
    meta = render_greenlight_videos(
        raw_clip_dir=args.raw_clip_dir.resolve(),
        pipeline_dir=args.pipeline_dir.resolve(),
        out_2d_mp4=args.out_2d_mp4.resolve(),
        out_3d_dynamics_mp4=args.out_3d_dynamics_mp4.resolve(),
        fps=args.fps if args.fps > 0 else None,
        out_width=args.out_width,
        out_3d_mp4=args.out_3d_mp4.resolve() if args.out_3d_mp4 is not None else None,
        include_dynamics_panel=not args.no_dynamics_panel,
        show_titles=not args.no_titles,
    )
    manifest_path = args.out_2d_mp4.resolve().parent.parent / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing["greenlight_videos"] = meta
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[make_full_pipeline_dynamics_video] wrote {args.out_2d_mp4}")
    print(f"[make_full_pipeline_dynamics_video] wrote {args.out_3d_dynamics_mp4} ({meta['num_frames']} frames)")
    if args.out_3d_mp4 is not None:
        print(f"[make_full_pipeline_dynamics_video] wrote {args.out_3d_mp4}")


if __name__ == "__main__":
    main()

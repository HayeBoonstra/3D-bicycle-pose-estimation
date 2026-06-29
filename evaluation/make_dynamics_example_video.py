#!/usr/bin/env python3
"""Render an example 3D pose video for the dynamics time-series example clip.

Produces a composite MP4 aligned with ``figures/dynamics_timeseries.png``:
  - top: pred vs GT skeleton (overlay)
  - bottom: steer vs MuJoCo GT, roll vs kinematic GT (same frame as pred)
"""

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
from tqdm import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_dynamics_angles import (  # noqa: E402
    bicycle_roll_angle,
    bicycle_steer_angle,
)
from data_generation_pipeline_tools.visualize_bicycle_pose3d import (  # noqa: E402
    axis_limits_for_poses,
    motion_to_images_or_video,
    reorient_for_display,
    render_frame,
)
from evaluation.common import ensure_dir  # noqa: E402
from evaluation.metrics.dynamics import _steer_sign_to_reference  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _example_clip_id(exp_dir: Path) -> str | None:
    metrics = _load_json(exp_dir / "metrics.json")
    ts = metrics.get("dynamics", {}).get("time_series", {})
    cid = ts.get("clip_id")
    if cid:
        return str(cid)
    accepted = metrics.get("pose3d", {}).get("clip_filter", {}).get("accepted_clip_ids", [])
    if accepted:
        return str(accepted[0])
    return None


def _clip_arrays(npz_path: Path, clip_id: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    clip_ids = [str(c) for c in data["clip_ids"]]
    mask = np.array([c == clip_id for c in clip_ids])
    if not np.any(mask):
        raise ValueError(f"clip_id not found in {npz_path}: {clip_id}")
    pred = np.asarray(data["pred"][mask], dtype=np.float32)
    gt = np.asarray(data["gt"][mask], dtype=np.float32)
    out: dict[str, np.ndarray] = {"pred": pred, "gt": gt}
    if "steer_deg" in data:
        out["steer_deg"] = np.asarray(data["steer_deg"][mask], dtype=np.float32)
    if "roll_deg" in data:
        out["roll_deg"] = np.asarray(data["roll_deg"][mask], dtype=np.float32)
    return out


def _dynamics_signals(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pred = arrays["pred"]
    gt = arrays["gt"]
    pred_steer = np.rad2deg(bicycle_steer_angle(pred))
    pred_roll = np.rad2deg(bicycle_roll_angle(pred))
    gt_roll_kpt = np.rad2deg(bicycle_roll_angle(gt))
    if "steer_deg" in arrays:
        sign = _steer_sign_to_reference(pred_steer, arrays["steer_deg"])
        gt_steer_mujoco = sign * arrays["steer_deg"]
    else:
        gt_steer_mujoco = np.rad2deg(bicycle_steer_angle(gt))
    return {
        "pred_steer": pred_steer,
        "pred_roll": pred_roll,
        "gt_steer_mujoco": gt_steer_mujoco,
        "gt_roll_kpt": gt_roll_kpt,
    }


def _render_dynamics_panel(
    signals: dict[str, np.ndarray],
    *,
    frame_idx: int,
    clip_id: str,
) -> np.ndarray:
    n = len(signals["pred_steer"])
    t = np.arange(n)
    cur = int(np.clip(frame_idx, 0, n - 1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 3.2), sharex=True)
    short = clip_id.replace("clip_", "", 1)
    if len(short) > 48:
        short = short[:45] + "..."

    axes[0].plot(t, signals["pred_steer"], color="#1f77b4", linewidth=1.2, label="pred")
    axes[0].plot(t, signals["gt_steer_mujoco"], color="#ff7f0e", linewidth=1.2, label="gt MuJoCo")
    axes[0].axvline(cur, color="black", linewidth=1.0, alpha=0.5)
    axes[0].set_ylabel("Steer (deg)")
    axes[0].set_title(f"Dynamics example — {short}")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, signals["pred_roll"], color="#1f77b4", linewidth=1.2, label="pred")
    axes[1].plot(t, signals["gt_roll_kpt"], color="#ff7f0e", linewidth=1.2, label="gt")
    axes[1].axvline(cur, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set_ylabel("Roll (deg)")
    axes[1].set_xlabel("Frame")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def _stack_vertical(top: np.ndarray, bottom: np.ndarray, *, target_width: int | None = None) -> np.ndarray:
    """Resize bottom to match top width, then stack. Optionally pad to fixed width."""
    from PIL import Image

    top_im = Image.fromarray(top)
    bot_im = Image.fromarray(bottom)
    width = target_width or top_im.width
    if top_im.width != width:
        scale = width / top_im.width
        top_im = top_im.resize((width, max(1, int(top_im.height * scale))), Image.Resampling.LANCZOS)
    scale = width / bot_im.width
    new_h = max(1, int(bot_im.height * scale))
    bot_im = bot_im.resize((width, new_h), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (width, top_im.height + new_h))
    out.paste(top_im, (0, 0))
    out.paste(bot_im, (0, top_im.height))
    return np.asarray(out)


def render_composite_video(
    arrays: dict[str, np.ndarray],
    *,
    clip_id: str,
    out_mp4: Path,
    fps: int = 12,
    elev: float = 20.0,
    azim: float = -70.0,
) -> dict[str, Any]:
    pred = arrays["pred"]
    gt = arrays["gt"]
    signals = _dynamics_signals(arrays)

    pred_v = reorient_for_display(pred, "camera_up")
    gt_v = reorient_for_display(gt, "camera_up")
    lo, hi = axis_limits_for_poses(pred_v, gt_v)

    mpjpe_mm = float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)
    title = f"{clip_id}  |  MPJPE {mpjpe_mm:.1f} mm"

    frames_dir = ensure_dir(out_mp4.parent / f"{out_mp4.stem}_frames")
    writer = imageio.get_writer(out_mp4, fps=fps)
    frame_size: tuple[int, int] | None = None

    for t in tqdm(range(pred.shape[0]), desc="composite video"):
        metrics_text = [
            f"frame={t}",
            f"steer(pred)={signals['pred_steer'][t]:+.1f}°",
            f"steer(MJ)={signals['gt_steer_mujoco'][t]:+.1f}°",
            f"roll(pred)={signals['pred_roll'][t]:+.1f}°",
            f"roll(gt)={signals['gt_roll_kpt'][t]:+.1f}°",
        ]
        pose_rgb = render_frame(
            pred_v[t],
            gt_v[t],
            layout="overlay",
            lo=lo,
            hi=hi,
            elev=elev,
            azim=azim,
            invert_z=True,
            title=title,
            metrics_text=metrics_text,
        )
        dyn_rgb = _render_dynamics_panel(signals, frame_idx=t, clip_id=clip_id)
        composite = _stack_vertical(pose_rgb, dyn_rgb)
        if frame_size is None:
            frame_size = (composite.shape[1], composite.shape[0])
        elif (composite.shape[1], composite.shape[0]) != frame_size:
            from PIL import Image

            composite = np.asarray(
                Image.fromarray(composite).resize(frame_size, Image.Resampling.LANCZOS)
            )
        frame_path = frames_dir / f"frame_{t:04d}.png"
        imageio.imwrite(frame_path, composite)
        writer.append_data(composite)

    writer.close()
    return {
        "video": str(out_mp4),
        "frames_dir": str(frames_dir),
        "clip_id": clip_id,
        "num_frames": int(pred.shape[0]),
        "mpjpe_mm": mpjpe_mm,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render dynamics example clip video.")
    p.add_argument(
        "--exp-dir",
        type=Path,
        default=None,
        help="Experiment results dir (default: results/<experiment>).",
    )
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--experiment", type=str, default="1_full_detection_pipeline")
    p.add_argument("--clip-id", type=str, default=None, help="Override example clip (default: from metrics.json).")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument(
        "--also-pose-only",
        action="store_true",
        help="Also write a skeleton-only MP4 (no dynamics strip).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = (args.exp_dir or args.results_dir / args.experiment).resolve()
    npz_path = exp_dir / "preds_3d.npz"
    if not npz_path.is_file():
        raise SystemExit(f"Missing {npz_path}; run evaluation/extract.py first.")

    clip_id = args.clip_id or _example_clip_id(exp_dir)
    if not clip_id:
        raise SystemExit(f"Could not resolve example clip_id from {exp_dir / 'metrics.json'}")

    arrays = _clip_arrays(npz_path, clip_id)
    fig_dir = ensure_dir(exp_dir / "figures")
    out_mp4 = fig_dir / "dynamics_example_clip.mp4"

    meta = render_composite_video(arrays, clip_id=clip_id, out_mp4=out_mp4, fps=args.fps)
    (fig_dir / "dynamics_example_clip.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[make_dynamics_example_video] wrote {out_mp4} ({meta['num_frames']} frames, {clip_id})")

    if args.also_pose_only:
        pose_dir = ensure_dir(fig_dir / "dynamics_example_pose_only")
        pose_meta = motion_to_images_or_video(
            arrays["pred"],
            arrays["gt"],
            pose_dir,
            layout="split",
            write_video=True,
            fps=args.fps,
            elev=20.0,
            azim=-70.0,
            reorient="camera_up",
            title=f"{clip_id}  |  MPJPE {meta['mpjpe_mm']:.1f} mm",
            write_dynamics_plots=True,
        )
        print(f"[make_dynamics_example_video] pose-only video: {pose_meta.get('video')}")


if __name__ == "__main__":
    main()

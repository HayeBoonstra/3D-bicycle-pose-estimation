"""Render bicycle 3D poses (PoseMamba-style) to PNG frames or MP4.

Designed for root-relative 3D in the same joint order as training
(`k_bottom_bracket` = index 0), analogous to ``PoseMamba/vis.py`` and
``PoseMamba/tools/vis_h36m.py`` for Human3.6M.

Supported inputs
----------------
* **NumPy** ``.npy``: array shaped ``(T, J, 3)``.
* **NumPy** ``.npz``: first matching key among ``pred``, ``prediction``, ``poses``,
  ``poses_3d``, ``kpts3d_cam``, ``gt``, ``data_label``, or an explicit ``--npz-key``.
  If the array is 4D ``(N, T, J, 3)``, pass ``--sequence-index``.
* **Pickle** (PoseMamba training sample): dict with ``pred`` / ``prediction`` /
  ``poses_3d`` for predictions, or ``data_label`` / ``gt`` for ground truth in
  camera coordinates. Camera GT is converted to root-relative by default so it
  matches lifted model outputs.

Examples
--------
Compare model ``pred.npz`` to a row from ``sequences_*.npz`` (camera GT, root-subtracted by default)::

    python data_generation_pipeline_tools/visualize_bicycle_pose3d.py \\
        --pred pred.npz --gt data/posemamba_sequences/out/sequences_val.npz \\
        --sequence-index 0 --gt-npz-key kpts3d_cam --out /tmp/bike_vis --video

GT-only from a training ``.pkl`` (root-relative after subtraction)::

    python data_generation_pipeline_tools/visualize_bicycle_pose3d.py \\
        --pred path/to/clip_000000.pkl --pred-pkl-key data_label --subtract-root-pred \\
        --out /tmp/gt_only --video
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
from tqdm import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import (  # noqa: E402
    BICYCLE_KEYPOINT_NAMES,
    KEYPOINT_INDEX,
    BICYCLE_SKELETON_NAMES,
)

_J = len(BICYCLE_KEYPOINT_NAMES)
_NPZ_KEYS = ("pred", "prediction", "poses", "poses_3d", "kpts3d_cam", "gt", "data_label")


def skeleton_edge_indices() -> list[tuple[int, int]]:
    return [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]) for a, b in BICYCLE_SKELETON_NAMES]


def subtract_root(motion: np.ndarray, root_index: int = 0) -> np.ndarray:
    """Root-relative: subtract joint ``root_index`` at each frame."""
    if motion.ndim != 3:
        raise ValueError(f"Expected motion (T, J, 3), got shape {motion.shape}")
    return (motion - motion[:, root_index : root_index + 1, :]).astype(np.float32)


def reorient_for_display(motion: np.ndarray, mode: str) -> np.ndarray:
    """Apply a fixed axis remap for matplotlib viewing (does not change stored metrics)."""
    if mode in ("none", ""):
        return motion
    if mode == "camera_up":
        # Camera: X right, Y down (OpenCV) or Y up depending on export; Z forward.
        # Matplotlib 3D screen-up is plot-Z. Training pickles use Y-up in practice, so plot-Z = Y.
        out = motion.astype(np.float32, copy=True)
        x, y, z = out[..., 0].copy(), out[..., 1].copy(), out[..., 2].copy()
        out[..., 0] = x
        out[..., 1] = z
        out[..., 2] = y
        return out
    raise ValueError(f"Unknown reorient mode: {mode!r}")


def _as_time_joint_xyz(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        raise ValueError("Array is 4D; select a sequence with --sequence-index on the loader.")
    if arr.ndim != 3:
        raise ValueError(f"Expected a (T, J, 3) array, got shape {arr.shape}")
    if arr.shape[1] != _J or arr.shape[2] != 3:
        raise ValueError(f"Expected J={_J} joints and 3 coords, got shape {arr.shape}")
    return arr.astype(np.float32)


def _select_npz_array(data: Any, key: str | None) -> np.ndarray:
    if key is not None:
        if key not in data:
            raise KeyError(f"--npz-key {key!r} not in file; keys: {sorted(data.files)}")
        return np.asarray(data[key])
    for candidate in _NPZ_KEYS:
        if candidate in data:
            return np.asarray(data[candidate])
    raise KeyError(f"No known pose key in .npz (tried {_NPZ_KEYS}); keys: {sorted(data.files)}")


def load_motion(
    path: Path,
    *,
    npz_key: str | None = None,
    sequence_index: int | None = None,
    pkl_key: str | None = None,
) -> np.ndarray:
    """Load a single motion as ``(T, J, 3)`` float32."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return _as_time_joint_xyz(np.asarray(arr))

    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            arr = _select_npz_array(data, npz_key)
        if arr.ndim == 4:
            if sequence_index is None:
                raise ValueError(f"Array in {path} has shape {arr.shape}; pass --sequence-index.")
            if not (0 <= sequence_index < arr.shape[0]):
                raise IndexError(f"sequence_index {sequence_index} out of range for shape {arr.shape}")
            arr = arr[sequence_index]
        return _as_time_joint_xyz(np.asarray(arr))

    if suffix == ".pkl":
        with path.open("rb") as f:
            obj: Any = pickle.load(f)
        if not isinstance(obj, dict):
            raise TypeError(f"Expected pickle dict in {path}, got {type(obj)}")
        key_order: tuple[str, ...]
        if pkl_key is not None:
            key_order = (pkl_key,)
        else:
            key_order = ("pred", "prediction", "poses_3d", "poses", "data_label", "gt", "kpts3d_cam")
        chosen = None
        for k in key_order:
            if k in obj and obj[k] is not None:
                chosen = k
                break
        if chosen is None:
            raise KeyError(f"No pose array found in pickle {path}; keys: {sorted(obj.keys())}")
        arr = np.asarray(obj[chosen], dtype=np.float32)
        return _as_time_joint_xyz(arr)

    raise ValueError(f"Unsupported file type: {path} (use .npy, .npz, or .pkl)")


def axis_limits_for_poses(*poses: np.ndarray, padding_ratio: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([p.reshape(-1, 3) for p in poses if p.size], axis=0).reshape(-1, 3)
    lo = np.min(stacked, axis=0)
    hi = np.max(stacked, axis=0)
    span = np.maximum(hi - lo, 1e-4)
    pad = span * padding_ratio
    return lo - pad, hi + pad


def _style_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.xaxis.pane.fill = False  # type: ignore[attr-defined]
    ax.yaxis.pane.fill = False  # type: ignore[attr-defined]
    ax.zaxis.pane.fill = False  # type: ignore[attr-defined]


def _finalize_3d_axes(
    ax: plt.Axes,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    elev: float,
    azim: float,
    invert_z: bool,
) -> None:
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=elev, azim=azim)
    if invert_z:
        ax.invert_zaxis()
    _style_axes(ax)


def draw_skeleton(
    ax: plt.Axes,
    joints: np.ndarray,
    *,
    edgecolor: str,
    pointcolor: str | None = None,
    linewidth: float = 2.0,
    linestyle: str = "-",
) -> None:
    """Plot one frame ``joints`` shaped ``(J, 3)``."""

    for i, j in skeleton_edge_indices():
        seg = np.stack([joints[i], joints[j]], axis=0)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=edgecolor, lw=linewidth, ls=linestyle)
    if pointcolor is not None:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=pointcolor, s=18, depthshade=True)


def render_frame(
    pred: np.ndarray,
    gt: np.ndarray | None,
    *,
    layout: str,
    lo: np.ndarray,
    hi: np.ndarray,
    elev: float,
    azim: float,
    invert_z: bool = True,
    title: str | None = None,
) -> np.ndarray:
    """Return an RGB uint8 image for one time step."""

    if layout not in {"split", "overlay"}:
        raise ValueError("layout must be 'split' or 'overlay'")

    if layout == "split" and gt is not None:
        fig = plt.figure(figsize=(12.8, 5.4), dpi=120)
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        for ax, panel_title, pose, ec in (
            (ax0, "Prediction", pred, "#2F70AF"),
            (ax1, "Ground truth", gt, "#666666"),
        ):
            draw_skeleton(ax, pose, edgecolor=ec, pointcolor=ec)
            ax.set_title(panel_title, fontsize=12)
            _finalize_3d_axes(ax, lo, hi, elev=elev, azim=azim, invert_z=invert_z)
    else:
        fig = plt.figure(figsize=(7.2, 6.4), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        if gt is not None:
            draw_skeleton(ax, gt, edgecolor="#999999", linewidth=1.8, linestyle="--")
        draw_skeleton(ax, pred, edgecolor="#00457E", linewidth=2.4)
        ax.set_title("Prediction vs ground truth" if gt is not None else "Prediction", fontsize=12)
        _finalize_3d_axes(ax, lo, hi, elev=elev, azim=azim, invert_z=invert_z)

    if title:
        fig.suptitle(title, fontsize=11, y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def motion_to_images_or_video(
    pred: np.ndarray,
    gt: np.ndarray | None,
    out_dir: Path,
    *,
    layout: str,
    write_video: bool,
    fps: int,
    elev: float,
    azim: float,
    reorient: str = "none",
    invert_z: bool = True,
    title: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if gt is not None and gt.shape != pred.shape:
        raise ValueError(f"pred shape {pred.shape} does not match gt {gt.shape}")

    pred_v = reorient_for_display(pred, reorient)
    gt_v = reorient_for_display(gt, reorient) if gt is not None else None

    if gt_v is not None:
        lo, hi = axis_limits_for_poses(pred_v, gt_v)
    else:
        lo, hi = axis_limits_for_poses(pred_v)

    meta: dict[str, Any] = {
        "out_dir": str(out_dir),
        "frames": int(pred.shape[0]),
        "layout": layout,
        "has_gt": gt is not None,
    }

    if gt is not None:
        dif = pred - gt
        per_joint = np.mean(np.linalg.norm(dif, axis=-1), axis=0)
        meta["mpjpe_mean_m"] = float(np.mean(np.linalg.norm(dif, axis=-1)))
        meta["mpjpe_per_joint_m"] = {
            name: float(per_joint[i]) for i, name in enumerate(BICYCLE_KEYPOINT_NAMES)
        }
        meta["mpjpe_mean_mm"] = meta["mpjpe_mean_m"] * 1000.0

    for t in tqdm(range(pred.shape[0]), desc="frames"):
        rgb = render_frame(
            pred_v[t],
            gt_v[t] if gt_v is not None else None,
            layout=layout,
            lo=lo,
            hi=hi,
            elev=elev,
            azim=azim,
            invert_z=invert_z,
            title=title,
        )
        imageio.imwrite(out_dir / f"frame_{t:04d}.png", rgb)

    if write_video:
        video_path = out_dir.with_suffix(".mp4")
        writer = imageio.get_writer(video_path, fps=fps)
        for t in range(pred.shape[0]):
            writer.append_data(imageio.imread(out_dir / f"frame_{t:04d}.png"))
        writer.close()
        meta["video"] = str(video_path)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["summary_json"] = str(summary_path)
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize bicycle 3D poses (root-relative or matched GT).")
    p.add_argument("--pred", type=Path, required=True, help="Path to .npy, .npz, or .pkl containing a (T,J,3) sequence.")
    p.add_argument("--gt", type=Path, help="Optional second path for ground truth.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for frame PNGs.")
    p.add_argument("--layout", choices=("split", "overlay"), default="split", help="split: two panels; overlay: same 3D axes.")
    p.add_argument("--video", action="store_true", help="Also write out.mp4 next to the frame directory (basename).")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=-70.0)
    p.add_argument(
        "--reorient",
        choices=("none", "camera_up"),
        default="camera_up",
        help="Display-only axis remap so the bicycle appears upright (metrics use raw arrays).",
    )
    p.add_argument("--title", type=str, default=None, help="Optional figure title (e.g. MPJPE summary).")
    p.add_argument("--root-index", type=int, default=0, help="Root joint index (default: k_bottom_bracket).")
    p.set_defaults(subtract_root_pred=False, subtract_root_gt=True)
    p.add_argument(
        "--subtract-root-pred",
        dest="subtract_root_pred",
        action="store_true",
        help="Subtract root from --pred (use when pred is absolute camera 3D, e.g. kpts3d_cam).",
    )
    p.add_argument(
        "--no-subtract-root-gt",
        dest="subtract_root_gt",
        action="store_false",
        help="Do not subtract root from --gt (use when GT is already root-relative).",
    )
    p.add_argument("--sequence-index", type=int, help="When loading a batched .npz of shape (N,T,J,3), pick sample index N.")
    p.add_argument("--npz-key", type=str, help="Which array to read from an .npz (otherwise first known key).")
    p.add_argument("--pred-npz-key", type=str, help="Override --npz-key for --pred only.")
    p.add_argument("--gt-npz-key", type=str, help="Override --npz-key for --gt only.")
    p.add_argument("--pred-pkl-key", type=str, help="Which pickle key to read for --pred.")
    p.add_argument("--gt-pkl-key", type=str, help="Which pickle key to read for --gt.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred = load_motion(
        args.pred,
        npz_key=args.pred_npz_key or args.npz_key,
        sequence_index=args.sequence_index,
        pkl_key=args.pred_pkl_key,
    )
    if args.subtract_root_pred:
        pred = subtract_root(pred, args.root_index)

    gt = None
    if args.gt is not None:
        gt = load_motion(
            args.gt,
            npz_key=args.gt_npz_key or args.npz_key,
            sequence_index=args.sequence_index,
            pkl_key=args.gt_pkl_key,
        )
        if args.subtract_root_gt:
            gt = subtract_root(gt, args.root_index)

    meta = motion_to_images_or_video(
        pred,
        gt,
        args.out,
        layout=args.layout,
        write_video=args.video,
        fps=args.fps,
        elev=args.elev,
        azim=args.azim,
        reorient=args.reorient,
        title=args.title,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

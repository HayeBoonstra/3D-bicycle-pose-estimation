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
    PART_COLORS_GT,
    PART_COLORS_PRED,
    skeleton_edges_by_part,
    validate_skeleton_part_groups,
)

validate_skeleton_part_groups()

_J = len(BICYCLE_KEYPOINT_NAMES)
_NPZ_KEYS = ("pred", "prediction", "poses", "poses_3d", "kpts3d_cam", "gt", "data_label")
_DYN_EPS = 1e-6

_KP_BB = KEYPOINT_INDEX["k_bottom_bracket"]
_KP_SEAT_STAY = KEYPOINT_INDEX["k_seat_stay"]
_KP_SADDLE = KEYPOINT_INDEX["k_saddle"]
_KP_UHT = KEYPOINT_INDEX["k_upper_head_tube"]
_KP_LHT = KEYPOINT_INDEX["k_lower_head_tube"]
_KP_HB_MID = KEYPOINT_INDEX["k_handlebar_middle"]
_KP_FH_L = KEYPOINT_INDEX["k_front_hub_left"]
_KP_FH_R = KEYPOINT_INDEX["k_front_hub_right"]
_KP_RH_L = KEYPOINT_INDEX["k_rear_hub_left"]
_KP_RH_R = KEYPOINT_INDEX["k_rear_hub_right"]
_KP_RW_GND = KEYPOINT_INDEX["k_rear_wheel_ground"]
_KP_LP = KEYPOINT_INDEX["k_left_pedal"]
_SAGITTAL_PLANE_IDS = [_KP_BB, _KP_SEAT_STAY, _KP_SADDLE, _KP_UHT, _KP_LHT, _KP_HB_MID, _KP_RW_GND]

from data_generation_pipeline_tools.bicycle_dynamics_angles import (  # noqa: E402
    bicycle_roll_angle,
    bicycle_steer_angle,
)


def skeleton_edge_indices() -> list[tuple[int, int]]:
    return [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]) for a, b in BICYCLE_SKELETON_NAMES]


def _safe_normalize(v: np.ndarray, eps: float = _DYN_EPS) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def _signed_oriented(vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    dot = np.sum(vec * ref, axis=-1, keepdims=True)
    sign = np.where(dot >= 0.0, 1.0, -1.0)
    return vec * sign


def _sagittal_plane_normal(kpts: np.ndarray) -> np.ndarray:
    pts = kpts[:, _SAGITTAL_PLANE_IDS, :]  # (T, K, 3)
    centered = pts - pts.mean(axis=1, keepdims=True)
    cov = np.matmul(np.transpose(centered, (0, 2, 1)), centered)  # (T, 3, 3)
    eigvals, eigvecs = np.linalg.eigh(cov)
    min_idx = np.argmin(eigvals, axis=-1)
    n = eigvecs[np.arange(eigvecs.shape[0]), :, min_idx]  # (T, 3)
    ref = kpts[:, _KP_RH_R, :] - kpts[:, _KP_RH_L, :]
    return _safe_normalize(_signed_oriented(n, ref))


def _project_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - np.sum(v * n, axis=-1, keepdims=True) * n


def bicycle_crank_angle(kpts: np.ndarray) -> np.ndarray:
    """Signed crank angle (rad) from the left pedal arm in the sagittal plane.

    Uses the bottom-bracket-to-left-pedal vector projected into the crank plane,
    measured against the head-tube forward direction and crank axis
    (``cross(crank_axis, forward)``). Matches the single crank DOF in MuJoCo.
    """
    crank_axis = _sagittal_plane_normal(kpts)
    bb = kpts[:, _KP_BB, :]
    forward = _safe_normalize(_project_to_plane(kpts[:, _KP_LHT, :] - bb, crank_axis))
    in_plane_y = _safe_normalize(np.cross(crank_axis, forward))
    arm = _project_to_plane(kpts[:, _KP_LP, :] - bb, crank_axis)
    along_forward = np.sum(arm * forward, axis=-1)
    along_in_plane_y = np.sum(arm * in_plane_y, axis=-1)
    return _wrap_pi(np.arctan2(-along_forward, along_in_plane_y))


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)


def _angle_velocity(angle: np.ndarray) -> np.ndarray:
    if angle.shape[0] <= 1:
        return np.zeros((0,), dtype=np.float32)
    return _wrap_pi(angle[1:] - angle[:-1]).astype(np.float32)


def subtract_root(motion: np.ndarray, root_index: int = 0) -> np.ndarray:
    """Root-relative: subtract joint ``root_index`` at each frame."""
    if motion.ndim != 3:
        raise ValueError(f"Expected motion (T, J, 3), got shape {motion.shape}")
    return (motion - motion[:, root_index : root_index + 1, :]).astype(np.float32)


def reorient_for_display(motion: np.ndarray, mode: str) -> np.ndarray:
    """Apply a fixed axis remap for matplotlib viewing (does not change stored metrics)."""
    if mode in ("none", ""):
        return motion
    if mode == "camera_view":
        # OpenCV root-relative camera coords (X right, Y down, Z forward).
        # Map to a display frame where the plot looks along the optical axis:
        #   plot-X = image horizontal, plot-Y = -depth (near=+Y), plot-Z = image up.
        out = motion.astype(np.float32, copy=True)
        x, y, z = out[..., 0].copy(), out[..., 1].copy(), out[..., 2].copy()
        out[..., 0] = x
        out[..., 1] = -z
        out[..., 2] = -y
        return out
    if mode == "camera_up":
        # Legacy remap (plot-Z = camera Y, not image-up). Prefer camera_view for composites.
        out = motion.astype(np.float32, copy=True)
        x, y, z = out[..., 0].copy(), out[..., 1].copy(), out[..., 2].copy()
        out[..., 0] = x
        out[..., 1] = z
        out[..., 2] = y
        return out
    raise ValueError(f"Unknown reorient mode: {mode!r}")


def view_angles_for_reorient(mode: str) -> tuple[float, float]:
    """Default matplotlib elev/azim for a reorient mode."""
    if mode == "camera_view":
        # Look along +plot-Y (optical axis); plot-Z is image-up on screen.
        return 0.0, -90.0
    return 20.0, -70.0


def axis_limits_cube(*poses: np.ndarray, padding_ratio: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Equal-span axis limits so matplotlib 3D skeletons are not stretched."""
    lo, hi = axis_limits_for_poses(*poses, padding_ratio=padding_ratio)
    center = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo))
    return center - half, center + half


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


def write_dynamics_angle_plots(
    out_dir: Path,
    *,
    steer_pred_deg: np.ndarray,
    roll_pred_deg: np.ndarray,
    crank_pred_deg: np.ndarray,
    steer_gt_deg: np.ndarray | None = None,
    roll_gt_deg: np.ndarray | None = None,
    crank_gt_deg: np.ndarray | None = None,
) -> dict[str, str]:
    """Write steer/roll/crank angle time-series plots (pred vs optional GT)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(len(steer_pred_deg), dtype=np.int32)
    paths: dict[str, str] = {}
    has_gt = steer_gt_deg is not None and roll_gt_deg is not None and crank_gt_deg is not None

    fig, (ax_steer_roll, ax_crank) = plt.subplots(2, 1, figsize=(10.0, 7.5), dpi=120, sharex=True)
    ax_steer_roll.plot(frames, steer_pred_deg, color="#1f77b4", linewidth=1.8, label="steer pred (deg)")
    ax_steer_roll.plot(frames, roll_pred_deg, color="#ff7f0e", linewidth=1.8, label="roll pred (deg)")
    if has_gt:
        ax_steer_roll.plot(
            frames, steer_gt_deg, color="#1f77b4", linewidth=1.4, linestyle="--", label="steer gt (deg)"
        )
        ax_steer_roll.plot(
            frames, roll_gt_deg, color="#ff7f0e", linewidth=1.4, linestyle="--", label="roll gt (deg)"
        )
    ax_steer_roll.set_ylabel("Angle (deg)")
    ax_steer_roll.set_title("Steer / Roll vs Frame")
    ax_steer_roll.grid(True, alpha=0.3)
    ax_steer_roll.legend(loc="best", fontsize=9)

    ax_crank.plot(frames, crank_pred_deg, color="#9467bd", linewidth=1.8, label="crank pred (deg)")
    if has_gt:
        ax_crank.plot(
            frames, crank_gt_deg, color="#9467bd", linewidth=1.4, linestyle="--", label="crank gt (deg)"
        )
    ax_crank.set_xlabel("Frame")
    ax_crank.set_ylabel("Angle (deg)")
    ax_crank.set_title("Crank Angle vs Frame")
    ax_crank.grid(True, alpha=0.3)
    ax_crank.legend(loc="best", fontsize=9)
    fig.tight_layout()
    angles_path = out_dir / "dynamics_angles_plot.png"
    fig.savefig(angles_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["dynamics_angles_plot"] = str(angles_path)

    if has_gt:
        steer_err_deg = np.rad2deg(np.abs(_wrap_pi(np.deg2rad(steer_pred_deg - steer_gt_deg))))
        roll_err_deg = np.rad2deg(np.abs(_wrap_pi(np.deg2rad(roll_pred_deg - roll_gt_deg))))
        crank_err_deg = np.rad2deg(np.abs(_wrap_pi(np.deg2rad(crank_pred_deg - crank_gt_deg))))

        fig, (ax_steer_roll_err, ax_crank_err) = plt.subplots(2, 1, figsize=(10.0, 7.5), dpi=120, sharex=True)
        ax_steer_roll_err.plot(frames, steer_err_deg, color="#2ca02c", linewidth=1.8, label="|steer error| (deg)")
        ax_steer_roll_err.plot(frames, roll_err_deg, color="#d62728", linewidth=1.8, label="|roll error| (deg)")
        ax_steer_roll_err.set_ylabel("Absolute error (deg)")
        ax_steer_roll_err.set_title("Steer / Roll Absolute Errors vs Frame")
        ax_steer_roll_err.grid(True, alpha=0.3)
        ax_steer_roll_err.legend(loc="best", fontsize=9)

        ax_crank_err.plot(frames, crank_err_deg, color="#9467bd", linewidth=1.8, label="|crank error| (deg)")
        ax_crank_err.set_xlabel("Frame")
        ax_crank_err.set_ylabel("Absolute error (deg)")
        ax_crank_err.set_title("Crank Absolute Error vs Frame")
        ax_crank_err.grid(True, alpha=0.3)
        ax_crank_err.legend(loc="best", fontsize=9)
        fig.tight_layout()
        errors_path = out_dir / "dynamics_angle_errors_plot.png"
        fig.savefig(errors_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths["dynamics_angle_errors_plot"] = str(errors_path)

    return paths


def _style_axes(ax: plt.Axes, *, show_labels: bool = True) -> None:
    if show_labels:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
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
    equal_aspect: bool = False,
    show_labels: bool = True,
) -> None:
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    if equal_aspect:
        spans = hi - lo
        try:
            ax.set_box_aspect(spans.tolist())  # type: ignore[attr-defined]
        except AttributeError:
            pass
    ax.view_init(elev=elev, azim=azim)
    # if invert_z:
    #     ax.invert_zaxis()
    _style_axes(ax, show_labels=show_labels)


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


def draw_skeleton_grouped(
    ax: plt.Axes,
    joints: np.ndarray,
    *,
    part_colors: dict[str, str],
    linewidth: float = 2.0,
    linestyle: str = "-",
    pointcolor: str | None = None,
) -> None:
    """Plot skeleton with semantic part colors (frame/handlebars/wheel grouped)."""
    for part, edges in skeleton_edges_by_part().items():
        color = part_colors.get(part, "#00457E")
        for i, j in edges:
            seg = np.stack([joints[i], joints[j]], axis=0)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=linewidth, ls=linestyle)
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
    equal_aspect: bool = False,
    title: str | None = None,
    metrics_text: list[str] | None = None,
    show_titles: bool = True,
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
            if pose is pred:
                draw_skeleton_grouped(ax, pose, part_colors=PART_COLORS_PRED, linewidth=2.4)
            else:
                draw_skeleton(ax, pose, edgecolor=ec, pointcolor=ec)
            if show_titles:
                ax.set_title(panel_title, fontsize=12)
            _finalize_3d_axes(
                ax,
                lo,
                hi,
                elev=elev,
                azim=azim,
                invert_z=invert_z,
                equal_aspect=equal_aspect,
                show_labels=show_titles,
            )
    else:
        fig = plt.figure(figsize=(7.2, 6.4), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        if gt is not None:
            draw_skeleton_grouped(
                ax, gt, part_colors=PART_COLORS_GT, linewidth=1.8, linestyle="--"
            )
        draw_skeleton_grouped(ax, pred, part_colors=PART_COLORS_PRED, linewidth=2.4)
        if show_titles:
            ax.set_title("Prediction vs ground truth" if gt is not None else "Prediction", fontsize=12)
        _finalize_3d_axes(
            ax,
            lo,
            hi,
            elev=elev,
            azim=azim,
            invert_z=invert_z,
            equal_aspect=equal_aspect,
            show_labels=show_titles,
        )

    if show_titles and title:
        fig.suptitle(title, fontsize=11, y=0.98)
    if show_titles and metrics_text:
        fig.text(
            0.01,
            0.02,
            "  |  ".join(metrics_text),
            fontsize=9,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

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
    write_dynamics_plots: bool = True,
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

    steer_pred = bicycle_steer_angle(pred)
    roll_pred = bicycle_roll_angle(pred)
    crank_pred = bicycle_crank_angle(pred)
    steer_vel_pred = _angle_velocity(steer_pred)
    roll_vel_pred = _angle_velocity(roll_pred)
    crank_vel_pred = _angle_velocity(crank_pred)
    meta["dynamics_pred"] = {
        "steer_deg": np.rad2deg(steer_pred).astype(np.float32).tolist(),
        "roll_deg": np.rad2deg(roll_pred).astype(np.float32).tolist(),
        "crank_deg": np.rad2deg(crank_pred).astype(np.float32).tolist(),
        "steer_velocity_deg": np.rad2deg(steer_vel_pred).astype(np.float32).tolist(),
        "roll_velocity_deg": np.rad2deg(roll_vel_pred).astype(np.float32).tolist(),
        "crank_velocity_deg": np.rad2deg(crank_vel_pred).astype(np.float32).tolist(),
    }

    steer_gt: np.ndarray | None = None
    roll_gt: np.ndarray | None = None
    crank_gt: np.ndarray | None = None
    if gt is not None:
        steer_gt = bicycle_steer_angle(gt)
        roll_gt = bicycle_roll_angle(gt)
        crank_gt = bicycle_crank_angle(gt)
        steer_vel_gt = _angle_velocity(steer_gt)
        roll_vel_gt = _angle_velocity(roll_gt)
        crank_vel_gt = _angle_velocity(crank_gt)
        steer_err = np.abs(_wrap_pi(steer_pred - steer_gt))
        roll_err = np.abs(_wrap_pi(roll_pred - roll_gt))
        crank_err = np.abs(_wrap_pi(crank_pred - crank_gt))
        steer_vel_err = np.abs(_wrap_pi(steer_vel_pred - steer_vel_gt))
        roll_vel_err = np.abs(_wrap_pi(roll_vel_pred - roll_vel_gt))
        crank_vel_err = np.abs(_wrap_pi(crank_vel_pred - crank_vel_gt))
        meta["dynamics_gt"] = {
            "steer_deg": np.rad2deg(steer_gt).astype(np.float32).tolist(),
            "roll_deg": np.rad2deg(roll_gt).astype(np.float32).tolist(),
            "crank_deg": np.rad2deg(crank_gt).astype(np.float32).tolist(),
            "steer_velocity_deg": np.rad2deg(steer_vel_gt).astype(np.float32).tolist(),
            "roll_velocity_deg": np.rad2deg(roll_vel_gt).astype(np.float32).tolist(),
            "crank_velocity_deg": np.rad2deg(crank_vel_gt).astype(np.float32).tolist(),
        }
        meta["dynamics_error"] = {
            "steer_mae_deg": float(np.rad2deg(np.mean(steer_err))),
            "roll_mae_deg": float(np.rad2deg(np.mean(roll_err))),
            "crank_mae_deg": float(np.rad2deg(np.mean(crank_err))),
            "steer_velocity_mae_deg": float(np.rad2deg(np.mean(steer_vel_err))) if steer_vel_err.size else 0.0,
            "roll_velocity_mae_deg": float(np.rad2deg(np.mean(roll_vel_err))) if roll_vel_err.size else 0.0,
            "crank_velocity_mae_deg": float(np.rad2deg(np.mean(crank_vel_err))) if crank_vel_err.size else 0.0,
        }

    if write_dynamics_plots:
        steer_pred_deg = np.rad2deg(steer_pred).astype(np.float32)
        roll_pred_deg = np.rad2deg(roll_pred).astype(np.float32)
        crank_pred_deg = np.rad2deg(crank_pred).astype(np.float32)
        steer_gt_deg = np.rad2deg(steer_gt).astype(np.float32) if steer_gt is not None else None
        roll_gt_deg = np.rad2deg(roll_gt).astype(np.float32) if roll_gt is not None else None
        crank_gt_deg = np.rad2deg(crank_gt).astype(np.float32) if crank_gt is not None else None
        meta["dynamics_plots"] = write_dynamics_angle_plots(
            out_dir,
            steer_pred_deg=steer_pred_deg,
            roll_pred_deg=roll_pred_deg,
            crank_pred_deg=crank_pred_deg,
            steer_gt_deg=steer_gt_deg,
            roll_gt_deg=roll_gt_deg,
            crank_gt_deg=crank_gt_deg,
        )

    for t in tqdm(range(pred.shape[0]), desc="frames"):
        metrics_text = [
            f"steer(pred)={np.rad2deg(steer_pred[t]):+6.2f} deg",
            f"roll(pred)={np.rad2deg(roll_pred[t]):+6.2f} deg",
            f"crank(pred)={np.rad2deg(crank_pred[t]):+6.2f} deg",
        ]
        if steer_gt is not None and roll_gt is not None:
            metrics_text.extend(
                [
                    f"steer(gt)={np.rad2deg(steer_gt[t]):+6.2f} deg",
                    f"roll(gt)={np.rad2deg(roll_gt[t]):+6.2f} deg",
                    f"crank(gt)={np.rad2deg(crank_gt[t]):+6.2f} deg",
                ]
            )
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
            metrics_text=metrics_text,
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
        choices=("none", "camera_up", "camera_view"),
        default="camera_up",
        help="Display-only axis remap. camera_view aligns the 3D panel with the recording camera.",
    )
    p.add_argument("--title", type=str, default=None, help="Optional figure title (e.g. MPJPE summary).")
    p.add_argument(
        "--no-dynamics-plots",
        action="store_true",
        help="Skip dynamics angle plots (steer/roll/crank, and errors when GT is available).",
    )
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
        write_dynamics_plots=not args.no_dynamics_plots,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

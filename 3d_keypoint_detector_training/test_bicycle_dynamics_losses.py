"""Tests + visualization for the bicycle steer / roll dynamics losses.

This script proves end-to-end that
``PoseMamba.lib.model.loss.bicycle_steer_angle`` and ``bicycle_roll_angle``
behave as advertised, and that they produce numerically meaningful targets
for use as loss terms.

It runs up to three checks. Each can be enabled / disabled via CLI flags.

1. Synthetic round-trip (default ``--synthetic``):
   Build a canonical bike pose with a non-trivial rake. Impose a known
   rotation about the head-tube axis (steer) and about the camera-frame
   forward axis (roll). Verify that the extractor recovers the imposed
   angle to within numerical tolerance.

2. Extractor vs MuJoCo on GT pickles (default ``--pickles``):
   For every pickle under ``--data-root``, run the steer / roll extractors
   on the GT 3D keypoints and compare against ``dynamics_gt`` already
   stored in the pickle (sourced from the MuJoCo CSV). Steer is a
   bike-intrinsic quantity and should match the MuJoCo value to within
   numerical noise. Roll is computed in the camera frame here vs the world
   frame in MuJoCo, so we check that the *shape* matches (high Pearson r
   and small variance of the residual).

3. End-to-end with a checkpoint (``--checkpoint <path/to/best_epoch.bin>``):
   Run the lifter on a pickle, then plot per-frame steer / roll for the
   prediction, the keypoint-derived GT (= what the loss compares against),
   and the MuJoCo dynamics_gt (= what we actually care about physically).
   Saves a 2x1 figure per pickle.

All numerical reports are printed to stdout; all figures are saved to
``--out-dir`` (default ``training_outputs/dynamics_loss_test``).
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_POSEMAMBA_ROOT = _REPO_ROOT / "PoseMamba"

# Make the PoseMamba package importable without changing cwd.
for path in (_REPO_ROOT, _POSEMAMBA_ROOT, _SCRIPT_DIR):
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

from data_generation_pipeline_tools.bicycle_keypoint_schema import KEYPOINT_INDEX  # noqa: E402
from PoseMamba.lib.model.loss import (  # noqa: E402
    bicycle_roll_angle,
    bicycle_steer_angle,
    loss_bicycle_roll,
    loss_bicycle_roll_velocity,
    loss_bicycle_steer,
    loss_bicycle_steer_velocity,
)

# Avoid pulling matplotlib display backend.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic bike + utilities
# ---------------------------------------------------------------------------

def _canonical_bike() -> torch.Tensor:
    """Return a hand-crafted 18-keypoint bicycle, in a camera-like frame.

    OpenCV convention: X right, Y down, Z forward. The bike is approximately
    aligned with its longitudinal axis along +X, sagittal plane = X-Y plane.
    The head tube is given a non-zero rake (tilt in X-Y) so that any
    steer-angle bug due to rake / cosine error shows up immediately.
    """
    kp = torch.zeros(18, 3, dtype=torch.float64)
    kp[KEYPOINT_INDEX["k_bottom_bracket"]]    = torch.tensor([0.0, 0.0, 0.0])
    kp[KEYPOINT_INDEX["k_seat_stay"]]         = torch.tensor([-0.30, -0.40, 0.0])
    kp[KEYPOINT_INDEX["k_saddle"]]            = torch.tensor([-0.30, -0.60, 0.0])
    kp[KEYPOINT_INDEX["k_lower_head_tube"]]   = torch.tensor([0.60, 0.00, 0.0])
    # Rake: head tube tilts back (towards saddle) -> non-vertical in X-Y.
    kp[KEYPOINT_INDEX["k_upper_head_tube"]]   = torch.tensor([0.50, -0.50, 0.0])
    kp[KEYPOINT_INDEX["k_handlebar_left"]]    = torch.tensor([0.55, -0.55, -0.20])
    kp[KEYPOINT_INDEX["k_handlebar_middle"]]  = torch.tensor([0.55, -0.55, 0.00])
    kp[KEYPOINT_INDEX["k_handlebar_right"]]   = torch.tensor([0.55, -0.55, 0.20])
    kp[KEYPOINT_INDEX["k_front_hub_left"]]    = torch.tensor([0.80, 0.30, -0.05])
    kp[KEYPOINT_INDEX["k_front_hub_right"]]   = torch.tensor([0.80, 0.30, 0.05])
    kp[KEYPOINT_INDEX["k_front_wheel_back"]]  = torch.tensor([0.50, 0.30, 0.00])
    kp[KEYPOINT_INDEX["k_front_wheel_front"]] = torch.tensor([1.10, 0.30, 0.00])
    kp[KEYPOINT_INDEX["k_front_wheel_ground"]] = torch.tensor([0.80, 0.65, 0.00])
    kp[KEYPOINT_INDEX["k_rear_hub_left"]]     = torch.tensor([-0.60, 0.30, -0.05])
    kp[KEYPOINT_INDEX["k_rear_hub_right"]]    = torch.tensor([-0.60, 0.30, 0.05])
    kp[KEYPOINT_INDEX["k_rear_wheel_ground"]] = torch.tensor([-0.60, 0.65, 0.00])
    kp[KEYPOINT_INDEX["k_left_pedal"]]        = torch.tensor([0.00, 0.20, -0.20])
    kp[KEYPOINT_INDEX["k_right_pedal"]]       = torch.tensor([0.00, -0.20, 0.20])
    return kp


def _rodrigues(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    a = axis / (torch.linalg.norm(axis) + 1e-12)
    K = torch.tensor(
        [[0.0, -a[2].item(), a[1].item()],
         [a[2].item(), 0.0, -a[0].item()],
         [-a[1].item(), a[0].item(), 0.0]],
        dtype=axis.dtype,
    )
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return torch.eye(3, dtype=axis.dtype) + s * K + (1.0 - c) * (K @ K)


_FORK_KEYPOINT_NAMES = [
    "k_handlebar_left",
    "k_handlebar_middle",
    "k_handlebar_right",
    "k_front_hub_left",
    "k_front_hub_right",
    "k_front_wheel_back",
    "k_front_wheel_front",
    "k_front_wheel_ground",
]


def _rotate_fork(kpts: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rotate fork-rigid keypoints by ``angle_rad`` about the head-tube axis."""
    out = kpts.clone()
    e = kpts[KEYPOINT_INDEX["k_upper_head_tube"]] - kpts[KEYPOINT_INDEX["k_lower_head_tube"]]
    pivot = kpts[KEYPOINT_INDEX["k_lower_head_tube"]]
    R = _rodrigues(e, angle_rad)
    for name in _FORK_KEYPOINT_NAMES:
        idx = KEYPOINT_INDEX[name]
        out[idx] = (R @ (kpts[idx] - pivot)) + pivot
    return out


def _rotate_about_axis(kpts: torch.Tensor, axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    R = _rodrigues(axis, angle_rad)
    return (R @ kpts.T).T


# ---------------------------------------------------------------------------
# Numpy helpers around the torch loss helpers
# ---------------------------------------------------------------------------

def _angles_from_kpts_np(kpts_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame (steer_deg, roll_deg) numpy arrays from a (T, J, 3) clip."""
    t = torch.from_numpy(np.asarray(kpts_3d, dtype=np.float64))
    if t.dim() == 3:
        t = t.unsqueeze(0)  # (1, T, J, 3)
    steer = bicycle_steer_angle(t).squeeze(0).cpu().numpy()
    roll = bicycle_roll_angle(t).squeeze(0).cpu().numpy()
    return np.rad2deg(steer), np.rad2deg(roll)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    a0 = a - a.mean()
    b0 = b - b.mean()
    return float((a0 * b0).sum() / (np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum()) + 1e-12))


# ---------------------------------------------------------------------------
# 1) Synthetic round-trip
# ---------------------------------------------------------------------------

@dataclass
class SyntheticReport:
    steer_max_err_deg: float
    roll_max_err_deg: float
    rows: list[tuple[str, float, float, float]]  # (case, true_deg, measured_deg, err_deg)


def run_synthetic_tests(angles_deg: tuple[int, ...] = (-30, -20, -10, -5, 0, 5, 10, 20, 30)) -> SyntheticReport:
    bike = _canonical_bike()
    rows: list[tuple[str, float, float, float]] = []

    print("\n=== 1) Synthetic round-trip ===")
    print(f"{'case':>14s}  {'true_deg':>10s}  {'measured_deg':>14s}  {'err_deg':>10s}")
    print(f"{'-' * 14:>14s}  {'-' * 10:>10s}  {'-' * 14:>14s}  {'-' * 10:>10s}")

    steer_errs: list[float] = []
    for deg in angles_deg:
        rotated = _rotate_fork(bike, math.radians(deg))
        measured_rad = bicycle_steer_angle(rotated.unsqueeze(0).unsqueeze(0)).item()
        measured = math.degrees(measured_rad)
        err = abs(measured - deg)
        steer_errs.append(err)
        rows.append(("steer", float(deg), float(measured), float(err)))
        print(f"{'steer':>14s}  {deg:>10d}  {measured:>14.6f}  {err:>10.6f}")

    roll_errs: list[float] = []
    long_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    for deg in angles_deg:
        rolled = _rotate_about_axis(bike, long_axis, math.radians(deg))
        measured_rad = bicycle_roll_angle(rolled.unsqueeze(0).unsqueeze(0)).item()
        measured = math.degrees(measured_rad)
        err = abs(measured - deg)
        roll_errs.append(err)
        rows.append(("roll", float(deg), float(measured), float(err)))
        print(f"{'roll':>14s}  {deg:>10d}  {measured:>14.6f}  {err:>10.6f}")

    print(f"\n  steer max err = {max(steer_errs):.6e} deg")
    print(f"  roll  max err = {max(roll_errs):.6e} deg")

    return SyntheticReport(
        steer_max_err_deg=max(steer_errs),
        roll_max_err_deg=max(roll_errs),
        rows=rows,
    )


# ---------------------------------------------------------------------------
# 2) Extractor vs MuJoCo on GT pickles
# ---------------------------------------------------------------------------

@dataclass
class PickleReport:
    """Statistics from the extractor-vs-MuJoCo comparison.

    The training loss is invariant to (a) a global sign flip and (b) a
    per-clip constant offset between the extractor and MuJoCo, because both
    pred and gt are run through the same extractor. So we report
    sign-aligned (steer) and median-aligned (roll) residuals separately
    from the raw ones.
    """
    n_pickles: int
    steer_mae_raw_deg: float
    steer_mae_aligned_deg: float
    steer_max_err_aligned_deg: float
    roll_mae_raw_deg: float
    roll_mae_aligned_deg: float
    roll_max_err_aligned_deg: float
    median_steer_abs_r: float
    median_roll_abs_r: float
    steer_sign_to_mujoco: int  # +1 or -1, autodetected
    per_pickle_rows: list[tuple[str, int, float, float, float, float, float, float]]


def _collect_pickles(data_root: Path, splits: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for split in splits:
        subdir = data_root / "BICYCLE" / split
        if subdir.is_dir():
            found.extend(sorted(subdir.glob("*.pkl")))
    return found


def run_pickle_tests(
    data_root: Path,
    splits: tuple[str, ...],
    out_dir: Path,
    max_pickles: int | None,
    make_plot: bool,
) -> PickleReport | None:
    pkls = _collect_pickles(data_root, splits)
    if not pkls:
        print(f"\n=== 2) Extractor vs MuJoCo skipped (no pickles found under {data_root}) ===")
        return None

    if max_pickles is not None and max_pickles > 0:
        pkls = pkls[:max_pickles]

    print(f"\n=== 2) Extractor vs MuJoCo dynamics_gt on {len(pkls)} pickle(s) ===")
    print(f"  data_root = {data_root}")
    print(f"  splits    = {','.join(splits)}")

    rows: list[tuple[str, int, float, float, float, float, float, float]] = []
    per_clip: list[tuple[Path, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "extractor_vs_mujoco"
    if make_plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for pkl_path in pkls:
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        if "dynamics_gt" not in obj:
            continue
        dg = obj["dynamics_gt"]
        if not isinstance(dg, dict) or "steer_deg" not in dg or "roll_deg" not in dg:
            continue

        gt_3d = np.asarray(obj["data_label"], dtype=np.float64)
        steer_kpt_deg, roll_kpt_deg = _angles_from_kpts_np(gt_3d)
        steer_muj_deg = np.asarray(dg["steer_deg"], dtype=np.float64)
        roll_muj_deg = np.asarray(dg["roll_deg"], dtype=np.float64)

        T = min(steer_kpt_deg.shape[0], steer_muj_deg.shape[0])
        steer_kpt_deg = steer_kpt_deg[:T]
        roll_kpt_deg = roll_kpt_deg[:T]
        steer_muj_deg = steer_muj_deg[:T]
        roll_muj_deg = roll_muj_deg[:T]

        per_clip.append(
            (pkl_path, obj, steer_kpt_deg, steer_muj_deg, roll_kpt_deg, roll_muj_deg)
        )

    if not per_clip:
        print("  no pickles with dynamics_gt found")
        return None

    # Auto-detect global sign convention for steer (training loss is invariant
    # to a sign flip on the extractor output, so this is purely cosmetic for
    # reporting against MuJoCo).
    pos_err = sum(float(np.sum(np.abs(+1 * s_k - s_m))) for _, _, s_k, s_m, _, _ in per_clip)
    neg_err = sum(float(np.sum(np.abs(-1 * s_k - s_m))) for _, _, s_k, s_m, _, _ in per_clip)
    steer_sign = +1 if pos_err <= neg_err else -1

    header = (
        f"{'clip':>50s}  {'win':>3s}  "
        f"{'steer_mae*':>10s}  {'steer_maxe*':>11s}  {'steer_|r|':>9s}  "
        f"{'roll_mae*':>9s}  {'roll_maxe*':>10s}  {'roll_|r|':>8s}"
    )
    print(header)
    print("-" * len(header))
    print("  * = sign-aligned (steer) / per-clip median-aligned (roll)")

    steer_abs_rs: list[float] = []
    roll_abs_rs: list[float] = []
    raw_steer_total_err = 0.0
    raw_roll_total_err = 0.0
    aligned_steer_total_err = 0.0
    aligned_roll_total_err = 0.0
    aligned_steer_max = 0.0
    aligned_roll_max = 0.0
    n_total = 0

    for pkl_path, obj, steer_kpt, steer_muj, roll_kpt, roll_muj in per_clip:
        s_aligned_resid = steer_sign * steer_kpt - steer_muj
        r_aligned_resid = (roll_kpt - np.median(roll_kpt)) - (roll_muj - np.median(roll_muj))

        steer_mae_aligned = float(np.mean(np.abs(s_aligned_resid)))
        steer_maxe_aligned = float(np.max(np.abs(s_aligned_resid)))
        roll_mae_aligned = float(np.mean(np.abs(r_aligned_resid)))
        roll_maxe_aligned = float(np.max(np.abs(r_aligned_resid)))

        steer_r = _pearson(steer_kpt, steer_muj)
        roll_r = _pearson(roll_kpt, roll_muj)
        steer_abs_r = abs(steer_r) if not math.isnan(steer_r) else float("nan")
        roll_abs_r = abs(roll_r) if not math.isnan(roll_r) else float("nan")
        steer_abs_rs.append(steer_abs_r)
        roll_abs_rs.append(roll_abs_r)

        raw_steer_total_err += float(np.sum(np.abs(steer_kpt - steer_muj)))
        raw_roll_total_err += float(np.sum(np.abs(roll_kpt - roll_muj)))
        aligned_steer_total_err += float(np.sum(np.abs(s_aligned_resid)))
        aligned_roll_total_err += float(np.sum(np.abs(r_aligned_resid)))
        aligned_steer_max = max(aligned_steer_max, steer_maxe_aligned)
        aligned_roll_max = max(aligned_roll_max, roll_maxe_aligned)
        n_total += steer_kpt.size

        clip_id = obj.get("meta", {}).get("clip_id", pkl_path.stem)
        win_idx = obj.get("meta", {}).get("window_index", -1)
        rows.append(
            (
                str(clip_id),
                int(win_idx),
                steer_mae_aligned,
                steer_maxe_aligned,
                steer_abs_r,
                roll_mae_aligned,
                roll_maxe_aligned,
                roll_abs_r,
            )
        )

        print(
            f"{clip_id[:50]:>50s}  {win_idx:>3d}  "
            f"{steer_mae_aligned:>10.4f}  {steer_maxe_aligned:>11.4f}  {steer_abs_r:>9.4f}  "
            f"{roll_mae_aligned:>9.4f}  {roll_maxe_aligned:>10.4f}  {roll_abs_r:>8.4f}"
        )

        if make_plot:
            _plot_extractor_vs_mujoco(
                clip_id=str(clip_id),
                win_idx=int(win_idx),
                steer_kpt=steer_sign * steer_kpt,
                steer_muj=steer_muj,
                roll_kpt=roll_kpt,
                roll_muj=roll_muj,
                steer_sign=steer_sign,
                out_path=plot_dir / f"{pkl_path.stem}.png",
            )

    steer_mae_raw = raw_steer_total_err / n_total
    roll_mae_raw = raw_roll_total_err / n_total
    steer_mae_aligned_agg = aligned_steer_total_err / n_total
    roll_mae_aligned_agg = aligned_roll_total_err / n_total
    median_steer_abs_r = float(np.median(steer_abs_rs))
    median_roll_abs_r = float(np.median(roll_abs_rs))

    print("\n  aggregate:")
    print(f"    steer sign-to-mujoco            = {steer_sign:+d}")
    print(f"    steer MAE  raw / sign-aligned   = {steer_mae_raw:8.4f} / {steer_mae_aligned_agg:8.4f} deg")
    print(f"    steer maxE sign-aligned         = {aligned_steer_max:8.4f} deg")
    print(f"    steer median per-clip |Pearson| = {median_steer_abs_r:.4f}")
    print(f"    roll  MAE  raw / median-aligned = {roll_mae_raw:8.4f} / {roll_mae_aligned_agg:8.4f} deg")
    print(f"    roll  maxE median-aligned       = {aligned_roll_max:8.4f} deg")
    print(f"    roll  median per-clip |Pearson| = {median_roll_abs_r:.4f}")
    print("    Sign / per-clip offset differences are expected and do not affect the loss:")
    print("    extractor(pred) and extractor(gt) share the same convention.")

    if make_plot:
        s_kpt_all = np.concatenate([steer_sign * s_k for _, _, s_k, _, _, _ in per_clip])
        s_muj_all = np.concatenate([s_m for _, _, _, s_m, _, _ in per_clip])
        r_kpt_centered = np.concatenate([r_k - np.median(r_k) for _, _, _, _, r_k, _ in per_clip])
        r_muj_centered = np.concatenate([r_m - np.median(r_m) for _, _, _, _, _, r_m in per_clip])
        _plot_scatter(
            x=s_kpt_all,
            y=s_muj_all,
            xlabel=f"steer (deg) from extractor  [sign x {steer_sign:+d}]",
            ylabel="steer (deg) from MuJoCo CSV",
            title=f"Steer extractor vs MuJoCo  (median |r| = {median_steer_abs_r:.4f})",
            out_path=out_dir / "steer_scatter.png",
        )
        _plot_scatter(
            x=r_kpt_centered,
            y=r_muj_centered,
            xlabel="roll (deg), per-clip median removed (extractor, camera frame)",
            ylabel="roll (deg), per-clip median removed (MuJoCo, world frame)",
            title=f"Roll extractor vs MuJoCo  (median |r| = {median_roll_abs_r:.4f})",
            out_path=out_dir / "roll_scatter.png",
        )

    return PickleReport(
        n_pickles=len(rows),
        steer_mae_raw_deg=steer_mae_raw,
        steer_mae_aligned_deg=steer_mae_aligned_agg,
        steer_max_err_aligned_deg=aligned_steer_max,
        roll_mae_raw_deg=roll_mae_raw,
        roll_mae_aligned_deg=roll_mae_aligned_agg,
        roll_max_err_aligned_deg=aligned_roll_max,
        median_steer_abs_r=median_steer_abs_r,
        median_roll_abs_r=median_roll_abs_r,
        steer_sign_to_mujoco=steer_sign,
        per_pickle_rows=rows,
    )


def _plot_extractor_vs_mujoco(
    *,
    clip_id: str,
    win_idx: int,
    steer_kpt: np.ndarray,
    steer_muj: np.ndarray,
    roll_kpt: np.ndarray,
    roll_muj: np.ndarray,
    steer_sign: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t = np.arange(steer_kpt.shape[0])

    ax = axes[0]
    ax.plot(t, steer_muj, label="MuJoCo (CSV)", linewidth=1.5, alpha=0.85)
    sign_lbl = "" if steer_sign == 1 else f" [x {steer_sign:+d} sign-aligned]"
    ax.plot(t, steer_kpt, label=f"extractor on GT 3D{sign_lbl}", linewidth=1.2, linestyle="--")
    ax.set_ylabel("steer angle (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{clip_id}  (window {win_idx})  - extractor vs MuJoCo dynamics_gt", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, roll_muj, label="MuJoCo (world frame)", linewidth=1.5, alpha=0.85)
    ax.plot(t, roll_kpt, label="extractor on GT 3D (camera frame)", linewidth=1.2, linestyle="--")
    ax.set_xlabel("frame")
    ax.set_ylabel("roll angle (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_scatter(*, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=3, alpha=0.35)
    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--", label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) End-to-end with a checkpoint
# ---------------------------------------------------------------------------

@dataclass
class CheckpointReport:
    clip_path: Path
    mpjpe_mm: float
    steer_loss_deg: float
    steer_velocity_loss_deg: float
    roll_loss_deg: float
    roll_velocity_loss_deg: float


def run_checkpoint_test(
    checkpoint: Path,
    data_root: Path,
    splits: tuple[str, ...],
    out_dir: Path,
    num_clips: int,
) -> list[CheckpointReport]:
    pkls = _collect_pickles(data_root, splits)
    if not pkls:
        print(f"\n=== 3) Checkpoint test skipped (no pickles under {data_root}) ===")
        return []
    pkls = pkls[: max(1, num_clips)]

    from lift_from_2d_array import lift_2d_to_3d, load_posemamba_lifter, squeeze_batch
    from posemamba_bicycle_io import prepare_2d, prepare_gt_3d, to_batch_2d

    print(f"\n=== 3) End-to-end pred vs gt on {len(pkls)} pickle(s) with checkpoint ===")
    print(f"  checkpoint = {checkpoint}")

    model, cfg, device = load_posemamba_lifter(
        checkpoint,
        posemamba_root=_POSEMAMBA_ROOT,
        fallback_config=_REPO_ROOT / "3d_keypoint_detector_training" / "PoseMamba_train_bicycle.generated.yaml",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "checkpoint_pred_vs_gt"
    pred_dir.mkdir(parents=True, exist_ok=True)

    reports: list[CheckpointReport] = []
    for pkl_path in pkls:
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        motion_2d = prepare_2d(obj, mode="image_2d", no_conf=getattr(cfg, "no_conf", True))
        motion_2d = to_batch_2d(motion_2d)
        pred_3d = lift_2d_to_3d(model, cfg, device, motion_2d)  # (1, T, J, 3)
        pred_3d = squeeze_batch(pred_3d)  # (T, J, 3)
        gt_3d = prepare_gt_3d(np.asarray(obj["data_label"]), rootrel=getattr(cfg, "rootrel", True))

        # MPJPE in millimeters (synthetic units are meters in our pipeline).
        mpjpe_mm = float(np.linalg.norm(pred_3d - gt_3d, axis=-1).mean() * 1000.0)

        steer_pred_deg, roll_pred_deg = _angles_from_kpts_np(pred_3d)
        steer_gt_deg, roll_gt_deg = _angles_from_kpts_np(gt_3d)

        # Mirror the torch loss values so we can sanity check them too.
        pt_pred = torch.from_numpy(pred_3d.astype(np.float64)).unsqueeze(0)
        pt_gt = torch.from_numpy(gt_3d.astype(np.float64)).unsqueeze(0)
        steer_loss_deg = math.degrees(loss_bicycle_steer(pt_pred, pt_gt).item())
        steer_velocity_loss_deg = math.degrees(loss_bicycle_steer_velocity(pt_pred, pt_gt).item())
        roll_loss_deg = math.degrees(loss_bicycle_roll(pt_pred, pt_gt).item())
        roll_velocity_loss_deg = math.degrees(loss_bicycle_roll_velocity(pt_pred, pt_gt).item())

        dg = obj.get("dynamics_gt", {}) or {}
        steer_muj_deg = np.asarray(dg.get("steer_deg", []), dtype=np.float64)
        roll_muj_deg = np.asarray(dg.get("roll_deg", []), dtype=np.float64)

        # Align MuJoCo to the extractor's convention for visualization only:
        #   - steer: a global sign flip (see the pickle test for derivation).
        #     Auto-detect per clip so a future convention change still works.
        #   - roll: MuJoCo is world-frame and the extractor is camera-frame;
        #     they differ by a per-clip constant equal to camera tilt projected
        #     onto the bike's lateral axis. Subtract MuJoCo's median and add
        #     the GT extractor's median so the curves visually overlap.
        steer_muj_aligned: np.ndarray | None = None
        roll_muj_aligned: np.ndarray | None = None
        if steer_muj_deg.size:
            T_cmp = min(steer_muj_deg.shape[0], steer_gt_deg.shape[0])
            pos_err = float(np.mean(np.abs(steer_muj_deg[:T_cmp] - steer_gt_deg[:T_cmp])))
            neg_err = float(np.mean(np.abs(steer_muj_deg[:T_cmp] + steer_gt_deg[:T_cmp])))
            steer_sign_to_extractor = +1 if pos_err <= neg_err else -1
            steer_muj_aligned = steer_sign_to_extractor * steer_muj_deg
        if roll_muj_deg.size:
            T_cmp = min(roll_muj_deg.shape[0], roll_gt_deg.shape[0])
            roll_muj_aligned = (
                roll_muj_deg
                - np.median(roll_muj_deg[:T_cmp])
                + np.median(roll_gt_deg[:T_cmp])
            )

        clip_id = obj.get("meta", {}).get("clip_id", pkl_path.stem)
        win_idx = obj.get("meta", {}).get("window_index", -1)

        print(
            f"  {clip_id[:60]:>60s} win={win_idx:>2d}  "
            f"MPJPE={mpjpe_mm:6.1f} mm  "
            f"|steer|L1={steer_loss_deg:5.2f} deg  "
            f"|steer'|L1={steer_velocity_loss_deg:5.2f} deg/f  "
            f"|roll|L1={roll_loss_deg:5.2f} deg  "
            f"|roll'|L1={roll_velocity_loss_deg:5.2f} deg/f"
        )

        _plot_pred_vs_gt(
            clip_id=str(clip_id),
            win_idx=int(win_idx),
            mpjpe_mm=mpjpe_mm,
            steer_pred=steer_pred_deg,
            steer_gt=steer_gt_deg,
            steer_muj=steer_muj_aligned,
            roll_pred=roll_pred_deg,
            roll_gt=roll_gt_deg,
            roll_muj=roll_muj_aligned,
            steer_loss_deg=steer_loss_deg,
            roll_loss_deg=roll_loss_deg,
            out_path=pred_dir / f"{pkl_path.stem}.png",
        )

        reports.append(
            CheckpointReport(
                clip_path=pkl_path,
                mpjpe_mm=mpjpe_mm,
                steer_loss_deg=steer_loss_deg,
                steer_velocity_loss_deg=steer_velocity_loss_deg,
                roll_loss_deg=roll_loss_deg,
                roll_velocity_loss_deg=roll_velocity_loss_deg,
            )
        )

    return reports


def _plot_pred_vs_gt(
    *,
    clip_id: str,
    win_idx: int,
    mpjpe_mm: float,
    steer_pred: np.ndarray,
    steer_gt: np.ndarray,
    steer_muj: np.ndarray | None,
    roll_pred: np.ndarray,
    roll_gt: np.ndarray,
    roll_muj: np.ndarray | None,
    steer_loss_deg: float,
    roll_loss_deg: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    t = np.arange(steer_pred.shape[0])

    ax = axes[0]
    ax.plot(t, steer_gt, label="GT (extractor on data_label)", linewidth=1.6)
    ax.plot(t, steer_pred, label="prediction (extractor on lifter output)", linewidth=1.4, linestyle="--")
    if steer_muj is not None and steer_muj.size:
        ax.plot(
            t[: steer_muj.shape[0]],
            steer_muj,
            label="MuJoCo dynamics_gt (sign-aligned to extractor)",
            linewidth=1.0,
            alpha=0.7,
        )
    ax.set_ylabel("steer angle (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"{clip_id}  (window {win_idx})\nMPJPE={mpjpe_mm:.1f} mm  |  "
        f"|steer|L1 = {steer_loss_deg:.2f} deg  |  |roll|L1 = {roll_loss_deg:.2f} deg",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, roll_gt, label="GT (extractor on data_label, camera frame)", linewidth=1.6)
    ax.plot(t, roll_pred, label="prediction (extractor on lifter output)", linewidth=1.4, linestyle="--")
    if roll_muj is not None and roll_muj.size:
        ax.plot(
            t[: roll_muj.shape[0]],
            roll_muj,
            label="MuJoCo dynamics_gt (world frame, median-shifted to GT extractor)",
            linewidth=1.0,
            alpha=0.7,
        )
    ax.set_xlabel("frame")
    ax.set_ylabel("roll angle (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tests and visualizations for bicycle steer / roll dynamics losses.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/SmallSSD/3D-bicycle-pose-estimation/posemamba_training_sequences/PoseMamba_f243s81_detected2d"),
        help="Root containing BICYCLE/{train,val,test}/*.pkl",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test,val",
        help="Comma-separated list of split subfolders to scan under <data-root>/BICYCLE/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "training_outputs" / "dynamics_loss_test",
    )
    parser.add_argument(
        "--max-pickles",
        type=int,
        default=0,
        help="0 = all pickles; otherwise truncate the pickle test to this many clips.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional PoseMamba .bin checkpoint to run an end-to-end pred-vs-gt comparison.",
    )
    parser.add_argument(
        "--num-checkpoint-clips",
        type=int,
        default=3,
        help="Number of pickles to visualize when --checkpoint is supplied.",
    )
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-pickles", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving matplotlib figures.")
    parser.add_argument("--tol-deg", type=float, default=1e-3, help="Pass/fail tolerance for synthetic test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    if not args.skip_synthetic:
        report = run_synthetic_tests()
        if report.steer_max_err_deg > args.tol_deg:
            failures.append(
                f"synthetic steer error {report.steer_max_err_deg:.6e} deg > tol {args.tol_deg:.6e}"
            )
        if report.roll_max_err_deg > args.tol_deg:
            failures.append(
                f"synthetic roll error {report.roll_max_err_deg:.6e} deg > tol {args.tol_deg:.6e}"
            )

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    pickle_report: PickleReport | None = None
    if not args.skip_pickles:
        pickle_report = run_pickle_tests(
            data_root=args.data_root.resolve(),
            splits=splits,
            out_dir=out_dir,
            max_pickles=(args.max_pickles or None),
            make_plot=not args.no_plots,
        )
        if pickle_report is not None and pickle_report.n_pickles > 0:
            # The loss is invariant to sign / per-clip offset, so we judge on
            # (a) per-clip |Pearson r| (shape agreement) and (b) the aligned
            # MAE in degrees (what the loss would actually see if pred = gt).
            if pickle_report.median_steer_abs_r < 0.95:
                failures.append(
                    f"median per-clip |Pearson r| for steer = "
                    f"{pickle_report.median_steer_abs_r:.4f} < 0.95"
                )
            if pickle_report.median_roll_abs_r < 0.95:
                failures.append(
                    f"median per-clip |Pearson r| for roll = "
                    f"{pickle_report.median_roll_abs_r:.4f} < 0.95"
                )
            # Sign-aligned steer should match MuJoCo to <1 deg on average.
            if pickle_report.steer_mae_aligned_deg > 1.0:
                failures.append(
                    f"sign-aligned steer MAE vs MuJoCo "
                    f"{pickle_report.steer_mae_aligned_deg:.4f} deg > 1.0 deg"
                )
            # After removing the per-clip camera-tilt offset, the residual
            # between camera-frame and world-frame roll should be small.
            if pickle_report.roll_mae_aligned_deg > 2.0:
                failures.append(
                    f"median-aligned roll MAE vs MuJoCo "
                    f"{pickle_report.roll_mae_aligned_deg:.4f} deg > 2.0 deg"
                )

    if args.checkpoint is not None:
        run_checkpoint_test(
            checkpoint=args.checkpoint.resolve(),
            data_root=args.data_root.resolve(),
            splits=splits,
            out_dir=out_dir,
            num_clips=args.num_checkpoint_clips,
        )

    print(f"\nOutputs (figures, scatters) under: {out_dir}")

    if failures:
        print("\nFAIL:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nOK: all enabled tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

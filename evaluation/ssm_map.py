#!/usr/bin/env python3
"""Visualize SSM coupling maps among bicycle joints and frames (Figure 5 analog)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "3d_keypoint_detector_training"
POSEMAMBA_ROOT = REPO_ROOT / "PoseMamba"

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lift_from_2d_array import load_posemamba_lifter  # noqa: E402
from posemamba_bicycle_io import Input2DMode, load_sequence_pkl, prepare_2d, to_batch_2d  # noqa: E402
from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES  # noqa: E402
from evaluation.common import default_detected2d_test_dir, ensure_dir  # noqa: E402

JOINT_SHORT = [n.replace("k_", "") for n in BICYCLE_KEYPOINT_NAMES]


def _off_diagonal_values(mat: np.ndarray) -> np.ndarray:
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return mat[mask]


def _contrast_limits(mat: np.ndarray, *, lo_pct: float = 5.0, hi_pct: float = 95.0) -> tuple[float, float]:
    """Percentile limits from off-diagonal entries (exclude self-correlation on diagonal)."""
    off = _off_diagonal_values(mat)
    if off.size == 0:
        return float(np.min(mat)), float(np.max(mat))
    lo = float(np.percentile(off, lo_pct))
    hi = float(np.percentile(off, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _coupling_residual(mat: np.ndarray) -> np.ndarray:
    """Off-diagonal mean-subtracted coupling; diagonal masked for display."""
    residual = mat.astype(np.float64).copy()
    baseline = float(np.mean(_off_diagonal_values(mat)))
    residual -= baseline
    np.fill_diagonal(residual, np.nan)
    return residual


def _mask_diagonal_for_display(mat: np.ndarray) -> np.ndarray:
    out = mat.astype(np.float64).copy()
    np.fill_diagonal(out, np.nan)
    return out


def _enable_ssm_debug(model: torch.nn.Module) -> None:
    """Enable __DEBUG__ capture on all BiSTSSM submodules."""
    for module in model.modules():
        if module.__class__.__name__ in ("BiSTSSM", "BiSTSSMBlock", "SS2D"):
            module.__DEBUG__ = True  # type: ignore[attr-defined]


def _collect_coupling(
    model: torch.nn.Module,
    sample: torch.Tensor,
    *,
    num_frames: int = 243,
    num_joints: int = 18,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run forward pass and aggregate |C| coupling over SSM blocks."""
    _enable_ssm_debug(model)
    with torch.no_grad():
        _ = model(sample)

    joint_maps: list[np.ndarray] = []
    frame_maps: list[np.ndarray] = []
    block_tags: list[str] = []

    for name, module in model.named_modules():
        data = getattr(module, "__data__", None)
        if data is None or "Cs" not in data:
            continue
        cs = data["Cs"].detach().cpu().float().numpy()  # (B, K, N, L)
        _b, _k, _n, L = cs.shape
        T, J = num_frames, num_joints
        if T * J != L:
            T = int(np.sqrt(L))
            J = L // max(T, 1)
        cs_abs = np.abs(cs).mean(axis=(0, 1))  # (N, L)
        cs_map = cs_abs.mean(axis=0).reshape(T, J)  # average state dim

        # Joint-joint: correlation of SSM readout weights across time
        jj = np.corrcoef(cs_map.T)
        jj = np.nan_to_num(jj, nan=0.0)

        # Frame-frame: correlation across joints
        ff = np.corrcoef(cs_map)
        ff = np.nan_to_num(ff, nan=0.0)

        joint_maps.append(jj)
        frame_maps.append(ff)
        block_tags.append(name)

    if not joint_maps:
        raise RuntimeError("No SSM debug tensors captured; ensure BiSTSSM __DEBUG__ hook is active.")

    joint_coupling = np.mean(np.stack(joint_maps, axis=0), axis=0)
    frame_coupling = np.mean(np.stack(frame_maps, axis=0), axis=0)
    return joint_coupling, frame_coupling, block_tags


def _plot_matrix_heatmap(
    mat: np.ndarray,
    *,
    title: str,
    out_path: Path,
    cbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    figsize: tuple[float, float] = (10, 9),
    axis_label: str | None = None,
) -> None:
    n = mat.shape[0]
    labels = x_labels or [str(i) for i in range(n)]
    ylabels = y_labels or labels

    fig, ax = plt.subplots(figsize=figsize)
    plot_cmap = plt.get_cmap(cmap).copy()
    plot_cmap.set_bad(color="#ececec")

    if center is not None:
        limit = max(abs(float(np.nanmin(mat))), abs(float(np.nanmax(mat))), 1e-6)
        im = ax.imshow(mat, cmap=plot_cmap, vmin=-limit, vmax=limit)
    else:
        im = ax.imshow(mat, cmap=plot_cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if x_labels is not None:
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_yticklabels(ylabels, fontsize=8)
    if axis_label:
        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_joint_coupling_maps(
    mat: np.ndarray,
    out_dir: Path,
    *,
    lo_pct: float,
    hi_pct: float,
) -> None:
    n = mat.shape[0]
    labels = JOINT_SHORT[:n]
    lo, hi = _contrast_limits(mat, lo_pct=lo_pct, hi_pct=hi_pct)
    baseline = float(np.mean(_off_diagonal_values(mat)))

    _plot_matrix_heatmap(
        mat,
        title="SSM joint-joint coupling (absolute scale)",
        out_path=out_dir / "ssm_joint_joint_absolute.png",
        cbar_label="|correlation|",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        x_labels=labels,
        y_labels=labels,
    )
    _plot_matrix_heatmap(
        _mask_diagonal_for_display(mat),
        title=f"SSM joint-joint coupling (contrast: {lo_pct:g}–{hi_pct:g} pct off-diagonal)",
        out_path=out_dir / "ssm_joint_joint.png",
        cbar_label="|correlation|",
        cmap="viridis",
        vmin=lo,
        vmax=hi,
        x_labels=labels,
        y_labels=labels,
    )
    _plot_matrix_heatmap(
        _coupling_residual(mat),
        title=f"SSM joint-joint coupling residual (Δ from mean off-diagonal = {baseline:.3f})",
        out_path=out_dir / "ssm_joint_joint_residual.png",
        cbar_label="Δ |correlation|",
        cmap="RdBu_r",
        center=0.0,
        x_labels=labels,
        y_labels=labels,
    )


def _plot_frame_coupling_maps(
    mat: np.ndarray,
    out_dir: Path,
    *,
    lo_pct: float,
    hi_pct: float,
) -> None:
    lo, hi = _contrast_limits(mat, lo_pct=lo_pct, hi_pct=hi_pct)
    baseline = float(np.mean(_off_diagonal_values(mat)))

    _plot_matrix_heatmap(
        mat,
        title="SSM frame-frame coupling (absolute scale)",
        out_path=out_dir / "ssm_frame_frame_absolute.png",
        cbar_label="|correlation|",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        figsize=(8, 7),
        axis_label="Frame index",
    )
    _plot_matrix_heatmap(
        _mask_diagonal_for_display(mat),
        title=f"SSM frame-frame coupling (contrast: {lo_pct:g}–{hi_pct:g} pct off-diagonal)",
        out_path=out_dir / "ssm_frame_frame.png",
        cbar_label="|correlation|",
        cmap="viridis",
        vmin=lo,
        vmax=hi,
        figsize=(8, 7),
        axis_label="Frame index",
    )
    _plot_matrix_heatmap(
        _coupling_residual(mat),
        title=f"SSM frame-frame coupling residual (Δ from mean off-diagonal = {baseline:.3f})",
        out_path=out_dir / "ssm_frame_frame_residual.png",
        cbar_label="Δ |correlation|",
        cmap="RdBu_r",
        center=0.0,
        figsize=(8, 7),
        axis_label="Frame index",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SSM coupling heatmaps.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=TRAIN_DIR / "PoseMamba_train_bicycle.generated.yaml")
    p.add_argument(
        "--sample-pkl",
        type=Path,
        default=None,
        help="A sample pickle or directory of pickles (default: detected-2D BICYCLE test set).",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/ssm_maps")
    p.add_argument(
        "--contrast-lo-pct",
        type=float,
        default=5.0,
        help="Lower percentile for contrast-enhanced heatmaps (off-diagonal only).",
    )
    p.add_argument(
        "--contrast-hi-pct",
        type=float,
        default=95.0,
        help="Upper percentile for contrast-enhanced heatmaps (off-diagonal only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # load_posemamba_lifter chdirs into PoseMamba/; resolve paths first.
    out_dir = ensure_dir(args.out.resolve())
    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    sample_path = (args.sample_pkl or default_detected2d_test_dir()).resolve()

    exp_name = checkpoint.stem.replace("_best_epoch", "")
    model, cfg, device = load_posemamba_lifter(
        checkpoint,
        fallback_config=config,
        experiment_name=exp_name,
    )
    if sample_path.is_dir():
        pkls = sorted(sample_path.glob("*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"No pickles in {sample_path}")
        sample_path = pkls[0]

    motion = load_sequence_pkl(sample_path)
    motion_2d = prepare_2d(motion, Input2DMode.IMAGE_2D, no_conf=bool(getattr(cfg, "no_conf", True)))
    batch = to_batch_2d(motion_2d)
    tensor_in = torch.from_numpy(batch).to(device)
    run_in = tensor_in[:, :, :, :2] if getattr(cfg, "no_conf", True) else tensor_in

    joint_map, frame_map, block_tags = _collect_coupling(
        model,
        run_in,
        num_frames=int(getattr(cfg, "clip_len", 243)),
        num_joints=int(getattr(cfg, "num_joints", 18)),
    )

    _plot_joint_coupling_maps(
        joint_map,
        out_dir,
        lo_pct=args.contrast_lo_pct,
        hi_pct=args.contrast_hi_pct,
    )
    # Subsample frame map if too large
    if frame_map.shape[0] > 200:
        idx = np.linspace(0, frame_map.shape[0] - 1, 200, dtype=int)
        frame_map = frame_map[np.ix_(idx, idx)]
    _plot_frame_coupling_maps(
        frame_map,
        out_dir,
        lo_pct=args.contrast_lo_pct,
        hi_pct=args.contrast_hi_pct,
    )

    np.savez_compressed(
        out_dir / "ssm_coupling.npz",
        joint_joint=joint_map,
        frame_frame=frame_map,
        joint_names=np.array(BICYCLE_KEYPOINT_NAMES, dtype=object),
        block_tags=np.array(block_tags, dtype=object),
    )
    print(f"[ssm_map] captured {len(block_tags)} BiSTSSM blocks")
    print(f"[ssm_map] wrote coupling maps to {out_dir}")


if __name__ == "__main__":
    main()

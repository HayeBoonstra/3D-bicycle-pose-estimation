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
from evaluation.common import ensure_dir  # noqa: E402

JOINT_SHORT = [n.replace("k_", "") for n in BICYCLE_KEYPOINT_NAMES]


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


def _plot_joint_heatmap(mat: np.ndarray, title: str, out_path: Path) -> None:
    n = mat.shape[0]
    labels = JOINT_SHORT[:n]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="|correlation|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_heatmap(mat: np.ndarray, title: str, out_path: Path, axis_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SSM coupling heatmaps.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=TRAIN_DIR / "PoseMamba_train_bicycle.generated.yaml")
    p.add_argument(
        "--sample-pkl",
        type=Path,
        default=REPO_ROOT / "data/posemamba_training_sequences/PoseMamba_f243s81_detected2d/BICYCLE/test",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/ssm_maps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # load_posemamba_lifter chdirs into PoseMamba/; resolve paths first.
    out_dir = ensure_dir(args.out.resolve())
    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    sample_path = args.sample_pkl.resolve()

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

    _plot_joint_heatmap(joint_map, "SSM joint-joint coupling (mean over blocks)", out_dir / "ssm_joint_joint.png")
    # Subsample frame map if too large
    if frame_map.shape[0] > 200:
        idx = np.linspace(0, frame_map.shape[0] - 1, 200, dtype=int)
        frame_map = frame_map[np.ix_(idx, idx)]
    _plot_heatmap(frame_map, "SSM frame-frame coupling", out_dir / "ssm_frame_frame.png", "Frame index")

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

#!/usr/bin/env python3
"""Stitch overlapping stride-81 windows into one full-clip 3D pose video.

Training pickles are 243-frame windows from longer clips (e.g. 729 frames, stride 81
→ 7 windows). This script groups pickles by ``meta.clip_id``, runs lifting on each
window, averages predictions where windows overlap, and renders one MP4 per clip.

Examples
--------
One camera clip from val (all ``*_000000.pkl`` … ``*_000006.pkl`` for that cam)::

    python 3d_keypoint_detector_training/visualize_lifter_clip_video.py \\
        --checkpoint checkpoints/posemamba_gpu_run_2026_05_18_T_17_22_57/best_epoch.bin \\
        --input-dir data/posemamba_training_sequences/PoseMamba_f243s81/BICYCLE/val \\
        --clip-id PoseMamba_left_traj0000_cam06 \\
        --out training_outputs/lifter_viz

Per-window only (no stitching) — use instead::

    python 3d_keypoint_detector_training/3D_lifting_inference.py --video
    # set INPUT_SEQUENCE_PATH to a single .pkl or folder
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from posemamba_bicycle_io import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    Input2DMode,
    load_sequence_pkl,
    load_training_config,
    mpjpe_eval,
    prepare_2d,
    prepare_gt_3d,
    to_batch_2d,
)


def _load_model(checkpoint: Path, config: Path, posemamba_root: Path) -> tuple[Any, Any, torch.device]:
    os.chdir(posemamba_root)
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))
    from lib.utils.learning import load_backbone

    cfg = load_training_config(checkpoint, config)
    model = load_backbone(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).cuda() if torch.cuda.is_available() else model.to(device)
    ckpt = torch.load(str(checkpoint), map_location=device)
    state = ckpt["model_pos"]
    if not torch.cuda.is_available():
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, device


def _infer_window(model: Any, cfg: Any, device: torch.device, motion_file: dict, mode: Input2DMode) -> np.ndarray:
    input_2d = to_batch_2d(prepare_2d(motion_file, mode, no_conf=bool(getattr(cfg, "no_conf", True))))
    tensor_in = torch.from_numpy(input_2d).to(device)
    run_in = tensor_in[:, :, :, :2] if cfg.no_conf else tensor_in
    with torch.no_grad():
        pred = model(run_in)
        if cfg.rootrel:
            pred[:, :, 0, :] = 0
    return pred.detach().cpu().numpy()[0]


def _group_windows(input_dir: Path, clip_id: str | None) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.pkl")):
        with path.open("rb") as f:
            obj = pickle.load(f)
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        cid = str(meta.get("clip_id", path.stem.rsplit("_", 1)[0]))
        if clip_id is not None and cid != clip_id:
            continue
        groups[cid].append(path)
    def _window_st(p: Path) -> int:
        with p.open("rb") as f:
            obj = pickle.load(f)
        return int(obj.get("meta", {}).get("st", 0)) if isinstance(obj, dict) else 0

    for cid in groups:
        groups[cid].sort(key=_window_st)
    return dict(groups)


def _stitch_predictions(
    windows: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """Average overlapping root-relative preds. windows: (st, end, pred[T,J,3])."""
    if not windows:
        raise ValueError("no windows to stitch")
    full_t = max(end for _st, end, _ in windows)
    acc = np.zeros((full_t, windows[0][2].shape[1], 3), dtype=np.float64)
    cnt = np.zeros(full_t, dtype=np.float64)
    for st, end, pred in windows:
        t_len = min(len(pred), end - st)
        for i in range(t_len):
            t = st + i
            acc[t] += pred[i]
            cnt[t] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return (acc / cnt[:, None, None]).astype(np.float32)


def _stitch_gt(windows: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    full_t = max(end for _st, end, _ in windows)
    acc = np.zeros((full_t, windows[0][2].shape[1], 3), dtype=np.float64)
    cnt = np.zeros(full_t, dtype=np.float64)
    for st, end, gt in windows:
        t_len = min(len(gt), end - st)
        for i in range(t_len):
            t = st + i
            acc[t] += gt[i]
            cnt[t] += 1.0
    return (acc / np.maximum(cnt, 1.0)[:, None, None]).astype(np.float32)


def _render(
    npz_path: Path,
    out_dir: Path,
    *,
    fps: int,
    layout: str,
    title: str | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "data_generation_pipeline_tools" / "visualize_bicycle_pose3d.py"),
        "--pred",
        str(npz_path),
        "--npz-key",
        "pred",
        "--gt",
        str(npz_path),
        "--gt-npz-key",
        "gt",
        "--out",
        str(out_dir),
        "--layout",
        layout,
        "--no-subtract-root-gt",
        "--reorient",
        "camera_up",
        "--video",
        "--fps",
        str(fps),
    ]
    if title:
        cmd.extend(["--title", title])
    subprocess.run(cmd, check=True)
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch stride-81 windows and render full-clip video.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"PoseMamba .bin checkpoint (default: {DEFAULT_CHECKPOINT.relative_to(_REPO_ROOT)}).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_SCRIPT_DIR / "PoseMamba_train_bicycle.generated.yaml",
    )
    p.add_argument("--posemamba-root", type=Path, default=_REPO_ROOT / "PoseMamba")
    p.add_argument("--input-dir", type=Path, required=True, help="Folder of .pkl windows (e.g. BICYCLE/val).")
    p.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="One source clip (e.g. PoseMamba_left_traj0000_cam06). If omitted, process all clips in dir.",
    )
    p.add_argument("--out", type=Path, default=_REPO_ROOT / "training_outputs" / "lifter_clip_viz")
    p.add_argument("--input-2d-mode", choices=("image_2d", "image_2d_noisy"), default="image_2d")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--layout",
        choices=("split", "overlay", "both"),
        default="both",
        help="GT comparison: side-by-side panels, same-axis overlay, or render both.",
    )
    p.add_argument("--max-clips", type=int, default=0, help="0 = all clips in input-dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mode = Input2DMode(args.input_2d_mode)
    groups = _group_windows(input_dir, args.clip_id)
    if not groups:
        raise SystemExit(f"No .pkl files found under {input_dir}")

    print(f"[setup] Loading model from {args.checkpoint}", flush=True)
    model, cfg, device = _load_model(args.checkpoint.resolve(), args.config.resolve(), args.posemamba_root.resolve())

    clip_ids = sorted(groups.keys())
    if args.max_clips > 0:
        clip_ids = clip_ids[: args.max_clips]

    summary = []
    for cid in clip_ids:
        paths = groups[cid]
        print(f"[clip] {cid}: {len(paths)} window(s)", flush=True)
        pred_wins: list[tuple[int, int, np.ndarray]] = []
        gt_wins: list[tuple[int, int, np.ndarray]] = []
        for path in paths:
            motion = load_sequence_pkl(path)
            meta = motion.get("meta", {})
            st = int(meta.get("st", 0))
            end = int(meta.get("end", st + len(motion["data_label"])))
            pred = _infer_window(model, cfg, device, motion, mode)
            pred_wins.append((st, end, pred))
            if motion.get("data_label") is not None:
                gt = prepare_gt_3d(np.asarray(motion["data_label"]), rootrel=bool(getattr(cfg, "rootrel", True)))
                gt_wins.append((st, end, gt))

        pred_full = _stitch_predictions(pred_wins)
        gt_full = _stitch_gt(gt_wins) if gt_wins else None

        mpjpe: dict[str, float] = {}
        if gt_full is not None:
            mpjpe = mpjpe_eval(pred_full, gt_full, cfg)
            print(
                f"[clip] {cid} MPJPE: {mpjpe['mpjpe_m'] * 1000:.2f} mm "
                f"({pred_full.shape[0]} frames, {len(paths)} windows)",
                flush=True,
            )

        title = None
        if mpjpe:
            title = f"{cid}  |  MPJPE {mpjpe['mpjpe_m'] * 1000:.1f} mm (root-rel)"

        layouts = ["split", "overlay"] if args.layout == "both" else [args.layout]
        video_dirs: dict[str, str] = {}
        vis_meta: dict[str, Any] = {}

        with tempfile.TemporaryDirectory(prefix="clip_vis_", dir=str(out_root)) as tmp:
            npz = Path(tmp) / f"{cid}.npz"
            payload: dict[str, Any] = {"pred": pred_full}
            if gt_full is not None:
                payload["gt"] = gt_full
            np.savez_compressed(npz, **payload)

            for layout in layouts:
                suffix = "" if len(layouts) == 1 else f"_{layout}"
                out_dir = out_root / f"{cid}_full_clip_vis{suffix}"
                vis_meta[layout] = _render(
                    npz,
                    out_dir,
                    fps=args.fps,
                    layout=layout,
                    title=title,
                )
                video_dirs[layout] = str(out_dir)

        row: dict[str, Any] = {
            "clip_id": cid,
            "num_windows": len(paths),
            "full_frames": int(pred_full.shape[0]),
            "video_dirs": video_dirs,
        }
        if mpjpe:
            row["mpjpe_m"] = mpjpe["mpjpe_m"]
            row["mpjpe_mm"] = mpjpe["mpjpe_m"] * 1000.0
        if vis_meta:
            row["viz_summary"] = vis_meta
        summary.append(row)
        dirs = ", ".join(video_dirs.values())
        print(f"[clip] {cid} -> {dirs} ({pred_full.shape[0]} frames)", flush=True)

    mpjpe_rows = [r["mpjpe_mm"] for r in summary if "mpjpe_mm" in r]
    report: dict[str, Any] = {"clips": summary}
    if mpjpe_rows:
        report["mpjpe_mm_mean"] = float(np.mean(mpjpe_rows))
        report["mpjpe_mm_per_clip"] = {r["clip_id"]: r["mpjpe_mm"] for r in summary if "mpjpe_mm" in r}
        print(f"[done] Mean MPJPE across clips: {report['mpjpe_mm_mean']:.2f} mm", flush=True)

    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] Wrote {len(summary)} clip video(s) under {out_root}", flush=True)


if __name__ == "__main__":
    main()

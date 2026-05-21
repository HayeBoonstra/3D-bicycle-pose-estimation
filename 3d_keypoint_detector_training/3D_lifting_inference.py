"""Single-file 2D -> 3D lifting inference for bicycle keypoints.

Edit the configuration block below once, then run:

    python 3d_keypoint_detector_training/3D_lifting_inference.py

Optional video rendering:

    python 3d_keypoint_detector_training/3D_lifting_inference.py --video
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from posemamba_bicycle_io import (
    DEFAULT_CHECKPOINT,
    Input2DMode,
    Noise2DConfig,
    load_sequence_pkl,
    load_training_config,
    mpjpe_eval,
    prepare_2d,
    prepare_gt_3d,
    to_batch_2d,
)

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

POSEMAMBA_ROOT = REPO_ROOT / "PoseMamba"
CONFIG_PATH = REPO_ROOT / "3d_keypoint_detector_training" / "PoseMamba_train_bicycle.generated.yaml"
CHECKPOINT_PATH = DEFAULT_CHECKPOINT

INPUT_SEQUENCE_PATH = (
    REPO_ROOT
    / "data"
    / "posemamba_training_sequences"
    / "PoseMamba_f243s81"
    / "BICYCLE"
    / "val"
)

OUTPUT_DIR = REPO_ROOT / "training_outputs" / "inference_3d"

# image_2d | image_2d_noisy
INPUT_2D_MODE: str = "image_2d"
NOISE_2D_CFG = Noise2DConfig()

MODEL_DEPTH_OVERRIDE = 10
MODEL_DIM_FEAT_OVERRIDE: int | None = 64
MODEL_MAXLEN_OVERRIDE: int | None = None

SKIP_WINDOW_NPZ = True

VIDEO_FPS = 30
VIDEO_ELEV = 18.0
VIDEO_AZIM = 35.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D -> 3D lifting on .npz/.pkl sequence(s).")
    parser.add_argument("--video", action="store_true", help="Render MP4 per input window (full T frames).")
    return parser.parse_args()


def _collect_input_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".npz", ".pkl"}:
            raise ValueError(f"Input file must be .npz or .pkl, got: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.npz")) + sorted(path.glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"No .npz/.pkl files in directory: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _load_model() -> tuple[Any, Any, torch.device]:
    ckpt_path = CHECKPOINT_PATH.resolve()
    fallback_cfg = CONFIG_PATH.resolve()
    posemamba_root = POSEMAMBA_ROOT.resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not posemamba_root.is_dir():
        raise FileNotFoundError(f"PoseMamba root not found: {posemamba_root}")

    os.chdir(posemamba_root)
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))

    from lib.utils.learning import load_backbone

    cfg = load_training_config(ckpt_path, fallback_cfg)
    if MODEL_DEPTH_OVERRIDE is not None:
        cfg.depth = MODEL_DEPTH_OVERRIDE
    if MODEL_DIM_FEAT_OVERRIDE is not None:
        cfg.dim_feat = MODEL_DIM_FEAT_OVERRIDE
    if MODEL_MAXLEN_OVERRIDE is not None:
        cfg.maxlen = MODEL_MAXLEN_OVERRIDE
        cfg.clip_len = MODEL_MAXLEN_OVERRIDE

    model = load_backbone(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    else:
        model = model.to(device)

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state = checkpoint["model_pos"]
    if not torch.cuda.is_available():
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, device


def _resolve_input_mode() -> Input2DMode:
    return Input2DMode(INPUT_2D_MODE)


def _run_inference(model: Any, cfg: Any, device: torch.device, input_2d: np.ndarray) -> np.ndarray:
    tensor_in = torch.from_numpy(input_2d).to(device)
    run_in = tensor_in[:, :, :, :2] if cfg.no_conf else tensor_in

    with torch.no_grad():
        pred = model(run_in)
        if cfg.rootrel:
            pred[:, :, 0, :] = 0
    return pred.detach().cpu().numpy()


def _squeeze_single_sequence(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _render_video(npz_path: Path, out_dir: Path) -> None:
    has_gt = False
    with np.load(npz_path, allow_pickle=True) as data:
        has_gt = "gt" in data

    cmd = [
        sys.executable,
        str(REPO_ROOT / "data_generation_pipeline_tools" / "visualize_bicycle_pose3d.py"),
        "--pred",
        str(npz_path),
        "--npz-key",
        "pred",
        "--out",
        str(out_dir),
        "--video",
        "--fps",
        str(VIDEO_FPS),
        "--elev",
        str(VIDEO_ELEV),
        "--azim",
        str(VIDEO_AZIM),
    ]
    if has_gt:
        cmd.extend(["--gt", str(npz_path), "--gt-npz-key", "gt"])
    subprocess.run(cmd, check=True)


def main() -> None:
    t0 = time.time()
    args = parse_args()
    input_path = INPUT_SEQUENCE_PATH.resolve()
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_input_files(input_path)
    print(f"[setup] Found {len(files)} input file(s) in {input_path}", flush=True)
    print("[setup] Loading model checkpoint...", flush=True)
    model, cfg, device = _load_model()
    input_mode = _resolve_input_mode()
    clip_len = int(getattr(cfg, "clip_len", getattr(cfg, "maxlen", 243)))
    print(
        f"[setup] device={device} input_2d_mode={input_mode.value} clip_len={clip_len}",
        flush=True,
    )

    summary: list[dict[str, Any]] = []
    for i, input_file in enumerate(files, start=1):
        file_t0 = time.time()
        print(f"[{i}/{len(files)}] Loading: {input_file.name}", flush=True)

        if input_file.suffix.lower() == ".pkl":
            motion_file = load_sequence_pkl(input_file)
        else:
            with np.load(input_file, allow_pickle=True) as data:
                motion_file = {k: data[k] for k in data.files}

        motion_2d = prepare_2d(
            motion_file,
            input_mode,
            no_conf=bool(getattr(cfg, "no_conf", True)),
            noise_cfg=NOISE_2D_CFG,
        )
        input_2d = to_batch_2d(motion_2d)
        print(f"[{i}/{len(files)}] Lifting shape {tuple(input_2d.shape)}...", flush=True)
        pred_3d = _run_inference(model, cfg, device, input_2d)

        pred_sq = _squeeze_single_sequence(pred_3d)
        save_payload: dict[str, Any] = {
            "pred": pred_sq,
            "data_input": _squeeze_single_sequence(input_2d),
        }
        gt_raw = motion_file.get("data_label")
        if gt_raw is not None:
            save_payload["gt"] = prepare_gt_3d(np.asarray(gt_raw), rootrel=bool(getattr(cfg, "rootrel", True)))

        row: dict[str, Any] = {
            "input": str(input_file),
            "input_2d_mode": input_mode.value,
            "shape_in": list(input_2d.shape),
            "shape_pred": list(pred_sq.shape),
        }
        if "gt" in save_payload:
            metrics = mpjpe_eval(pred_sq, save_payload["gt"], cfg)
            row["mpjpe_m"] = metrics["mpjpe_m"]
            print(f"[{i}/{len(files)}] MPJPE={metrics['mpjpe_m']:.4f} m", flush=True)

        stem = input_file.stem
        out_npz = output_dir / f"{stem}_lifted3d.npz"
        if not SKIP_WINDOW_NPZ:
            np.savez_compressed(out_npz, **save_payload)
            row["output"] = str(out_npz)

        summary.append(row)

        if args.video:
            vis_out_dir = output_dir / f"{stem}_vis"
            vis_out_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="lift_vis_", dir=str(output_dir)) as tmp:
                tmp_npz = Path(tmp) / f"{stem}_vis.npz"
                np.savez_compressed(tmp_npz, **save_payload)
                _render_video(tmp_npz, vis_out_dir)
            row["video_dir"] = str(vis_out_dir)
            print(f"[{i}/{len(files)}] Video -> {vis_out_dir}", flush=True)

        print(f"[{i}/{len(files)}] Done in {time.time() - file_t0:.1f}s", flush=True)

    summary_path = output_dir / "summary.json"
    mpjpe_vals = [r["mpjpe_m"] for r in summary if "mpjpe_m" in r]
    payload = {
        "input_2d_mode": input_mode.value,
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "files": summary,
        "mpjpe_mean_m": float(np.mean(mpjpe_vals)) if mpjpe_vals else None,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] Wrote {summary_path}", flush=True)
    print(f"[done] Total elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

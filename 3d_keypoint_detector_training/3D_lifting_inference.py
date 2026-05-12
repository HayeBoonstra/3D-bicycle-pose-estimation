"""Single-file 2D -> 3D lifting inference for bicycle keypoints.

Edit the configuration block below once, then run:

    conda run -n posemamba python 3d_keypoint_detector_training/3D_lifting_inference.py

Optional video rendering:

    conda run -n posemamba python 3d_keypoint_detector_training/3D_lifting_inference.py --video
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

# ---------------------------------------------------------------------------
# User configuration (edit these paths/values; no shell path arguments needed)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
POSEMAMBA_ROOT = REPO_ROOT / "PoseMamba"
CONFIG_PATH = REPO_ROOT / "3d_keypoint_detector_training" / "PoseMamba_train_bicycle.generated.yaml"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "posemamba_gpu_run_2026_05_11_T_19_54_48" / "best_epoch.bin"

# Can be either:
#   - a single .npz or .pkl sequence file
#   - a directory containing many .npz/.pkl files
INPUT_SEQUENCE_PATH = (
    REPO_ROOT
    / "data"
    / "posemamba_training_sequences"
    / "PoseMamba_f27s1"
    / "BICYCLE"
    / "zigzag_input"
)
OUTPUT_DIR = REPO_ROOT / "training_outputs" / "inference_3d"

# Override model shape only when checkpoint/config differ.
MODEL_DEPTH_OVERRIDE = None
MODEL_DIM_FEAT_OVERRIDE: int | None = None
MODEL_MAXLEN_OVERRIDE: int | None = None

# Which key in input .npz/.pkl contains 2D keypoints; if None, auto-detect.
INPUT_2D_KEY: str | None = None

# Video settings (used only with --video).
VIDEO_FPS = 30
VIDEO_ELEV = 18.0
VIDEO_AZIM = 35.0
# Render fix for camera-frame "upside down" look.
VIDEO_FLIP_Z = True

# When True, skip writing per-window .npz files to disk.
SKIP_WINDOW_NPZ = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D -> 3D lifting on .npz/.pkl sequence(s).")
    parser.add_argument("--video", action="store_true", help="Render MP4 for each output sequence.")
    return parser.parse_args()


def _load_2d_array(input_path: Path, key: str | None) -> np.ndarray:
    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        with np.load(input_path, allow_pickle=True) as data:
            if key is not None:
                if key not in data:
                    raise KeyError(f"{input_path}: key {key!r} not found. Keys: {sorted(data.files)}")
                return np.asarray(data[key], dtype=np.float32)

            for candidate in ("data_input", "keypoints_2d", "poses_2d", "pose2d", "input_2d", "kpts2d"):
                if candidate in data:
                    return np.asarray(data[candidate], dtype=np.float32)

            raise KeyError(
                f"{input_path}: could not auto-detect 2D keypoints key. "
                f"Tried data_input/keypoints_2d/poses_2d/pose2d/input_2d/kpts2d; keys={sorted(data.files)}"
            )

    if suffix == ".pkl":
        with input_path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise TypeError(f"{input_path}: expected dict in pickle, got {type(obj)}")

        if key is not None:
            if key not in obj:
                raise KeyError(f"{input_path}: key {key!r} not found. Keys: {sorted(obj.keys())}")
            return np.asarray(obj[key], dtype=np.float32)

        for candidate in ("data_input", "keypoints_2d", "poses_2d", "pose2d", "input_2d", "kpts2d"):
            if candidate in obj and obj[candidate] is not None:
                return np.asarray(obj[candidate], dtype=np.float32)

        raise KeyError(
            f"{input_path}: could not auto-detect 2D keypoints key. "
            f"Tried data_input/keypoints_2d/poses_2d/pose2d/input_2d/kpts2d; keys={sorted(obj.keys())}"
        )

    raise ValueError(f"Unsupported input format: {input_path}")


def _to_model_input(arr: np.ndarray) -> np.ndarray:
    """Normalize to shape (N, T, J, C) where C is 2 or 3."""
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected (T,J,C) or (N,T,J,C), got {arr.shape}")
    if arr.shape[-1] < 2:
        raise ValueError(f"Expected at least xy channels in last dimension, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


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
    cfg_path = CONFIG_PATH.resolve()
    ckpt_path = CHECKPOINT_PATH.resolve()
    posemamba_root = POSEMAMBA_ROOT.resolve()

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not posemamba_root.is_dir():
        raise FileNotFoundError(f"PoseMamba root not found: {posemamba_root}")

    os.chdir(posemamba_root)
    if str(posemamba_root) not in sys.path:
        sys.path.insert(0, str(posemamba_root))

    from lib.utils.learning import load_backbone
    from lib.utils.tools import get_config

    cfg = get_config(str(cfg_path))
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


def _run_inference(model: Any, cfg: Any, device: torch.device, input_2d: np.ndarray) -> np.ndarray:
    tensor_in = torch.from_numpy(input_2d).to(device)
    run_in = tensor_in[:, :, :, :2] if cfg.no_conf else tensor_in

    with torch.no_grad():
        pred = model(run_in)
        if cfg.rootrel:
            pred[:, :, 0, :] = 0
        if getattr(cfg, "eval_snap_xy_to_input", False) and cfg.gt_2d:
            pred = pred.clone()
            pred[..., :2] = run_in[..., :2]
    return pred.detach().cpu().numpy()


def _squeeze_single_sequence(arr: np.ndarray) -> np.ndarray:
    """Convert (1, T, J, C) -> (T, J, C) for downstream visualizers."""
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _window_index_from_stem(stem: str) -> int | None:
    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _clip_id_from_stem(stem: str) -> str:
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def _center_frame_3d(arr: np.ndarray) -> np.ndarray:
    """Take center frame from (T,J,3) or (1,T,J,3)."""
    arr = _squeeze_single_sequence(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,J,3) after squeeze, got {arr.shape}")
    return arr[arr.shape[0] // 2]


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


def _apply_render_orientation(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    if VIDEO_FLIP_Z:
        out[..., 2] *= -1.0
    return out


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
    print(f"[setup] Model ready on device={device}", flush=True)

    summary: list[dict[str, Any]] = []
    clip_windows: dict[str, list[tuple[int, np.ndarray, np.ndarray | None]]] = {}
    for i, input_file in enumerate(files, start=1):
        file_t0 = time.time()
        print(f"[{i}/{len(files)}] Loading input: {input_file.name}", flush=True)
        arr = _load_2d_array(input_file, INPUT_2D_KEY)
        input_2d = _to_model_input(arr)
        print(f"[{i}/{len(files)}] Running 3D lifting for shape {tuple(input_2d.shape)}...", flush=True)
        pred_3d = _run_inference(model, cfg, device, input_2d)

        stem = input_file.stem
        save_payload: dict[str, Any] = {
            "pred": _squeeze_single_sequence(pred_3d),
            "data_input": _squeeze_single_sequence(input_2d),
        }
        if input_file.suffix.lower() == ".npz":
            with np.load(input_file, allow_pickle=True) as src:
                if "gt" in src:
                    save_payload["gt"] = _squeeze_single_sequence(np.asarray(src["gt"]))
                elif "data_label" in src:
                    save_payload["gt"] = _squeeze_single_sequence(np.asarray(src["data_label"]))
        else:
            with input_file.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                if "gt" in obj and obj["gt"] is not None:
                    save_payload["gt"] = _squeeze_single_sequence(np.asarray(obj["gt"]))
                elif "data_label" in obj and obj["data_label"] is not None:
                    save_payload["gt"] = _squeeze_single_sequence(np.asarray(obj["data_label"]))
        out_npz = output_dir / f"{stem}_lifted3d.npz"
        if not SKIP_WINDOW_NPZ:
            np.savez_compressed(out_npz, **save_payload)

        row = {"input": str(input_file), "shape_in": list(input_2d.shape)}
        if not SKIP_WINDOW_NPZ:
            row["output"] = str(out_npz)
        summary.append(row)
        if not SKIP_WINDOW_NPZ:
            print(f"[{i}/{len(files)}] Saved: {out_npz.name}", flush=True)
        else:
            print(f"[{i}/{len(files)}] Window processed (npz skipped).", flush=True)

        if args.video:
            idx = _window_index_from_stem(stem)
            clip_id = _clip_id_from_stem(stem)
            center_pred = _center_frame_3d(save_payload["pred"])
            center_gt = _center_frame_3d(save_payload["gt"]) if "gt" in save_payload else None
            if idx is None:
                idx = i
            clip_windows.setdefault(clip_id, []).append((idx, center_pred, center_gt))

        print(f"[{i}/{len(files)}] Finished in {time.time() - file_t0:.1f}s", flush=True)

    if args.video:
        print(f"[video] Stitching full-sequence videos for {len(clip_windows)} clip(s)...", flush=True)
        for clip_id, items in clip_windows.items():
            items.sort(key=lambda x: x[0])
            pred_seq = np.stack([x[1] for x in items], axis=0).astype(np.float32)
            pred_seq = _apply_render_orientation(pred_seq)
            payload: dict[str, Any] = {"pred": pred_seq}
            gt_items = [x[2] for x in items]
            if all(g is not None for g in gt_items):
                gt_seq = np.stack([g for g in gt_items if g is not None], axis=0).astype(np.float32)
                payload["gt"] = _apply_render_orientation(gt_seq)
            elif any(g is not None for g in gt_items):
                print(f"[video] Warning: partial GT for clip {clip_id}; rendering prediction-only video.", flush=True)

            vis_out_dir = output_dir / f"{clip_id}_full_sequence_vis"
            vis_out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[video] Rendering clip {clip_id} -> {vis_out_dir}", flush=True)
            with tempfile.TemporaryDirectory(prefix="lift_vis_", dir=str(output_dir)) as tmp:
                tmp_npz = Path(tmp) / f"{clip_id}_tmp_vis_input.npz"
                np.savez_compressed(tmp_npz, **payload)
                _render_video(tmp_npz, vis_out_dir)
            print(f"[video] Done clip {clip_id}", flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] Wrote {len(summary)} file(s) to {output_dir}", flush=True)
    print(f"[done] Summary: {summary_path}", flush=True)
    print(f"[done] Total elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract 3D predictions + GT dynamics from a PoseMamba checkpoint.

Default inference (``window``): one model forward per test pickle at the training
window length (T=243 for f243s81, T=27 for f27s9, etc.) with **no** temporal PE
extension. Overlapping windows from the same clip are fused with center-pick.

``full_sequence``: stitch each clip's windows into a contiguous timeline, then run
one forward per contiguous segment with temporally interpolated PE when T exceeds
the training window. Useful for deployment visualization, not for matching training.

Other modes:
  - ``center_pick``: alias for ``window``
  - ``center_slide``: stride-1 padded sliding (slow)
  - ``overlap_mean``: overlap averaging (legacy)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "3d_keypoint_detector_training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lift_from_2d_array import (  # noqa: E402
    checkpoint_train_maxlen,
    lift_2d_to_3d,
    lift_2d_to_3d_sequence,
    load_posemamba_lifter,
    squeeze_batch,
)
from posemamba_bicycle_io import Input2DMode, load_sequence_pkl, prepare_2d, prepare_gt_3d  # noqa: E402
from data_generation_pipeline_tools.bicycle_dynamics_angles import (  # noqa: E402
    bicycle_roll_angle,
    bicycle_steer_angle,
)
from data_generation_pipeline_tools.visualize_bicycle_pose3d import bicycle_crank_angle  # noqa: E402
from evaluation.common import (  # noqa: E402
    Input2DSource,
    contiguous_segment_slices,
    test_dir_for_input_2d,
    ensure_dir,
)

InferenceMode = Literal["window", "full_sequence", "center_pick", "center_slide", "overlap_mean"]


def _normalize_inference_mode(mode: str) -> InferenceMode:
    """Map legacy aliases to canonical mode names."""
    if mode == "center_pick":
        return "window"
    return mode  # type: ignore[return-value]


def _group_test_windows(test_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(test_dir.glob("*.pkl")):
        obj = load_sequence_pkl(path)
        meta = obj.get("meta", {})
        cid = str(meta.get("clip_id", path.stem.rsplit("_", 1)[0]))
        groups[cid].append(path)

    def _st(p: Path) -> int:
        obj = load_sequence_pkl(p)
        return int(obj.get("meta", {}).get("st", 0))

    return {k: sorted(v, key=_st) for k, v in groups.items()}


def _input_2d_source_for_summary(test_dir: Path, input_2d: Input2DSource) -> str:
    manifest_path = test_dir.parents[1] / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest.get("input_2d_source")
        if source:
            return str(source)

    first_pkl = next(iter(sorted(test_dir.glob("*.pkl"))), None)
    if first_pkl is not None:
        meta = load_sequence_pkl(first_pkl).get("meta", {})
        source = meta.get("input_2d_source")
        if source:
            return str(source)

    return "gt_projection" if input_2d == "gt" else "rtmpose_detection_bbox"


def _stitch_windows(windows: list[tuple[int, int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average overlapping timeline segments."""
    full_t = max(end for _st, end, _ in windows)
    feat_shape = windows[0][2].shape[1:]
    acc = np.zeros((full_t, *feat_shape), dtype=np.float64)
    cnt = np.zeros(full_t, dtype=np.float64)
    for st, end, arr in windows:
        t_len = min(len(arr), end - st)
        for i in range(t_len):
            t = st + i
            acc[t] += arr[i]
            cnt[t] += 1.0
    support = np.where(cnt > 0.0)[0]
    out = np.zeros_like(acc, dtype=np.float32)
    cnt_bc = cnt[support].reshape(-1, *([1] * (acc.ndim - 1)))
    out[support] = (acc[support] / cnt_bc).astype(np.float32)
    return out[support], support.astype(np.int32)


def _center_pick_predictions(
    windows: list[tuple[int, int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """One prediction per frame from the window where that frame is nearest the center."""
    full_t = max(end for _st, end, _ in windows)
    feat_shape = windows[0][2].shape[1:]
    best_dist = np.full(full_t, np.inf, dtype=np.float64)
    out = np.zeros((full_t, *feat_shape), dtype=np.float32)
    covered = np.zeros(full_t, dtype=bool)

    for st, end, arr in windows:
        t_len = min(len(arr), end - st)
        window_center = st + (t_len - 1) * 0.5
        for i in range(t_len):
            t = st + i
            dist = abs(t - window_center)
            if dist < best_dist[t]:
                best_dist[t] = dist
                out[t] = arr[i]
                covered[t] = True

    support = np.where(covered)[0]
    return out[support], support.astype(np.int32)


def _stitch_dynamics(windows: list[tuple[int, int, np.ndarray]], support_idx: np.ndarray) -> np.ndarray:
    full_t = max(end for _st, end, _ in windows)
    acc = np.zeros(full_t, dtype=np.float64)
    cnt = np.zeros(full_t, dtype=np.float64)
    for st, end, arr in windows:
        t_len = min(len(arr), end - st)
        for i in range(t_len):
            acc[st + i] += arr[i]
            cnt[st + i] += 1.0
    cnt = np.maximum(cnt[support_idx], 1.0)
    return (acc[support_idx] / cnt).astype(np.float32)


def _load_window_segments(
    paths: list[Path],
    cfg: Any,
) -> tuple[
    list[tuple[int, int, np.ndarray]],
    list[tuple[int, int, np.ndarray]],
    list[tuple[int, int, np.ndarray]],
    list[tuple[int, int, np.ndarray]],
]:
    input_wins: list[tuple[int, int, np.ndarray]] = []
    gt_wins: list[tuple[int, int, np.ndarray]] = []
    steer_wins: list[tuple[int, int, np.ndarray]] = []
    roll_wins: list[tuple[int, int, np.ndarray]] = []

    for pkl_path in paths:
        motion = load_sequence_pkl(pkl_path)
        meta = motion.get("meta", {})
        st = int(meta.get("st", 0))
        end = int(meta.get("end", st + len(motion["data_input"])))

        motion_2d = prepare_2d(motion, Input2DMode.IMAGE_2D, no_conf=bool(getattr(cfg, "no_conf", True)))
        gt = prepare_gt_3d(np.asarray(motion["data_label"]), rootrel=bool(getattr(cfg, "rootrel", True)))

        input_wins.append((st, end, motion_2d))
        gt_wins.append((st, end, gt))

        dyn = motion.get("dynamics_gt", {})
        if dyn:
            steer_wins.append((st, end, np.asarray(dyn.get("steer_deg", []), dtype=np.float32)))
            roll_wins.append((st, end, np.asarray(dyn.get("roll_deg", []), dtype=np.float32)))

    return input_wins, gt_wins, steer_wins, roll_wins


def _infer_window_batch(
    model: Any,
    cfg: Any,
    device: Any,
    input_wins: list[tuple[int, int, np.ndarray]],
    *,
    batch_size: int,
    extend_temporal: bool = False,
) -> list[tuple[int, int, np.ndarray]]:
    """Run the lifter on pre-sliced windows, batched along the batch dimension."""
    pred_wins: list[tuple[int, int, np.ndarray]] = []
    batch_starts = list(range(0, len(input_wins), batch_size))
    for batch_start in batch_starts:
        chunk = input_wins[batch_start : batch_start + batch_size]
        arrays = [arr for _st, _end, arr in chunk]
        try:
            batch_pred = lift_2d_to_3d(
                model, cfg, device, np.stack(arrays, axis=0), extend_temporal=extend_temporal
            )
        except torch.cuda.OutOfMemoryError:
            if len(chunk) <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            half = max(1, len(chunk) // 2)
            pred_wins.extend(
                _infer_window_batch(
                    model,
                    cfg,
                    device,
                    chunk[:half],
                    batch_size=half,
                    extend_temporal=extend_temporal,
                )
            )
            pred_wins.extend(
                _infer_window_batch(
                    model,
                    cfg,
                    device,
                    chunk[half:],
                    batch_size=max(1, len(chunk) - half),
                    extend_temporal=extend_temporal,
                )
            )
            continue

        for (st, end, _arr), pred in zip(chunk, batch_pred):
            pred_wins.append((st, end, squeeze_batch(pred) if pred.ndim == 4 else pred))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_wins


def _reconstruct_clip_from_windows(
    paths: list[Path],
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    input_wins, gt_wins, steer_wins, roll_wins = _load_window_segments(paths, cfg)
    motion_2d_full, support = _stitch_windows(input_wins)
    gt_full, _ = _stitch_windows(gt_wins)
    steer_full = _stitch_dynamics(steer_wins, support) if steer_wins else None
    roll_full = _stitch_dynamics(roll_wins, support) if roll_wins else None
    return motion_2d_full, gt_full, steer_full, roll_full, support


def _full_sequence_predict(
    model: Any,
    cfg: Any,
    device: Any,
    motion_2d: np.ndarray,
    frame_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One forward pass per contiguous segment (full measurement, no maxlen chunking)."""
    segments = contiguous_segment_slices(frame_idx)
    pred_parts: list[np.ndarray] = []
    segment_ids: list[int] = []
    for seg_id, seg in enumerate(segments):
        seg_2d = motion_2d[seg]
        seg_pred = lift_2d_to_3d_sequence(model, cfg, device, seg_2d)
        pred_parts.append(seg_pred)
        segment_ids.extend([seg_id] * len(seg_pred))
    if not pred_parts:
        empty = np.zeros((0, motion_2d.shape[1], 3), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.int32)
    return np.concatenate(pred_parts, axis=0), np.asarray(segment_ids, dtype=np.int32)


def _default_batch_size(window_size: int, *, inference_mode: InferenceMode) -> int:
    if inference_mode == "center_slide":
        return max(1, min(16, 1500 // max(window_size, 1)))
    if inference_mode == "full_sequence":
        return 1
    return max(1, min(32, 4000 // max(window_size, 1)))


def _center_slide_predict(
    model: Any,
    cfg: Any,
    device: Any,
    motion_2d_full: np.ndarray,
    *,
    window_size: int,
    batch_size: int,
) -> np.ndarray:
    """Stride-1 center-frame sliding (slow; deployment-exact)."""
    num_frames = int(motion_2d_full.shape[0])
    if num_frames == 0:
        return np.zeros((0, motion_2d_full.shape[1], 3), dtype=np.float32)

    radius = window_size // 2
    center_idx = radius
    preds = np.zeros((num_frames, motion_2d_full.shape[1], 3), dtype=np.float32)

    batch_start = 0
    while batch_start < num_frames:
        current_bs = min(batch_size, num_frames - batch_start)
        batch_centers = list(range(batch_start, batch_start + current_bs))
        windows = []
        for center in batch_centers:
            idx = np.clip(
                np.arange(center - radius, center - radius + window_size),
                0,
                num_frames - 1,
            )
            windows.append(motion_2d_full[idx])
        try:
            batch_pred = lift_2d_to_3d(model, cfg, device, np.stack(windows, axis=0))
        except torch.cuda.OutOfMemoryError:
            if current_bs <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"[extract] CUDA OOM: reducing batch_size to {batch_size}", flush=True)
            continue

        for i, center in enumerate(batch_centers):
            preds[center] = batch_pred[i, center_idx]
        batch_start += current_bs

    return preds


def _fuse_predictions(
    pred_wins: list[tuple[int, int, np.ndarray]],
    *,
    inference_mode: InferenceMode,
) -> tuple[np.ndarray, np.ndarray]:
    if inference_mode == "overlap_mean":
        return _stitch_windows(pred_wins)
    return _center_pick_predictions(pred_wins)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract 3D preds for evaluation.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=TRAIN_DIR / "PoseMamba_train_bicycle.generated.yaml")
    p.add_argument(
        "--input-2d",
        choices=("detected", "gt"),
        default="detected",
        help="2D input source: detected (RTMPose+RF-DETR, default) or gt (oracle projected 2D)",
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="BICYCLE test split pickles (default: auto from --input-2d and checkpoint)",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument(
        "--inference-mode",
        choices=("window", "full_sequence", "center_pick", "center_slide", "overlap_mean"),
        default="window",
        help="window: one forward per test pickle at train T (default); "
        "full_sequence: stitched clip with PE extension",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Windows per forward pass (0 = auto)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint.resolve()
    input_2d: Input2DSource = args.input_2d
    base_exp_name = args.experiment_name or ckpt.parent.name
    if args.experiment_name is None and input_2d == "gt":
        exp_name = f"{base_exp_name}_gt2d"
    else:
        exp_name = base_exp_name
    out_dir = ensure_dir(args.out.resolve() / exp_name)
    test_dir = (
        args.test_dir.resolve()
        if args.test_dir is not None
        else test_dir_for_input_2d(ckpt, input_2d)
    )
    inference_mode: InferenceMode = _normalize_inference_mode(args.inference_mode)

    model, cfg, device = load_posemamba_lifter(
        ckpt,
        fallback_config=args.config.resolve(),
        experiment_name=exp_name,
    )
    window_size = checkpoint_train_maxlen(cfg)
    batch_size = (
        int(args.batch_size)
        if int(args.batch_size) > 0
        else _default_batch_size(window_size, inference_mode=inference_mode)
    )
    print(
        f"[extract] input_2d={input_2d} test_dir={test_dir} "
        f"mode={inference_mode} train_pe={window_size} batch_size={batch_size}",
        flush=True,
    )
    groups = _group_test_windows(test_dir)
    input_2d_source = _input_2d_source_for_summary(test_dir, input_2d)

    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    all_2d: list[np.ndarray] = []
    all_clip_ids: list[str] = []
    all_frame_idx: list[int] = []
    all_segment_ids: list[int] = []
    all_steer: list[float] = []
    all_roll: list[float] = []
    clip_summaries: list[dict[str, Any]] = []

    for clip_id, paths in groups.items():
        segment_ids: np.ndarray | None = None

        if inference_mode == "full_sequence":
            motion_2d_full, gt_full, steer_full, roll_full, support = _reconstruct_clip_from_windows(
                paths, cfg
            )
            pred_clip, segment_ids = _full_sequence_predict(
                model,
                cfg,
                device,
                motion_2d_full,
                support,
            )
            input_2d_clip = motion_2d_full[..., :2]
        elif inference_mode == "center_slide":
            motion_2d_full, gt_full, steer_full, roll_full, support = _reconstruct_clip_from_windows(
                paths, cfg
            )
            pred_clip = _center_slide_predict(
                model,
                cfg,
                device,
                motion_2d_full,
                window_size=window_size,
                batch_size=batch_size,
            )
            input_2d_clip = motion_2d_full[..., :2]
        else:
            input_wins, gt_wins, steer_wins, roll_wins = _load_window_segments(paths, cfg)
            pred_wins = _infer_window_batch(
                model,
                cfg,
                device,
                input_wins,
                batch_size=batch_size,
                extend_temporal=False,
            )
            fuse_mode: InferenceMode = "overlap_mean" if inference_mode == "overlap_mean" else "window"
            pred_clip, support = _fuse_predictions(pred_wins, inference_mode=fuse_mode)
            gt_stitched, gt_support = _stitch_windows(gt_wins)
            input_2d_clip, _ = _stitch_windows(input_wins)
            gt_lookup = {int(f): i for i, f in enumerate(gt_support)}
            gt_full = np.stack([gt_stitched[gt_lookup[int(f)]] for f in support], axis=0)
            steer_full = _stitch_dynamics(steer_wins, support) if steer_wins else None
            roll_full = _stitch_dynamics(roll_wins, support) if roll_wins else None

        if len(pred_clip) != len(gt_full):
            raise RuntimeError(
                f"{clip_id}: pred/gt length mismatch ({len(pred_clip)} vs {len(gt_full)})"
            )

        all_pred.append(pred_clip)
        all_gt.append(gt_full)
        all_2d.append(input_2d_clip[..., :2])
        all_clip_ids.extend([clip_id] * len(support))
        all_frame_idx.extend(support.tolist())
        if segment_ids is not None:
            all_segment_ids.extend(segment_ids.tolist())
        if steer_full is not None:
            all_steer.extend(steer_full.tolist())
            all_roll.extend(roll_full.tolist())

        mpjpe = float(np.linalg.norm(pred_clip - gt_full, axis=-1).mean())
        clip_summaries.append(
            {
                "clip_id": clip_id,
                "num_frames": int(len(support)),
                "num_windows": len(paths),
                "num_contiguous_segments": len(contiguous_segment_slices(support)),
                "mpjpe_m": mpjpe,
            }
        )

    pred_all = np.concatenate(all_pred, axis=0)
    gt_all = np.concatenate(all_gt, axis=0)
    input_all = np.concatenate(all_2d, axis=0)

    save_kwargs: dict[str, Any] = {
        "pred": pred_all,
        "gt": gt_all,
        "data_input": input_all,
        "clip_ids": np.array(all_clip_ids, dtype=object),
        "frame_idx": np.array(all_frame_idx, dtype=np.int32),
        "inference_mode": np.array(inference_mode),
        "pred_steer_deg": np.rad2deg(bicycle_steer_angle(pred_all)).astype(np.float32),
        "pred_roll_deg": np.rad2deg(bicycle_roll_angle(pred_all)).astype(np.float32),
        "pred_crank_deg": np.rad2deg(bicycle_crank_angle(pred_all)).astype(np.float32),
        "gt_steer_deg": np.rad2deg(bicycle_steer_angle(gt_all)).astype(np.float32),
        "gt_roll_deg": np.rad2deg(bicycle_roll_angle(gt_all)).astype(np.float32),
    }
    if all_segment_ids:
        save_kwargs["segment_id"] = np.asarray(all_segment_ids, dtype=np.int32)
    if all_steer:
        save_kwargs["steer_deg"] = np.asarray(all_steer, dtype=np.float32)
        save_kwargs["roll_deg"] = np.asarray(all_roll, dtype=np.float32)

    npz_path = out_dir / "preds_3d.npz"
    np.savez_compressed(npz_path, **save_kwargs)

    summary = {
        "experiment": exp_name,
        "checkpoint": str(ckpt),
        "input_2d": input_2d,
        "input_2d_source": input_2d_source,
        "test_dir": str(test_dir),
        "inference_mode": inference_mode,
        "train_pe_len": window_size,
        "num_clips": len(groups),
        "num_frames": int(pred_all.shape[0]),
        "clips": clip_summaries,
    }
    (out_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[extract] wrote {npz_path} ({pred_all.shape[0]} frames, {len(groups)} clips, "
        f"mode={inference_mode} train_pe={window_size})"
    )


if __name__ == "__main__":
    main()

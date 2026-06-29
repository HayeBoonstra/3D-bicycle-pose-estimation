#!/usr/bin/env python3
"""Extract stitched 3D predictions + GT dynamics from a PoseMamba checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "3d_keypoint_detector_training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lift_from_2d_array import lift_2d_to_3d, load_posemamba_lifter, squeeze_batch  # noqa: E402
from posemamba_bicycle_io import Input2DMode, load_sequence_pkl, prepare_2d, prepare_gt_3d  # noqa: E402
from data_generation_pipeline_tools.visualize_bicycle_pose3d import (  # noqa: E402
    bicycle_crank_angle,
    bicycle_roll_angle,
    bicycle_steer_angle,
)
from evaluation.common import ensure_dir  # noqa: E402


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


def _stitch_windows(windows: list[tuple[int, int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    full_t = max(end for _st, end, _ in windows)
    acc = np.zeros((full_t, windows[0][2].shape[1], 3), dtype=np.float64)
    cnt = np.zeros(full_t, dtype=np.float64)
    for st, end, arr in windows:
        t_len = min(len(arr), end - st)
        for i in range(t_len):
            t = st + i
            acc[t] += arr[i]
            cnt[t] += 1.0
    support = np.where(cnt > 0.0)[0]
    out = np.zeros_like(acc, dtype=np.float32)
    out[support] = (acc[support] / cnt[support, None, None]).astype(np.float32)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract stitched 3D preds for evaluation.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=TRAIN_DIR / "PoseMamba_train_bicycle.generated.yaml")
    p.add_argument(
        "--test-dir",
        type=Path,
        default=REPO_ROOT / "data/posemamba_training_sequences/PoseMamba_f243s81_detected2d/BICYCLE/test",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--experiment-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint.resolve()
    exp_name = args.experiment_name or ckpt.parent.name
    out_dir = ensure_dir(args.out / exp_name)

    model, cfg, device = load_posemamba_lifter(ckpt, fallback_config=args.config.resolve())
    groups = _group_test_windows(args.test_dir.resolve())

    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    all_2d: list[np.ndarray] = []
    all_clip_ids: list[str] = []
    all_frame_idx: list[int] = []
    all_steer: list[float] = []
    all_roll: list[float] = []
    clip_summaries: list[dict[str, Any]] = []

    for clip_id, paths in groups.items():
        pred_wins: list[tuple[int, int, np.ndarray]] = []
        gt_wins: list[tuple[int, int, np.ndarray]] = []
        steer_wins: list[tuple[int, int, np.ndarray]] = []
        roll_wins: list[tuple[int, int, np.ndarray]] = []
        input_2d_wins: list[tuple[int, int, np.ndarray]] = []

        for pkl_path in paths:
            motion = load_sequence_pkl(pkl_path)
            meta = motion.get("meta", {})
            st = int(meta.get("st", 0))
            end = int(meta.get("end", st + len(motion["data_input"])))

            motion_2d = prepare_2d(motion, Input2DMode.IMAGE_2D, no_conf=bool(getattr(cfg, "no_conf", True)))
            pred = squeeze_batch(lift_2d_to_3d(model, cfg, device, motion_2d))
            gt = prepare_gt_3d(np.asarray(motion["data_label"]), rootrel=bool(getattr(cfg, "rootrel", True)))

            pred_wins.append((st, end, pred))
            gt_wins.append((st, end, gt))
            input_2d_wins.append((st, end, motion_2d))

            dyn = motion.get("dynamics_gt", {})
            if dyn:
                steer_wins.append((st, end, np.asarray(dyn.get("steer_deg", []), dtype=np.float32)))
                roll_wins.append((st, end, np.asarray(dyn.get("roll_deg", []), dtype=np.float32)))

        pred_clip, support = _stitch_windows(pred_wins)
        gt_clip, _ = _stitch_windows(gt_wins)
        input_2d_clip, _ = _stitch_windows(input_2d_wins)

        steer_clip = roll_clip = None
        if steer_wins:
            steer_clip = _stitch_dynamics(steer_wins, support)
            roll_clip = _stitch_dynamics(roll_wins, support)

        all_pred.append(pred_clip)
        all_gt.append(gt_clip)
        all_2d.append(input_2d_clip[..., :2])
        all_clip_ids.extend([clip_id] * len(support))
        all_frame_idx.extend(support.tolist())
        if steer_clip is not None:
            all_steer.extend(steer_clip.tolist())
            all_roll.extend(roll_clip.tolist())

        mpjpe = float(np.linalg.norm(pred_clip - gt_clip, axis=-1).mean())
        clip_summaries.append({"clip_id": clip_id, "num_frames": int(len(support)), "mpjpe_m": mpjpe})

    pred_all = np.concatenate(all_pred, axis=0)
    gt_all = np.concatenate(all_gt, axis=0)
    input_all = np.concatenate(all_2d, axis=0)

    save_kwargs: dict[str, Any] = {
        "pred": pred_all,
        "gt": gt_all,
        "data_input": input_all,
        "clip_ids": np.array(all_clip_ids, dtype=object),
        "frame_idx": np.array(all_frame_idx, dtype=np.int32),
        "pred_steer_deg": np.rad2deg(bicycle_steer_angle(pred_all)).astype(np.float32),
        "pred_roll_deg": np.rad2deg(bicycle_roll_angle(pred_all)).astype(np.float32),
        "pred_crank_deg": np.rad2deg(bicycle_crank_angle(pred_all)).astype(np.float32),
        "gt_steer_deg": np.rad2deg(bicycle_steer_angle(gt_all)).astype(np.float32),
        "gt_roll_deg": np.rad2deg(bicycle_roll_angle(gt_all)).astype(np.float32),
    }
    if all_steer:
        save_kwargs["steer_deg"] = np.asarray(all_steer, dtype=np.float32)
        save_kwargs["roll_deg"] = np.asarray(all_roll, dtype=np.float32)

    npz_path = out_dir / "preds_3d.npz"
    np.savez_compressed(npz_path, **save_kwargs)

    summary = {
        "experiment": exp_name,
        "checkpoint": str(ckpt),
        "num_clips": len(groups),
        "num_frames": int(pred_all.shape[0]),
        "clips": clip_summaries,
    }
    (out_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[extract] wrote {npz_path} ({pred_all.shape[0]} frames, {len(groups)} clips)")


if __name__ == "__main__":
    main()

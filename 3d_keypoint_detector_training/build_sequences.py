"""Build PoseMamba-ready sequence datasets from synthetic clip annotations."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES, KEYPOINT_INDEX


@dataclass
class ClipData:
    clip_id: str
    fps: int
    image_wh: tuple[int, int]
    frame_idx: np.ndarray
    points_2d: np.ndarray  # [F, J, 2]
    conf_2d: np.ndarray  # [F, J]
    points_3d_world: np.ndarray  # [F, J, 3]
    points_3d_cam: np.ndarray  # [F, J, 3]
    valid_3d: np.ndarray  # [F, J]
    missing_mask: np.ndarray  # [F, J]
    occluded_mask: np.ndarray  # [F, J]
    in_front_mask: np.ndarray  # [F, J]
    K: np.ndarray  # [3, 3]
    R: np.ndarray  # [3, 3]
    t: np.ndarray  # [3]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sort_2d_frames(annotation_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(annotation_dir.glob("keypoints_2d_frame_*.json")):
        rows.append(_load_json(path))
    return sorted(rows, key=lambda item: int(item["frame_index"]))


def _normalize_2d(points_2d: np.ndarray, bboxes_xywh: np.ndarray) -> np.ndarray:
    centers = np.stack([bboxes_xywh[:, 0] + bboxes_xywh[:, 2] * 0.5, bboxes_xywh[:, 1] + bboxes_xywh[:, 3] * 0.5], axis=-1)
    scales = np.maximum(1.0, np.maximum(bboxes_xywh[:, 2], bboxes_xywh[:, 3]))
    return ((points_2d - centers[:, None, :]) / scales[:, None, None]).astype(np.float32)


def _assert_joint_order(names: list[str]) -> None:
    if names != BICYCLE_KEYPOINT_NAMES:
        raise ValueError("Joint names do not match canonical bicycle keypoint order.")


def _read_clip(clip_dir: Path) -> ClipData:
    annotation_dir = clip_dir / "per_frame_annotations"
    k3d_path = clip_dir / "keypoints_3d.jsonl"
    camera_path = clip_dir / "camera.json"
    if not annotation_dir.exists() or not k3d_path.exists() or not camera_path.exists():
        raise FileNotFoundError(f"Clip missing required files: {clip_dir}")

    frames2d = _sort_2d_frames(annotation_dir)
    rows3d = sorted(_load_jsonl(k3d_path), key=lambda item: int(item["frame_index"]))
    camera = _load_json(camera_path)

    if len(frames2d) != len(rows3d):
        raise ValueError(f"Frame count mismatch in clip {clip_dir.name}: 2D={len(frames2d)} 3D={len(rows3d)}")

    if rows3d and "joint_names" in rows3d[0]:
        _assert_joint_order(list(rows3d[0]["joint_names"]))

    F = len(frames2d)
    J = len(BICYCLE_KEYPOINT_NAMES)
    points_2d = np.zeros((F, J, 2), dtype=np.float32)
    conf_2d = np.zeros((F, J), dtype=np.float32)
    bboxes = np.zeros((F, 4), dtype=np.float32)
    points_3d_world = np.zeros((F, J, 3), dtype=np.float32)
    points_3d_cam = np.zeros((F, J, 3), dtype=np.float32)
    valid_3d = np.zeros((F, J), dtype=np.uint8)
    missing_mask = np.zeros((F, J), dtype=np.uint8)
    occluded_mask = np.zeros((F, J), dtype=np.uint8)
    in_front_mask = np.zeros((F, J), dtype=np.uint8)
    frame_idx = np.zeros((F,), dtype=np.int32)

    for i, (row2d, row3d) in enumerate(zip(frames2d, rows3d)):
        frame_idx[i] = int(row2d["frame_index"])
        bboxes[i] = np.asarray(row2d.get("gt_bbox_xywh", [0.0, 0.0, float(row2d["image_width"]), float(row2d["image_height"])]), dtype=np.float32)

        for kp in row2d["keypoints"]:
            j = KEYPOINT_INDEX[kp["name"]]
            points_2d[i, j] = np.asarray([kp["x"], kp["y"]], dtype=np.float32)
            conf_2d[i, j] = float(kp.get("v", 0)) / 2.0

        kps_world = row3d.get("kps_world", [])
        if len(kps_world) != J:
            world_from_named = [None] * J
            for kp in row3d.get("keypoints", []):
                if kp.get("name") in KEYPOINT_INDEX:
                    world_from_named[KEYPOINT_INDEX[kp["name"]]] = kp.get("world")
            kps_world = world_from_named

        kps_cam = row3d.get("kps_camera")
        if not isinstance(kps_cam, list) or len(kps_cam) != J:
            kps_cam = [None] * J
            R = np.asarray(camera["R"], dtype=np.float32)
            t = np.asarray(camera["t"], dtype=np.float32)
            for j, pw in enumerate(kps_world):
                if pw is None:
                    continue
                pw_np = np.asarray(pw, dtype=np.float32)
                kps_cam[j] = (R @ pw_np + t).tolist()

        valid_row = row3d.get("valid_3d")
        if not isinstance(valid_row, list) or len(valid_row) != J:
            valid_row = [0 if pw is None else 1 for pw in kps_world]
        missing_row = row3d.get("missing_mask")
        if not isinstance(missing_row, list) or len(missing_row) != J:
            missing_row = [1 - int(v) for v in valid_row]
        occluded_row = row3d.get("occluded_mask")
        if not isinstance(occluded_row, list) or len(occluded_row) != J:
            occluded_row = [0] * J
        in_front_row = row3d.get("in_front_mask")
        if not isinstance(in_front_row, list) or len(in_front_row) != J:
            in_front_row = [1 if cam is not None and cam[2] > 0 else 0 for cam in kps_cam]

        for j in range(J):
            if kps_world[j] is not None:
                points_3d_world[i, j] = np.asarray(kps_world[j], dtype=np.float32)
            if kps_cam[j] is not None:
                points_3d_cam[i, j] = np.asarray(kps_cam[j], dtype=np.float32)
            valid_3d[i, j] = np.uint8(valid_row[j])
            missing_mask[i, j] = np.uint8(missing_row[j])
            occluded_mask[i, j] = np.uint8(occluded_row[j])
            in_front_mask[i, j] = np.uint8(in_front_row[j])

    points_2d_norm = _normalize_2d(points_2d, bboxes)
    return ClipData(
        clip_id=rows3d[0]["clip_id"],
        fps=int(rows3d[0].get("fps", 24)),
        image_wh=tuple(camera.get("image_size", [0, 0])),
        frame_idx=frame_idx,
        points_2d=points_2d_norm,
        conf_2d=conf_2d,
        points_3d_world=points_3d_world,
        points_3d_cam=points_3d_cam,
        valid_3d=valid_3d,
        missing_mask=missing_mask,
        occluded_mask=occluded_mask,
        in_front_mask=in_front_mask,
        K=np.asarray(camera["K"], dtype=np.float32),
        R=np.asarray(camera["R"], dtype=np.float32),
        t=np.asarray(camera["t"], dtype=np.float32),
    )


def _window_clip(clip: ClipData, window_size: int, stride: int) -> list[dict]:
    radius = window_size // 2
    idxs = list(range(0, clip.points_2d.shape[0], stride))
    samples: list[dict] = []
    for center in idxs:
        frame_ids = np.clip(np.arange(center - radius, center + radius + 1), 0, clip.points_2d.shape[0] - 1)
        k2d = clip.points_2d[frame_ids]
        conf = clip.conf_2d[frame_ids]
        k3d = clip.points_3d_cam[frame_ids]
        valid = clip.valid_3d[frame_ids]
        samples.append(
            {
                "clip_id": clip.clip_id,
                "frame_idx": clip.frame_idx[frame_ids],
                "kpts2d": k2d,
                "kpts2d_conf": conf,
                "kpts3d_cam": k3d,
                "valid_mask": valid,
                "K": clip.K,
                "R": clip.R,
                "t": clip.t,
                "image_wh": np.asarray(clip.image_wh, dtype=np.int32),
            }
        )
    return samples


def _split_clip_ids(clip_ids: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, set[str]]:
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")
    rng = random.Random(seed)
    # Split over unique clip IDs so each logical clip lands in exactly one split.
    ids = list(dict.fromkeys(clip_ids))
    rng.shuffle(ids)
    n = len(ids)
    # Floor counts so train absorbs rounding remainder (~80/10/10 for default ratios).
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    if n_test + n_val > n:
        n_val = max(0, n - n_test)
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    train_ids = set(ids[n_test + n_val :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _write_posemamba_split(samples: list[dict], split_dir: Path, use_confidence: bool) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        k2d = sample["kpts2d"]
        if use_confidence:
            model_input = np.concatenate([k2d, sample["kpts2d_conf"][..., None]], axis=-1).astype(np.float32)
        else:
            model_input = k2d.astype(np.float32)
        payload = {"data_input": model_input, "data_label": sample["kpts3d_cam"].astype(np.float32)}
        with (split_dir / f"{sample['clip_id']}_{i:06d}.pkl").open("wb") as f:
            pickle.dump(payload, f)


def build_sequences(args: argparse.Namespace) -> None:
    raw_root = args.raw_root
    clip_dirs = [path for path in sorted(raw_root.iterdir()) if path.is_dir() and (path / "keypoints_3d.jsonl").exists()]
    if not clip_dirs:
        raise RuntimeError(f"No clips with keypoints_3d.jsonl found in {raw_root}")

    clips = [_read_clip(path) for path in clip_dirs]
    split_map = _split_clip_ids([clip.clip_id for clip in clips], args.val_ratio, args.test_ratio, args.seed)

    split_samples: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for clip in clips:
        split = "train"
        if clip.clip_id in split_map["val"]:
            split = "val"
        elif clip.clip_id in split_map["test"]:
            split = "test"
        split_samples[split].extend(_window_clip(clip, args.window_size, args.stride))

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    for split, samples in split_samples.items():
        if not samples:
            continue
        npz_payload = {
            "kpts2d": np.asarray([sample["kpts2d"] for sample in samples], dtype=np.float32),
            "kpts2d_conf": np.asarray([sample["kpts2d_conf"] for sample in samples], dtype=np.float32),
            "kpts3d_cam": np.asarray([sample["kpts3d_cam"] for sample in samples], dtype=np.float32),
            "valid_mask": np.asarray([sample["valid_mask"] for sample in samples], dtype=np.uint8),
            "K": np.asarray([sample["K"] for sample in samples], dtype=np.float32),
            "R": np.asarray([sample["R"] for sample in samples], dtype=np.float32),
            "t": np.asarray([sample["t"] for sample in samples], dtype=np.float32),
            "image_wh": np.asarray([sample["image_wh"] for sample in samples], dtype=np.int32),
            "frame_idx": np.asarray([sample["frame_idx"] for sample in samples], dtype=np.int32),
            "clip_id": np.asarray([sample["clip_id"] for sample in samples]),
        }
        np.savez_compressed(out_root / f"sequences_{split}.npz", **npz_payload)

    posemamba_subset = out_root / f"PoseMamba_f{args.window_size}s{args.stride}" / "BICYCLE"
    _write_posemamba_split(split_samples["train"], posemamba_subset / "train", use_confidence=args.use_confidence)
    _write_posemamba_split(split_samples["val"], posemamba_subset / "val", use_confidence=args.use_confidence)
    _write_posemamba_split(split_samples["test"], posemamba_subset / "test", use_confidence=args.use_confidence)

    split_clip_counts = {k: len(v) for k, v in split_map.items()}
    meta = {
        "joint_names": BICYCLE_KEYPOINT_NAMES,
        "window_size": args.window_size,
        "stride": args.stride,
        "split_ratios": {
            "val": args.val_ratio,
            "test": args.test_ratio,
            "train_implied": max(0.0, 1.0 - args.val_ratio - args.test_ratio),
        },
        "split_clip_counts": split_clip_counts,
        "splits": {k: sorted(list(v)) for k, v in split_map.items()},
        "normalization": "bbox_center_scale",
        "coord_frame_target": "camera",
    }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    run_qa(out_root)


def run_qa(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        path = output_root / f"sequences_{split}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        J = len(BICYCLE_KEYPOINT_NAMES)
        for key in ("kpts2d", "kpts2d_conf", "kpts3d_cam", "valid_mask", "frame_idx", "clip_id"):
            if key not in data:
                raise ValueError(f"{path.name} missing required key {key}")
        if data["kpts2d"].shape[2] != J or data["kpts3d_cam"].shape[2] != J:
            raise ValueError(f"{path.name} has wrong joint dimension.")
        if data["kpts2d"].shape[1] != data["kpts3d_cam"].shape[1]:
            raise ValueError(f"{path.name} has inconsistent temporal dimensions.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PoseMamba-ready sequence dataset from synthetic clips.")
    parser.add_argument("--raw-root", type=Path, default=Path("raw_renders"))
    parser.add_argument("--output-root", type=Path, default=Path("data/posemamba_sequences"))
    parser.add_argument("--window-size", type=int, default=27)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--use-confidence", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.qa_only:
        run_qa(args.output_root)
        return
    build_sequences(args)


if __name__ == "__main__":
    main()


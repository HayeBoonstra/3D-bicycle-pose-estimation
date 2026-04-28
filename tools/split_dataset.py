"""Create deterministic clip-level train/val/test splits from raw renders."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_RENDERS_DIR = REPO_ROOT / "raw_renders"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "bicycle_pose_dataset"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw rendered clips by clip id.")
    parser.add_argument("--raw-renders", type=Path, default=DEFAULT_RAW_RENDERS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATASET_DIR / "splits.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--no-stratify", action="store_true")
    return parser.parse_args()


def _clip_scene_id(clip_dir: Path) -> str:
    config_path = clip_dir / "render_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("scene_id"):
            return str(data["scene_id"])
    parts = clip_dir.name.split("_")
    if len(parts) >= 3 and parts[0] == "clip":
        return "_".join(parts[1:-1])
    return "unknown"


def _discover_clips(raw_renders_dir: Path) -> list[tuple[str, str]]:
    clips: list[tuple[str, str]] = []
    for clip_dir in sorted(raw_renders_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        if not (clip_dir / "per_frame_annotations").exists():
            continue
        clips.append((clip_dir.name, _clip_scene_id(clip_dir)))
    if not clips:
        raise RuntimeError(f"No rendered clips found in {raw_renders_dir}")
    return clips


def _counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * (1.0 - train_ratio - val_ratio)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def _split_group(
    clip_ids: list[str],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[str]]:
    shuffled = clip_ids[:]
    rng.shuffle(shuffled)
    n_train, n_val, _ = _counts(len(shuffled), train_ratio, val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def build_splits(
    clips: list[tuple[str, str]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    stratify: bool,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}

    if stratify:
        by_scene: dict[str, list[str]] = defaultdict(list)
        for clip_id, scene_id in clips:
            by_scene[scene_id].append(clip_id)
        groups = by_scene.values()
    else:
        groups = [[clip_id for clip_id, _ in clips]]

    for group in groups:
        split = _split_group(group, rng, train_ratio, val_ratio)
        for name in result:
            result[name].extend(split[name])

    for name in result:
        result[name].sort()
    return result


def main() -> None:
    args = _parse_args()
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--train + --val + --test must sum to 1.0")

    clips = _discover_clips(args.raw_renders)
    splits = build_splits(
        clips,
        seed=args.seed,
        train_ratio=args.train,
        val_ratio=args.val,
        stratify=not args.no_stratify,
    )
    payload = {
        **splits,
        "_meta": {
            "seed": args.seed,
            "train_ratio": args.train,
            "val_ratio": args.val,
            "test_ratio": args.test,
            "stratified_by_scene_id": not args.no_stratify,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote splits to {args.out}")


if __name__ == "__main__":
    main()

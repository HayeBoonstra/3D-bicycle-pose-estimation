"""Run 2D keypoint inference with a trained MMPose checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

try:
    from data_generation_pipeline_tools.bicycle_keypoint_schema import (
        BICYCLE_KEYPOINT_NAMES,
        BICYCLE_SKELETON_NAMES,
    )
except ModuleNotFoundError:
    # Support direct script execution from subdirectories by adding repo root.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data_generation_pipeline_tools.bicycle_keypoint_schema import (
        BICYCLE_KEYPOINT_NAMES,
        BICYCLE_SKELETON_NAMES,
    )


def _to_jsonable(value):
    """Recursively convert NumPy/MMEngine values into JSON-safe Python types."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy should exist, but keep robust.
        np = None

    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D inference with MMPose.")
    parser.add_argument("--config", type=Path, required=True, help="Path to MMPose config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pth.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Single image path or a directory containing images.",
    )
    parser.add_argument(
        "--vis-out-dir",
        type=Path,
        default=Path("training_outputs/inference_2d/vis"),
        help="Directory for rendered visualization images.",
    )
    parser.add_argument(
        "--pred-out-dir",
        type=Path,
        default=Path("training_outputs/inference_2d/preds"),
        help="Directory for per-image prediction json files.",
    )
    parser.add_argument(
        "--summary-jsonl",
        type=Path,
        default=Path("training_outputs/inference_2d/predictions.jsonl"),
        help="Path for a consolidated prediction JSONL.",
    )
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images.")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize image order before applying --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--draw-skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw skeleton links on output visualization images.",
    )
    parser.add_argument(
        "--use-bicycle-skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override visualizer skeleton links with bicycle keypoint schema.",
    )
    return parser.parse_args()


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    for pattern in patterns:
        for image_path in sorted(path.rglob(pattern)):
            yield image_path


def _rename_inferencer_outputs(
    vis_out_dir: Path,
    pred_out_dir: Path,
    image_path: Path,
    frame_index: int,
) -> None:
    """MMPose names outputs from the image basename; many clips share frame_0000.

    Rename the saved vis image and pred json to a single ascending sequence.
    """
    suffix = image_path.suffix.lower() if image_path.suffix else ".png"
    stem = image_path.stem

    src_vis = vis_out_dir / image_path.name
    dst_vis = vis_out_dir / f"frame_{frame_index:06d}{suffix}"
    if src_vis.exists() and src_vis.resolve() != dst_vis.resolve():
        dst_vis.unlink(missing_ok=True)
        src_vis.rename(dst_vis)

    src_pred = pred_out_dir / f"{stem}.json"
    dst_pred = pred_out_dir / f"frame_{frame_index:06d}.json"
    if src_pred.exists() and src_pred.resolve() != dst_pred.resolve():
        dst_pred.unlink(missing_ok=True)
        src_pred.rename(dst_pred)


def _set_bicycle_skeleton(inferencer) -> bool:
    """Inject the shared bicycle skeleton metadata into MMPose visualizer."""
    visualizer = None
    if hasattr(inferencer, "inferencer"):
        visualizer = getattr(inferencer.inferencer, "visualizer", None)
    if visualizer is None:
        visualizer = getattr(inferencer, "visualizer", None)
    if visualizer is None:
        return False

    dataset_meta = dict(getattr(visualizer, "dataset_meta", {}) or {})
    keypoint_index = {name: idx for idx, name in enumerate(BICYCLE_KEYPOINT_NAMES)}
    skeleton_links = [
        (keypoint_index[start], keypoint_index[end]) for start, end in BICYCLE_SKELETON_NAMES
    ]
    dataset_meta["keypoint_names"] = list(BICYCLE_KEYPOINT_NAMES)
    dataset_meta["skeleton_links"] = skeleton_links
    # Keep color arrays in sync with metadata lengths expected by visualizer.
    dataset_meta["keypoint_colors"] = [(255, 0, 0)] * len(BICYCLE_KEYPOINT_NAMES)
    dataset_meta["skeleton_link_colors"] = [(0, 255, 0)] * len(skeleton_links)
    dataset_meta["link_color"] = dataset_meta["skeleton_link_colors"]
    dataset_meta["point_color"] = dataset_meta["keypoint_colors"]
    if hasattr(visualizer, "set_dataset_meta"):
        visualizer.set_dataset_meta(dataset_meta, skeleton_style="mmpose")
    else:
        visualizer.dataset_meta = dataset_meta
    return True


def main() -> None:
    args = _parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    from mmpose.apis import MMPoseInferencer

    args.vis_out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_out_dir.mkdir(parents=True, exist_ok=True)
    args.summary_jsonl.parent.mkdir(parents=True, exist_ok=True)

    inferencer = MMPoseInferencer(
        pose2d=str(args.config),
        pose2d_weights=str(args.checkpoint),
        device=args.device,
    )
    if args.use_bicycle_skeleton and not _set_bicycle_skeleton(inferencer):
        print("Warning: could not access visualizer to apply bicycle skeleton.")

    image_paths = list(_iter_images(args.input))
    if args.shuffle and len(image_paths) > 1:
        rng = random.Random(args.seed)
        rng.shuffle(image_paths)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    print(f"Running inference on {len(image_paths)} image(s)...")
    with args.summary_jsonl.open("w", encoding="utf-8") as f:
        for frame_index, image_path in enumerate(image_paths):
            result_iter = inferencer(
                str(image_path),
                pred_out_dir=str(args.pred_out_dir),
                vis_out_dir=str(args.vis_out_dir),
                return_vis=False,
                draw_skeleton=args.draw_skeleton,
            )
            result = next(result_iter)
            _rename_inferencer_outputs(
                args.vis_out_dir,
                args.pred_out_dir,
                image_path,
                frame_index,
            )
            row = {
                "image_path": str(image_path),
                "predictions": _to_jsonable(result.get("predictions", [])),
            }
            f.write(json.dumps(row) + "\n")
            done = frame_index + 1
            print(f"[{done}/{len(image_paths)}] {image_path}")

    print("Done.")
    print(f"- Visualizations: {args.vis_out_dir}")
    print(f"- Per-image predictions: {args.pred_out_dir}")
    print(f"- Consolidated JSONL: {args.summary_jsonl}")


if __name__ == "__main__":
    main()

"""Run 2D keypoint inference with a trained MMPose checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


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
        default=Path("outputs/inference_2d/vis"),
        help="Directory for rendered visualization images.",
    )
    parser.add_argument(
        "--pred-out-dir",
        type=Path,
        default=Path("outputs/inference_2d/preds"),
        help="Directory for per-image prediction json files.",
    )
    parser.add_argument(
        "--summary-jsonl",
        type=Path,
        default=Path("outputs/inference_2d/predictions.jsonl"),
        help="Path for a consolidated prediction JSONL.",
    )
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images.")
    return parser.parse_args()


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    for pattern in patterns:
        for image_path in sorted(path.rglob(pattern)):
            yield image_path


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

    image_paths = list(_iter_images(args.input))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    print(f"Running inference on {len(image_paths)} image(s)...")
    with args.summary_jsonl.open("w", encoding="utf-8") as f:
        for index, image_path in enumerate(image_paths, start=1):
            result_iter = inferencer(
                str(image_path),
                pred_out_dir=str(args.pred_out_dir),
                vis_out_dir=str(args.vis_out_dir),
                return_vis=False,
            )
            result = next(result_iter)
            row = {
                "image_path": str(image_path),
                "predictions": _to_jsonable(result.get("predictions", [])),
            }
            f.write(json.dumps(row) + "\n")
            print(f"[{index}/{len(image_paths)}] {image_path}")

    print("Done.")
    print(f"- Visualizations: {args.vis_out_dir}")
    print(f"- Per-image predictions: {args.pred_out_dir}")
    print(f"- Consolidated JSONL: {args.summary_jsonl}")


if __name__ == "__main__":
    main()

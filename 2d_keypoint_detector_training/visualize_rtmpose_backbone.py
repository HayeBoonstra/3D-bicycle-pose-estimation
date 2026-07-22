#!/usr/bin/env python3
"""Visualize RTMPose CSPNeXt backbone intermediates for one bicycle crop.

The script saves five thesis-friendly steps:

1. RF-DETR/input bbox crop.
2. Early CSPNeXt feature maps.
3. Middle CSPNeXt feature maps.
4. Late CSPNeXt feature maps.
5. Final RTMPose keypoints overlaid on the crop.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import (  # noqa: E402
    BICYCLE_KEYPOINT_NAMES,
    BICYCLE_SKELETON_NAMES,
    KEYPOINT_INDEX,
)

DEFAULT_CONFIG = REPO_ROOT / "2d_keypoint_detector_training" / "rtmpose_bicycle_full.py"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "training_outputs" / "mmpose_bicycle_rtmpose_l_gpu" / "best_coco_AP_epoch_160.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training_outputs" / "rtmpose_backbone_visualization"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CSPNeXt backbone feature maps from a trained bicycle RTMPose model."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="MMPose config path.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Trained RTMPose checkpoint path.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "Input image or directory containing frames. Required unless --detections-jsonl is used. "
            "If a directory is given, --frame-id selects the frame, otherwise the first image is used."
        ),
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Bicycle bbox in image xyxy pixels. Defaults to the full image.",
    )
    parser.add_argument(
        "--detections-jsonl",
        type=Path,
        default=None,
        help="Pipeline detections JSONL with image_path and bbox_xyxy fields.",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help=(
            "Frame id to read from --detections-jsonl or image directory. "
            "Defaults to the first row/image."
        ),
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Run RF-DETR on the selected image and use its best bicycle bbox.",
    )
    parser.add_argument(
        "--rfdetr-model",
        default="rfdetr-2xlarge",
        help="RF-DETR model id used with --auto-detect.",
    )
    parser.add_argument(
        "--det-confidence",
        type=float,
        default=0.5,
        help="RF-DETR confidence threshold used with --auto-detect.",
    )
    parser.add_argument(
        "--rfdetr-python",
        type=Path,
        default=None,
        help=(
            "Optional Python executable from a dedicated RF-DETR environment. "
            "When set, --auto-detect is run in that environment and RTMPose stays in this process."
        ),
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--channels",
        type=int,
        default=16,
        help="Number of feature-map channels to show per backbone stage.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.05,
        help="Minimum keypoint score to draw in the final overlay.",
    )
    parser.add_argument(
        "--layer-names",
        nargs=3,
        default=None,
        metavar=("EARLY", "MIDDLE", "LATE"),
        help=(
            "Optional exact backbone child layer names to visualize. "
            "If omitted, the script uses the first/middle/last captured backbone stages."
        ),
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save raw selected activation tensors as selected_activations.npz.",
    )
    parser.add_argument(
        "--include-head",
        action="store_true",
        help=(
            "Also save RTMCC head intermediates: keypoint spatial maps, GAU token embedding, "
            "and final SimCC x/y distributions."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    images: list[Path] = []
    for extension in sorted(IMAGE_EXTENSIONS):
        images.extend(sorted(path.glob(f"*{extension}")))
    return images


def resolve_image_path(path: Path, frame_id: int | None) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Image path not found: {path}")

    images = iter_image_paths(path)
    if not images:
        raise FileNotFoundError(f"No images found in directory: {path}")
    if frame_id is None:
        return images[0]
    if frame_id < 0 or frame_id >= len(images):
        raise RuntimeError(f"frame_id={frame_id} is outside image directory range 0..{len(images) - 1}")
    return images[frame_id]


def resolve_input(args: argparse.Namespace) -> tuple[Path, list[float] | None]:
    if args.detections_jsonl is None:
        if args.image is None:
            raise ValueError("Provide --image, or provide --detections-jsonl.")
        return resolve_image_path(args.image, args.frame_id), list(args.bbox) if args.bbox is not None else None

    rows = read_jsonl(args.detections_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {args.detections_jsonl}")

    row = rows[0]
    if args.frame_id is not None:
        matches = [r for r in rows if int(r.get("frame_id", -1)) == args.frame_id]
        if not matches:
            raise RuntimeError(f"frame_id={args.frame_id} not found in {args.detections_jsonl}")
        row = matches[0]

    image_path = Path(row["image_path"])
    bbox = row.get("bbox_xyxy") or row.get("det_bbox_xyxy") or args.bbox
    return image_path, list(bbox) if bbox is not None else None


def detect_best_bbox(
    image_path: Path,
    *,
    model_id: str,
    confidence: float,
    rfdetr_python: Path | None = None,
) -> tuple[list[float], float]:
    if rfdetr_python is not None:
        cmd = [
            str(rfdetr_python),
            str(Path(__file__).resolve()),
            "--detect-only",
            "--image",
            str(image_path),
            "--rfdetr-model",
            model_id,
            "--det-confidence",
            str(confidence),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "RF-DETR subprocess failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        for line in reversed(completed.stdout.splitlines()):
            try:
                row = json.loads(line)
                return [float(v) for v in row["bbox_xyxy"]], float(row.get("score", 0.0))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        raise RuntimeError(f"RF-DETR subprocess did not print a bbox JSON row:\n{completed.stdout}")

    os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")
    from keypoint_detector_pipeline.detect_rfdetr import RFDETRDetector

    detector = RFDETRDetector(model_id=model_id, confidence=confidence)
    detections = detector.detect_image(image_path, frame_id=0)
    if not detections:
        raise RuntimeError(
            f"RF-DETR found no bicycle in {image_path} at confidence {confidence}. "
            "Try lowering --det-confidence or pass --bbox manually."
        )
    best = detections[0]
    return [float(v) for v in best["bbox_xyxy"]], float(best.get("score", 0.0))


def clamp_bbox(bbox: list[float] | None, image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    if bbox is None:
        return [0.0, 0.0, float(width - 1), float(height - 1)]

    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(x1 + 1.0, min(x2, float(width)))
    y2 = max(y1 + 1.0, min(y2, float(height)))
    return [x1, y1, x2, y2]


def crop_and_resize(
    image: Image.Image,
    bbox: list[float],
    size: tuple[int, int],
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    crop = image.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    return crop.resize(size, Image.Resampling.BICUBIC)


def get_config_input_size(config_path: Path) -> tuple[int, int]:
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    input_size = tuple(cfg.codec["input_size"])
    if len(input_size) != 2:
        raise RuntimeError(f"Unexpected codec.input_size: {input_size}")
    return int(input_size[0]), int(input_size[1])


def register_backbone_hooks(model) -> tuple[dict[str, np.ndarray], list[Any]]:
    activations: dict[str, np.ndarray] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(tensor, "detach") and tensor.ndim == 4:
                activations[name] = tensor.detach().float().cpu().numpy()[0]

        return hook

    for name, module in model.backbone.named_children():
        handles.append(module.register_forward_hook(make_hook(name)))

    return activations, handles


def tensor_to_numpy_first_item(value: Any) -> np.ndarray | None:
    tensor = value[0] if isinstance(value, (tuple, list)) else value
    if not hasattr(tensor, "detach"):
        return None
    array = tensor.detach().float().cpu().numpy()
    if array.ndim >= 1:
        array = array[0]
    return array


def register_head_hooks(model) -> tuple[dict[str, np.ndarray], list[Any]]:
    activations: dict[str, np.ndarray] = {}
    handles = []
    target_layers = ("final_layer", "mlp", "gau", "cls_x", "cls_y")

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            array = tensor_to_numpy_first_item(output)
            if array is not None:
                activations[name] = array

        return hook

    for name in target_layers:
        module = getattr(model.head, name, None)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(name)))

    return activations, handles


def select_activation_names(
    activations: dict[str, np.ndarray],
    layer_names: list[str] | None,
) -> list[str]:
    names = list(activations.keys())
    if not names:
        raise RuntimeError("No backbone activations were captured.")

    if layer_names is not None:
        missing = [name for name in layer_names if name not in activations]
        if missing:
            available = ", ".join(names)
            raise RuntimeError(f"Requested layer(s) not captured: {missing}. Available: {available}")
        return layer_names

    if len(names) >= 3:
        return [names[0], names[len(names) // 2], names[-1]]
    if len(names) == 2:
        return [names[0], names[1], names[1]]
    return [names[0], names[0], names[0]]


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(values.min())
    max_value = float(values.max())
    if max_value > 1e-8:
        values = values / max_value
    return values


def activation_to_pca_rgb(activation: np.ndarray) -> np.ndarray:
    """Project CxHxW feature maps to an RGB image via 3-component PCA."""
    if activation.ndim != 3:
        raise RuntimeError(f"Expected CxHxW activation, got shape {activation.shape}")

    channels, height, width = activation.shape
    features = activation.transpose(1, 2, 0).reshape(height * width, channels).astype(np.float64)
    centered = features - features.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3:
        raise RuntimeError(f"Need at least 3 spatial locations for PCA RGB, got {centered.shape[0]}")

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:3].T

    rgb = np.zeros_like(projected, dtype=np.float32)
    for channel_idx in range(3):
        channel = projected[:, channel_idx]
        low = float(channel.min())
        high = float(channel.max())
        rgb[:, channel_idx] = (channel - low) / (high - low) if high - low > 1e-8 else 0.5
    return rgb.reshape(height, width, 3)


def save_pca_rgb_map(rgb: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation="bilinear")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_grid(
    activation: np.ndarray,
    out_path: Path,
    title: str,
    *,
    channels: int,
) -> None:
    import matplotlib.pyplot as plt

    if activation.ndim != 3:
        raise RuntimeError(f"Expected CxHxW activation, got shape {activation.shape}")

    channel_scores = activation.reshape(activation.shape[0], -1).std(axis=1)
    selected = np.argsort(channel_scores)[-channels:][::-1]
    num_cols = min(4, len(selected))
    num_rows = int(np.ceil(len(selected) / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.2, num_rows * 2.0))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, channel_idx in zip(axes_arr, selected):
        channel = normalize_array(activation[channel_idx])
        ax.imshow(channel, cmap="magma")
        ax.set_title(f"ch {int(channel_idx)}", fontsize=8)
        ax.axis("off")
    for ax in axes_arr[len(selected) :]:
        ax.axis("off")

    fig.suptitle(f"{title} ({activation.shape[0]} channels, {activation.shape[1]}x{activation.shape[2]})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_keypoint_map_grid(
    activation: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if activation.ndim != 3:
        raise RuntimeError(f"Expected KxHxW head activation, got shape {activation.shape}")

    num_maps = activation.shape[0]
    num_cols = 6
    num_rows = int(np.ceil(num_maps / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.1, num_rows * 2.0))
    axes_arr = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes_arr[:num_maps]):
        ax.imshow(normalize_array(activation[idx]), cmap="magma")
        name = BICYCLE_KEYPOINT_NAMES[idx] if idx < len(BICYCLE_KEYPOINT_NAMES) else f"kpt_{idx}"
        ax.set_title(name.replace("k_", ""), fontsize=7)
        ax.axis("off")
    for ax in axes_arr[num_maps:]:
        ax.axis("off")
    fig.suptitle(f"{title} ({activation.shape[0]} keypoint maps, {activation.shape[1]}x{activation.shape[2]})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_token_embedding_heatmap(
    embedding: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if embedding.ndim != 2:
        raise RuntimeError(f"Expected KxD token embedding, got shape {embedding.shape}")

    normalized = normalize_array(embedding)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(normalized, aspect="auto", cmap="viridis")
    labels = [
        name.replace("k_", "").replace("_", "\n")
        for name in BICYCLE_KEYPOINT_NAMES[: embedding.shape[0]]
    ]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("hidden feature dimension")
    ax.set_title(f"{title} ({embedding.shape[0]} keypoint tokens x {embedding.shape[1]} hidden dims)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    row_min = values.min(axis=1, keepdims=True)
    row_max = values.max(axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, 1e-8)
    return (values - row_min) / denom


def softmax_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-8)


def save_simcc_distributions(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if pred_x.ndim != 2 or pred_y.ndim != 2:
        raise RuntimeError(f"Expected KxW/KxH SimCC logits, got {pred_x.shape} and {pred_y.shape}")

    labels = [
        name.replace("k_", "").replace("_", "\n")
        for name in BICYCLE_KEYPOINT_NAMES[: pred_x.shape[0]]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, values, title, xlabel in (
        (axes[0], pred_x, "SimCC x distributions", "x bin"),
        (axes[1], pred_y, "SimCC y distributions", "y bin"),
    ):
        im = ax.imshow(normalize_rows(values), aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Final RTMCC coordinate distributions before SimCC decoding")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_simcc_heatmap(pred_x: np.ndarray, pred_y: np.ndarray) -> np.ndarray:
    if pred_x.ndim != 2 or pred_y.ndim != 2:
        raise RuntimeError(f"Expected KxW/KxH SimCC logits, got {pred_x.shape} and {pred_y.shape}")

    prob_x = softmax_rows(pred_x)
    prob_y = softmax_rows(pred_y)
    heatmaps = prob_y[:, :, None] * prob_x[:, None, :]
    return normalize_array(heatmaps.max(axis=0))


def save_simcc_heatmap(pred_x: np.ndarray, pred_y: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    aggregate_heatmap = aggregate_simcc_heatmap(pred_x, pred_y)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    im = ax.imshow(aggregate_heatmap, cmap="magma", interpolation="bilinear")
    ax.set_title("Aggregated reconstructed SimCC heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_single_simcc_distribution(
    values: np.ndarray,
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    bins_on_y_axis: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if values.ndim != 2:
        raise RuntimeError(f"Expected KxBins SimCC logits, got {values.shape}")

    figsize = (4.2, 9.5) if bins_on_y_axis else (9.5, 4.2)
    fig, ax = plt.subplots(figsize=figsize)
    display_values = normalize_rows(values)
    if bins_on_y_axis:
        display_values = display_values.T
    im = ax.imshow(display_values, aspect="auto", cmap="magma")
    ax.set_title(title)
    if bins_on_y_axis:
        ax.set_xlabel("keypoint index")
        ax.set_ylabel(xlabel)
        ax.set_xticks([])
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("keypoint index")
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_keypoint_overlay(
    crop: Image.Image,
    bbox: list[float],
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    score_thr: float,
) -> Image.Image:
    overlay = crop.copy()
    draw = ImageDraw.Draw(overlay)
    crop_w, crop_h = overlay.size
    x1, y1, x2, y2 = bbox
    sx = crop_w / max(1.0, x2 - x1)
    sy = crop_h / max(1.0, y2 - y1)
    points = np.zeros_like(keypoints, dtype=np.float32)
    points[:, 0] = (keypoints[:, 0] - x1) * sx
    points[:, 1] = (keypoints[:, 1] - y1) * sy

    edges = [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]) for a, b in BICYCLE_SKELETON_NAMES]
    for i, j in edges:
        if i >= len(points) or j >= len(points):
            continue
        if scores[i] < score_thr or scores[j] < score_thr:
            continue
        draw.line([tuple(points[i]), tuple(points[j])], fill=(0, 220, 255), width=3)

    for idx, (x, y) in enumerate(points):
        if idx >= len(scores) or scores[idx] < score_thr:
            continue
        radius = 4
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(255, 100, 0),
            outline=(255, 255, 255),
            width=1,
        )
    return overlay


def add_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 30), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((8, 8), label, fill=(0, 0, 0), font=font)
    return canvas


def save_combined_panel(paths: list[Path], labels: list[str], out_path: Path) -> None:
    panels = [add_label(Image.open(path).convert("RGB"), label) for path, label in zip(paths, labels)]
    panel_width = 360
    resized = []
    for panel in panels:
        scale = panel_width / panel.width
        resized.append(panel.resize((panel_width, int(panel.height * scale)), Image.Resampling.LANCZOS))

    gap = 16
    total_width = panel_width * len(resized) + gap * (len(resized) - 1)
    total_height = max(panel.height for panel in resized)
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    x_offset = 0
    for panel in resized:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel_width + gap
    canvas.save(out_path, quality=95)


def unpack_prediction(data_sample) -> tuple[np.ndarray, np.ndarray]:
    instances = data_sample.pred_instances
    keypoints = np.asarray(instances.keypoints[0], dtype=np.float32)
    scores = np.asarray(instances.keypoint_scores[0], dtype=np.float32)
    return keypoints, scores


def main() -> None:
    args = parse_args()
    if args.detect_only:
        image_path, _ = resolve_input(args)
        bbox, score = detect_best_bbox(
            image_path,
            model_id=args.rfdetr_model,
            confidence=args.det_confidence,
        )
        print(json.dumps({"image_path": str(image_path), "bbox_xyxy": bbox, "score": score}))
        return

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    image_path, bbox = resolve_input(args)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise RuntimeError(f"Unsupported image extension: {image_path.suffix}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    detection_score = None
    if args.auto_detect:
        print(f"[rtmpose-backbone-vis] Running RF-DETR on {image_path}...")
        bbox, detection_score = detect_best_bbox(
            image_path,
            model_id=args.rfdetr_model,
            confidence=args.det_confidence,
            rfdetr_python=args.rfdetr_python,
        )
        print(f"[rtmpose-backbone-vis] Best bicycle bbox score: {detection_score:.3f}")
    bbox_xyxy = clamp_bbox(bbox, image.size)
    input_size = get_config_input_size(args.config)

    from mmpose.apis import inference_topdown, init_model

    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    model.eval()
    if hasattr(model, "test_cfg"):
        model.test_cfg["flip_test"] = False

    activations, handles = register_backbone_hooks(model)
    head_activations, head_handles = register_head_hooks(model) if args.include_head else ({}, [])
    try:
        results = inference_topdown(
            model,
            np.asarray(image),
            bboxes=np.asarray([bbox_xyxy], dtype=np.float32),
            bbox_format="xyxy",
        )
    finally:
        for handle in handles:
            handle.remove()
        for handle in head_handles:
            handle.remove()

    if not results:
        raise RuntimeError("RTMPose returned no predictions.")

    selected_names = select_activation_names(activations, args.layer_names)
    keypoints, scores = unpack_prediction(results[0])

    crop = crop_and_resize(image, bbox_xyxy, input_size)
    crop_path = args.out_dir / "01_input_rfdetr_crop.png"
    early_path = args.out_dir / "02_early_cspnext_features.png"
    middle_path = args.out_dir / "03_middle_cspnext_features.png"
    late_path = args.out_dir / "04_late_cspnext_features.png"
    overlay_path = args.out_dir / "05_rtmpose_keypoints_overlay.png"
    combined_path = args.out_dir / "rtmpose_backbone_five_steps.png"
    metadata_path = args.out_dir / "metadata.json"
    head_map_path = args.out_dir / "06_head_keypoint_spatial_maps.png"
    head_embedding_path = args.out_dir / "07_head_gau_keypoint_embedding.png"
    simcc_heatmap_path = args.out_dir / "08_head_simcc_heatmap.png"
    simcc_x_path = args.out_dir / "09_head_simcc_x_distributions.png"
    simcc_y_path = args.out_dir / "10_head_simcc_y_distributions.png"

    crop.save(crop_path)
    save_feature_grid(
        activations[selected_names[0]],
        early_path,
        f"Early CSPNeXt feature maps: {selected_names[0]}",
        channels=args.channels,
    )
    save_pca_rgb_map(
        activation_to_pca_rgb(activations[selected_names[1]]),
        middle_path,
        f"Middle CSPNeXt PCA RGB: {selected_names[1]}",
    )
    save_feature_grid(
        activations[selected_names[2]],
        late_path,
        f"Late CSPNeXt feature maps: {selected_names[2]}",
        channels=args.channels,
    )
    draw_keypoint_overlay(crop, bbox_xyxy, keypoints, scores, score_thr=args.score_thr).save(overlay_path)

    head_outputs = {}
    if args.include_head:
        if "final_layer" in head_activations:
            save_keypoint_map_grid(
                head_activations["final_layer"],
                head_map_path,
                "RTMCC head spatial maps after final convolution",
            )
            head_outputs["keypoint_spatial_maps"] = str(head_map_path)
        if "gau" in head_activations:
            save_token_embedding_heatmap(
                head_activations["gau"],
                head_embedding_path,
                "RTMCC GAU keypoint-token representation",
            )
            head_outputs["gau_keypoint_embedding"] = str(head_embedding_path)
        if "cls_x" in head_activations and "cls_y" in head_activations:
            save_simcc_heatmap(
                head_activations["cls_x"],
                head_activations["cls_y"],
                simcc_heatmap_path,
            )
            head_outputs["simcc_heatmap"] = str(simcc_heatmap_path)
            save_single_simcc_distribution(
                head_activations["cls_x"],
                simcc_x_path,
                title="X-axis SimCC coordinate distributions",
                xlabel="x bin",
            )
            head_outputs["simcc_x_distributions"] = str(simcc_x_path)
            save_single_simcc_distribution(
                head_activations["cls_y"],
                simcc_y_path,
                title="Y-axis SimCC coordinate distributions",
                xlabel="y bin",
                bins_on_y_axis=True,
            )
            head_outputs["simcc_y_distributions"] = str(simcc_y_path)

    save_combined_panel(
        [crop_path, early_path, middle_path, late_path, overlay_path],
        [
            "1. RF-DETR crop",
            "2. Early backbone",
            "3. Middle backbone",
            "4. Late backbone",
            "5. RTMPose keypoints",
        ],
        combined_path,
    )

    metadata = {
        "image_path": str(image_path),
        "bbox_xyxy": bbox_xyxy,
        "auto_detect": bool(args.auto_detect),
        "detection_score": detection_score,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "input_size_wh": list(input_size),
        "captured_backbone_layers": {
            name: list(value.shape) for name, value in activations.items()
        },
        "captured_head_layers": {
            name: list(value.shape) for name, value in head_activations.items()
        },
        "selected_layers": {
            "early": selected_names[0],
            "middle": selected_names[1],
            "late": selected_names[2],
        },
        "outputs": {
            "input_crop": str(crop_path),
            "early_features": str(early_path),
            "middle_features": str(middle_path),
            "late_features": str(late_path),
            "keypoints_overlay": str(overlay_path),
            "combined_panel": str(combined_path),
        },
    }
    if head_outputs:
        metadata["outputs"].update(head_outputs)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.save_npz:
        np.savez_compressed(
            args.out_dir / "selected_activations.npz",
            early=activations[selected_names[0]],
            middle=activations[selected_names[1]],
            late=activations[selected_names[2]],
        )

    print("[rtmpose-backbone-vis] Wrote:")
    for output in metadata["outputs"].values():
        print(f"  {output}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()

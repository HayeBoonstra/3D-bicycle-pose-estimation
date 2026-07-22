#!/usr/bin/env python3
"""Visualize RF-DETR intermediate representations for one bicycle frame.

The script uses the native PyTorch RF-DETR backend (not ONNX) so backbone and
decoder tensors can be captured with forward hooks.

Saved panels:

1. Input frame.
2. Early DINOv2 patch features.
3. Middle DINOv2 transformer-layer features.
4. Late multi-scale projector features.
5. Bicycle-query deformable cross-attention heatmap.
6. Final RF-DETR bicycle bbox overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "training_outputs" / "rfdetr_visualization"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BICYCLE_CLASS_NAME = "bicycle"
MIDDLE_LAYER_INDEX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RF-DETR backbone/decoder intermediates for one image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image or directory containing frames.",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help="Frame index when --image is a directory. Defaults to the first image.",
    )
    parser.add_argument("--model-id", default="rfdetr-2xlarge", help="RF-DETR model id.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--channels",
        type=int,
        default=16,
        help="Number of channels to show in the late projector feature grid.",
    )
    parser.add_argument(
        "--middle-layer",
        type=int,
        default=MIDDLE_LAYER_INDEX,
        help="DINOv2 encoder layer index used for the middle feature panel.",
    )
    return parser.parse_args()


def set_inference_env() -> None:
    os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
    os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")


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


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(values.min())
    max_value = float(values.max())
    if max_value > 1e-8:
        values = values / max_value
    return values


def pca_rgb_from_features(features: np.ndarray) -> np.ndarray:
    """Project (N, C) feature vectors to RGB via the top three PCA components."""
    if features.ndim != 2:
        raise RuntimeError(f"Expected NxC features, got shape {features.shape}")

    features = features.astype(np.float64)
    centered = features - features.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3:
        raise RuntimeError(f"Need at least 3 tokens for PCA RGB, got {centered.shape[0]}")

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:3].T

    rgb = np.zeros_like(projected, dtype=np.float32)
    for channel_idx in range(3):
        channel = projected[:, channel_idx]
        low = float(channel.min())
        high = float(channel.max())
        rgb[:, channel_idx] = (channel - low) / (high - low) if high - low > 1e-8 else 0.5
    return rgb


def windowed_tokens_to_pca_map(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim != 3:
        raise RuntimeError(f"Expected WxTxC windowed tokens, got shape {tokens.shape}")

    window_shapes: list[tuple[int, int]] = []
    spatial_tokens: list[np.ndarray] = []
    for window_idx in range(tokens.shape[0]):
        seq = tokens[window_idx, 1:, :]
        side = int(round(np.sqrt(seq.shape[0])))
        if side * side != seq.shape[0]:
            raise RuntimeError(f"Window token count {seq.shape[0]} is not square.")
        window_shapes.append((side, side))
        spatial_tokens.append(seq)

    all_rgb = pca_rgb_from_features(np.concatenate(spatial_tokens, axis=0))
    if len(window_shapes) == 1:
        side = window_shapes[0][0]
        return all_rgb.reshape(side, side, 3)

    grid_size = int(round(np.sqrt(len(window_shapes))))
    if grid_size * grid_size != len(window_shapes):
        raise RuntimeError(f"Expected a square window grid, got {len(window_shapes)} windows.")

    tile_h, tile_w = window_shapes[0]
    stitched = np.zeros((grid_size * tile_h, grid_size * tile_w, 3), dtype=np.float32)
    offset = 0
    for window_idx, (tile_h, tile_w) in enumerate(window_shapes):
        num_tokens = tile_h * tile_w
        window_rgb = all_rgb[offset : offset + num_tokens].reshape(tile_h, tile_w, 3)
        offset += num_tokens
        row = window_idx // grid_size
        col = window_idx % grid_size
        y0 = row * tile_h
        x0 = col * tile_w
        stitched[y0 : y0 + tile_h, x0 : x0 + tile_w] = window_rgb
    return stitched


def save_rgb_map(rgb: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation="bilinear")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_spatial_map(map_2d: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(normalize_array(map_2d), cmap="magma", interpolation="bilinear")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
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


def flat_tokens_to_map(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim == 3:
        tokens = tokens[0]
    if tokens.ndim != 2:
        raise RuntimeError(f"Expected NxC tokens, got shape {tokens.shape}")
    side = int(round(np.sqrt(tokens.shape[0])))
    if side * side != tokens.shape[0]:
        raise RuntimeError(f"Token count {tokens.shape[0]} is not square.")
    return tokens.reshape(side, side, -1).std(axis=-1)


def windowed_tokens_to_map(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim != 3:
        raise RuntimeError(f"Expected WxTxC windowed tokens, got shape {tokens.shape}")

    window_maps: list[np.ndarray] = []
    for window_idx in range(tokens.shape[0]):
        seq = tokens[window_idx, 1:, :]
        side = int(round(np.sqrt(seq.shape[0])))
        if side * side != seq.shape[0]:
            raise RuntimeError(f"Window token count {seq.shape[0]} is not square.")
        window_maps.append(seq.reshape(side, side, -1).std(axis=-1))

    if len(window_maps) == 1:
        return window_maps[0]

    grid_size = int(round(np.sqrt(len(window_maps))))
    if grid_size * grid_size != len(window_maps):
        raise RuntimeError(
            f"Expected a square window grid, got {len(window_maps)} windows."
        )

    tile_h, tile_w = window_maps[0].shape
    stitched = np.zeros((grid_size * tile_h, grid_size * tile_w), dtype=np.float32)
    for window_idx, window_map in enumerate(window_maps):
        row = window_idx // grid_size
        col = window_idx % grid_size
        y0 = row * tile_h
        x0 = col * tile_w
        stitched[y0 : y0 + tile_h, x0 : x0 + tile_w] = window_map
    return stitched


def build_attention_heatmap(
    sampling_locations: np.ndarray,
    attention_weights: np.ndarray,
    spatial_shapes: np.ndarray,
    query_index: int,
) -> np.ndarray:
    height, width = int(spatial_shapes[0, 0]), int(spatial_shapes[0, 1])
    heatmap = np.zeros((height, width), dtype=np.float32)
    loc = sampling_locations[0, query_index]
    weights = attention_weights[0, query_index]
    num_heads, num_levels, num_points, _ = loc.shape
    for head_idx in range(num_heads):
        for level_idx in range(num_levels):
            for point_idx in range(num_points):
                x_norm, y_norm = loc[head_idx, level_idx, point_idx]
                weight = float(weights[head_idx, level_idx * num_points + point_idx])
                x_idx = int(round(float(x_norm) * max(width - 1, 0)))
                y_idx = int(round(float(y_norm) * max(height - 1, 0)))
                heatmap[y_idx, x_idx] += weight
    return normalize_array(heatmap)


def draw_detection_overlay(
    image: Image.Image,
    bbox_xyxy: list[float],
    *,
    score: float,
    label: str,
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 80), width=4)
    caption = f"{label} {score:.2f}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text_y = max(0, y1 - 18)
    draw.rectangle([x1, text_y, x1 + 8 * len(caption), text_y + 16], fill=(0, 180, 60))
    draw.text((x1 + 2, text_y + 1), caption, fill=(0, 0, 0), font=font)
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


class ActivationCapture:
    def __init__(self) -> None:
        self.values: dict[str, np.ndarray] = {}
        self.handles: list[Any] = []

    def _store(self, name: str, tensor: Any) -> None:
        if hasattr(tensor, "detach"):
            array = tensor.detach().float().cpu().numpy()
            if array.ndim >= 1 and array.shape[0] == 1:
                array = array[0]
            self.values[name] = array

    def hook_tensor(self, name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self._store(name, tensor)

        return hook

    def patch_cross_attention(self, cross_attn_module) -> None:
        import torch

        original_forward = cross_attn_module.forward

        def wrapped_forward(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask=None,
        ):
            module = cross_attn_module
            batch_size, num_queries, _ = query.shape
            sampling_offsets = module.sampling_offsets(query).view(
                batch_size,
                num_queries,
                module.n_heads,
                module.n_levels,
                module.n_points,
                2,
            )
            attention_weights = module.attention_weights(query).view(
                batch_size,
                num_queries,
                module.n_heads,
                module.n_levels * module.n_points,
            )
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
                )
                sampling_locations = (
                    reference_points[:, :, None, :, None, :]
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                )
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / module.n_points
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
                )
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
            self.values["sampling_locations"] = sampling_locations.detach().float().cpu().numpy()
            self.values["attention_weights"] = attention_weights.detach().float().cpu().numpy()
            self.values["spatial_shapes"] = input_spatial_shapes.detach().cpu().numpy()
            return original_forward(
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask,
            )

        cross_attn_module.forward = wrapped_forward

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def select_best_bicycle_query(pred_logits: np.ndarray, class_names: list[str]) -> tuple[int, float]:
    import torch

    if BICYCLE_CLASS_NAME not in class_names:
        raise RuntimeError(f"Class '{BICYCLE_CLASS_NAME}' not found in model labels.")
    bicycle_id = class_names.index(BICYCLE_CLASS_NAME)
    scores = torch.from_numpy(pred_logits).sigmoid().numpy()
    query_index = int(scores[:, bicycle_id].argmax())
    return query_index, float(scores[query_index, bicycle_id])


def select_best_bicycle_detection(detections, class_names: list[str]) -> tuple[list[float], float] | tuple[None, None]:
    if len(detections.xyxy) == 0:
        return None, None
    bicycle_id = class_names.index(BICYCLE_CLASS_NAME)
    best_bbox = None
    best_score = -1.0
    for bbox, class_id, score in zip(detections.xyxy, detections.class_id, detections.confidence):
        if int(class_id) != bicycle_id:
            continue
        score_value = float(score)
        if score_value > best_score:
            best_score = score_value
            best_bbox = [float(v) for v in bbox.tolist()]
    return best_bbox, best_score


def load_torch_model(model_id: str, device: str):
    import torch
    from inference_models import AutoModel

    torch_device = torch.device(device)
    return AutoModel.from_pretrained(model_id, device=torch_device, backend="torch")


def main() -> None:
    args = parse_args()
    set_inference_env()

    image_path = resolve_image_path(args.image, args.frame_id)
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise RuntimeError(f"Unsupported image extension: {image_path.suffix}")

    import torch
    from inference_models.models.rfdetr.rfdetr_base_pytorch import nested_tensor_from_tensor_list

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)

    model = load_torch_model(args.model_id, args.device)
    tensor, preprocess_meta = model.pre_process(image_np)
    core = model._model
    backbone = core.backbone[0]

    capture = ActivationCapture()
    capture.handles.append(
        backbone.encoder.encoder.embeddings.patch_embeddings.register_forward_hook(
            capture.hook_tensor("patch_tokens")
        )
    )
    capture.handles.append(
        backbone.encoder.encoder.encoder.layer[args.middle_layer].register_forward_hook(
            capture.hook_tensor("middle_layer")
        )
    )
    capture.handles.append(
        backbone.projector.register_forward_hook(capture.hook_tensor("projector"))
    )
    capture.patch_cross_attention(core.transformer.decoder.layers[-1].cross_attn)

    with torch.inference_mode():
        samples = nested_tensor_from_tensor_list(tensor)
        raw_outputs = core(samples)

    detections = model.post_process(raw_outputs, preprocess_meta, confidence=args.confidence)[0]
    pred_logits = raw_outputs["pred_logits"][0].detach().float().cpu().numpy()
    query_index, query_score = select_best_bicycle_query(pred_logits, model.class_names)
    bicycle_bbox, bicycle_score = select_best_bicycle_detection(detections, model.class_names)
    capture.close()

    patch_map = flat_tokens_to_map(capture.values["patch_tokens"])
    middle_map = windowed_tokens_to_pca_map(capture.values["middle_layer"])
    projector = capture.values["projector"]
    if projector.ndim == 4:
        projector = projector[0]
    attention_map = build_attention_heatmap(
        capture.values["sampling_locations"],
        capture.values["attention_weights"],
        capture.values["spatial_shapes"],
        query_index,
    )

    input_path = args.out_dir / "01_input_frame.png"
    early_path = args.out_dir / "02_early_dinov2_patch_features.png"
    middle_path = args.out_dir / "03_middle_dinov2_layer_features.png"
    late_path = args.out_dir / "04_late_projector_features.png"
    attention_path = args.out_dir / "05_bicycle_query_attention.png"
    overlay_path = args.out_dir / "06_detection_bbox_overlay.png"
    combined_path = args.out_dir / "rfdetr_six_steps.png"
    metadata_path = args.out_dir / "metadata.json"

    image.save(input_path)
    save_spatial_map(patch_map, early_path, "Early DINOv2 patch features")
    save_rgb_map(
        middle_map,
        middle_path,
        f"Middle DINOv2 layer {args.middle_layer} features (PCA RGB)",
    )
    save_feature_grid(
        projector,
        late_path,
        "Late multi-scale projector features",
        channels=args.channels,
    )
    save_spatial_map(
        attention_map,
        attention_path,
        f"Bicycle query cross-attention (query {query_index}, score {query_score:.2f})",
    )

    if bicycle_bbox is not None and bicycle_score is not None:
        draw_detection_overlay(
            image,
            bicycle_bbox,
            score=bicycle_score,
            label=BICYCLE_CLASS_NAME,
        ).save(overlay_path)
    else:
        image.save(overlay_path)

    save_combined_panel(
        [input_path, early_path, middle_path, late_path, attention_path, overlay_path],
        [
            "1. Input frame",
            "2. Early DINOv2 patches",
            "3. Middle DINOv2 layer",
            "4. Late projector",
            "5. Bicycle attention",
            "6. RF-DETR bbox",
        ],
        combined_path,
    )

    metadata = {
        "image_path": str(image_path),
        "model_id": args.model_id,
        "device": args.device,
        "confidence_threshold": args.confidence,
        "bicycle_query_index": query_index,
        "bicycle_query_score": query_score,
        "bicycle_bbox_xyxy": bicycle_bbox,
        "bicycle_detection_score": bicycle_score,
        "captured_tensors": {name: list(value.shape) for name, value in capture.values.items()},
        "outputs": {
            "input_frame": str(input_path),
            "early_patch_features": str(early_path),
            "middle_layer_features": str(middle_path),
            "late_projector_features": str(late_path),
            "bicycle_query_attention": str(attention_path),
            "detection_overlay": str(overlay_path),
            "combined_panel": str(combined_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[rfdetr-vis] Wrote:")
    for output in metadata["outputs"].values():
        print(f"  {output}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()

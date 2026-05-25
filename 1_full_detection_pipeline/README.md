# Full image-sequence → 3D keypoint pipeline

Runs three stages on a **243-frame** PNG sequence:

1. **RF-DETR** (`conda` env `rfdetr`) — bicycle bounding boxes  
2. **RTMPose bicycle** (`mmpose`) — 18 2D keypoints per frame (RF-DETR bbox + top-down warp)  
3. **PoseMamba** (`posemamba`) — root-relative camera 3D sequence  

Final artifact: **`output/keypoints_3d.npz`** with array `pred` of shape **(243, 18, 3)**.

## Prerequisites

| Env | Setup |
|-----|--------|
| `rfdetr` | [Bounding box detector/environment_setup.md](../Bounding%20box%20detector/environment_setup.md) |
| `mmpose` | [2d_keypoint_detector_training/install_mmpose_env.sh](../2d_keypoint_detector_training/install_mmpose_env.sh) |
| `posemamba` | [PoseMamba/README.md](../PoseMamba/README.md) + selective_scan kernel |

Checkpoints (local paths, not in git):

- `training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_175.pth.pth` (or your best `.pth`)
- `checkpoints/posemamba_bicycle_<timestamp>/best_epoch.bin` (+ `config.yaml` in same dir)

**Lifter checkpoint:** train on **detected-2D** pickles (`PoseMamba_f243s81_detected2d`), not the older oracle GT-2D corpus (`PoseMamba_f243s81`). See [`3d_keypoint_detector_training/README.md`](../3d_keypoint_detector_training/README.md). Build data with `generate_blender_posemamba_dataset.sh`, then `DATASET_TAG=detected2d ./3d_keypoint_detector_training/start_training.sh`.

## Quick start

```bash
cd /home/hayepc/3D-bicycle-pose-estimation
chmod +x 1_full_detection_pipeline/run_full_pipeline.sh

./1_full_detection_pipeline/run_full_pipeline.sh \
  --frames-dir 1_full_detection_pipeline/input_sequence \
  --output-dir 1_full_detection_pipeline/output \
  --mmpose-checkpoint training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_175.pth \
  --lifter-checkpoint checkpoints/posemamba_bicycle_2026_05_19_T_06_11_25/best_epoch.bin
```

Use `--resume` to skip stages whose outputs already exist.

Use `--no-visualize` to skip rendering. Use `--no-video` for frame PNGs only.

## Intermediate visualizations

After inference (enabled by default), `visualize_intermediates.py` writes:

| Stage | Frames | Video |
|-------|--------|-------|
| Bounding boxes | `output/vis/detections/frames/` | `output/vis/detections/detections.mp4` |
| 2D keypoints + skeleton | `output/vis/keypoints_2d/frames/` | `output/vis/keypoints_2d/keypoints_2d.mp4` |
| 3D lift (matplotlib) | `output/vis/keypoints_3d/frame_*.png` | `output/vis/keypoints_3d.mp4` |

Re-render without re-running models:

```bash
conda run -n mmpose python 1_full_detection_pipeline/visualize_intermediates.py \
  --output-dir 1_full_detection_pipeline/output --stages detections 2d

conda run -n posemamba python 1_full_detection_pipeline/visualize_intermediates.py \
  --output-dir 1_full_detection_pipeline/output --stages 3d
```

3D viz uses `--reorient camera_up` so the bicycle appears upright in the matplotlib view.

## Output layout

```
1_full_detection_pipeline/output/
├── detections.jsonl       # stage 1 (debug)
├── keypoints_2d.jsonl     # stage 2 (debug)
├── keypoints_3d.npz       # final: pred (243, 18, 3)
└── vis/                   # intermediate visualizations (optional)
    ├── detections/
    ├── keypoints_2d/
    └── keypoints_3d/
```

### `keypoints_3d.npz` keys

| Key | Shape | Description |
|-----|-------|-------------|
| `pred` | (243, 18, 3) | Root-relative camera 3D; joint 0 (`k_bottom_bracket`) ≈ 0 |
| `data_input` | (243, 18, 2) | Bbox-normalized 2D fed to the lifter |
| `frame_ids` | (243,) | Frame indices from detection order |
| `keypoint_names` | (18,) | Joint names (schema order) |

## Stage 2 pose modes

`run_full_pipeline.sh` and `stage2_pose2d.py` default to **`--pose-mode detection_bbox`**:

| Mode | Behavior |
|------|----------|
| `detection_bbox` | RF-DETR bbox on the full frame; RTMPose via `inference_topdown` (same `GetBBoxCenterScale` + `TopdownAffine` as training). Stage 3 uses the detection bbox. |
| `full_image` | MMPose on the full frame without a detector bbox (like `infer_2d.py`). Stage 3 bbox derived from keypoints. |
| `auto` | `detection_bbox` only when bbox area ≥ `--min-det-bbox-area-frac` (default 1%). |

Override in the shell script: `--pose-mode full_image`.

## Data contracts

- **Frame order:** sorted glob `*.png` in `--frames-dir` (`0001.png` … `0243.png`).
- **v1 length:** exactly **243** frames (matches PoseMamba `clip_len`).
- **2D normalization (stage 3):** per-frame bbox center + `max(1, max(w,h))` scale — same as training (`build_sequences.py` / `sequence_builder.py`).

## Run stages individually

```bash
conda run -n rfdetr python 1_full_detection_pipeline/stage1_detect.py \
  --frames-dir 1_full_detection_pipeline/input_sequence \
  --output-dir 1_full_detection_pipeline/output

conda run -n mmpose python 1_full_detection_pipeline/stage2_pose2d.py \
  --frames-dir 1_full_detection_pipeline/input_sequence \
  --output-dir 1_full_detection_pipeline/output \
  --mmpose-config 2d_keypoint_detector_training/rtmpose_bicycle_full.py \
  --mmpose-checkpoint training_outputs/mmpose_bicycle_rtmpose_l_gpu/epoch_340.pth \
  --pose-mode detection_bbox

conda run -n posemamba python 1_full_detection_pipeline/stage3_lift3d.py \
  --frames-dir 1_full_detection_pipeline/input_sequence \
  --output-dir 1_full_detection_pipeline/output \
  --lifter-checkpoint checkpoints/posemamba_bicycle_2026_05_19_T_06_11_25/best_epoch.bin
```

## Related code

- Shared lifting: [`3d_keypoint_detector_training/lift_from_2d_array.py`](../3d_keypoint_detector_training/lift_from_2d_array.py)
- Single-process alternative (one env): [`keypoint_detector_pipeline/run_inference.py`](../keypoint_detector_pipeline/run_inference.py) — not recommended for split conda stacks

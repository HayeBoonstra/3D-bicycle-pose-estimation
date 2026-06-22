# Bicycle PoseMamba 3D lifting pipeline

Synthetic bicycle data → PoseMamba training → inference on **bbox-normalized 2D** (`data_input`).

**Training always uses `data_input`.** The old `gt_2d` oracle path is disabled for `BICYCLE` in `MotionDataset3D` regardless of YAML flags.

## Two training corpora

| Corpus | Folder | `data_input` source | Normalization bbox | Use case |
|--------|--------|---------------------|--------------------|----------|
| **GT 2D (oracle)** | `PoseMamba_f243s81/` | Projected GT from `keypoints_2d_frame_*.json` | `gt_bbox_xywh` | High val MPJPE; **not** matched to real RTMPose |
| **Detected 2D** | `PoseMamba_f243s81_detected2d/` | RTMPose `detection_bbox` sidecars (RF-DETR + top-down) | **RF-DETR** `det_bbox_xyxy` | **Use for** [`1_full_detection_pipeline`](../1_full_detection_pipeline/) and Swapfiets |

Manifest fields: `input_2d_source` (`gt_projection`, `rtmpose_detection_bbox`, `rtmpose_full_image`, `rtmpose_keypoint_bbox`), `bbox_source` (`gt`, `detection`, `keypoints`).

Do **not** resume an oracle-GT-2D checkpoint when training on detected-2D pickles.

## Data flow (detected 2D — recommended for deployment)

```
generate_blender_posemamba_dataset.sh   →  data/raw_blender_posemamba/ (≥729 frames/clip)
export_clip_detections.py (rfdetr)        →  {clip}/detections.jsonl
export_detected_2d.py (mmpose)            →  keypoints_2d_detected_frame_*.json
build_sequences.py --input-2d detected    →  PoseMamba_f243s81_detected2d/BICYCLE/{train,val,test}/*.pkl
DATASET_TAG=detected2d start_training.sh →  posemamba_weights/run_NNN/
```

MuJoCo-only raw clips (no `frames/*.png`) cannot run RTMPose export; use Blender renders or the dev smoke script below.

## Data flow (GT 2D — legacy / ablation)

```
generate_mujoco_direct_dataset.sh  →  raw clips (FRAMES≥729 recommended)
build_sequences.py                 →  PoseMamba_f243s81/BICYCLE/{train,val,test}/*.pkl
start_training.sh                  →  posemamba_weights/run_NNN/
```

Each `.pkl`: **T=243**, **J=18**, `data_input` (2D), `data_label` (camera 3D GT).

## Blender long-sequence dataset

Large outputs can live on a secondary SSD; see [`data/README.md`](../data/README.md).

```bash
cd /home/hayepc/3D-bicycle-pose-estimation
bash data_generation_pipeline_tools/setup_secondary_data_disk.sh   # once: SSD + symlinks
chmod +x data_generation_pipeline_tools/generate_blender_posemamba_dataset.sh

# Defaults: auto-use /mnt/SmallSSD when mounted, RTMPose export, detected2d pickles
NUM_CLIPS=20 SYNC_WINDOW_SIZE=729 bash data_generation_pipeline_tools/generate_blender_posemamba_dataset.sh

# Train on detected-2D corpus
DATASET_TAG=detected2d ./3d_keypoint_detector_training/start_training.sh
```

Environment variables: `RAW_ROOT`, `SEQUENCE_ROOT`, `SKIP_RENDER`, `SKIP_DETECTION`, `SKIP_DETECTED_2D`, `RFDETR_MODEL`, `DET_CONFIDENCE`, `POSE_MODE`, `MMPOSE_CHECKPOINT`, etc. (see script header).

**Camera framing (RTMPose):** Default `CAMERA_MODE=track` — the camera follows the bicycle each frame (offset sampled once: distance 4–12 m, bbox ≥ 4% of image). This avoids empty frames when the bike crosses a large scene. Labels stay in **per-frame camera space** (compatible with root-relative PoseMamba). Legacy `CAMERA_MODE=fixed` keeps a world-fixed camera. Post-render QA checks `gt_bbox_xywh` area in `qa_raw_annotations.py`.

Rebuild pickles only (after RF-DETR + RTMPose export):

```bash
python 3d_keypoint_detector_training/build_sequences.py \
  --raw-root data/raw_blender_posemamba \
  --output-root data/posemamba_training_sequences \
  --input-2d detected --bbox-source detection --dataset-tag detected2d
```

## Dev smoke test (one clip, viz PNGs as frames)

```bash
./3d_keypoint_detector_training/smoke_detected2d_corpus.sh
```

## Retrain from scratch

```bash
# Detected-2D (matches full pipeline)
DATASET_TAG=detected2d ./3d_keypoint_detector_training/start_training.sh

# Oracle GT-2D ablation
./3d_keypoint_detector_training/start_training.sh
```

Optional: `NOISE_2D=1` for extra 2D noise during training (after detected-2D baseline works).

Checkpoints: `posemamba_weights/run_NNN/` (auto) or `posemamba_weights/<name>/` with `EXPERIMENT_NAME=<name>`. Use `best_epoch.bin` for inference.

## Eval and inference

```bash
python 3d_keypoint_detector_training/eval_lifter.py \
  --checkpoint posemamba_weights/run_001/best_epoch.bin

python 3d_keypoint_detector_training/visualize_lifter_clip_video.py \
  --checkpoint posemamba_weights/run_001/best_epoch.bin \
  --input-dir /home/hayepc/3D-bicycle-pose-estimation/data/validation_input/ \
  --clip-id clip_evening_street_scene_1973818957_000043 \
  --out training_outputs/lifter_clip_viz
```

Point `--input-dir` at the corpus you trained on (`PoseMamba_f243s81` vs `PoseMamba_f243s81_detected2d`).

Shared I/O: [`posemamba_bicycle_io.py`](posemamba_bicycle_io.py).

## QA

| Script | Purpose |
|--------|---------|
| `qa_raw_annotations.py` | GT contract, reprojection |
| `qa_detected_2d.py` | GT vs RTMPose pixel RMSE per clip |
| `build_sequences.py` | Writes `dataset_manifest.json` with `input_2d_source` / `bbox_source` |

## Video visualization (stride 81)

| Goal | Tool | Output |
|------|------|--------|
| One window (243 frames) | `3D_lifting_inference.py --video` | `{stem}_vis/*.mp4` |
| Full clip (~729 frames) | `visualize_lifter_clip_video.py` | `{clip_id}_full_clip_vis_{split,overlay}/*.mp4` |

Test MPJPE during training is on `BICYCLE/test` pickles (same `data_input` as inference).

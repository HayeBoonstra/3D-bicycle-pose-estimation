# Bicycle PoseMamba 3D lifting pipeline

Synthetic bicycle data → PoseMamba training → inference on **bbox-normalized image 2D** (`data_input`), matching RTMPose/YOLO-style inputs.

## Data flow

```
generate_mujoco_direct_dataset.sh  →  raw clips (FRAMES≥729 recommended)
build_sequences.py                 →  PoseMamba_f243s81/BICYCLE/{train,val,test}/*.pkl
train_lifter.py                    →  checkpoints/posemamba_bicycle/ (or posemamba_gpu_run_*)
3D_lifting_inference.py            →  training_outputs/inference_3d/
```

Each `.pkl`: **T=243**, **J=18**, `data_input` (image 2D), `data_label` (camera 3D GT).

Windowing: PoseMamba [`split_clips`](../../PoseMamba/lib/utils/utils_data.py) (contiguous windows). See `data/posemamba_training_sequences/PoseMamba_f243s81/dataset_manifest.json`.

## Commands

```bash
# Regenerate data (export + build_sequences)
bash data_generation_pipeline_tools/generate_mujoco_direct_dataset.sh

# Train (optional detector noise)
./3d_keypoint_detector_training/start_training.sh
NOISE_2D=1 ./3d_keypoint_detector_training/start_training.sh

# Eval on test split (runs each epoch during training too)
python 3d_keypoint_detector_training/eval_lifter.py \
  --checkpoint checkpoints/posemamba_gpu_run_2026_05_18_T_17_22_57/best_epoch.bin

# Inference — set INPUT_2D_MODE = image_2d in 3D_lifting_inference.py
python 3d_keypoint_detector_training/3D_lifting_inference.py --video
```

Shared I/O: [`posemamba_bicycle_io.py`](posemamba_bicycle_io.py).

## Video visualization (stride 81)

Stride 81 does **not** mean 3-frame videos. Each `.pkl` is still **243 frames** (one model window). A ~729-frame source clip becomes **7** overlapping train windows (or **3** val windows with stride 243).

| Goal | Tool | Output |
|------|------|--------|
| One window (243 frames) | `3D_lifting_inference.py --video` on a single `.pkl` | `{stem}_vis/*.mp4` |
| Full source clip (~729 frames) | `visualize_lifter_clip_video.py` (stitch windows) | `{clip_id}_full_clip_vis_{split,overlay}/*.mp4` |

Full-clip example (all windows for one camera, overlap-averaged). By default renders **split** (pred | GT) and **overlay** (same axes), upright via `camera_up` reorient, MPJPE in title + `summary.json`:

```bash
python 3d_keypoint_detector_training/visualize_lifter_clip_video.py \
  --input-dir data/posemamba_training_sequences/PoseMamba_f243s81/BICYCLE/val \
  --clip-id PoseMamba_left_traj0000_cam06 \
  --out training_outputs/lifter_clip_viz \
  --layout both   # split | overlay | both (default)
```

You do **not** need a separate dataset for viz — use existing pickles and stitch by `meta.clip_id` + `meta.st`.

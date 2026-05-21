# Bicycle PoseMamba 3D lifting pipeline

Synthetic bicycle data → PoseMamba training → inference on **bbox-normalized image 2D** (`data_input`), matching RTMPose/YOLO-style inputs.

**Training always uses `data_input`.** The old `gt_2d` oracle path is disabled for `BICYCLE` in `MotionDataset3D` regardless of YAML flags.

## Data flow

```
generate_mujoco_direct_dataset.sh  →  raw clips (FRAMES≥729 recommended)
build_sequences.py                 →  PoseMamba_f243s81/BICYCLE/{train,val,test}/*.pkl
train_lifter.py / start_training.sh →  checkpoints/posemamba_bicycle_<timestamp>/
3D_lifting_inference.py            →  training_outputs/inference_3d/
```

Each `.pkl`: **T=243**, **J=18**, `data_input` (image 2D), `data_label` (camera 3D GT).

## Retrain from scratch (image 2D)

Do **not** resume from `posemamba_gpu_run_2026_05_18_*` — that run used oracle camera 2D (`gt_2d: true`).

```bash
cd /home/hayepc/3D-bicycle-pose-estimation

# Optional: regenerate sequences if data changed
bash data_generation_pipeline_tools/generate_mujoco_direct_dataset.sh

# Fresh training (conda env posemamba)
./3d_keypoint_detector_training/start_training.sh
```

Optional flags:

```bash
NOISE_2D=1 ./3d_keypoint_detector_training/start_training.sh   # detector noise aug
EPOCHS=120 BATCH_SIZE=5 DIM_FEAT=64 ./3d_keypoint_detector_training/start_training.sh
```

Checkpoints land under `checkpoints/posemamba_bicycle_<YYYY_MM_DD_T_HH_MM_SS>/`. Use `best_epoch.bin` for inference.

## Eval and inference

```bash
python 3d_keypoint_detector_training/eval_lifter.py \
  --checkpoint checkpoints/posemamba_bicycle_<run>/best_epoch.bin

python 3d_keypoint_detector_training/visualize_lifter_clip_video.py \
  --checkpoint checkpoints/posemamba_bicycle_<run>/best_epoch.bin \
  --input-dir data/posemamba_training_sequences/PoseMamba_f243s81/BICYCLE/val \
  --clip-id PoseMamba_left_traj0000_cam06 \
  --out training_outputs/lifter_clip_viz
```

Shared I/O: [`posemamba_bicycle_io.py`](posemamba_bicycle_io.py).

## Video visualization (stride 81)

| Goal | Tool | Output |
|------|------|--------|
| One window (243 frames) | `3D_lifting_inference.py --video` | `{stem}_vis/*.mp4` |
| Full clip (~729 frames) | `visualize_lifter_clip_video.py` | `{clip_id}_full_clip_vis_{split,overlay}/*.mp4` |

Test MPJPE during training is on `BICYCLE/test` pickles (same `data_input` as inference).

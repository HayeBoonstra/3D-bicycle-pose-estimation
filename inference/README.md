# Full image-sequence → 3D keypoint pipeline

Runs four stages on a **243-frame** PNG sequence:

1. **RF-DETR** (`conda` env `rfdetr`) — bicycle bounding boxes  
2. **RTMPose bicycle** (`mmpose`) — 18 2D keypoints per frame  
3. **PoseMamba** (`posemamba`) — root-relative camera 3D sequence  
4. **Kinematics** — roll and steer from 3D keypoints (`common/bicycle_dynamics_angles.py`)

Final artifacts: **`output/keypoints_3d.npz`**, **`output/dynamics.npz`**.

## Prerequisites

| Env | Setup |
|-----|--------|
| `rfdetr` | [training/bbox/environment_setup.md](../training/bbox/environment_setup.md) |
| `mmpose` | [training/pose2d/install_mmpose_env.sh](../training/pose2d/install_mmpose_env.sh) |
| `posemamba` | [third_party/PoseMamba/README.md](../third_party/PoseMamba/README.md) + selective_scan kernel |

Checkpoints: see [WEIGHTS.md](../WEIGHTS.md). Train the lifter on **detected-2D** pickles (`PoseMamba_f243s81_detected2d`).

## Quick start

```bash
cd /path/to/3D-bicycle-pose-estimation
chmod +x inference/run_pipeline.sh

./inference/run_pipeline.sh \
  --frames-dir inference/input_sequence \
  --output-dir inference/output \
  --mmpose-checkpoint training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_175.pth \
  --lifter-checkpoint posemamba_weights/run_001/best_epoch.bin
```

Use `--resume`, `--no-visualize`, or `--no-video` as needed.

## Output layout

```
inference/output/
├── detections.jsonl
├── keypoints_2d.jsonl
├── keypoints_3d.npz
├── dynamics.npz          # steer, roll (rad)
├── dynamics.json
└── vis/                  # optional
```

Shared library code lives in `inference/lib/` (detection, 2D pose, schema, ROI helpers).

## Related code

- Lifting: [`training/pose3d/lift_from_2d_array.py`](../training/pose3d/lift_from_2d_array.py)
- Training data build: [`training/pose3d/build_sequences.py`](../training/pose3d/build_sequences.py)

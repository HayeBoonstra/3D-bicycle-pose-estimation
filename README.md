# 3D bicycle pose estimation

Monocular RGB video (static camera) → **18 semantic 3D keypoints** → **roll / steer** kinematics, using a synthetic-data-trained **PoseMamba** (SSM) lifter with **RF-DETR** + **RTMPose** at inference.

**Research target:** MPJPE &lt; 40 mm on BICYCLE test clips with deployment-matched detected 2D input.

## Repository map

| Path | Purpose |
|------|---------|
| [`synthetic_data/`](synthetic_data/README.md) | MuJoCo + Blender dataset generation |
| [`training/pose2d/`](training/pose2d/) | Bicycle RTMPose (MMPose) training |
| [`training/pose3d/`](training/pose3d/README.md) | PoseMamba lifter training + pickle build |
| [`training/bbox/`](training/bbox/) | RF-DETR conda setup notes |
| [`inference/`](inference/README.md) | End-to-end RGB → 3D → dynamics |
| [`third_party/PoseMamba/`](third_party/PoseMamba/) | Vendored lifter + [bicycle patches](third_party/PATCHES.md) |
| [`common/`](common/) | J=18 schema and dynamics angle helpers |
| [`evaluation/`](evaluation/) | Metrics and figure scripts |
| [`experiments/`](experiments/) | Ablation configs and runners |

Weights and corpora are **not** committed — see [`WEIGHTS.md`](WEIGHTS.md) and [`data/README.md`](data/README.md).

## Conda environments

Three envs match the inference stack:

1. **`rfdetr`** — bbox detection  
2. **`mmpose`** — 2D keypoints  
3. **`posemamba`** — 3D lifting + training  

Setup: `training/bbox/environment_setup.md`, `training/pose2d/install_mmpose_env.sh`, `training/pose3d/posemamba_env.sh`.

## Quick start

**Train 3D lifter (detected-2D corpus):**

```bash
bash training/pose3d/start_training.sh
```

**Run inference on a frame sequence (243 frames):**

```bash
chmod +x inference/run_pipeline.sh
./inference/run_pipeline.sh --frames-dir inference/input_sequence --output-dir inference/output
```

Outputs: `keypoints_3d.npz`, `dynamics.npz` (roll/steer), optional `vis/` when visualization is enabled.

**Generate synthetic training data:**

```bash
bash synthetic_data/pipeline/generate_blender_posemamba_dataset.sh
```

## Tests

```bash
PYTHONPATH=. pytest tests/ -q
```

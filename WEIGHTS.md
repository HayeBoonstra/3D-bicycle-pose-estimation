# Model weights (not in git)

Place checkpoints locally or on shared storage; paths below are defaults used by scripts.

| Component | Source | Typical path |
|-----------|--------|----------------|
| RF-DETR 2x-large | Roboflow / `inference` env | Model id `rfdetr-2xlarge` (downloaded on first use) |
| RTMPose bicycle (J=18) | Train with `training/pose2d/start_training.sh` | `training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_*.pth` |
| PoseMamba lifter | Train with `training/pose3d/start_training.sh` | `posemamba_weights/run_NNN/best_epoch.bin` |

Inference overrides: `inference/run_pipeline.sh --mmpose-checkpoint … --lifter-checkpoint …`

Synthetic training pickles: `data/posemamba_training_sequences/PoseMamba_f243s81_detected2d/` (see `data/README.md` for SSD symlinks).

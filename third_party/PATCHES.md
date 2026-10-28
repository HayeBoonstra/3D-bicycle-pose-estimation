# PoseMamba bicycle patches

Upstream [PoseMamba](https://github.com/...) is vendored under `third_party/PoseMamba/`. Bicycle-specific changes are limited to:

| File | Change |
|------|--------|
| `lib/data/dataset_motion_3d.py` | `BICYCLE` subset; always uses pickle `data_input` (detected 2D); optional 2D noise via `training/pose3d/posemamba_bicycle_io.py` |
| `lib/model/loss.py` | J=18 limb graph from `common.bicycle_keypoint_schema`; steer/roll dynamics losses |
| `train.py` | Dynamics loss weights (`lambda_steer`, `lambda_roll`, …) and eval Protocol #3–#6 |
| `lib/utils/utils_data.py` | Bicycle left/right flip pairs when `J >= 18` |

Build the selective-scan kernel before training:

```bash
bash training/pose3d/build_posemamba_kernel.sh
```

Training entry point: `training/pose3d/start_training.sh` (invokes `train.py` in this tree).

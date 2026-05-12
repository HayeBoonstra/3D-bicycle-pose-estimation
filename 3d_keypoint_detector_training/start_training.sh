#!/usr/bin/env bash
set -euo pipefail

# Keep PoseMamba-specific build variables from leaking into runtime.
unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX

BATCH_SIZE="${BATCH_SIZE:-64}"

RUN_ARGS=(
  /home/hayepc/3D-bicycle-pose-estimation/3d_keypoint_detector_training/train_lifter.py
  --conda-env posemamba
  --posemamba-root /home/hayepc/3D-bicycle-pose-estimation/PoseMamba
  --sequence-root /home/hayepc/3D-bicycle-pose-estimation/data/posemamba_training_sequences
  --window-size 27
  --stride 1
  --batch-size "${BATCH_SIZE}"
  --checkpoint-dir /home/hayepc/3D-bicycle-pose-estimation/checkpoints/posemamba_gpu_run
  --gt-2d
  --epochs 40
)

if [[ "${CONDA_DEFAULT_ENV:-}" == "posemamba" ]]; then
  python "${RUN_ARGS[@]}"
else
  conda run -n posemamba python "${RUN_ARGS[@]}"
fi


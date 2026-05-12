#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hayepc/3D-bicycle-pose-estimation}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/raw_mujoco_direct_smoke}"
SEQUENCE_ROOT="${SEQUENCE_ROOT:-${REPO_ROOT}/data/posemamba_sequences_mujoco_direct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/posemamba_mujoco_direct_smoke}"
WINDOW_SIZE="${WINDOW_SIZE:-27}"
STRIDE="${STRIDE:-1}"

"${REPO_ROOT}/3d_keypoint_detector_training/generate_mujoco_direct_dataset.sh"

RUN_ARGS=(
  "${REPO_ROOT}/3d_keypoint_detector_training/train_lifter.py"
  --conda-env "${CONDA_ENV}"
  --posemamba-root "${REPO_ROOT}/PoseMamba"
  --sequence-root "${SEQUENCE_ROOT}"
  --window-size "${WINDOW_SIZE}"
  --stride "${STRIDE}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --generated-config "${CHECKPOINT_DIR}/PoseMamba_train_mujoco_direct_smoke.generated.yaml"
  --epochs 1
  --batch-size 4
  --max-batches 1
  --max-eval-batches 1
)

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  python "${RUN_ARGS[@]}"
else
  conda run -n "${CONDA_ENV}" python "${RUN_ARGS[@]}"
fi

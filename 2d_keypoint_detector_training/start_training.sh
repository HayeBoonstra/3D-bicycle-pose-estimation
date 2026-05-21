#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hayepc/3D-bicycle-pose-estimation}"
MMPPOSE_ENV="${REPO_ROOT}/2d_keypoint_detector_training/mmpose_env.sh"
CONFIG="${REPO_ROOT}/2d_keypoint_detector_training/rtmpose_bicycle_full.py"
WORK_DIR="${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${CONDA_DEFAULT_ENV:-}" == "mmpose" ]]; then
  unset _PYTHON_SYSCONFIGDATA_NAME
  unset CC CXX CUDAHOSTCXX
  mim train mmpose "${CONFIG}" --work-dir "${WORK_DIR}" --launcher none "$@"
else
  "${MMPPOSE_ENV}" env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    mim train mmpose "${CONFIG}" --work-dir "${WORK_DIR}" --launcher none "$@"
fi

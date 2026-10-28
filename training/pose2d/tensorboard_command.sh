#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hayepc/3D-bicycle-pose-estimation}"
LOGDIR="${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu"

unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX

if [[ "${CONDA_DEFAULT_ENV:-}" == "mmpose" ]]; then
  tensorboard --logdir "${LOGDIR}" --port 6006
else
  conda run -n mmpose tensorboard --logdir "${LOGDIR}" --port 6006
fi

#!/usr/bin/env bash
set -euo pipefail

unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX

REPO_ROOT="${REPO_ROOT:-/home/hayepc/3D-bicycle-pose-estimation}"
WINDOW_SIZE="${WINDOW_SIZE:-243}"
STRIDE="${STRIDE:-81}"
BATCH_SIZE="${BATCH_SIZE:-5}"
EPOCHS="${EPOCHS:-120}"
DIM_FEAT="${DIM_FEAT:-64}"
CHECKPOINT_FREQUENCY="${CHECKPOINT_FREQUENCY:-10}"
NOISE_2D="${NOISE_2D:-0}"

RUN_ARGS=(
  "${REPO_ROOT}/3d_keypoint_detector_training/train_lifter.py"
  --conda-env posemamba
  --posemamba-root "${REPO_ROOT}/PoseMamba"
  --sequence-root "${REPO_ROOT}/data/posemamba_training_sequences"
  --window-size "${WINDOW_SIZE}"
  --stride "${STRIDE}"
  --batch-size "${BATCH_SIZE}"
  --dim-feat "${DIM_FEAT}"
  --checkpoint-frequency "${CHECKPOINT_FREQUENCY}"
  --checkpoint-dir "${REPO_ROOT}/checkpoints/posemamba_bicycle"
  --epochs "${EPOCHS}"
)

if [[ "${NOISE_2D}" == "1" ]]; then
  RUN_ARGS+=(--noise-2d)
fi

RESUME_FROM="${1:-${RESUME_CHECKPOINT:-}}"
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ "${RESUME_FROM}" == *.sh ]]; then
    echo "error: first argument must be a .bin checkpoint, not a script: ${RESUME_FROM}" >&2
    echo "  Usage: ./3d_keypoint_detector_training/start_training.sh" >&2
    echo "  Or:    ./3d_keypoint_detector_training/start_training.sh checkpoints/posemamba_bicycle/<run>/latest_epoch.bin" >&2
    exit 1
  fi
  RUN_ARGS+=(--resume "${RESUME_FROM}")
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "posemamba" ]]; then
  python "${RUN_ARGS[@]}"
else
  conda run -n posemamba python "${RUN_ARGS[@]}"
fi

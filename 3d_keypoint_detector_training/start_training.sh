#!/usr/bin/env bash
set -euo pipefail

unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX

REPO_ROOT="${REPO_ROOT:-/home/hayepc/3D-bicycle-pose-estimation}"
WINDOW_SIZE="${WINDOW_SIZE:-243}"
STRIDE="${STRIDE:-81}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-120}"
DIM_FEAT="${DIM_FEAT:-64}"
# YAML depth: each unit is one spatial + one temporal block; paper table N = 2 * DEPTH.
# Presets: S/B -> 10 (N=20), L/X -> 20 (N=40). Override DEPTH explicitly if needed.
DEPTH="${DEPTH:-10}"
CHECKPOINT_FREQUENCY="${CHECKPOINT_FREQUENCY:-10}"
NOISE_2D="${NOISE_2D:-0}"
DATASET_TAG="${DATASET_TAG:-detected2d}"
LAMBDA_STEER="${LAMBDA_STEER:-0.0}"
LAMBDA_STEER_VELOCITY="${LAMBDA_STEER_VELOCITY:-0.0}"
LAMBDA_ROLL="${LAMBDA_ROLL:-0.0}"
LAMBDA_ROLL_VELOCITY="${LAMBDA_ROLL_VELOCITY:-0.0}"
# Optional: EXPERIMENT_NAME=my_ablation -> posemamba_weights/my_ablation/ (default: run_001, run_002, ...)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

export CUDA_VISIBLE_DEVICES=0

RUN_ARGS=(
  "${REPO_ROOT}/3d_keypoint_detector_training/train_lifter.py"
  --conda-env posemamba
  --posemamba-root "${REPO_ROOT}/PoseMamba"
  --sequence-root "${REPO_ROOT}/data/posemamba_training_sequences"
  --window-size "${WINDOW_SIZE}"
  --stride "${STRIDE}"
  --batch-size "${BATCH_SIZE}"
  --dim-feat "${DIM_FEAT}"
  --depth "${DEPTH}"
  --checkpoint-frequency "${CHECKPOINT_FREQUENCY}"
  --checkpoint-dir "${REPO_ROOT}/posemamba_weights"
  --epochs "${EPOCHS}"
  --lambda-steer "${LAMBDA_STEER}"
  --lambda-steer-velocity "${LAMBDA_STEER_VELOCITY}"
  --lambda-roll "${LAMBDA_ROLL}"
  --lambda-roll-velocity "${LAMBDA_ROLL_VELOCITY}"
)

if [[ "${NOISE_2D}" == "1" ]]; then
  RUN_ARGS+=(--noise-2d)
fi
if [[ -n "${DATASET_TAG}" ]]; then
  RUN_ARGS+=(--dataset-tag "${DATASET_TAG}")
fi
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  RUN_ARGS+=(--experiment-name "${EXPERIMENT_NAME}")
fi

echo "[train] model: dim_feat=${DIM_FEAT} depth=${DEPTH} (paper N=$((DEPTH * 2)) STE+TTE blocks)" >&2
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  echo "[train] experiment: ${EXPERIMENT_NAME} -> posemamba_weights/${EXPERIMENT_NAME}/" >&2
else
  echo "[train] experiment: (auto) next posemamba_weights/run_NNN/" >&2
fi
echo "[train] BICYCLE training uses data_input only (gt_2d disabled in dataset)." >&2
if [[ -n "${DATASET_TAG}" ]]; then
  echo "[train] dataset tag: ${DATASET_TAG} (e.g. detected2d = RTMPose 2D + GT bbox)" >&2
fi
echo "[train] For a clean image-2D model, start without resuming old posemamba_gpu_run_* checkpoints." >&2

RESUME_FROM="${1:-${RESUME_CHECKPOINT:-}}"
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ "${RESUME_FROM}" == *.sh ]]; then
    echo "error: first argument must be a .bin checkpoint, not a script: ${RESUME_FROM}" >&2
    echo "  Usage: ./3d_keypoint_detector_training/start_training.sh" >&2
    echo "  Or:    ./3d_keypoint_detector_training/start_training.sh posemamba_weights/<run>/latest_epoch.bin" >&2
    echo "  Named: EXPERIMENT_NAME=posemamba_b_dim128 ./3d_keypoint_detector_training/start_training.sh" >&2
    exit 1
  fi
  RUN_ARGS+=(--resume "${RESUME_FROM}")
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "posemamba" ]]; then
  python "${RUN_ARGS[@]}"
else
  conda run -n posemamba python "${RUN_ARGS[@]}"
fi

#!/usr/bin/env bash
# Optional thin wrapper (generate already runs build_sequences).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STAGE="${STAGE:-build}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/raw_3D_keypoint_annotations}"
SEQUENCE_ROOT="${SEQUENCE_ROOT:-${REPO_ROOT}/data/posemamba_training_sequences}"
WINDOW_SIZE="${WINDOW_SIZE:-243}"
STRIDE="${STRIDE:-81}"
FRAMES="${FRAMES:-729}"

case "${STAGE}" in
  generate)
    FRAMES="${FRAMES}" WINDOW_SIZE="${WINDOW_SIZE}" STRIDE="${STRIDE}" \
      bash "${REPO_ROOT}/data_generation_pipeline_tools/generate_mujoco_direct_dataset.sh"
    ;;
  build)
    python "${REPO_ROOT}/3d_keypoint_detector_training/build_sequences.py" \
      --raw-root "${RAW_ROOT}" \
      --output-root "${SEQUENCE_ROOT}" \
      --window-size "${WINDOW_SIZE}" \
      --stride "${STRIDE}" \
      --eval-stride "${WINDOW_SIZE}" \
      --val-ratio "${VAL_RATIO:-0.1}" \
      --test-ratio "${TEST_RATIO:-0.1}" \
      --seed "${SEED:-7}"
    ;;
  train)
    bash "${REPO_ROOT}/3d_keypoint_detector_training/start_training.sh" "${@}"
    ;;
  eval)
    CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=path/to/best_epoch.bin}"
    python "${REPO_ROOT}/3d_keypoint_detector_training/eval_lifter.py" \
      --checkpoint "${CHECKPOINT}"
    ;;
  infer)
    python "${REPO_ROOT}/3d_keypoint_detector_training/3D_lifting_inference.py" "${@}"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}. Use: generate|build|train|eval|infer" >&2
    exit 1
    ;;
esac

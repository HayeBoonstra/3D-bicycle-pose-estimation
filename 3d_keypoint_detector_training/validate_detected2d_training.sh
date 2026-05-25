#!/usr/bin/env bash
# Production validation after building PoseMamba_f243s81_detected2d corpus.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEQUENCE_ROOT="${SEQUENCE_ROOT:-${REPO_ROOT}/data/posemamba_training_sequences}"
DATA_TAG="${DATASET_TAG:-detected2d}"
WINDOW_SIZE="${WINDOW_SIZE:-243}"
STRIDE="${STRIDE:-81}"
CKPT="${1:-}"

DATA_ROOT="${SEQUENCE_ROOT}/PoseMamba_f${WINDOW_SIZE}s${STRIDE}_${DATA_TAG}"
VAL_DIR="${DATA_ROOT}/BICYCLE/val"

if [[ ! -d "${DATA_ROOT}/BICYCLE/train" ]]; then
  echo "error: missing train pickles at ${DATA_ROOT}/BICYCLE/train" >&2
  echo "  Run: bash data_generation_pipeline_tools/generate_blender_posemamba_dataset.sh" >&2
  exit 1
fi

echo "[validate] corpus=${DATA_ROOT}"
if [[ -f "${DATA_ROOT}/dataset_manifest.json" ]]; then
  python3 -c "import json; m=json.load(open('${DATA_ROOT}/dataset_manifest.json')); print('  input_2d_source:', m.get('input_2d_source')); print('  bbox_source:', m.get('bbox_source'))"
fi

if [[ -z "${CKPT}" ]]; then
  echo "[validate] Training fresh lifter (DATASET_TAG=${DATA_TAG})..."
  DATASET_TAG="${DATA_TAG}" SEQUENCE_ROOT="${SEQUENCE_ROOT}" "${REPO_ROOT}/3d_keypoint_detector_training/start_training.sh"
  CKPT="$(ls -t "${REPO_ROOT}"/checkpoints/posemamba_bicycle_*/best_epoch.bin 2>/dev/null | head -1)"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "error: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

echo "[validate] eval_lifter on ${CKPT}"
conda run -n posemamba python "${REPO_ROOT}/3d_keypoint_detector_training/eval_lifter.py" \
  --checkpoint "${CKPT}"

if [[ -d "${VAL_DIR}" ]] && [[ -n "$(ls -A "${VAL_DIR}"/*.pkl 2>/dev/null || true)" ]]; then
  echo "[validate] val pickles present at ${VAL_DIR}"
fi

echo "[validate] full pipeline on Swapfiets input_sequence"
"${REPO_ROOT}/1_full_detection_pipeline/run_full_pipeline.sh" \
  --frames-dir "${REPO_ROOT}/1_full_detection_pipeline/input_sequence" \
  --output-dir "${REPO_ROOT}/1_full_detection_pipeline/output_detected2d_val" \
  --lifter-checkpoint "${CKPT}" \
  --no-visualize

echo "[validate] Done. 3D output: 1_full_detection_pipeline/output_detected2d_val/keypoints_3d.npz"

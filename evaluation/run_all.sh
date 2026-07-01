#!/usr/bin/env bash
# End-to-end results pipeline: extract -> stats -> figures
#
# Usage:
#   ./evaluation/run_all.sh
#   CHECKPOINT=path/to/best_epoch.bin ./evaluation/run_all.sh
#
# Data paths (optional; auto-detects /mnt/SmallSSD/3D-bicycle-pose-estimation if repo data/ missing):
#   RAW_ROOT=/mnt/SmallSSD/3D-bicycle-pose-estimation/raw_blender_posemamba
#   DATA_ROOT=/mnt/SmallSSD/3D-bicycle-pose-estimation/posemamba_training_sequences
#   BICYCLE_POSE_DATA_ROOT=/path/to/bicycle_pose_dataset
#
# Stage-1/2 static-frame eval (bicycle_pose_dataset):
#   SKIP_STATIC_STAGE12=1          skip RF-DETR + RTMPose on static frames
#   STATIC_STAGE12_SPLIT=test      COCO split (default: test)
#   STATIC_STAGE12_LIMIT=50        optional image cap for quick runs
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
RAW_ROOT="${RAW_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-}"
BICYCLE_POSE_DATA_ROOT="${BICYCLE_POSE_DATA_ROOT:-${REPO_ROOT}/data/bicycle_pose_dataset}"
STATIC_STAGE12_SPLIT="${STATIC_STAGE12_SPLIT:-test}"
STATIC_STAGE12_LIMIT="${STATIC_STAGE12_LIMIT:-}"
SKIP_STATIC_STAGE12="${SKIP_STATIC_STAGE12:-0}"

export RAW_ROOT DATA_ROOT

run_py() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "$@"
  else
    conda run -n "${CONDA_ENV}" python "$@"
  fi
}

LIFTERINPUT_RECORDS="${RESULTS_DIR}/stage12_lifterinput_records.jsonl"
STATIC_RECORDS="${RESULTS_DIR}/stage12_static_records.jsonl"
STATIC_CACHE="${RESULTS_DIR}/stage12_static_cache"
EXTRACT_STATIC="${REPO_ROOT}/evaluation/extract_stage12_static.py"

STATIC_LIMIT_ARGS=()
if [[ -n "${STATIC_STAGE12_LIMIT}" ]]; then
  STATIC_LIMIT_ARGS=(--limit "${STATIC_STAGE12_LIMIT}")
fi

echo "[run_all] stage-1/2 lifter-input extraction"
run_py "${REPO_ROOT}/evaluation/extract_stage12_lifterinput.py" --out "${LIFTERINPUT_RECORDS}"

if [[ "${SKIP_STATIC_STAGE12}" != "1" ]]; then
  echo "[run_all] stage-1/2 static-frame detection (rfdetr env, split=${STATIC_STAGE12_SPLIT})"
  conda run -n rfdetr python "${EXTRACT_STATIC}" \
    --dataset-root "${BICYCLE_POSE_DATA_ROOT}" \
    --split "${STATIC_STAGE12_SPLIT}" \
    --cache-dir "${STATIC_CACHE}" \
    --out "${STATIC_RECORDS}" \
    --run-detection \
    --skip-pose \
    --resume \
    "${STATIC_LIMIT_ARGS[@]}"

  echo "[run_all] stage-1/2 static-frame pose (mmpose env)"
  conda run -n mmpose python "${EXTRACT_STATIC}" \
    --dataset-root "${BICYCLE_POSE_DATA_ROOT}" \
    --split "${STATIC_STAGE12_SPLIT}" \
    --cache-dir "${STATIC_CACHE}" \
    --out "${STATIC_RECORDS}" \
    --skip-detection \
    --run-pose \
    --resume \
    "${STATIC_LIMIT_ARGS[@]}"
else
  echo "[run_all] skipping static-frame stage-1/2 (SKIP_STATIC_STAGE12=1)"
fi

if [[ -n "${CHECKPOINT}" ]]; then
  exp_name="$(basename "$(dirname "${CHECKPOINT}")")"
  echo "[run_all] 3D extraction for ${exp_name}"
  run_py "${REPO_ROOT}/evaluation/extract.py" \
    --checkpoint "${CHECKPOINT}" \
    --out "${RESULTS_DIR}" \
    --experiment-name "${exp_name}"
else
  echo "[run_all] extracting all checkpoints under posemamba_weights/"
  shopt -s nullglob
  for ckpt in "${REPO_ROOT}"/posemamba_weights/*/best_epoch.bin; do
    exp_name="$(basename "$(dirname "${ckpt}")")"
    if [[ -f "${RESULTS_DIR}/${exp_name}/preds_3d.npz" ]]; then
      echo "[skip] ${exp_name}: preds_3d.npz exists"
      continue
    fi
    run_py "${REPO_ROOT}/evaluation/extract.py" \
      --checkpoint "${ckpt}" \
      --out "${RESULTS_DIR}" \
      --experiment-name "${exp_name}"
  done
fi

echo "[run_all] computing statistics"
run_py "${REPO_ROOT}/evaluation/compute_stats.py" --results-dir "${RESULTS_DIR}"

echo "[run_all] generating figures"
run_py "${REPO_ROOT}/evaluation/make_figures.py" --results-dir "${RESULTS_DIR}"

if [[ -n "${CHECKPOINT}" ]]; then
  exp_name="$(basename "$(dirname "${CHECKPOINT}")")"
  echo "[run_all] dynamics example video for ${exp_name}"
  run_py "${REPO_ROOT}/evaluation/make_dynamics_example_video.py" \
    --results-dir "${RESULTS_DIR}" \
    --experiment "${exp_name}" || echo "[warn] dynamics example video failed (non-fatal)"
fi

if [[ -n "${CHECKPOINT}" ]]; then
  echo "[run_all] SSM coupling map (optional)"
  run_py "${REPO_ROOT}/evaluation/ssm_map.py" \
    --checkpoint "${CHECKPOINT}" \
    --out "${RESULTS_DIR}/ssm_maps" || echo "[warn] ssm_map failed (non-fatal)"
fi

echo "[run_all] done -> ${RESULTS_DIR}"
echo "[run_all]   stage12_lifterinput_metrics.json"
echo "[run_all]   stage12_static_metrics.json (if static eval ran)"

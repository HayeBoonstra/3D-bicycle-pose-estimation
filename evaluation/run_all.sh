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
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
RAW_ROOT="${RAW_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-}"

export RAW_ROOT DATA_ROOT

run_py() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "$@"
  else
    conda run -n "${CONDA_ENV}" python "$@"
  fi
}

echo "[run_all] stage-1/2 extraction"
run_py "${REPO_ROOT}/evaluation/extract_stage12.py" --out "${RESULTS_DIR}/stage12_records.jsonl"

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

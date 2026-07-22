#!/usr/bin/env bash
# Extract + stats + figures for GT-trained capacity models (S/B/L/X).
#
# Each checkpoint is evaluated on its matching GT corpus test split
# (PoseMamba_f243s81_gt), not the detected-2D corpus.
#
# Usage:
#   ./evaluation/run_gt_capacity_eval.sh
#   SKIP_EXTRACT=1 ./evaluation/run_gt_capacity_eval.sh
#   DATA_ROOT=/mnt/SmallSSD/.../posemamba_training_sequences ./evaluation/run_gt_capacity_eval.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
DATA_ROOT="${DATA_ROOT:-/mnt/SmallSSD/3D-bicycle-pose-estimation/posemamba_training_sequences}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-0}"
EXTRACT_INFERENCE_MODE="${EXTRACT_INFERENCE_MODE:-window}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

export DATA_ROOT

run_py() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "$@"
  else
    conda run -n "${CONDA_ENV}" python "$@"
  fi
}

GT_CAPACITY_CKPTS=(
  "${REPO_ROOT}/posemamba_weights/capacity_s_gt/best_epoch.bin"
  "${REPO_ROOT}/posemamba_weights/capacity_b_gt/best_epoch.bin"
  "${REPO_ROOT}/posemamba_weights/capacity_l_gt/best_epoch.bin"
  "${REPO_ROOT}/posemamba_weights/capacity_x_gt/best_epoch.bin"
)

gt_test_dir="${DATA_ROOT}/PoseMamba_f243s81_gt/BICYCLE/test"
if [[ ! -d "${gt_test_dir}" ]]; then
  echo "error: GT test corpus not found: ${gt_test_dir}" >&2
  echo "  Run: ./experiments/build_gt_training_corpus.sh" >&2
  exit 1
fi

if [[ "${SKIP_EXTRACT}" != "1" ]]; then
  for ckpt in "${GT_CAPACITY_CKPTS[@]}"; do
    if [[ ! -f "${ckpt}" ]]; then
      echo "error: checkpoint not found: ${ckpt}" >&2
      exit 1
    fi
    exp_name="$(basename "$(dirname "${ckpt}")")"
    if [[ "${FORCE_EXTRACT}" != "1" && -f "${RESULTS_DIR}/${exp_name}/preds_3d.npz" ]]; then
      summary="${RESULTS_DIR}/${exp_name}/extract_summary.json"
      if [[ -f "${summary}" ]] \
        && grep -q "\"test_dir\".*PoseMamba_f243s81_gt" "${summary}" \
        && grep -q "\"inference_mode\": \"${EXTRACT_INFERENCE_MODE}\"" "${summary}"; then
        echo "[skip] ${exp_name}: GT corpus preds_3d.npz exists (${EXTRACT_INFERENCE_MODE})"
        continue
      fi
      echo "[re-extract] ${exp_name}: updating GT training evaluation (${EXTRACT_INFERENCE_MODE})"
    fi
    echo "[run_gt_capacity_eval] extracting ${exp_name} from ${ckpt}"
    run_py "${REPO_ROOT}/evaluation/extract.py" \
      --checkpoint "${ckpt}" \
      --out "${RESULTS_DIR}" \
      --experiment-name "${exp_name}" \
      --inference-mode "${EXTRACT_INFERENCE_MODE}" \
      --batch-size "${EXTRACT_BATCH_SIZE}"
  done
else
  echo "[run_gt_capacity_eval] skipping extraction (SKIP_EXTRACT=1)"
fi

echo "[run_gt_capacity_eval] computing statistics"
run_py "${REPO_ROOT}/evaluation/compute_stats.py" --results-dir "${RESULTS_DIR}"

echo "[run_gt_capacity_eval] generating figures"
run_py "${REPO_ROOT}/evaluation/make_figures.py" --results-dir "${RESULTS_DIR}"

echo "[run_gt_capacity_eval] done -> ${RESULTS_DIR}"
echo "[run_gt_capacity_eval]   tables/pose3d_capacity_gt.tex"
echo "[run_gt_capacity_eval]   tables/pose3d_capacity_detected_vs_gt.tex"
echo "[run_gt_capacity_eval]   figures/capacity_gt.png"
echo "[run_gt_capacity_eval]   figures/capacity_detected_vs_gt.png"

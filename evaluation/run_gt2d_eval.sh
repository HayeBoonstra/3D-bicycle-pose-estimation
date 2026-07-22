#!/usr/bin/env bash
# Run oracle (GT-2D) 3D lifting evaluation for capacity S/B/L/X checkpoints.
#
# Isolates PoseMamba lifter error from RF-DETR + RTMPose front-end error by lifting
# bbox-normalized GT-projected 2D keypoints instead of detected 2D.
#
# Prerequisites:
#   1. Oracle corpus: ./experiments/build_gt2d_corpus.sh
#   2. Trained checkpoints: posemamba_weights/capacity_{s,b,l,x}/best_epoch.bin
#
# Usage:
#   ./evaluation/run_gt2d_eval.sh
#   DATA_ROOT=/mnt/SmallSSD/.../posemamba_training_sequences ./evaluation/run_gt2d_eval.sh
#   SKIP_EXTRACT=1 ./evaluation/run_gt2d_eval.sh   # stats/figures only
#
# Environment:
#   DATA_ROOT              PoseMamba sequence root (default: SSD if mounted)
#   RESULTS_DIR            Output root (default: results/)
#   EXTRACT_INFERENCE_MODE Inference mode passed to extract.py (default: window)
#   EXTRACT_BATCH_SIZE     Batch size for extract.py (default: 0 = auto)
#   SKIP_EXTRACT           Set to 1 to skip extraction and only run stats/figures
#   FORCE_EXTRACT          Set to 1 to overwrite existing GT-2D predictions
#   CONDA_ENV              Conda env for Python (default: posemamba)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
DATA_ROOT="${DATA_ROOT:-}"
EXTRACT_INFERENCE_MODE="${EXTRACT_INFERENCE_MODE:-window}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

export DATA_ROOT

run_py() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "$@"
  else
    conda run -n "${CONDA_ENV}" python "$@"
  fi
}

CAPACITY_CKPTS=(
  "${REPO_ROOT}/posemamba_weights/capacity_s/best_epoch.bin"
  "${REPO_ROOT}/posemamba_weights/capacity_b/best_epoch.bin"
  "${REPO_ROOT}/posemamba_weights/capacity_l/best_epoch.bin"
  "${REPO_ROOT}/posemamba_weights/capacity_x/best_epoch.bin"
)

if [[ "${SKIP_EXTRACT}" != "1" ]]; then
  gt2d_test_dir=""
  if [[ -n "${DATA_ROOT}" ]]; then
    gt2d_test_dir="${DATA_ROOT}/PoseMamba_f243s81/BICYCLE/test"
  else
    gt2d_test_dir="/mnt/SmallSSD/3D-bicycle-pose-estimation/posemamba_training_sequences/PoseMamba_f243s81/BICYCLE/test"
  fi
  if [[ ! -d "${gt2d_test_dir}" ]]; then
    echo "error: GT-2D test corpus not found: ${gt2d_test_dir}" >&2
    echo "  Run: ./experiments/build_gt2d_corpus.sh" >&2
    exit 1
  fi

  for ckpt in "${CAPACITY_CKPTS[@]}"; do
    if [[ ! -f "${ckpt}" ]]; then
      echo "error: checkpoint not found: ${ckpt}" >&2
      exit 1
    fi
    exp_name="$(basename "$(dirname "${ckpt}")")_gt2d"
    if [[ "${FORCE_EXTRACT}" != "1" && -f "${RESULTS_DIR}/${exp_name}/preds_3d.npz" ]]; then
      summary="${RESULTS_DIR}/${exp_name}/extract_summary.json"
      if [[ -f "${summary}" ]] \
        && grep -q "\"input_2d\": \"gt\"" "${summary}" \
        && grep -q "\"inference_mode\": \"${EXTRACT_INFERENCE_MODE}\"" "${summary}"; then
        echo "[skip] ${exp_name}: GT-2D preds_3d.npz exists (${EXTRACT_INFERENCE_MODE})"
        continue
      fi
      echo "[re-extract] ${exp_name}: updating GT-2D evaluation (${EXTRACT_INFERENCE_MODE})"
    fi
    echo "[run_gt2d_eval] extracting ${exp_name} from ${ckpt}"
    run_py "${REPO_ROOT}/evaluation/extract.py" \
      --checkpoint "${ckpt}" \
      --input-2d gt \
      --out "${RESULTS_DIR}" \
      --inference-mode "${EXTRACT_INFERENCE_MODE}" \
      --batch-size "${EXTRACT_BATCH_SIZE}"
  done
else
  echo "[run_gt2d_eval] skipping extraction (SKIP_EXTRACT=1)"
fi

echo "[run_gt2d_eval] computing statistics"
run_py "${REPO_ROOT}/evaluation/compute_stats.py" --results-dir "${RESULTS_DIR}"

echo "[run_gt2d_eval] generating figures"
run_py "${REPO_ROOT}/evaluation/make_figures.py" --results-dir "${RESULTS_DIR}"

echo "[run_gt2d_eval] done -> ${RESULTS_DIR}"
echo "[run_gt2d_eval]   tables/pose3d_capacity_gt2d.tex"
echo "[run_gt2d_eval]   tables/pose3d_capacity_detected_vs_gt2d.tex"
echo "[run_gt2d_eval]   figures/capacity_detected_vs_gt2d.png"

#!/usr/bin/env bash
# Train temporal-window ablation experiments sequentially on a server.
#
# Usage:
#   DATA_ROOT=/mnt/SmallSSD/.../posemamba_training_sequences ./experiments/run_window_ablations.sh
#   ./experiments/run_window_ablations.sh window_t81
#
# Prerequisite: ./experiments/build_window_corpora.sh
#               python experiments/make_window_configs.py
#
# T=243 baseline is capacity_b (main analysis), not retrained here.
# Checkpoints land in ${CHECKPOINT_BASE}/window_t{T}/ (separate from headline ablations).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/posemamba_training_sequences}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${REPO_ROOT}/posemamba_weights}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIGS_DIR="${REPO_ROOT}/experiments/configs"
MANIFEST="${CONFIGS_DIR}/window_experiments.json"
POSEMAMBA_ROOT="${REPO_ROOT}/PoseMamba"
FILTER="${1:-}"

unset _PYTHON_SYSCONFIGDATA_NAME CC CXX CUDAHOSTCXX
export CUDA_VISIBLE_DEVICES

mkdir -p "${CHECKPOINT_BASE}"

find_completed_best() {
  local exp_name="$1"
  local run_dir="${CHECKPOINT_BASE}/${exp_name}"
  local legacy

  if [[ -f "${run_dir}/best_epoch.bin" ]]; then
    echo "${run_dir}/best_epoch.bin"
    return 0
  fi

  shopt -s nullglob
  local legacy_runs=("${CHECKPOINT_BASE}/${exp_name}"_*/)
  shopt -u nullglob
  for legacy in "${legacy_runs[@]}"; do
    if [[ -f "${legacy}/best_epoch.bin" ]]; then
      echo "${legacy}/best_epoch.bin"
      return 0
    fi
  done
  return 1
}

find_resume_dir() {
  local exp_name="$1"
  local run_dir="${CHECKPOINT_BASE}/${exp_name}"
  local candidate newest="" newest_mtime=0 mtime

  if [[ -f "${run_dir}/latest_epoch.bin" && ! -f "${run_dir}/best_epoch.bin" ]]; then
    echo "${run_dir}"
    return 0
  fi

  shopt -s nullglob
  local legacy_runs=("${CHECKPOINT_BASE}/${exp_name}"_*/)
  shopt -u nullglob
  for candidate in "${legacy_runs[@]}"; do
    if [[ ! -f "${candidate}/latest_epoch.bin" || -f "${candidate}/best_epoch.bin" ]]; then
      continue
    fi
    mtime="$(stat -c %Y "${candidate}/latest_epoch.bin" 2>/dev/null || stat -f %m "${candidate}/latest_epoch.bin")"
    if (( mtime > newest_mtime )); then
      newest_mtime="${mtime}"
      newest="${candidate}"
    fi
  done

  if [[ -n "${newest}" ]]; then
    echo "${newest}"
    return 0
  fi
  return 1
}

corpus_for_experiment() {
  local exp_name="$1"
  python3 - "${MANIFEST}" "${exp_name}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
name = sys.argv[2]
if name not in manifest:
    raise SystemExit(f"unknown experiment: {name}")
print(manifest[name]["corpus_subdir"])
PY
}

run_one() {
  local exp_name="$1"
  local cfg_src="${CONFIGS_DIR}/${exp_name}.yaml"
  local corpus_subdir

  if [[ ! -f "${cfg_src}" ]]; then
    echo "error: missing config for ${exp_name}: ${cfg_src}" >&2
    exit 1
  fi
  if [[ -n "${FILTER}" && "${exp_name}" != "${FILTER}" ]]; then
    return 0
  fi

  corpus_subdir="$(corpus_for_experiment "${exp_name}")"
  if [[ ! -d "${DATA_ROOT}/${corpus_subdir}/BICYCLE/train" ]]; then
    echo "error: corpus not found for ${exp_name}: ${DATA_ROOT}/${corpus_subdir}" >&2
    echo "  Run: ./experiments/build_window_corpora.sh" >&2
    exit 1
  fi

  local run_dir="${CHECKPOINT_BASE}/${exp_name}"
  local completed_best resume_dir ckpt_dir resume_arg=()

  if completed_best="$(find_completed_best "${exp_name}")"; then
    echo "[skip] ${exp_name}: completed (${completed_best})"
    return 0
  fi

  mkdir -p "${run_dir}"
  local cfg_mat="${run_dir}/train_config.yaml"
  sed "s|__DATA_ROOT__|${DATA_ROOT}|g" "${cfg_src}" > "${cfg_mat}"

  if resume_dir="$(find_resume_dir "${exp_name}")"; then
    ckpt_dir="${resume_dir}"
    resume_arg=(-r "${ckpt_dir}/latest_epoch.bin")
    echo "[resume] ${exp_name} from ${ckpt_dir}/latest_epoch.bin"
  else
    ckpt_dir="${run_dir}"
    echo "[train] ${exp_name} -> ${ckpt_dir}"
  fi

  echo "[train] config: ${cfg_mat}"
  echo "[train] data_root: ${DATA_ROOT}/${corpus_subdir}"

  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    (
      export POSEMAMBA_CHECKPOINT_RUN_DIR="${ckpt_dir}"
      cd "${POSEMAMBA_ROOT}"
      python train.py \
        --config "${cfg_mat}" \
        --checkpoint "${ckpt_dir}" \
        "${resume_arg[@]}" \
        2>&1 | tee -a "${run_dir}/train.log"
    )
  else
    (
      export POSEMAMBA_CHECKPOINT_RUN_DIR="${ckpt_dir}"
      cd "${POSEMAMBA_ROOT}"
      conda run -n "${CONDA_ENV}" python train.py \
        --config "${cfg_mat}" \
        --checkpoint "${ckpt_dir}" \
        "${resume_arg[@]}" \
        2>&1 | tee -a "${run_dir}/train.log"
    )
  fi
}

if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: missing ${MANIFEST}. Run: python experiments/make_window_configs.py" >&2
  exit 1
fi

mapfile -t experiment_names < <(
  python3 - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in manifest:
    print(name)
PY
)

if [[ ${#experiment_names[@]} -eq 0 ]]; then
  echo "error: no experiments listed in ${MANIFEST}" >&2
  exit 1
fi

for exp_name in "${experiment_names[@]}"; do
  run_one "${exp_name}"
done

echo "[done] all window ablation runs finished (or skipped)"

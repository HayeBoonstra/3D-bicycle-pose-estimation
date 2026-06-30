#!/usr/bin/env bash
# Train all headline ablation experiments sequentially on a server.
#
# Usage:
#   DATA_ROOT=/path/to/posemamba_training_sequences ./experiments/run_ablations.sh
#   DATA_ROOT=/mnt/SmallSSD/.../posemamba_training_sequences ./experiments/run_ablations.sh capacity_b
#
# Set DATA_ROOT to the parent of PoseMamba_f243s81_detected2d (not the corpus folder itself).
#
# Completed runs are skipped when best_epoch.bin exists in the experiment stub dir or any
# legacy timestamped sibling (e.g. capacity_b_2026_06_30_T_07_08_18/). Interrupted runs
# resume from latest_epoch.bin in the newest matching run directory.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/posemamba_training_sequences}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${REPO_ROOT}/posemamba_weights}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIGS_DIR="${REPO_ROOT}/experiments/configs"
MANIFEST="${CONFIGS_DIR}/experiments.json"
POSEMAMBA_ROOT="${REPO_ROOT}/PoseMamba"
FILTER="${1:-}"

unset _PYTHON_SYSCONFIGDATA_NAME CC CXX CUDAHOSTCXX
export CUDA_VISIBLE_DEVICES

if [[ ! -d "${DATA_ROOT}/PoseMamba_f243s81_detected2d" ]]; then
  echo "error: detected-2D corpus not found at ${DATA_ROOT}/PoseMamba_f243s81_detected2d" >&2
  exit 1
fi

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

run_one() {
  local exp_name="$1"
  local cfg_src="${CONFIGS_DIR}/${exp_name}.yaml"

  if [[ ! -f "${cfg_src}" ]]; then
    echo "error: missing config for ${exp_name}: ${cfg_src}" >&2
    exit 1
  fi
  if [[ -n "${FILTER}" && "${exp_name}" != "${FILTER}" ]]; then
    return 0
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
  echo "[train] data_root: ${DATA_ROOT}/PoseMamba_f243s81_detected2d"

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
  echo "error: missing ${MANIFEST}. Run: python experiments/make_configs.py" >&2
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

echo "[done] all ablation runs finished (or skipped)"

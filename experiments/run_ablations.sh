#!/usr/bin/env bash
# Train all headline ablation experiments sequentially on a server.
#
# Usage:
#   DATA_ROOT=/path/to/posemamba_training_sequences ./experiments/run_ablations.sh
#   DATA_ROOT=/mnt/SmallSSD/.../posemamba_training_sequences ./experiments/run_ablations.sh capacity_b
#
# Set DATA_ROOT to the parent of PoseMamba_f243s81_detected2d (not the corpus folder itself).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/posemamba_training_sequences}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${REPO_ROOT}/posemamba_weights}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIGS_DIR="${REPO_ROOT}/experiments/configs"
POSEMAMBA_ROOT="${REPO_ROOT}/PoseMamba"
FILTER="${1:-}"

unset _PYTHON_SYSCONFIGDATA_NAME CC CXX CUDAHOSTCXX
export CUDA_VISIBLE_DEVICES

if [[ ! -d "${DATA_ROOT}/PoseMamba_f243s81_detected2d" ]]; then
  echo "error: detected-2D corpus not found at ${DATA_ROOT}/PoseMamba_f243s81_detected2d" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_BASE}"

run_one() {
  local cfg_src="$1"
  local exp_name
  exp_name="$(basename "${cfg_src}" .yaml)"
  if [[ -n "${FILTER}" && "${exp_name}" != "${FILTER}" ]]; then
    return 0
  fi

  local run_dir="${CHECKPOINT_BASE}/${exp_name}"
  local best="${run_dir}/best_epoch.bin"
  if [[ -f "${best}" ]]; then
    echo "[skip] ${exp_name}: ${best} already exists"
    return 0
  fi

  mkdir -p "${run_dir}"
  local cfg_mat="${run_dir}/train_config.yaml"
  sed "s|__DATA_ROOT__|${DATA_ROOT}|g" "${cfg_src}" > "${cfg_mat}"

  echo "[train] ${exp_name} -> ${run_dir}"
  echo "[train] config: ${cfg_mat}"
  echo "[train] data_root: ${DATA_ROOT}/PoseMamba_f243s81_detected2d"

  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    (
      cd "${POSEMAMBA_ROOT}"
      python train.py \
        --config "${cfg_mat}" \
        --checkpoint "${run_dir}" \
        2>&1 | tee "${run_dir}/train.log"
    )
  else
    (
      cd "${POSEMAMBA_ROOT}"
      conda run -n "${CONDA_ENV}" python train.py \
        --config "${cfg_mat}" \
        --checkpoint "${run_dir}" \
        2>&1 | tee "${run_dir}/train.log"
    )
  fi
}

shopt -s nullglob
configs=("${CONFIGS_DIR}"/*.yaml)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "error: no YAML configs in ${CONFIGS_DIR}. Run: python experiments/make_configs.py" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  run_one "${cfg}"
done

echo "[done] all ablation runs finished (or skipped)"

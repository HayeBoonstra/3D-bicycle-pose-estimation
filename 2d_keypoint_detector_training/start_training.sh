#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MMPPOSE_ENV="${REPO_ROOT}/2d_keypoint_detector_training/mmpose_env.sh"
CONFIG="${REPO_ROOT}/2d_keypoint_detector_training/rtmpose_bicycle_full.py"
WORK_DIR="${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu"

# Resume (optional):
#   ./start_training.sh --resume              # latest checkpoint in WORK_DIR
#   RESUME=1 ./start_training.sh              # same
#   RESUME=/path/to/epoch_50.pth ./start_training.sh
#   AUTO_RESUME=1 ./start_training.sh         # resume only if last_checkpoint exists
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pull --resume / --resume=PATH out of "$@" so they are not passed through to mim twice.
EXTRA_MIM_ARGS=()
RESUME_MODE="${RESUME:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_MODE=auto
      shift
      ;;
    --resume=*)
      RESUME_MODE="${1#*=}"
      shift
      ;;
    *)
      EXTRA_MIM_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${EXTRA_MIM_ARGS[@]}"

RESUME_ARGS=()
if [[ "${AUTO_RESUME:-0}" == "1" && -z "${RESUME_MODE}" && -f "${WORK_DIR}/last_checkpoint" ]]; then
  RESUME_MODE=auto
fi
if [[ -n "${RESUME_MODE}" ]]; then
  if [[ "${RESUME_MODE}" == "1" || "${RESUME_MODE}" == "auto" || "${RESUME_MODE}" == "true" ]]; then
    if [[ ! -f "${WORK_DIR}/last_checkpoint" ]]; then
      echo "[train] ERROR: resume requested but missing ${WORK_DIR}/last_checkpoint"
      echo "[train]   Start fresh (no --resume) or set RESUME=/path/to/checkpoint.pth"
      exit 1
    fi
    mapfile -t _ckpt < "${WORK_DIR}/last_checkpoint"
    echo "[train] Resuming from latest: ${_ckpt[0]}"
    RESUME_ARGS=(--resume auto)
  else
    if [[ ! -f "${RESUME_MODE}" ]]; then
      echo "[train] ERROR: checkpoint not found: ${RESUME_MODE}"
      exit 1
    fi
    echo "[train] Resuming from: ${RESUME_MODE}"
    RESUME_ARGS=(--resume "${RESUME_MODE}")
  fi
fi

# Every Python process (including DataLoader workers) uses file-backed tensor sharing.
_install_mp_preload() {
  local py site_pkgs pth
  py="$(command -v python 2>/dev/null || true)"
  [[ -z "${py}" ]] && return 0
  site_pkgs="$("${py}" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)"
  [[ -z "${site_pkgs}" ]] && return 0
  pth="${site_pkgs}/bicycle_mmpose_dataloader.pth"
  printf '%s\n' \
    "import sys; sys.path.insert(0, '${REPO_ROOT}/2d_keypoint_detector_training'); import preload_mp" \
    >"${pth}"
}

# PyTorch DataLoader passes tensors via FDs; DDP + many workers can hit the default ulimit.
if ulimit -n 4096 2>/dev/null; then
  : # raised soft limit
elif ulimit -n 2048 2>/dev/null; then
  :
fi

_cpu_count() {
  nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8
}

# Per-GPU dataloader workers (num_workers is per DDP rank).
# Multi-GPU uses a lower cap — many workers × 2 ranks exhaust file descriptors.
_compute_train_workers() {
  local cpus="$1" gpus="$2"
  local divisor=6 max_w=32 min_w=8
  if ((gpus >= 2)); then
    divisor=10
    max_w=12
    min_w=4
  fi
  local w=$((cpus / gpus / divisor))
  ((w < min_w)) && w="${min_w}"
  ((w > max_w)) && w="${max_w}"
  echo "${w}"
}

# GPU / batch settings (batch_size is PER GPU under DDP, not global):
#   NUM_GPUS=1 TRAIN_BATCH_SIZE=64  # ~8GB GPU
#   NUM_GPUS=2 TRAIN_BATCH_SIZE=64  # default: 64/GPU → 128 global (same VRAM as 1×8GB run)
#   NUM_GPUS=2 TRAIN_BATCH_SIZE=96  # try after stable training if headroom on 24GB
if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    NUM_GPUS=1
  fi
fi

CPU_COUNT="$(_cpu_count)"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-$(_compute_train_workers "${CPU_COUNT}" "${NUM_GPUS}")}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-$((TRAIN_NUM_WORKERS / 2))}"
((VAL_NUM_WORKERS < 4)) && VAL_NUM_WORKERS=4

if [[ "${NUM_GPUS}" -ge 2 ]]; then
  LAUNCHER="${LAUNCHER:-pytorch}"
  # 64/GPU fits ~8GB at 256×320; 128/GPU OOMs on 24GB (activations + AdamW + EMA).
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
else
  LAUNCHER="${LAUNCHER:-none}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

MIM_ARGS=(
  train mmpose "${CONFIG}"
  --work-dir "${WORK_DIR}"
  "${RESUME_ARGS[@]}"
  --launcher "${LAUNCHER}"
  --gpus "${NUM_GPUS}"
  --cfg-options
  "train_dataloader.batch_size=${TRAIN_BATCH_SIZE}"
  "val_dataloader.batch_size=${VAL_BATCH_SIZE}"
  "test_dataloader.batch_size=${VAL_BATCH_SIZE}"
  "train_dataloader.num_workers=${TRAIN_NUM_WORKERS}"
  "val_dataloader.num_workers=${VAL_NUM_WORKERS}"
  "test_dataloader.num_workers=${VAL_NUM_WORKERS}"
  "train_dataloader.multiprocessing_context=spawn"
  "val_dataloader.multiprocessing_context=spawn"
  "test_dataloader.multiprocessing_context=spawn"
)

TOTAL_LOADER_WORKERS=$((TRAIN_NUM_WORKERS * NUM_GPUS))
echo "[train] CPUs=${CPU_COUNT} GPUs=${NUM_GPUS} launcher=${LAUNCHER}"
echo "[train] train_batch=${TRAIN_BATCH_SIZE}/GPU (global=$((TRAIN_BATCH_SIZE * NUM_GPUS))) val_batch=${VAL_BATCH_SIZE}/GPU"
echo "[train] train_workers=${TRAIN_NUM_WORKERS}/GPU (${TOTAL_LOADER_WORKERS} loader procs) val_workers=${VAL_NUM_WORKERS}/GPU OMP_NUM_THREADS=${OMP_NUM_THREADS}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "mmpose" ]]; then
  unset _PYTHON_SYSCONFIGDATA_NAME
  unset CC CXX CUDAHOSTCXX
  _install_mp_preload
  mim "${MIM_ARGS[@]}" "$@"
else
  _PTH_LINE="import sys; sys.path.insert(0, '${REPO_ROOT}/2d_keypoint_detector_training'); import preload_mp"
  "${MMPPOSE_ENV}" bash -c "
    set -euo pipefail
    py=\$(command -v python)
    site_pkgs=\$(\"\${py}\" -c 'import site; print(site.getsitepackages()[0])')
    printf '%s\n' '${_PTH_LINE}' > \"\${site_pkgs}/bicycle_mmpose_dataloader.pth\"
    exec mim $(printf '%q ' "${MIM_ARGS[@]}") $(printf '%q ' "$@")
  "
fi

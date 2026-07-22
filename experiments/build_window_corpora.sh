#!/usr/bin/env bash
# Re-chunk detected-2D training corpora at multiple temporal window sizes.
#
# Stride scales as T/3 (e.g. f243s81 -> f27s9, f81s27, f121s40, f162s54) so the overlap
# fraction matches the main-analysis baseline. T=243 is omitted — use capacity_b there.
#
# Reuses existing per-frame RTMPose + RF-DETR annotations; only clip boundaries change.
# Uses window-level train/val/test splits (same as PoseMamba_f243s81_detected2d / capacity_b):
# every clip contributes train windows; val/test hold out ~10% of windows each, covering all 43+ test clips.
#
# Usage:
#   ./experiments/build_window_corpora.sh
#   RAW_ROOT=/mnt/SmallSSD/.../raw_blender_posemamba \
#     OUTPUT_ROOT=/mnt/SmallSSD/.../posemamba_training_sequences \
#     ./experiments/build_window_corpora.sh
#
# Environment:
#   RAW_ROOT          Source clips with detected-2D sidecars (default: SSD path if mounted)
#   OUTPUT_ROOT       Parent of PoseMamba_f{T}s{S}_detected2d folders
#   WINDOW_SIZES      Space-separated T values (default: "27 81 121 162")
#   STRIDE_DIVISOR    Stride = T / STRIDE_DIVISOR (default: 3, matching f243s81)
#   SPLIT_SEED        Random seed for window splits (default: 7, matches f243s81)
#   FORCE_REBUILD       Set to 1 to rebuild even if corpus folders exist
#   CONDA_ENV         Conda env with torch + repo deps (default: posemamba)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SSD_BASE="/mnt/SmallSSD/3D-bicycle-pose-estimation"
RAW_ROOT="${RAW_ROOT:-${SSD_BASE}/raw_blender_posemamba}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SSD_BASE}/posemamba_training_sequences}"
WINDOW_SIZES="${WINDOW_SIZES:-27 81 121 162}"
STRIDE_DIVISOR="${STRIDE_DIVISOR:-3}"
SPLIT_SEED="${SPLIT_SEED:-7}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
BUILD_SCRIPT="${REPO_ROOT}/3d_keypoint_detector_training/build_sequences.py"

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python3 "$@"
  else
    conda run -n "${CONDA_ENV}" python3 "$@"
  fi
}

stride_for_window() {
  local window_size="$1"
  local stride=$((window_size / STRIDE_DIVISOR))
  if (( stride < 1 )); then
    echo "error: T=${window_size} too small for STRIDE_DIVISOR=${STRIDE_DIVISOR}" >&2
    exit 1
  fi
  echo "${stride}"
}

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "error: raw clip root not found: ${RAW_ROOT}" >&2
  echo "  Mount /mnt/SmallSSD or set RAW_ROOT to your clip annotations." >&2
  exit 1
fi

if ! run_python -c "import torch" >/dev/null 2>&1; then
  echo "error: conda env '${CONDA_ENV}' cannot import torch (needed by build_sequences.py)" >&2
  echo "  Activate it or set CONDA_ENV=..." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

build_one() {
  local window_size="$1"
  local stride
  stride="$(stride_for_window "${window_size}")"
  local corpus_dir="${OUTPUT_ROOT}/PoseMamba_f${window_size}s${stride}_detected2d"
  if [[ "${FORCE_REBUILD}" != "1" && -f "${corpus_dir}/dataset_manifest.json" ]]; then
    local ok
    ok="$(python3 - "${corpus_dir}/dataset_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = manifest.get("split_mode", "")
train_clips = manifest.get("split_clip_counts", {}).get("train", 0)
test_clips = manifest.get("split_clip_counts", {}).get("test", 0)
train_windows = manifest.get("split_sample_counts", {}).get("train", 0)
print("ok" if mode == "window" and train_clips >= 40 and test_clips >= 43 and train_windows > 0 else "rebuild")
PY
)"
    if [[ "${ok}" == "ok" ]]; then
      echo "[skip] T=${window_size} s=${stride}: window-split corpus OK at ${corpus_dir}"
      return 0
    fi
    echo "[rebuild] T=${window_size} s=${stride}: corpus needs window-level split -> ${corpus_dir}"
  fi

  echo "[build] T=${window_size} stride=${stride} (T/${STRIDE_DIVISOR}) split=window -> ${corpus_dir}"
  run_python "${BUILD_SCRIPT}" \
    --raw-root "${RAW_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --window-size "${window_size}" \
    --stride "${stride}" \
    --eval-stride "${window_size}" \
    --slice-style contiguous \
    --split-mode window \
    --seed "${SPLIT_SEED}" \
    --input-2d detected \
    --bbox-source detection \
    --dataset-tag detected2d

  if [[ -f "${corpus_dir}/dataset_manifest.json" ]]; then
    python3 - "${corpus_dir}/dataset_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = manifest.get("split_sample_counts", {})
clip_counts = manifest.get("split_clip_counts", {})
print(
    f"  clips train={clip_counts.get('train', 0)} val={clip_counts.get('val', 0)} "
    f"test={clip_counts.get('test', 0)} | "
    f"windows train={counts.get('train', 0)} val={counts.get('val', 0)} test={counts.get('test', 0)}"
)
PY
  fi
}

echo "[build_window_corpora] RAW_ROOT=${RAW_ROOT}"
echo "[build_window_corpora] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[build_window_corpora] WINDOW_SIZES=${WINDOW_SIZES} (stride=T/${STRIDE_DIVISOR})"
echo "[build_window_corpora] T=243 baseline: PoseMamba_f243s81_detected2d (main analysis / capacity_b)"

for t in ${WINDOW_SIZES}; do
  build_one "${t}"
done

echo "[done] window corpora ready under ${OUTPUT_ROOT}"

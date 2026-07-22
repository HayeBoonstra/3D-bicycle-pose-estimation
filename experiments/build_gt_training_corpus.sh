#!/usr/bin/env bash
# Build full train/val/test PoseMamba corpus from projected GT 2D keypoints.
#
# Uses Blender-exported keypoints_2d_frame_*.json normalized in gt_bbox_xywh
# (bicycle mesh projection). Same windowing and split policy as
# PoseMamba_f243s81_detected2d so capacity GT runs are directly comparable.
#
# Usage:
#   ./experiments/build_gt_training_corpus.sh
#   RAW_ROOT=/mnt/SmallSSD/.../raw_blender_posemamba \
#     OUTPUT_ROOT=/mnt/SmallSSD/.../posemamba_training_sequences \
#     ./experiments/build_gt_training_corpus.sh
#
# Environment:
#   RAW_ROOT          Source clips with GT 2D sidecars (default: SSD path if mounted)
#   OUTPUT_ROOT       Parent of PoseMamba_f243s81_gt folder
#   SPLIT_SEED        Random seed for window splits (default: 7, matches detected2d)
#   FORCE_REBUILD     Set to 1 to rebuild even if corpus manifest exists
#   CONDA_ENV         Conda env with torch + repo deps (default: posemamba)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SSD_BASE="/mnt/SmallSSD/3D-bicycle-pose-estimation"
RAW_ROOT="${RAW_ROOT:-${SSD_BASE}/raw_blender_posemamba}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SSD_BASE}/posemamba_training_sequences}"
SPLIT_SEED="${SPLIT_SEED:-7}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
BUILD_SCRIPT="${REPO_ROOT}/3d_keypoint_detector_training/build_sequences.py"

WINDOW_SIZE=243
STRIDE=81
DATASET_TAG=gt
CORPUS_DIR="${OUTPUT_ROOT}/PoseMamba_f${WINDOW_SIZE}s${STRIDE}_${DATASET_TAG}"

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python3 "$@"
  else
    conda run -n "${CONDA_ENV}" python3 "$@"
  fi
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

if [[ "${FORCE_REBUILD}" != "1" && -f "${CORPUS_DIR}/dataset_manifest.json" ]]; then
  ok="$(python3 - "${CORPUS_DIR}/dataset_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = manifest.get("split_mode", "")
input_2d = manifest.get("input_2d", "")
train_clips = manifest.get("split_clip_counts", {}).get("train", 0)
test_clips = manifest.get("split_clip_counts", {}).get("test", 0)
train_windows = manifest.get("split_sample_counts", {}).get("train", 0)
print(
    "ok"
    if mode == "window"
    and input_2d == "gt"
    and manifest.get("bbox_source") == "gt"
    and train_clips >= 40
    and test_clips >= 43
    and train_windows > 0
    else "rebuild"
)
PY
)"
  if [[ "${ok}" == "ok" ]]; then
    echo "[skip] GT training corpus OK at ${CORPUS_DIR}"
    exit 0
  fi
  echo "[rebuild] GT training corpus needs refresh -> ${CORPUS_DIR}"
fi

echo "[build_gt_training_corpus] RAW_ROOT=${RAW_ROOT}"
echo "[build_gt_training_corpus] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[build_gt_training_corpus] bbox_source=gt (gt_bbox_xywh from annotations)"
echo "[build_gt_training_corpus] -> ${CORPUS_DIR}"

run_python "${BUILD_SCRIPT}" \
  --raw-root "${RAW_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --window-size "${WINDOW_SIZE}" \
  --stride "${STRIDE}" \
  --eval-stride "${WINDOW_SIZE}" \
  --slice-style contiguous \
  --split-mode window \
  --seed "${SPLIT_SEED}" \
  --input-2d gt \
  --bbox-source gt \
  --dataset-tag "${DATASET_TAG}"

if [[ -f "${CORPUS_DIR}/dataset_manifest.json" ]]; then
  python3 - "${CORPUS_DIR}/dataset_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = manifest.get("split_sample_counts", {})
clip_counts = manifest.get("split_clip_counts", {})
print(
    f"[done] clips train={clip_counts.get('train', 0)} val={clip_counts.get('val', 0)} "
    f"test={clip_counts.get('test', 0)} | "
    f"windows train={counts.get('train', 0)} val={counts.get('val', 0)} test={counts.get('test', 0)}"
)
print(f"[done] input_2d={manifest.get('input_2d')} bbox_source={manifest.get('bbox_source')}")
print(f"[done] input_2d_source={manifest.get('input_2d_source')}")
PY
fi

echo "[done] GT training corpus ready at ${CORPUS_DIR}"

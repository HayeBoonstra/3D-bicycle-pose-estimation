#!/usr/bin/env bash
# Build oracle (GT-projected 2D) PoseMamba evaluation corpus.
#
# Produces PoseMamba_f243s81/BICYCLE/test/ with bbox-normalized GT 2D in
# data_input. The test split is built from the detected-2D reference pickles, so
# clip IDs, frame spans, data_label, and dynamics_gt match detected-2D exactly.
#
# By default GT keypoints are normalized in the RF-DETR detection bbox frame.
# This isolates RTMPose keypoint error while preserving the lifter input coordinate
# system used during detected-2D training/evaluation.
#
# Usage:
#   ./experiments/build_gt2d_corpus.sh
#   RAW_ROOT=/mnt/SmallSSD/.../raw_blender_posemamba \
#     OUTPUT_ROOT=/mnt/SmallSSD/.../posemamba_training_sequences \
#     ./experiments/build_gt2d_corpus.sh
#
# Environment:
#   RAW_ROOT          Source clips with GT 2D sidecars (default: SSD path if mounted)
#   OUTPUT_ROOT       Parent of PoseMamba_f243s81 folder
#   REFERENCE_TEST_DIR Detected-2D reference test pickles
#   BBOX_SOURCE       detection (default, RF-DETR bbox) or gt
#   FORCE_REBUILD     Set to 1 to rebuild even if corpus manifest exists
#   CONDA_ENV         Conda env with torch + repo deps (default: posemamba)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SSD_BASE="/mnt/SmallSSD/3D-bicycle-pose-estimation"
RAW_ROOT="${RAW_ROOT:-${SSD_BASE}/raw_blender_posemamba}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SSD_BASE}/posemamba_training_sequences}"
BBOX_SOURCE="${BBOX_SOURCE:-detection}"
REFERENCE_TEST_DIR="${REFERENCE_TEST_DIR:-${OUTPUT_ROOT}/PoseMamba_f243s81_detected2d/BICYCLE/test}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
CONDA_ENV="${CONDA_ENV:-posemamba}"
BUILD_SCRIPT="${REPO_ROOT}/experiments/build_gt2d_test_from_reference.py"

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

mkdir -p "${OUTPUT_ROOT}"

corpus_dir="${OUTPUT_ROOT}/PoseMamba_f243s81"
test_dir="${corpus_dir}/BICYCLE/test"
manifest="${corpus_dir}/dataset_manifest.json"

if [[ "${FORCE_REBUILD}" != "1" && -f "${manifest}" ]]; then
  ok="$(python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
input_2d = manifest.get("input_2d", "")
bbox_source = manifest.get("bbox_source", "")
reference = manifest.get("reference_test_dir", "")
test_windows = manifest.get("split_sample_counts", {}).get("test", 0)
print(
    "ok"
    if input_2d == "gt" and bbox_source == "detection" and reference and test_windows > 0
    else "rebuild"
)
PY
)"
  if [[ "${ok}" == "ok" ]]; then
    echo "[skip] GT-2D oracle test corpus OK at ${test_dir}"
    exit 0
  fi
  echo "[rebuild] GT-2D test corpus needs refresh -> ${test_dir}"
fi

if [[ ! -d "${REFERENCE_TEST_DIR}" ]]; then
  echo "error: reference detected-2D test dir not found: ${REFERENCE_TEST_DIR}" >&2
  echo "  Build detected-2D sequences first or set REFERENCE_TEST_DIR=..." >&2
  exit 1
fi

echo "[build_gt2d_corpus] RAW_ROOT=${RAW_ROOT}"
echo "[build_gt2d_corpus] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[build_gt2d_corpus] REFERENCE_TEST_DIR=${REFERENCE_TEST_DIR}"
echo "[build_gt2d_corpus] BBOX_SOURCE=${BBOX_SOURCE}"
echo "[build_gt2d_corpus] -> ${test_dir}"

run_python "${BUILD_SCRIPT}" \
  --raw-root "${RAW_ROOT}" \
  --reference-test-dir "${REFERENCE_TEST_DIR}" \
  --out-test-dir "${test_dir}" \
  --bbox-source "${BBOX_SOURCE}"

if [[ -f "${manifest}" ]]; then
  python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = manifest.get("split_sample_counts", {})
clip_counts = manifest.get("split_clip_counts", {})
print(
    f"[done] clips test={clip_counts.get('test', 0)} | "
    f"windows test={counts.get('test', 0)}"
)
print(f"[done] input_2d={manifest.get('input_2d')} source={manifest.get('input_2d_source')}")
print(f"[done] bbox_source={manifest.get('bbox_source')}")
PY
fi

echo "[done] GT-2D oracle test corpus ready at ${test_dir}"

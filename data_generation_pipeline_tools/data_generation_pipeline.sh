#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/data_generation_pipeline_tools"
DEFAULT_DATASET_DIR="${REPO_ROOT}/data/bicycle_pose_dataset"
DEFAULT_SPLITS="${DEFAULT_DATASET_DIR}/splits.json"
DEFAULT_TRAIN_JSON="${DEFAULT_DATASET_DIR}/annotations/train.json"

NUM_CLIPS=1
SEED=32
ENCODE_VIDEO=0
SKIP_VIS=0
BLENDER_BIN="blender"
RAW_RENDERS_DIR="${REPO_ROOT}/raw_renders"
DATASET_DIR="${DEFAULT_DATASET_DIR}"
OUTSIDE_VISIBILITY="unlabeled"
SYNC_WINDOW_SIZE=80
DISABLE_SYNC_CAMERA_WINDOW=0
PARALLEL_JOBS=2

usage() {
  cat <<EOF
Run full synthetic-data pipeline end-to-end.

Usage:
  $(basename "$0") [options]

Options:
  --num-clips N            Number of clips to render (default: ${NUM_CLIPS})
  --seed N                 Seed for rendering/splitting (default: ${SEED})
  --blender PATH           Blender executable (default: ${BLENDER_BIN})
  --raw-renders DIR        Output raw renders dir (default: ${RAW_RENDERS_DIR})
  --dataset-dir DIR        Output dataset dir (default: ${DATASET_DIR})
  --outside-visibility V   convert_to_coco policy: occluded|unlabeled (default: ${OUTSIDE_VISIBILITY})
  --sync-window-size N     Frames per clip around sampled camera frame (default: ${SYNC_WINDOW_SIZE})
  --parallel-jobs N        Number of Blender clips to render concurrently (default: ${PARALLEL_JOBS})
  --no-sync-camera-window  Disable camera-sampled frame-window synchronization
  --encode-video           Encode MP4 during render + overlay stage
  --skip-visualize         Skip visualize_coco stage
  -h, --help               Show this help
EOF
}

resolve_blender_bin() {
  local requested="$1"
  if command -v "$requested" >/dev/null 2>&1; then
    command -v "$requested"
    return 0
  fi

  # In this project setup Blender is often an interactive-shell alias:
  # alias blender='cd ~/Desktop/blender; ./blender'
  # Non-interactive scripts cannot resolve aliases, so we probe common paths.
  if [[ "$requested" == "blender" ]]; then
    local candidates=(
      "$HOME/Desktop/blender/blender"
      "/usr/bin/blender"
      "/snap/bin/blender"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
      if [[ -x "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    done
  fi

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-clips)
      NUM_CLIPS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --blender)
      BLENDER_BIN="$2"
      shift 2
      ;;
    --raw-renders)
      RAW_RENDERS_DIR="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --outside-visibility)
      OUTSIDE_VISIBILITY="$2"
      shift 2
      ;;
    --sync-window-size)
      SYNC_WINDOW_SIZE="$2"
      shift 2
      ;;
    --parallel-jobs)
      PARALLEL_JOBS="$2"
      shift 2
      ;;
    --no-sync-camera-window)
      DISABLE_SYNC_CAMERA_WINDOW=1
      shift
      ;;
    --encode-video)
      ENCODE_VIDEO=1
      shift
      ;;
    --skip-visualize)
      SKIP_VIS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH." >&2
  exit 1
fi

if ! RESOLVED_BLENDER_BIN="$(resolve_blender_bin "$BLENDER_BIN")"; then
  echo "Blender executable not found: $BLENDER_BIN" >&2
  echo "Tip: pass --blender /absolute/path/to/blender" >&2
  exit 1
fi

if [[ "$NUM_CLIPS" -lt 1 ]]; then
  echo "--num-clips must be >= 1" >&2
  exit 1
fi
if [[ "$SYNC_WINDOW_SIZE" -lt 1 ]]; then
  echo "--sync-window-size must be >= 1" >&2
  exit 1
fi
if [[ "$PARALLEL_JOBS" -lt 1 ]]; then
  echo "--parallel-jobs must be >= 1" >&2
  exit 1
fi

SPLITS_PATH="${DATASET_DIR}/splits.json"
TRAIN_JSON_PATH="${DATASET_DIR}/annotations/train.json"
VIS_ARGS=()
if [[ "$ENCODE_VIDEO" -eq 1 ]]; then
  VIS_ARGS+=(--encode-video)
fi

echo "[pipeline] repo root: ${REPO_ROOT}"
echo "[pipeline] blender: ${RESOLVED_BLENDER_BIN}"
echo "[pipeline] validating registry..."
python3 "${TOOLS_DIR}/scene_registry.py"

echo "[pipeline] rendering ${NUM_CLIPS} clip(s)..."
RENDER_ARGS=(
  python3 "${TOOLS_DIR}/batch_render.py"
  --num-clips "${NUM_CLIPS}"
  --seed "${SEED}"
  --blender "${RESOLVED_BLENDER_BIN}"
  --sync-window-size "${SYNC_WINDOW_SIZE}"
  --jobs "${PARALLEL_JOBS}"
  --out "${RAW_RENDERS_DIR}"
)
if [[ "$ENCODE_VIDEO" -eq 1 ]]; then
  RENDER_ARGS+=(--encode-video)
fi
if [[ "$DISABLE_SYNC_CAMERA_WINDOW" -eq 1 ]]; then
  RENDER_ARGS+=(--no-sync-camera-window)
fi
"${RENDER_ARGS[@]}"

RAW_CLIP_COUNT="$(python3 - <<'PY' "${RAW_RENDERS_DIR}"
from pathlib import Path
import sys
raw = Path(sys.argv[1])
count = sum(1 for p in raw.iterdir() if p.is_dir() and (p / "per_frame_annotations").exists())
print(count)
PY
)"
echo "[pipeline] discovered raw clips with annotations: ${RAW_CLIP_COUNT}"
if [[ "${RAW_CLIP_COUNT}" -lt 1 ]]; then
  echo "[pipeline] ERROR: no raw clips found under ${RAW_RENDERS_DIR}" >&2
  exit 1
fi

echo "[pipeline] creating train/val/test split..."
python3 "${TOOLS_DIR}/split_dataset.py" \
  --raw-renders "${RAW_RENDERS_DIR}" \
  --out "${SPLITS_PATH}" \
  --seed "${SEED}"

echo "[pipeline] converting to COCO..."
python3 "${TOOLS_DIR}/convert_to_coco.py" \
  --raw-renders "${RAW_RENDERS_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --splits "${SPLITS_PATH}" \
  --outside-visibility "${OUTSIDE_VISIBILITY}"

COCO_IMAGE_COUNT="$(python3 - <<'PY' "${TRAIN_JSON_PATH}"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data.get("images", [])))
PY
)"
echo "[pipeline] COCO train images: ${COCO_IMAGE_COUNT}"
if [[ "${COCO_IMAGE_COUNT}" -lt 1 ]]; then
  echo "[pipeline] ERROR: train COCO has zero images at ${TRAIN_JSON_PATH}" >&2
  exit 1
fi
PARALLEL_JOBS="$(nproc 2>/dev/null || echo 1)"
if [[ "$SKIP_VIS" -eq 0 ]]; then
  echo "[pipeline] generating overlays..."
  python3 "${TOOLS_DIR}/visualize_coco.py" \
    --coco "${TRAIN_JSON_PATH}" \
    --dataset-dir "${DATASET_DIR}" \
    --jobs "${PARALLEL_JOBS}" \
    "${VIS_ARGS[@]}"

  OVERLAY_COUNT="$(python3 - <<'PY' "${DATASET_DIR}/overlays"
from pathlib import Path
import sys
root = Path(sys.argv[1])
print(sum(1 for _ in root.rglob("*.png")) if root.exists() else 0)
PY
)"
  echo "[pipeline] overlay frames written: ${OVERLAY_COUNT}"
  if [[ "${OVERLAY_COUNT}" -lt 1 ]]; then
    echo "[pipeline] ERROR: overlays directory is empty at ${DATASET_DIR}/overlays" >&2
    exit 1
  fi
else
  echo "[pipeline] skipping visualization stage."
fi

echo "[pipeline] done."
echo "  raw renders : ${RAW_RENDERS_DIR}"
echo "  splits      : ${SPLITS_PATH}"
echo "  coco train  : ${TRAIN_JSON_PATH}"

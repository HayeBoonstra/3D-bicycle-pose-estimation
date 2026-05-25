#!/usr/bin/env bash
# Blender renders -> RF-DETR bboxes -> RTMPose 2D sidecars -> PoseMamba pickles (detected-2D + GT 3D).
set -euo pipefail

resolve_blender_bin() {
  local requested="$1"
  if command -v "$requested" >/dev/null 2>&1; then
    command -v "$requested"
    return 0
  fi
  if [[ "$requested" == "blender" ]]; then
    local candidate
    for candidate in \
      "${BLENDER_PATH:-}" \
      "$HOME/Desktop/blender/blender" \
      "/usr/bin/blender" \
      "/snap/bin/blender"; do
      if [[ -n "$candidate" && -x "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    done
  elif [[ -x "$requested" ]]; then
    echo "$requested"
    return 0
  fi
  return 1
}

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Large artifacts: prefer secondary SSD (217G+ on /mnt/SmallSSD). Symlink via setup_secondary_data_disk.sh
# so data/... still appears inside the repo in the IDE.
SECONDARY_DATA_ROOT="${SECONDARY_DATA_ROOT:-/mnt/SmallSSD/3D-bicycle-pose-estimation}"
USE_SECONDARY_SSD="${USE_SECONDARY_SSD:-auto}"  # auto | 1 | 0

_resolve_data_roots() {
  local use_ssd=0
  case "${USE_SECONDARY_SSD}" in
    1|yes|true|TRUE) use_ssd=1 ;;
    0|no|false|FALSE) use_ssd=0 ;;
    auto)
      if [[ -d "/mnt/SmallSSD" ]] && mkdir -p "${SECONDARY_DATA_ROOT}" 2>/dev/null; then
        use_ssd=1
      fi
      ;;
    *)
      echo "error: USE_SECONDARY_SSD must be auto, 1, or 0 (got: ${USE_SECONDARY_SSD})" >&2
      exit 1
      ;;
  esac

  if [[ "${use_ssd}" -eq 1 ]]; then
    if ! mkdir -p "${SECONDARY_DATA_ROOT}/raw_blender_posemamba" "${SECONDARY_DATA_ROOT}/posemamba_training_sequences" 2>/dev/null; then
      echo "error: USE_SECONDARY_SSD=1 but cannot write to ${SECONDARY_DATA_ROOT}" >&2
      echo "  Run: bash data_generation_pipeline_tools/setup_secondary_data_disk.sh" >&2
      exit 1
    fi
    RAW_ROOT="${RAW_ROOT:-${SECONDARY_DATA_ROOT}/raw_blender_posemamba}"
    SEQUENCE_ROOT="${SEQUENCE_ROOT:-${SECONDARY_DATA_ROOT}/posemamba_training_sequences}"
  else
    RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/raw_blender_posemamba}"
    SEQUENCE_ROOT="${SEQUENCE_ROOT:-${REPO_ROOT}/data/posemamba_training_sequences}"
  fi
}

_resolve_data_roots
WINDOW_SIZE="${WINDOW_SIZE:-243}"
STRIDE="${STRIDE:-81}"
# Long clips: >=729 frames => multiple train windows (243, stride 81).
SYNC_WINDOW_SIZE="${SYNC_WINDOW_SIZE:-729}"
NUM_CLIPS="${NUM_CLIPS:-50}"
# Batch RNG seed: picks scene (weighted) + camera_seed per clip. Same SEED => same choices.
# Use SEED=random for a new draw each run, or SEED=42 for reproducibility.
SEED="${SEED:-random}"
OVERWRITE="${OVERWRITE:-0}"
# RENDER_JOBS=1 (default): one clip at a time; batch_render gives Blender all CPU cores (-t 0).
# RENDER_JOBS>1: render multiple clips in parallel; cores are split across Blender processes.
RENDER_JOBS="${RENDER_JOBS:-1}"
# Optional override for Blender -t (0 = all logical CPUs). Leave unset for automatic choice.
BLENDER_THREADS="${BLENDER_THREADS:-}"
BLENDER="${BLENDER:-blender}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_DETECTION="${SKIP_DETECTION:-0}"
SKIP_DETECTED_2D="${SKIP_DETECTED_2D:-0}"
MMPOSE_CONFIG="${MMPOSE_CONFIG:-${REPO_ROOT}/2d_keypoint_detector_training/rtmpose_bicycle_full.py}"
MMPOSE_CHECKPOINT="${MMPOSE_CHECKPOINT:-${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_175.pth}"
RFDETR_MODEL="${RFDETR_MODEL:-rfdetr-2xlarge}"
DET_CONFIDENCE="${DET_CONFIDENCE:-0.5}"
POSE_MODE="${POSE_MODE:-detection_bbox}"
# Keep the bicycle large enough in frame for RTMPose (reject far zoomed-out cameras).
export CAMERA_MIN_DISTANCE="${CAMERA_MIN_DISTANCE:-4.0}"
export CAMERA_MAX_DISTANCE="${CAMERA_MAX_DISTANCE:-12.0}"
export CAMERA_MIN_BBOX_AREA_FRAC="${CAMERA_MIN_BBOX_AREA_FRAC:-0.04}"
export CAMERA_MAX_BBOX_AREA_FRAC="${CAMERA_MAX_BBOX_AREA_FRAC:-0.80}"
export CAMERA_MIN_VISIBLE_KEYPOINTS="${CAMERA_MIN_VISIBLE_KEYPOINTS:-14}"
export CAMERA_MIN_VISIBLE_FRAME_RATIO="${CAMERA_MIN_VISIBLE_FRAME_RATIO:-0.9}"
export CAMERA_MAX_LOW_BBOX_FRAME_FRAC="${CAMERA_MAX_LOW_BBOX_FRAME_FRAC:-0.50}"
# When 1, post-render QA skips (not fails) clips with poor framing; re-render those with OVERWRITE=1.
export QA_ALLOW_BBOX_FRAMING_FAILURES="${QA_ALLOW_BBOX_FRAMING_FAILURES:-1}"
export CAMERA_FIT_MARGIN="${CAMERA_FIT_MARGIN:-1.25}"
export CAMERA_MODE="${CAMERA_MODE:-track}"

if [[ "${SEED}" == "random" ]]; then
  SEED="$((RANDOM * 32768 + RANDOM))"
fi

echo "[blender-posemamba] RAW_ROOT=${RAW_ROOT}"
echo "[blender-posemamba] batch SEED=${SEED} NUM_CLIPS=${NUM_CLIPS} (same values => same scene + camera_seed; use SEED=random for new picks)"
echo "[blender-posemamba] SEQUENCE_ROOT=${SEQUENCE_ROOT}"
if [[ "${RAW_ROOT}" == "${SECONDARY_DATA_ROOT}"* ]]; then
  echo "[blender-posemamba] storage: secondary SSD (${SECONDARY_DATA_ROOT})"
fi
echo "[blender-posemamba] SYNC_WINDOW_SIZE=${SYNC_WINDOW_SIZE} (frames per clip)"
echo "[blender-posemamba] camera mode=${CAMERA_MODE}, distance ${CAMERA_MIN_DISTANCE}-${CAMERA_MAX_DISTANCE}m, min bbox ${CAMERA_MIN_BBOX_AREA_FRAC}"
echo "[blender-posemamba] pose-mode=${POSE_MODE}, rfdetr=${RFDETR_MODEL}, det-confidence=${DET_CONFIDENCE}"
if [[ -n "${BLENDER_THREADS}" ]]; then
  echo "[blender-posemamba] render jobs=${RENDER_JOBS}, blender_threads=${BLENDER_THREADS}"
else
  echo "[blender-posemamba] render jobs=${RENDER_JOBS} (blender_threads=auto: all CPUs when jobs=1)"
fi

if [[ "${SKIP_RENDER}" != "1" ]]; then
  if ! RESOLVED_BLENDER="$(resolve_blender_bin "${BLENDER}")"; then
    echo "error: Blender executable not found: ${BLENDER}" >&2
    echo "  Set BLENDER=/absolute/path/to/blender or BLENDER_PATH=..." >&2
    echo "  (common on this machine: \$HOME/Desktop/blender/blender)" >&2
    exit 1
  fi
  echo "[blender-posemamba] blender: ${RESOLVED_BLENDER}"
  BATCH_RENDER_ARGS=(
    --out "${RAW_ROOT}"
    --num-clips "${NUM_CLIPS}"
    --seed "${SEED}"
    --blender "${RESOLVED_BLENDER}"
    --sync-window-size "${SYNC_WINDOW_SIZE}"
    --jobs "${RENDER_JOBS}"
    --camera-min-distance "${CAMERA_MIN_DISTANCE}"
    --camera-max-distance "${CAMERA_MAX_DISTANCE}"
    --camera-min-bbox-area-frac "${CAMERA_MIN_BBOX_AREA_FRAC}"
    --camera-max-bbox-area-frac "${CAMERA_MAX_BBOX_AREA_FRAC}"
    --camera-min-visible-keypoints "${CAMERA_MIN_VISIBLE_KEYPOINTS}"
    --camera-min-visible-frame-ratio "${CAMERA_MIN_VISIBLE_FRAME_RATIO}"
    --camera-fit-margin "${CAMERA_FIT_MARGIN}"
    --camera-mode "${CAMERA_MODE}"
  )
  if [[ -n "${BLENDER_THREADS}" ]]; then
    BATCH_RENDER_ARGS+=(--blender-threads "${BLENDER_THREADS}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    BATCH_RENDER_ARGS+=(--overwrite)
  fi
  python "${REPO_ROOT}/data_generation_pipeline_tools/batch_render.py" "${BATCH_RENDER_ARGS[@]}"
fi

_run_py() {
  local env_name="$1"
  shift
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    python "$@"
  else
    conda run -n "${env_name}" python "$@"
  fi
}

QA_ARGS=(
  --raw-root "${RAW_ROOT}"
  --min-gt-bbox-area-frac "${CAMERA_MIN_BBOX_AREA_FRAC}"
  --max-low-bbox-frame-frac "${CAMERA_MAX_LOW_BBOX_FRAME_FRAC}"
)
if [[ "${QA_ALLOW_BBOX_FRAMING_FAILURES}" == "1" ]]; then
  QA_ARGS+=(--allow-bbox-framing-failures)
fi
_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/qa_raw_annotations.py" "${QA_ARGS[@]}"

if [[ "${SKIP_DETECTED_2D}" != "1" ]]; then
  if [[ "${SKIP_DETECTION}" != "1" ]]; then
    _run_py rfdetr "${REPO_ROOT}/3d_keypoint_detector_training/export_clip_detections.py" \
      --raw-root "${RAW_ROOT}" \
      --rfdetr-model "${RFDETR_MODEL}" \
      --det-confidence "${DET_CONFIDENCE}"
  fi
  _run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/export_detected_2d.py" \
    --raw-root "${RAW_ROOT}" \
    --mmpose-config "${MMPOSE_CONFIG}" \
    --mmpose-checkpoint "${MMPOSE_CHECKPOINT}" \
    --pose-mode "${POSE_MODE}"
fi

_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/qa_detected_2d.py" \
  --raw-root "${RAW_ROOT}"

_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/build_sequences.py" \
  --raw-root "${RAW_ROOT}" \
  --output-root "${SEQUENCE_ROOT}" \
  --window-size "${WINDOW_SIZE}" \
  --stride "${STRIDE}" \
  --eval-stride "${WINDOW_SIZE}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --seed "${SEED}" \
  --input-2d detected \
  --bbox-source detection \
  --dataset-tag detected2d

echo "[blender-posemamba] Done. Pickles: ${SEQUENCE_ROOT}/PoseMamba_f${WINDOW_SIZE}s${STRIDE}_detected2d/BICYCLE/"
echo "[blender-posemamba] Train: DATASET_TAG=detected2d ./3d_keypoint_detector_training/start_training.sh"

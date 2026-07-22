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
    TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${SECONDARY_DATA_ROOT}/mujoco_blender_trajectories}"
  else
    RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/raw_blender_posemamba}"
    SEQUENCE_ROOT="${SEQUENCE_ROOT:-${REPO_ROOT}/data/posemamba_training_sequences}"
    TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${REPO_ROOT}/data/mujoco_blender_trajectories}"
  fi
}

_resolve_data_roots
WINDOW_SIZE="${WINDOW_SIZE:-243}"
STRIDE="${STRIDE:-81}"

SYNC_WINDOW_SIZE="${SYNC_WINDOW_SIZE:-2187}"
TRAJECTORY_FRAMES="${TRAJECTORY_FRAMES:-2187}"
NUM_CLIPS="${NUM_CLIPS:-10}"
# Batch RNG seed: picks camera_seed per clip; scene choice uses weighted random unless BALANCE_SCENES=1.
# Use SEED=random for a new draw each run, or SEED=42 for reproducibility.
SEED="${SEED:-random}"
# 1 = scan existing clips under RAW_ROOT and assign new scenes to even out per-scene counts (default).
# 0 = weighted random scene picks from scenes.yaml (legacy behavior).
BALANCE_SCENES="${BALANCE_SCENES:-1}"
if [[ "${SEED}" == "random" ]]; then
  SEED="$((RANDOM * 32768 + RANDOM))"
fi
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-${NUM_CLIPS}}"
CAMERAS_PER_TRAJECTORY="${CAMERAS_PER_TRAJECTORY:-1}"
USE_MUJOCO_TRAJECTORIES="${USE_MUJOCO_TRAJECTORIES:-1}"
GENERATE_TRAJECTORIES="${GENERATE_TRAJECTORIES:-1}"
# 1 = regenerate MuJoCo CSVs when safe (no on-disk clips bound to that trajectory).
# 0 = reuse existing CSVs (default — preserves clip/trajectory history).
TRAJECTORY_ALWAYS_REGENERATE="${TRAJECTORY_ALWAYS_REGENERATE:-0}"
# 1 = append NUM_TRAJECTORIES NEW trajectories per run (default).
# 0 = legacy contract: ensure indices 00000..NUM_TRAJECTORIES-1 exist.
TRAJECTORY_APPEND_MODE="${TRAJECTORY_APPEND_MODE:-1}"
TRAJECTORY_MANIFEST="${TRAJECTORY_MANIFEST:-${TRAJECTORY_ROOT}/manifest.csv}"
TRAJECTORY_PATTERN="${TRAJECTORY_PATTERN:-composite}"
TRAJECTORY_SEED="${TRAJECTORY_SEED:-${SEED}}"
TRAJECTORY_PHYSICS_HZ="${TRAJECTORY_PHYSICS_HZ:-200}"
TRAJECTORY_DISPLAY_HZ="${TRAJECTORY_DISPLAY_HZ:-60}"
TRAJECTORY_MIN_SPEED_MPS="${TRAJECTORY_MIN_SPEED_MPS:-3.5}"
TRAJECTORY_MAX_SPEED_MPS="${TRAJECTORY_MAX_SPEED_MPS:-8.0}"
TRAJECTORY_MAX_YAW_RATE_DEG_S="${TRAJECTORY_MAX_YAW_RATE_DEG_S:-25.0}"
TRAJECTORY_COMPOSITE_PROFILE="${TRAJECTORY_COMPOSITE_PROFILE:-stable}"
TRAJECTORY_MAX_ROLL_DEG="${TRAJECTORY_MAX_ROLL_DEG:-40.0}"
TRAJECTORY_GEN_RETRIES="${TRAJECTORY_GEN_RETRIES:-8}"
TRAJECTORY_SEGMENT_MIN_SECONDS="${TRAJECTORY_SEGMENT_MIN_SECONDS:-3.0}"
TRAJECTORY_SEGMENT_MAX_SECONDS="${TRAJECTORY_SEGMENT_MAX_SECONDS:-7.0}"
OVERWRITE="${OVERWRITE:-0}"
# RENDER_JOBS=1 (default): one clip at a time; batch_render gives Blender all CPU cores (-t 0).
# RENDER_JOBS>1: render multiple clips in parallel; cores are split across Blender processes.
RENDER_JOBS="${RENDER_JOBS:-1}"
# Optional override for Blender -t (0 = all logical CPUs). Leave unset for automatic choice.
BLENDER_THREADS="${BLENDER_THREADS:-}"
BLENDER="${BLENDER:-blender}"
MUJOCO_PYTHON="${MUJOCO_PYTHON:-}"
MUJOCO_CONDA_ENV="${MUJOCO_CONDA_ENV:-posemamba}"
RENDER_VERBOSE_PROGRESS="${RENDER_VERBOSE_PROGRESS:-0}"
RENDER_FORMAT="${RENDER_FORMAT:-PNG}"
BLENDER_RESOLUTION_PERCENTAGE="${BLENDER_RESOLUTION_PERCENTAGE:-100}"
BLENDER_CYCLES_SAMPLES="${BLENDER_CYCLES_SAMPLES:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SEQUENCE_SPLIT_MODE="${SEQUENCE_SPLIT_MODE:-window}"
# Frame cleanup mode: none | end | per_clip
# - none: keep rendered images
# - end: remove all clip_*/frames after pickle build
# - per_clip: remove clip_*/frames immediately after each clip finishes detected-2d export
CLEANUP_RENDER_FRAMES_MODE="${CLEANUP_RENDER_FRAMES_MODE:-per_clip}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_DETECTION="${SKIP_DETECTION:-0}"
SKIP_DETECTED_2D="${SKIP_DETECTED_2D:-0}"
SKIP_BUILD_SEQUENCES="${SKIP_BUILD_SEQUENCES:-0}"
MMPOSE_CONFIG="${MMPOSE_CONFIG:-${REPO_ROOT}/2d_keypoint_detector_training/rtmpose_bicycle_full.py}"
MMPOSE_CHECKPOINT="${MMPOSE_CHECKPOINT:-${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_175.pth}"
RFDETR_MODEL="${RFDETR_MODEL:-rfdetr-2xlarge}"
DET_CONFIDENCE="${DET_CONFIDENCE:-0.3}"
POSE_MODE="${POSE_MODE:-detection_bbox}"
DETECTED_EXPORT_RESUME="${DETECTED_EXPORT_RESUME:-1}"
DETECTED_SHARD_INDEX="${DETECTED_SHARD_INDEX:-}"
DETECTED_NUM_SHARDS="${DETECTED_NUM_SHARDS:-}"
MMPOSE_DEVICE="${MMPOSE_DEVICE:-cuda:0}"
# Keep the bicycle large enough in frame for RTMPose (reject far zoomed-out cameras).
export CAMERA_MIN_DISTANCE="${CAMERA_MIN_DISTANCE:-4.0}"
export CAMERA_MAX_DISTANCE="${CAMERA_MAX_DISTANCE:-10.0}"
export CAMERA_MIN_BBOX_AREA_FRAC="${CAMERA_MIN_BBOX_AREA_FRAC:-0.01}"
export CAMERA_MAX_BBOX_AREA_FRAC="${CAMERA_MAX_BBOX_AREA_FRAC:-0.80}"
export CAMERA_MIN_VISIBLE_KEYPOINTS="${CAMERA_MIN_VISIBLE_KEYPOINTS:-14}"
export CAMERA_MIN_VISIBLE_FRAME_RATIO="${CAMERA_MIN_VISIBLE_FRAME_RATIO:-0.9}"
export CAMERA_MAX_LOW_BBOX_FRAME_FRAC="${CAMERA_MAX_LOW_BBOX_FRAME_FRAC:-0.50}"
# When 1, post-render QA skips (not fails) clips with poor framing; re-render those with OVERWRITE=1.
export QA_ALLOW_BBOX_FRAMING_FAILURES="${QA_ALLOW_BBOX_FRAMING_FAILURES:-1}"
export CAMERA_FIT_MARGIN="${CAMERA_FIT_MARGIN:-1.25}"
export CAMERA_MODE="${CAMERA_MODE:-track}"

echo "[blender-posemamba] RAW_ROOT=${RAW_ROOT}"
echo "[blender-posemamba] batch SEED=${SEED} NUM_CLIPS=${NUM_CLIPS} BALANCE_SCENES=${BALANCE_SCENES} (use SEED=random for new camera seeds)"
echo "[blender-posemamba] SEQUENCE_ROOT=${SEQUENCE_ROOT}"
if [[ "${RAW_ROOT}" == "${SECONDARY_DATA_ROOT}"* ]]; then
  echo "[blender-posemamba] storage: secondary SSD (${SECONDARY_DATA_ROOT})"
fi
echo "[blender-posemamba] SYNC_WINDOW_SIZE=${SYNC_WINDOW_SIZE} (frames per clip)"
if [[ "${USE_MUJOCO_TRAJECTORIES}" == "1" ]]; then
  echo "[blender-posemamba] MuJoCo trajectories: ${NUM_TRAJECTORIES} x ${CAMERAS_PER_TRAJECTORY} camera(s), frames=${TRAJECTORY_FRAMES}, manifest=${TRAJECTORY_MANIFEST}"
fi
echo "[blender-posemamba] trajectory cache policy: always_regenerate=${TRAJECTORY_ALWAYS_REGENERATE}"
echo "[blender-posemamba] camera mode=${CAMERA_MODE}, distance ${CAMERA_MIN_DISTANCE}-${CAMERA_MAX_DISTANCE}m, min bbox ${CAMERA_MIN_BBOX_AREA_FRAC} (QA hard floor ${CAMERA_MIN_BBOX_HARD_FLOOR:-${CAMERA_MIN_BBOX_AREA_FRAC}})"
echo "[blender-posemamba] pose-mode=${POSE_MODE}, rfdetr=${RFDETR_MODEL}, det-confidence=${DET_CONFIDENCE}"
echo "[blender-posemamba] render-format=${RENDER_FORMAT}, resolution=${BLENDER_RESOLUTION_PERCENTAGE}%, cycles_samples=${BLENDER_CYCLES_SAMPLES}"
echo "[blender-posemamba] frame cleanup mode=${CLEANUP_RENDER_FRAMES_MODE}"
if [[ -n "${BLENDER_THREADS}" ]]; then
  echo "[blender-posemamba] render jobs=${RENDER_JOBS}, blender_threads=${BLENDER_THREADS}"
else
  echo "[blender-posemamba] render jobs=${RENDER_JOBS} (blender_threads=auto: all CPUs when jobs=1)"
fi

_run_mujoco_py() {
  if [[ -n "${MUJOCO_PYTHON}" ]]; then
    "${MUJOCO_PYTHON}" "$@"
  elif [[ "${CONDA_DEFAULT_ENV:-}" == "${MUJOCO_CONDA_ENV}" ]]; then
    python "$@"
  else
    conda run -n "${MUJOCO_CONDA_ENV}" python "$@"
  fi
}

_run_py() {
  local env_name="$1"
  shift
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    python "$@"
  else
    conda run -n "${env_name}" python "$@"
  fi
}

if [[ "${SKIP_RENDER}" != "1" ]]; then
  _run_py posemamba "${REPO_ROOT}/data_generation_pipeline_tools/sync_pipeline_state.py" \
    --raw-root "${RAW_ROOT}" \
    --trajectory-root "${TRAJECTORY_ROOT}" \
    --trajectory-manifest "${TRAJECTORY_MANIFEST}" \
    --pattern "${TRAJECTORY_PATTERN}" \
    --num-trajectories "${NUM_TRAJECTORIES}" \
    --trajectory-frames "${TRAJECTORY_FRAMES}" \
    --display-hz "${TRAJECTORY_DISPLAY_HZ}" \
    --seed-base "${TRAJECTORY_SEED}" \
    $([[ "${TRAJECTORY_ALWAYS_REGENERATE}" == "1" ]] && echo --always-regenerate || echo --no-always-regenerate) \
    $([[ "${TRAJECTORY_APPEND_MODE}" == "1" ]] && echo --append || echo --no-append) \
    $([[ "${OVERWRITE}" == "1" ]] && echo --overwrite) \
    --audit-only

  if [[ "${USE_MUJOCO_TRAJECTORIES}" == "1" && "${GENERATE_TRAJECTORIES}" == "1" ]]; then
    mkdir -p "${TRAJECTORY_ROOT}"
    _run_mujoco_py "${REPO_ROOT}/data_generation_pipeline_tools/ensure_mujoco_trajectories.py" \
      --trajectory-root "${TRAJECTORY_ROOT}" \
      --trajectory-manifest "${TRAJECTORY_MANIFEST}" \
      --raw-root "${RAW_ROOT}" \
      --pattern "${TRAJECTORY_PATTERN}" \
      --num-trajectories "${NUM_TRAJECTORIES}" \
      --trajectory-frames "${TRAJECTORY_FRAMES}" \
      --display-hz "${TRAJECTORY_DISPLAY_HZ}" \
      --seed-base "${TRAJECTORY_SEED}" \
      --physics-hz "${TRAJECTORY_PHYSICS_HZ}" \
      $([[ "${TRAJECTORY_ALWAYS_REGENERATE}" == "1" ]] && echo --always-regenerate || echo --no-always-regenerate) \
      $([[ "${TRAJECTORY_APPEND_MODE}" == "1" ]] && echo --append || echo --no-append) \
      $([[ "${OVERWRITE}" == "1" ]] && echo --overwrite) \
      --gen-retries "${TRAJECTORY_GEN_RETRIES}" \
      --min-target-velocity-mps "${TRAJECTORY_MIN_SPEED_MPS}" \
      --max-target-velocity-mps "${TRAJECTORY_MAX_SPEED_MPS}" \
      --max-yaw-rate-deg-s "${TRAJECTORY_MAX_YAW_RATE_DEG_S}" \
      --composite-profile "${TRAJECTORY_COMPOSITE_PROFILE}" \
      --max-roll-deg "${TRAJECTORY_MAX_ROLL_DEG}" \
      --segment-min-seconds "${TRAJECTORY_SEGMENT_MIN_SECONDS}" \
      --segment-max-seconds "${TRAJECTORY_SEGMENT_MAX_SECONDS}"
  fi
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
    --render-format "${RENDER_FORMAT}"
    --resolution-percentage "${BLENDER_RESOLUTION_PERCENTAGE}"
    --cycles-samples "${BLENDER_CYCLES_SAMPLES}"
    --camera-min-distance "${CAMERA_MIN_DISTANCE}"
    --camera-max-distance "${CAMERA_MAX_DISTANCE}"
    --camera-min-bbox-area-frac "${CAMERA_MIN_BBOX_AREA_FRAC}"
    --camera-max-bbox-area-frac "${CAMERA_MAX_BBOX_AREA_FRAC}"
    --camera-min-visible-keypoints "${CAMERA_MIN_VISIBLE_KEYPOINTS}"
    --camera-min-visible-frame-ratio "${CAMERA_MIN_VISIBLE_FRAME_RATIO}"
    --camera-fit-margin "${CAMERA_FIT_MARGIN}"
    --camera-mode "${CAMERA_MODE}"
  )
  if [[ "${USE_MUJOCO_TRAJECTORIES}" == "1" ]]; then
    BATCH_RENDER_ARGS=(
      "${BATCH_RENDER_ARGS[@]}"
      --trajectory-manifest "${TRAJECTORY_MANIFEST}"
      --cameras-per-trajectory "${CAMERAS_PER_TRAJECTORY}"
      --num-clips "${NUM_TRAJECTORIES}"
    )
  fi
  if [[ -n "${BLENDER_THREADS}" ]]; then
    BATCH_RENDER_ARGS+=(--blender-threads "${BLENDER_THREADS}")
  fi
  if [[ "${RENDER_VERBOSE_PROGRESS}" == "1" ]]; then
    BATCH_RENDER_ARGS+=(--verbose-render-progress)
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    BATCH_RENDER_ARGS+=(--overwrite)
  fi
  if [[ "${BALANCE_SCENES}" == "1" ]]; then
    BATCH_RENDER_ARGS+=(--balance-scenes)
  fi
  python "${REPO_ROOT}/data_generation_pipeline_tools/batch_render.py" "${BATCH_RENDER_ARGS[@]}"

  _run_py posemamba "${REPO_ROOT}/data_generation_pipeline_tools/sync_pipeline_state.py" \
    --raw-root "${RAW_ROOT}" \
    --trajectory-root "${TRAJECTORY_ROOT}" \
    --trajectory-manifest "${TRAJECTORY_MANIFEST}" \
    --pattern "${TRAJECTORY_PATTERN}" \
    --num-trajectories "${NUM_TRAJECTORIES}" \
    --trajectory-frames "${TRAJECTORY_FRAMES}" \
    --display-hz "${TRAJECTORY_DISPLAY_HZ}" \
    --seed-base "${TRAJECTORY_SEED}" \
    --rebuild-raw-manifest
fi

QA_ARGS=(
  --raw-root "${RAW_ROOT}"
  --min-gt-bbox-area-frac "${CAMERA_MIN_BBOX_AREA_FRAC}"
  --min-gt-bbox-hard-floor "${CAMERA_MIN_BBOX_HARD_FLOOR:-${CAMERA_MIN_BBOX_AREA_FRAC}}"
  --max-low-bbox-frame-frac "${CAMERA_MAX_LOW_BBOX_FRAME_FRAC}"
)
if [[ "${QA_ALLOW_BBOX_FRAMING_FAILURES}" == "1" ]]; then
  QA_ARGS+=(--allow-bbox-framing-failures)
fi
_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/qa_raw_annotations.py" "${QA_ARGS[@]}"

if [[ "${SKIP_DETECTED_2D}" != "1" ]]; then
  DETECTED_COMMON_ARGS=()
  if [[ "${DETECTED_EXPORT_RESUME}" == "1" ]]; then
    DETECTED_COMMON_ARGS+=(--resume)
  fi
  if [[ -n "${DETECTED_SHARD_INDEX}" || -n "${DETECTED_NUM_SHARDS}" ]]; then
    DETECTED_COMMON_ARGS+=(--shard-index "${DETECTED_SHARD_INDEX}" --num-shards "${DETECTED_NUM_SHARDS}")
  fi
  DETECTED_2D_ARGS=("${DETECTED_COMMON_ARGS[@]}")
  if [[ "${CLEANUP_RENDER_FRAMES_MODE}" == "per_clip" ]]; then
    DETECTED_2D_ARGS+=(--cleanup-frames)
  fi
  if [[ "${SKIP_DETECTION}" != "1" ]]; then
    _run_py rfdetr "${REPO_ROOT}/3d_keypoint_detector_training/export_clip_detections.py" \
      --raw-root "${RAW_ROOT}" \
      --rfdetr-model "${RFDETR_MODEL}" \
      --det-confidence "${DET_CONFIDENCE}" \
      "${DETECTED_COMMON_ARGS[@]}"
  fi
  _run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/export_detected_2d.py" \
    --raw-root "${RAW_ROOT}" \
    --mmpose-config "${MMPOSE_CONFIG}" \
    --mmpose-checkpoint "${MMPOSE_CHECKPOINT}" \
    --pose-mode "${POSE_MODE}" \
    --device "${MMPOSE_DEVICE}" \
    "${DETECTED_2D_ARGS[@]}"
fi

_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/qa_detected_2d.py" \
  --raw-root "${RAW_ROOT}"

if [[ "${SKIP_BUILD_SEQUENCES}" != "1" ]]; then
  _run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/build_sequences.py" \
    --raw-root "${RAW_ROOT}" \
    --output-root "${SEQUENCE_ROOT}" \
    --window-size "${WINDOW_SIZE}" \
    --stride "${STRIDE}" \
    --eval-stride "${WINDOW_SIZE}" \
    --val-ratio "${VAL_RATIO}" \
    --test-ratio "${TEST_RATIO}" \
    --split-mode "${SEQUENCE_SPLIT_MODE}" \
    --seed "${SEED}" \
    --input-2d detected \
    --bbox-source detection \
    --dataset-tag detected2d
else
  echo "[blender-posemamba] SKIP_BUILD_SEQUENCES=1 — skipping PoseMamba pickle build"
fi

if [[ "${CLEANUP_RENDER_FRAMES_MODE}" == "end" ]]; then
  deleted_frames_dirs=0
  for clip_dir in "${RAW_ROOT}"/clip_*; do
    [[ -d "${clip_dir}" ]] || continue
    if [[ -d "${clip_dir}/frames" ]]; then
      rm -rf "${clip_dir}/frames"
      deleted_frames_dirs=$((deleted_frames_dirs + 1))
    fi
  done
  echo "[blender-posemamba] cleanup: removed ${deleted_frames_dirs} clip frame folder(s) under ${RAW_ROOT}"
  echo "[blender-posemamba] cleanup: set CLEANUP_RENDER_FRAMES_MODE=none to keep rendered images"
fi

echo "[blender-posemamba] Done. Pickles: ${SEQUENCE_ROOT}/PoseMamba_f${WINDOW_SIZE}s${STRIDE}_detected2d/BICYCLE/"
echo "[blender-posemamba] Train: DATASET_TAG=detected2d ./3d_keypoint_detector_training/start_training.sh"

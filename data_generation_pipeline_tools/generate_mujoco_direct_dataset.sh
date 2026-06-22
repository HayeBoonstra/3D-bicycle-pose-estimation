#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hayepc/3D-bicycle-pose-estimation}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/raw_3D_keypoint_annotations}"
SEQUENCE_ROOT="${SEQUENCE_ROOT:-${REPO_ROOT}/data/posemamba_training_sequences}"
# Long clips so split_clips(window=243, stride=81) yields multiple contiguous training windows (H36M-style).
FRAMES="${FRAMES:-729}"
FPS="${FPS:-60}"
PATTERNS="${PATTERNS:-straight,left,right,sine,zigzag}"
TRAJECTORIES_PER_PATTERN="${TRAJECTORIES_PER_PATTERN:-40}"
NUM_CAMERAS="${NUM_CAMERAS:-16}"
MIN_CAMERA_DISTANCE="${MIN_CAMERA_DISTANCE:-4.0}"
MAX_CAMERA_DISTANCE="${MAX_CAMERA_DISTANCE:-14.0}"
MIN_CAMERA_ELEVATION_DEG="${MIN_CAMERA_ELEVATION_DEG:-5.0}"
MAX_CAMERA_ELEVATION_DEG="${MAX_CAMERA_ELEVATION_DEG:-55.0}"
MIN_FOV_DEG="${MIN_FOV_DEG:-35.0}"
MAX_FOV_DEG="${MAX_FOV_DEG:-80.0}"
MIN_VISIBLE_KEYPOINTS="${MIN_VISIBLE_KEYPOINTS:-18}"
MIN_VISIBLE_FRAME_RATIO="${MIN_VISIBLE_FRAME_RATIO:-1.0}"
CAMERA_FIT_MARGIN="${CAMERA_FIT_MARGIN:-1.15}"
CAMERA_MAX_TRIES="${CAMERA_MAX_TRIES:-1000}"
MIN_SPEED_MPS="${MIN_SPEED_MPS:-2.0}"
MAX_SPEED_MPS="${MAX_SPEED_MPS:-8.0}"
MIN_CRANK_HZ="${MIN_CRANK_HZ:-0.8}"
MAX_CRANK_HZ="${MAX_CRANK_HZ:-2.4}"
MIN_TURN_RATE_DEG="${MIN_TURN_RATE_DEG:-5.0}"
MAX_TURN_RATE_DEG="${MAX_TURN_RATE_DEG:-35.0}"
MIN_SINE_YAW_DEG="${MIN_SINE_YAW_DEG:-10.0}"
MAX_SINE_YAW_DEG="${MAX_SINE_YAW_DEG:-45.0}"
MIN_SINE_FREQUENCY_HZ="${MIN_SINE_FREQUENCY_HZ:-0.05}"
MAX_SINE_FREQUENCY_HZ="${MAX_SINE_FREQUENCY_HZ:-1.00}"
# Match PoseMamba_train_h36m_B.yaml: clip_len 243, offline stride 81 (MB3D_f243s81).
# See 3d_keypoint_detector_training/README.md — YAML data_stride is not applied at train time.
WINDOW_SIZE="${WINDOW_SIZE:-243}"
STRIDE="${STRIDE:-81}"
SEED="${SEED:-7}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
VISUALIZE="${VISUALIZE:-1}"
VIS_MAX_CLIPS="${VIS_MAX_CLIPS:-2}"
VIS_MAX_FRAMES="${VIS_MAX_FRAMES:-${FRAMES}}"
VIS_FRAME_STEP="${VIS_FRAME_STEP:-1}"
VIS_COORD_FRAME="${VIS_COORD_FRAME:-bicycle}"
# When 1, run ffmpeg after frames (requires ffmpeg in PATH). Set VIS_ENCODE_VIDEO=0 to skip.
VIS_ENCODE_VIDEO="${VIS_ENCODE_VIDEO:-1}"
VIS_FPS="${VIS_FPS:-30}"

IFS=',' read -r -a PATTERN_LIST <<< "${PATTERNS}"
pattern_idx=0
for pattern in "${PATTERN_LIST[@]}"; do
  pattern="$(echo "${pattern}" | xargs)"
  if [[ -z "${pattern}" ]]; then
    continue
  fi
  for trajectory_idx in $(seq 0 $((TRAJECTORIES_PER_PATTERN - 1))); do
    trajectory_seed=$((SEED + trajectory_idx + 1000 * pattern_idx))
    clip_id="PoseMamba_${pattern}_traj$(printf '%04d' "${trajectory_idx}")"
    python "${REPO_ROOT}/Mujoco_bicycle_path_generator/export_posemamba_annotations.py" \
      --out "${RAW_ROOT}" \
      --clip-id "${clip_id}" \
      --scene-id "PoseMamba_training_data" \
      --pattern "${pattern}" \
      --trajectory-seed "${trajectory_seed}" \
      --frames "${FRAMES}" \
      --fps "${FPS}" \
      --num-cameras "${NUM_CAMERAS}" \
      --seed "${trajectory_seed}" \
      --min-camera-distance "${MIN_CAMERA_DISTANCE}" \
      --max-camera-distance "${MAX_CAMERA_DISTANCE}" \
      --min-camera-elevation-deg "${MIN_CAMERA_ELEVATION_DEG}" \
      --max-camera-elevation-deg "${MAX_CAMERA_ELEVATION_DEG}" \
      --min-fov-deg "${MIN_FOV_DEG}" \
      --max-fov-deg "${MAX_FOV_DEG}" \
      --min-visible-keypoints "${MIN_VISIBLE_KEYPOINTS}" \
      --min-visible-frame-ratio "${MIN_VISIBLE_FRAME_RATIO}" \
      --camera-fit-margin "${CAMERA_FIT_MARGIN}" \
      --camera-max-tries "${CAMERA_MAX_TRIES}" \
      --min-speed-mps "${MIN_SPEED_MPS}" \
      --max-speed-mps "${MAX_SPEED_MPS}" \
      --min-crank-hz "${MIN_CRANK_HZ}" \
      --max-crank-hz "${MAX_CRANK_HZ}" \
      --min-turn-rate-deg "${MIN_TURN_RATE_DEG}" \
      --max-turn-rate-deg "${MAX_TURN_RATE_DEG}" \
      --min-sine-yaw-deg "${MIN_SINE_YAW_DEG}" \
      --max-sine-yaw-deg "${MAX_SINE_YAW_DEG}" \
      --min-sine-frequency-hz "${MIN_SINE_FREQUENCY_HZ}" \
      --max-sine-frequency-hz "${MAX_SINE_FREQUENCY_HZ}"
  done
  pattern_idx=$((pattern_idx + 1))
done

python "${REPO_ROOT}/3d_keypoint_detector_training/qa_raw_annotations.py" \
  --raw-root "${RAW_ROOT}"

if [[ "${VISUALIZE}" == "1" ]]; then
  VIS_PY_ARGS=(
    --raw-root "${RAW_ROOT}"
    --max-clips "${VIS_MAX_CLIPS}"
    --max-frames "${VIS_MAX_FRAMES}"
    --frame-step "${VIS_FRAME_STEP}"
    --coord-frame "${VIS_COORD_FRAME}"
  )
  if [[ "${VIS_ENCODE_VIDEO}" == "1" ]]; then
    VIS_PY_ARGS+=(--encode-video --fps "${VIS_FPS}")
  fi
  python "${REPO_ROOT}/data_generation_pipeline_tools/visualize_raw_annotations.py" "${VIS_PY_ARGS[@]}"
fi

python "${REPO_ROOT}/3d_keypoint_detector_training/build_sequences.py" \
  --raw-root "${RAW_ROOT}" \
  --output-root "${SEQUENCE_ROOT}" \
  --window-size "${WINDOW_SIZE}" \
  --stride "${STRIDE}" \
  --eval-stride "${WINDOW_SIZE}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --seed "${SEED}"

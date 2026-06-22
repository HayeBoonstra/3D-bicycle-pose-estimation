#!/usr/bin/env bash
set -euo pipefail

# End-to-end: image sequence -> keypoints_3d.npz (243 frames)
# Stage 1: RF-DETR bboxes. Stage 2: RTMPose keypoints via inference_topdown on full frames.
# Uses conda envs: rfdetr, mmpose, posemamba

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="${REPO_ROOT}/1_full_detection_pipeline"

FRAMES_DIR="${REPO_ROOT}/1_full_detection_pipeline/input_sequence"
OUTPUT_DIR="${REPO_ROOT}/1_full_detection_pipeline/output"
MMPOSE_CONFIG="${REPO_ROOT}/2d_keypoint_detector_training/rtmpose_bicycle_full.py"
MMPOSE_CHECKPOINT="${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_175.pth"
LIFTER_CHECKPOINT="${REPO_ROOT}/1_full_detection_pipeline/posemamba_X_best_epoch.bin"
LIFTER_CONFIG="${REPO_ROOT}/1_full_detection_pipeline/PoseMamba_train_bicycle_X.generated.yaml"
RFDETR_MODEL="rfdetr-2xlarge"
DET_CONFIDENCE="0.5"
POSE_MODE="detection_bbox"
RESUME=0
VISUALIZE=1
NO_VIDEO=0
VIS_FPS=30

unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --frames-dir PATH          Input image directory (default: input_sequence)
  --output-dir PATH          Output directory (default: output)
  --mmpose-config PATH       RTMPose config .py
  --mmpose-checkpoint PATH   RTMPose .pth weights
  --lifter-checkpoint PATH   PoseMamba best_epoch.bin
  --lifter-config PATH       PoseMamba fallback YAML
  --rfdetr-model ID          RF-DETR model id (default: rfdetr-2xlarge)
  --det-confidence FLOAT     Detection threshold (default: 0.5)
  --pose-mode MODE           Stage 2: detection_bbox | full_image | auto (default: detection_bbox)
  --resume                   Skip stages whose outputs already exist
  --no-visualize             Skip intermediate visualization step
  --no-video                 Write vis frame PNGs only (no MP4)
  --vis-fps FPS              Frame rate for visualization videos (default: 30)
  -h, --help                 Show this help

Requires conda envs: rfdetr, mmpose, posemamba
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frames-dir) FRAMES_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --mmpose-config) MMPOSE_CONFIG="$2"; shift 2 ;;
    --mmpose-checkpoint) MMPOSE_CHECKPOINT="$2"; shift 2 ;;
    --lifter-checkpoint) LIFTER_CHECKPOINT="$2"; shift 2 ;;
    --lifter-config) LIFTER_CONFIG="$2"; shift 2 ;;
    --rfdetr-model) RFDETR_MODEL="$2"; shift 2 ;;
    --det-confidence) DET_CONFIDENCE="$2"; shift 2 ;;
    --pose-mode) POSE_MODE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no-visualize) VISUALIZE=0; shift ;;
    --no-video) NO_VIDEO=1; shift ;;
    --vis-fps) VIS_FPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

FRAMES_DIR="$(cd "${FRAMES_DIR}" && pwd)"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

STAGE_ARGS=(
  --frames-dir "${FRAMES_DIR}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${RESUME}" -eq 1 ]]; then
  STAGE_ARGS+=(--resume)
fi

echo "[pipeline] frames: ${FRAMES_DIR}"
echo "[pipeline] output: ${OUTPUT_DIR}"
echo "[pipeline] stage2 pose-mode: ${POSE_MODE}"

conda run -n rfdetr python "${PIPELINE_DIR}/stage1_detect.py" \
  "${STAGE_ARGS[@]}" \
  --rfdetr-model "${RFDETR_MODEL}" \
  --det-confidence "${DET_CONFIDENCE}"

conda run -n mmpose python "${PIPELINE_DIR}/stage2_pose2d.py" \
  "${STAGE_ARGS[@]}" \
  --mmpose-config "${MMPOSE_CONFIG}" \
  --mmpose-checkpoint "${MMPOSE_CHECKPOINT}" \
  --pose-mode "${POSE_MODE}"

conda run -n posemamba python "${PIPELINE_DIR}/stage3_lift3d.py" \
  "${STAGE_ARGS[@]}" \
  --lifter-checkpoint "${LIFTER_CHECKPOINT}" \
  --lifter-config "${LIFTER_CONFIG}"

echo "[pipeline] Done. Final output: ${OUTPUT_DIR}/keypoints_3d.npz"

if [[ "${VISUALIZE}" -eq 1 ]]; then
  VIS_ARGS=(
    "${PIPELINE_DIR}/visualize_intermediates.py"
    --output-dir "${OUTPUT_DIR}"
    --fps "${VIS_FPS}"
  )
  if [[ "${RESUME}" -eq 1 ]]; then
    VIS_ARGS+=(--resume)
  fi
  if [[ "${NO_VIDEO}" -eq 1 ]]; then
    VIS_ARGS+=(--no-video)
  fi
  echo "[pipeline] Rendering intermediate visualizations..."
  conda run -n mmpose python "${VIS_ARGS[@]}" --stages detections 2d
  conda run -n posemamba python "${VIS_ARGS[@]}" --stages 3d
  echo "[pipeline] Visualizations: ${OUTPUT_DIR}/vis/"
fi

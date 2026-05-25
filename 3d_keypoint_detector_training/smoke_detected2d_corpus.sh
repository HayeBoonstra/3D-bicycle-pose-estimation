#!/usr/bin/env bash
# Smoke-test detected-2D corpus build using MuJoCo viz PNGs as frames (dev only).
# Full Blender corpus: data_generation_pipeline_tools/generate_blender_posemamba_dataset.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_RAW="${SRC_RAW:-${REPO_ROOT}/data/raw_3D_keypoint_annotations}"
SMOKE_RAW="${SMOKE_RAW:-${REPO_ROOT}/data/raw_blender_posemamba_smoke}"
CLIP_ID="${CLIP_ID:-PoseMamba_left_traj0000_cam00}"
LIMIT_CLIPS="${LIMIT_CLIPS:-1}"
LIMIT_FRAMES="${LIMIT_FRAMES:-243}"
WINDOW_SIZE="${WINDOW_SIZE:-243}"
STRIDE="${STRIDE:-81}"
MAX_BATCHES="${MAX_BATCHES:-2}"
EPOCHS="${EPOCHS:-1}"

echo "[smoke] Preparing ${SMOKE_RAW}/${CLIP_ID} from ${SRC_RAW}"
rm -rf "${SMOKE_RAW}"
mkdir -p "${SMOKE_RAW}/${CLIP_ID}"
cp -a "${SRC_RAW}/${CLIP_ID}/per_frame_annotations" "${SMOKE_RAW}/${CLIP_ID}/"
cp -a "${SRC_RAW}/${CLIP_ID}/keypoints_3d.jsonl" "${SMOKE_RAW}/${CLIP_ID}/"
cp -a "${SRC_RAW}/${CLIP_ID}/camera.json" "${SMOKE_RAW}/${CLIP_ID}/"
cp -a "${SRC_RAW}/${CLIP_ID}/render_config.json" "${SMOKE_RAW}/${CLIP_ID}/" 2>/dev/null || true
mkdir -p "${SMOKE_RAW}/${CLIP_ID}/frames"
cp -a "${SRC_RAW}/visualizations/${CLIP_ID}/frame_"*.png "${SMOKE_RAW}/${CLIP_ID}/frames/"

_run_py() {
  local env_name="$1"
  shift
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    python "$@"
  else
    conda run -n "${env_name}" python "$@"
  fi
}

_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/qa_raw_annotations.py" --raw-root "${SMOKE_RAW}"

_run_py rfdetr "${REPO_ROOT}/3d_keypoint_detector_training/export_clip_detections.py" \
  --raw-root "${SMOKE_RAW}" --limit-clips "${LIMIT_CLIPS}" --limit-frames "${LIMIT_FRAMES}"

_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/export_detected_2d.py" \
  --raw-root "${SMOKE_RAW}" --limit-clips "${LIMIT_CLIPS}" --limit-frames "${LIMIT_FRAMES}" \
  --pose-mode detection_bbox

_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/qa_detected_2d.py" --raw-root "${SMOKE_RAW}"

_run_py mmpose "${REPO_ROOT}/3d_keypoint_detector_training/build_sequences.py" \
  --raw-root "${SMOKE_RAW}" \
  --output-root "${REPO_ROOT}/data/posemamba_training_sequences_smoke" \
  --window-size "${WINDOW_SIZE}" \
  --stride "${STRIDE}" \
  --eval-stride "${WINDOW_SIZE}" \
  --val-ratio 0.0 \
  --test-ratio 0.0 \
  --input-2d detected \
  --bbox-source detection \
  --dataset-tag detected2d

DATA_ROOT="${REPO_ROOT}/data/posemamba_training_sequences_smoke/PoseMamba_f${WINDOW_SIZE}s${STRIDE}_detected2d"
echo "[smoke] Pickles at ${DATA_ROOT}/BICYCLE/train"

_run_py posemamba "${REPO_ROOT}/3d_keypoint_detector_training/train_lifter.py" \
  --sequence-root "${REPO_ROOT}/data/posemamba_training_sequences_smoke" \
  --dataset-tag detected2d \
  --dim-feat 64 \
  --epochs "${EPOCHS}" \
  --max-batches "${MAX_BATCHES}" \
  --no-eval \
  --checkpoint-dir "${REPO_ROOT}/checkpoints/posemamba_bicycle_smoke_detected2d"

CKPT="${REPO_ROOT}/checkpoints/posemamba_bicycle_smoke_detected2d"
if [[ -f "${CKPT}/best_epoch.bin" ]]; then
  _run_py posemamba "${REPO_ROOT}/3d_keypoint_detector_training/eval_lifter.py" \
    --checkpoint "${CKPT}/best_epoch.bin" \
    --config "${REPO_ROOT}/3d_keypoint_detector_training/PoseMamba_train_bicycle.generated.yaml"
fi

echo "[smoke] Done. For production: generate_blender_posemamba_dataset.sh then DATASET_TAG=detected2d start_training.sh"

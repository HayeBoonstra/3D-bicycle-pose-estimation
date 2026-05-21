#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hayepc/3D-bicycle-pose-estimation}"

unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX

RUN_ARGS=(
  "${REPO_ROOT}/2d_keypoint_detector_training/infer_2d.py"
  --config "${REPO_ROOT}/2d_keypoint_detector_training/rtmpose_bicycle_full.py"
  --checkpoint "${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu/epoch_340.pth"
  --input "${REPO_ROOT}/data/bicycle_pose_dataset/images/val"
  --vis-out-dir "${REPO_ROOT}/training_outputs/inference_2d/vis"
  --pred-out-dir "${REPO_ROOT}/training_outputs/inference_2d/preds"
  --summary-jsonl "${REPO_ROOT}/training_outputs/inference_2d/predictions.jsonl"
  --device cuda:0
  --limit 50
)

if [[ "${CONDA_DEFAULT_ENV:-}" == "mmpose" ]]; then
  python "${RUN_ARGS[@]}"
else
  conda run -n mmpose python "${RUN_ARGS[@]}"
fi

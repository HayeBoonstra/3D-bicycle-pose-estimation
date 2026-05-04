source .venv-mmpose311/bin/activate

REPO_ROOT=~/3D-bicycle-pose-estimation
python ${REPO_ROOT}/2d_keypoint_detector_training/infer_2d.py \
  --config ${REPO_ROOT}/2d_keypoint_detector_training/rtmpose_bicycle_full.py \
  --checkpoint ${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu/epoch_300.pth \
  --input ${REPO_ROOT}/data/bicycle_pose_dataset/images/val \
  --vis-out-dir ${REPO_ROOT}/training_outputs/inference_2d/vis \
  --pred-out-dir ${REPO_ROOT}/training_outputs/inference_2d/preds \
  --summary-jsonl ${REPO_ROOT}/training_outputs/inference_2d/predictions.jsonl \
  --device cuda:0 \
  --limit 50
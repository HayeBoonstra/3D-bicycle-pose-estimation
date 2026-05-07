unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX
source /home/hayepc/3D-bicycle-pose-estimation/.venv-mmpose311/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python /home/hayepc/3D-bicycle-pose-estimation/.venv-mmpose311/lib/python3.11/site-packages/mmpose/.mim/tools/train.py \
  /home/hayepc/3D-bicycle-pose-estimation/2d_keypoint_detector_training/rtmpose_bicycle_full.py \
  --work-dir /home/hayepc/3D-bicycle-pose-estimation/training_outputs/mmpose_bicycle_rtmpose_l_gpu \
  --launcher none
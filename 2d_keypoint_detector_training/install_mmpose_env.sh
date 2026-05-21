#!/usr/bin/env bash
# One-time setup for the `mmpose` conda environment (RTMPose / MMPose 1.x).
set -euo pipefail

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mmpose

echo "[install] Python: $(python --version)"
echo "[install] Installing PyTorch (CUDA 12.1; matches prebuilt mmcv 2.1 wheels)..."
pip install -U pip wheel
# openmim pins setuptools~=60.2; keep a version that still exposes pkg_resources for mim.
pip install "setuptools>=60.2.0,<70"
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

echo "[install] Installing OpenMMLab stack via openmim..."
pip install -U openmim
mim install mmengine
# mmcv 2.1.x is required by mmdet 3.3 (<2.2.0); only published for torch 2.1 + cu121.
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
mim install "mmdet>=3.1.0,<4.0.0"
# mmpose pulls chumpy; --no-build-isolation avoids PEP517 build env issues on py3.11.
pip install "mmpose==1.3.2" --no-build-isolation

echo "[install] Pin numpy/opencv for torch 2.1 + mmcv binary wheels..."
pip install "numpy<2,>=1.23" "opencv-python>=4.8,<4.10"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIM_RUNTIME="$(python -c "import importlib.util; from pathlib import Path; s=importlib.util.find_spec('mmpose'); print(Path(s.origin).parent / '.mim/configs/_base_/default_runtime.py')")"
mkdir -p "${REPO_ROOT}/2d_keypoint_detector_training/configs/_base_"
cp "${MIM_RUNTIME}" "${REPO_ROOT}/2d_keypoint_detector_training/configs/_base_/default_runtime.py"
echo "[install] Synced default_runtime.py into 2d_keypoint_detector_training/configs/_base_/"

echo "[install] Training / inference utilities..."
pip install tensorboard

python - <<'PY'
import mmpose
import mmcv
import mmdet
import mmengine
import torch
from pathlib import Path

mim_cfg = Path(mmpose.__file__).resolve().parent / ".mim" / "configs" / "_base_" / "default_runtime.py"
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmpose:", mmpose.__version__)
print("default_runtime:", mim_cfg, "exists:", mim_cfg.is_file())
PY

echo "[install] Done. Activate with: conda activate mmpose"

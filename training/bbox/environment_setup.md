# RF-DETR conda environment (`rfdetr`)

`rfdetr-2xlarge` uses the ONNX GPU backend, which needs `onnxruntime-gpu` and `pycuda`.

## Create / fix the environment

```bash
conda create -n rfdetr python=3.12 -y
conda activate rfdetr

# PyTorch with CUDA (match your driver; cu130 works on recent drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install inference supervision Pillow

# ONNX GPU backend for Roboflow inference models (required for rfdetr-* on GPU)
pip install "inference-models[onnx-cu12]"
```

## Verify

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expect: CUDAExecutionProvider (and optionally TensorrtExecutionProvider)

python bbox_detector.py
```

## CPU-only fallback

If you cannot install GPU extras, force CPU ONNX before loading the model:

```bash
export ONNXRUNTIME_EXECUTION_PROVIDERS="CPUExecutionProvider"
```

Inference will be slower but does not require `pycuda`.

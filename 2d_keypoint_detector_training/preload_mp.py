"""Installed into the active env via start_training.sh (.pth). Sets PyTorch MP sharing."""

try:
    import torch.multiprocessing as _mp

    _mp.set_sharing_strategy("file_system")
except Exception:
    pass

"""Temporal 2D->3D lifting model and inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class _NumpyFallbackLifter:
    def __call__(self, windows: np.ndarray, conf_windows: np.ndarray) -> np.ndarray:
        center = windows.shape[1] // 2
        xy = windows[:, center]
        conf = conf_windows[:, center]
        z = np.maximum(0.0, 1.0 - np.linalg.norm(xy, axis=-1)) * conf
        return np.concatenate([xy, z[..., None]], axis=-1).astype(np.float32)


class TemporalSSMLifter:
    def __init__(self, num_keypoints: int, hidden_dim: int = 128):
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self._mode = "numpy"
        self._model = None
        try:
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self, nk: int, hd: int):
                    super().__init__()
                    in_dim = nk * 3
                    self.encoder = nn.Linear(in_dim, hd)
                    self.gru = nn.GRU(hd, hd, batch_first=True)
                    self.head = nn.Linear(hd, nk * 3)

                def forward(self, x):
                    x = self.encoder(x)
                    y, _ = self.gru(x)
                    y = y[:, y.shape[1] // 2]
                    y = self.head(y)
                    return y

            self._model = Net(num_keypoints, hidden_dim)
            self._mode = "torch"
        except Exception:
            self._model = _NumpyFallbackLifter()
            self._mode = "numpy"

    def load_weights(self, checkpoint: Path | None) -> None:
        if self._mode != "torch" or checkpoint is None or not checkpoint.exists():
            return
        import torch

        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    def infer(self, windows: np.ndarray, conf_windows: np.ndarray) -> np.ndarray:
        if windows.size == 0:
            return np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
        if self._mode == "numpy":
            return self._model(windows, conf_windows)

        import torch

        x = np.concatenate([windows, conf_windows[..., None]], axis=-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        with torch.no_grad():
            y = self._model(torch.from_numpy(x).float()).cpu().numpy()
        return y.reshape(y.shape[0], self.num_keypoints, 3).astype(np.float32)


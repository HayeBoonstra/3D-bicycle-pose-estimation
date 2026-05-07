"""Temporal 3D lifter wrapper with PoseMamba backend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - fallback-only environments
    torch = None


class TemporalSSMLifter:
    """Loads PoseMamba checkpoints and predicts frame-centered 3D keypoints.

    If PoseMamba dependencies are unavailable, this class gracefully falls back to a
    deterministic baseline that lifts 2D windows into 3D by returning centered 2D
    coordinates with zero depth.
    """

    def __init__(
        self,
        num_keypoints: int,
        window_size: int = 27,
        config_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.window_size = window_size
        self.config_path = config_path
        self._backend = "fallback"
        self._model = None
        self._device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self._use_confidence = False

    def _try_build_posemamba(self) -> bool:
        if torch is None:
            return False

        repo_root = Path(__file__).resolve().parents[1]
        posemamba_root = repo_root / "PoseMamba"
        if not posemamba_root.exists():
            return False

        if str(posemamba_root) not in sys.path:
            sys.path.insert(0, str(posemamba_root))

        try:
            from lib.model.PoseMamba import PoseMamba  # type: ignore
        except Exception:
            return False

        model_kwargs = {
            "num_frame": self.window_size,
            "num_joints": self.num_keypoints,
            "in_chans": 2,
            "embed_dim_ratio": 64,
            "depth": 6,
            "mlp_ratio": 2.0,
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
        }
        self._model = PoseMamba(**model_kwargs)
        self._model.to(self._device)
        self._model.eval()
        self._backend = "posemamba"
        self._use_confidence = False
        return True

    def load_weights(self, weights_path: Path | None) -> None:
        if not self._try_build_posemamba():
            return
        if weights_path is None:
            return
        if torch is None:
            return
        if not weights_path.exists():
            raise FileNotFoundError(f"Lifter checkpoint not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self._device)
        state_dict = checkpoint.get("model_pos", checkpoint.get("state_dict", checkpoint))
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        self._model.load_state_dict(state_dict, strict=False)
        self._model.eval()

    def _fallback_infer(self, windows: np.ndarray) -> np.ndarray:
        center = windows[:, windows.shape[1] // 2]
        out = np.zeros((windows.shape[0], self.num_keypoints, 3), dtype=np.float32)
        out[:, :, :2] = center
        return out

    def infer(self, windows: np.ndarray, conf_windows: np.ndarray | None = None) -> np.ndarray:
        if windows.size == 0:
            return np.zeros((0, self.num_keypoints, 3), dtype=np.float32)

        if self._backend != "posemamba" or self._model is None or torch is None:
            return self._fallback_infer(windows)

        inputs = windows
        if self._use_confidence and conf_windows is not None:
            inputs = np.concatenate([windows, conf_windows[..., None]], axis=-1)

        with torch.no_grad():
            x = torch.as_tensor(inputs, dtype=torch.float32, device=self._device)
            pred = self._model(x)  # [N, T, J, 3]
            center = pred[:, pred.shape[1] // 2, :, :]
            return center.detach().cpu().numpy().astype(np.float32)

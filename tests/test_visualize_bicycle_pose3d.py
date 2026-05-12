from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES
from data_generation_pipeline_tools.visualize_bicycle_pose3d import (
    load_motion,
    motion_to_images_or_video,
    subtract_root,
)


class VisualizeBicyclePose3DTests(unittest.TestCase):
    def test_subtract_root_zeroes_root_joint(self) -> None:
        t, j = 4, len(BICYCLE_KEYPOINT_NAMES)
        x = np.random.randn(t, j, 3).astype(np.float32)
        rr = subtract_root(x, root_index=0)
        np.testing.assert_allclose(rr[:, 0, :], 0.0, atol=1e-6)

    def test_motion_to_images_writes_frames_and_summary(self) -> None:
        t, j = 3, len(BICYCLE_KEYPOINT_NAMES)
        rng = np.random.default_rng(0)
        pred = rng.standard_normal((t, j, 3), dtype=np.float32)
        pred = subtract_root(pred, 0)
        gt = pred + 0.01 * rng.standard_normal((t, j, 3), dtype=np.float32)
        gt = subtract_root(gt, 0)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "frames"
            meta = motion_to_images_or_video(
                pred,
                gt,
                out,
                layout="overlay",
                write_video=False,
                fps=10,
                elev=12.0,
                azim=70.0,
            )
            self.assertEqual(meta["frames"], t)
            self.assertTrue(meta["has_gt"])
            self.assertIn("mpjpe_mean_m", meta)
            for i in range(t):
                self.assertTrue((out / f"frame_{i:04d}.png").exists())
            summary = Path(meta["summary_json"])
            self.assertTrue(summary.exists())
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["frames"], t)

    def test_load_npy_roundtrip(self) -> None:
        t, j = 2, len(BICYCLE_KEYPOINT_NAMES)
        arr = np.zeros((t, j, 3), dtype=np.float32)
        arr[:, :, 0] = np.linspace(0, 1, t)[:, None]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.npy"
            np.save(path, arr)
            loaded = load_motion(path)
            np.testing.assert_array_equal(loaded, arr)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.lift3d_ssm import TemporalSSMLifter
from pipeline.sequence_builder import build_temporal_windows


class PipelineSmokeTests(unittest.TestCase):
    def test_temporal_windows_and_lifter_shapes(self):
        frames = 8
        keypoints = 18
        points_2d = np.random.rand(frames, keypoints, 2).astype(np.float32) * 100.0
        conf = np.ones((frames, keypoints), dtype=np.float32)
        bboxes = np.tile(np.asarray([[10.0, 20.0, 200.0, 240.0]], dtype=np.float32), (frames, 1))

        windows, conf_windows = build_temporal_windows(points_2d, conf, bboxes, window_size=5)
        self.assertEqual(windows.shape, (frames, 5, keypoints, 2))
        self.assertEqual(conf_windows.shape, (frames, 5, keypoints))

        model = TemporalSSMLifter(num_keypoints=keypoints)
        pred = model.infer(windows, conf_windows)
        self.assertEqual(pred.shape, (frames, keypoints, 3))


if __name__ == "__main__":
    unittest.main()


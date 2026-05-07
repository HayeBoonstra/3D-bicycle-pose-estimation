from __future__ import annotations

import unittest

import numpy as np

from keypoint_detector_pipeline.world_transform import CameraModel, camera_to_world, world_to_camera


class WorldTransformTests(unittest.TestCase):
    def test_camera_world_roundtrip(self):
        camera = CameraModel(
            K=np.eye(3, dtype=np.float32),
            R=np.eye(3, dtype=np.float32),
            t=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        )
        points_world = np.asarray(
            [[[0.0, 0.0, 4.0], [1.0, 1.0, 5.0]], [[-1.0, 2.0, 3.5], [2.0, -1.0, 6.0]]],
            dtype=np.float32,
        )
        points_cam = world_to_camera(points_world, camera)
        points_back = camera_to_world(points_cam, camera)
        self.assertTrue(np.allclose(points_world, points_back, atol=1e-5))


if __name__ == "__main__":
    unittest.main()


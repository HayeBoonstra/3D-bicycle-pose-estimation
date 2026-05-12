from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_TOOLS = REPO_ROOT / "Mujoco_bicycle_path_generator"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MUJOCO_TOOLS) not in sys.path:
    sys.path.insert(0, str(MUJOCO_TOOLS))

from bicycle_constructor import Bicycle
from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES, KEYPOINT_INDEX
from export_posemamba_annotations import look_at_camera, sample_virtual_cameras
from keypoint_detector_pipeline.world_transform import project_to_image, world_to_camera


class MujocoDirectExportTests(unittest.TestCase):
    def test_bicycle_xml_contains_canonical_keypoint_sites(self):
        bicycle = Bicycle()
        bicycle.create_bicycle_variables()
        xml = bicycle.create_bicycle_model()
        for name in BICYCLE_KEYPOINT_NAMES:
            self.assertIn(f'name="{name}"', xml)

    def test_front_wheel_cardinal_keypoints_are_not_rotated_by_fork_angle(self):
        import mujoco

        generator_dir = REPO_ROOT / "Mujoco_bicycle_path_generator"
        if str(generator_dir) not in sys.path:
            sys.path.insert(0, str(generator_dir))
        from world_contructor import World

        bicycle = Bicycle()
        bicycle.create_bicycle_variables()
        model = mujoco.MjModel.from_xml_string(World().create_world())
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        def site(name: str) -> np.ndarray:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            self.assertGreaterEqual(site_id, 0)
            return np.asarray(data.site_xpos[site_id], dtype=np.float32)

        hub = 0.5 * (site("k_front_hub_left") + site("k_front_hub_right"))
        radius = bicycle.wheel_size / 2.0
        self.assertTrue(np.allclose(site("k_front_wheel_back"), hub + np.asarray([-radius, 0.0, 0.0]), atol=1e-5))
        self.assertTrue(np.allclose(site("k_front_wheel_front"), hub + np.asarray([radius, 0.0, 0.0]), atol=1e-5))
        self.assertTrue(np.allclose(site("k_front_wheel_ground"), hub + np.asarray([0.0, 0.0, -radius]), atol=1e-5))

    def test_look_at_camera_projects_target_to_image_center(self):
        camera = look_at_camera(
            np.asarray([4.0, 0.0, 1.2], dtype=np.float32),
            np.asarray([0.0, 0.0, 0.5], dtype=np.float32),
            width=1280,
            height=720,
            fov_deg=45.0,
        )
        point_world = np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32)
        point_camera = world_to_camera(point_world, camera.model)
        projected = project_to_image(point_camera, camera.model)
        self.assertGreater(point_camera[0, 2], 0.0)
        self.assertTrue(np.allclose(projected[0], [640.0, 360.0], atol=1e-3))

    def test_camera_sampling_is_deterministic(self):
        kwargs = {
            "target": np.asarray([0.0, 0.0, 0.5], dtype=np.float32),
            "points_world_by_frame": np.asarray(
                [
                    [[0.0, 0.0, 0.5], [0.6, 0.0, 0.5]],
                    [[0.1, 0.0, 0.5], [0.7, 0.0, 0.5]],
                ],
                dtype=np.float32,
            ),
            "count": 3,
            "seed": 11,
            "width": 1280,
            "height": 720,
            "min_fov_deg": 35.0,
            "max_fov_deg": 75.0,
            "min_distance": 4.0,
            "max_distance": 6.0,
            "min_elevation_deg": 8.0,
            "max_elevation_deg": 45.0,
            "min_visible_keypoints": 1,
            "min_visible_frame_ratio": 1.0,
            "max_tries_per_camera": 20,
            "fit_margin": 1.15,
        }
        first = sample_virtual_cameras(**kwargs)
        second = sample_virtual_cameras(**kwargs)
        for left, right in zip(first, second):
            self.assertTrue(np.allclose(left.position, right.position))
            self.assertTrue(np.allclose(left.model.K, right.model.K))
            self.assertTrue(np.allclose(left.model.R, right.model.R))
            self.assertTrue(np.allclose(left.model.t, right.model.t))


if __name__ == "__main__":
    unittest.main()

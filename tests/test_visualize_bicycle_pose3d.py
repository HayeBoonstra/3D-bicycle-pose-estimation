from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_generation_pipeline_tools.bicycle_keypoint_schema import (
    BICYCLE_KEYPOINT_NAMES,
    BICYCLE_SKELETON_NAMES,
    KEYPOINT_INDEX,
    SKELETON_PART_GROUPS,
    validate_skeleton_part_groups,
)
from data_generation_pipeline_tools.visualize_bicycle_pose3d import (
    bicycle_crank_angle,
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

    def test_bicycle_steer_matches_posemamba_loss(self) -> None:
        from data_generation_pipeline_tools.bicycle_dynamics_angles import bicycle_steer_angle
        from PoseMamba.lib.model.loss import bicycle_steer_angle as steer_loss
        import torch

        kp = np.zeros((5, len(BICYCLE_KEYPOINT_NAMES), 3), dtype=np.float64)
        kp[:, KEYPOINT_INDEX["k_bottom_bracket"]] = [0.0, 0.0, 0.0]
        kp[:, KEYPOINT_INDEX["k_lower_head_tube"]] = [0.60, 0.00, 0.0]
        kp[:, KEYPOINT_INDEX["k_upper_head_tube"]] = [0.50, -0.50, 0.0]
        kp[:, KEYPOINT_INDEX["k_rear_hub_left"]] = [-0.60, 0.30, -0.05]
        kp[:, KEYPOINT_INDEX["k_rear_hub_right"]] = [-0.60, 0.30, 0.05]
        kp[:, KEYPOINT_INDEX["k_front_hub_left"]] = [0.80, 0.30, -0.05]
        kp[:, KEYPOINT_INDEX["k_front_hub_right"]] = [0.80, 0.30, 0.05]

        np_steer = bicycle_steer_angle(kp.astype(np.float32))
        torch_steer = steer_loss(torch.from_numpy(kp).unsqueeze(0)).squeeze(0).numpy()
        np.testing.assert_allclose(np_steer, torch_steer, rtol=0, atol=1e-5)

    def test_bicycle_crank_angle_at_neutral_pose(self) -> None:
        kp = np.zeros((1, len(BICYCLE_KEYPOINT_NAMES), 3), dtype=np.float32)
        kp[0, KEYPOINT_INDEX["k_bottom_bracket"]] = [0.0, 0.0, 0.0]
        kp[0, KEYPOINT_INDEX["k_seat_stay"]] = [-0.30, -0.40, 0.0]
        kp[0, KEYPOINT_INDEX["k_saddle"]] = [-0.30, -0.60, 0.0]
        kp[0, KEYPOINT_INDEX["k_lower_head_tube"]] = [0.60, 0.00, 0.0]
        kp[0, KEYPOINT_INDEX["k_upper_head_tube"]] = [0.50, -0.50, 0.0]
        kp[0, KEYPOINT_INDEX["k_handlebar_middle"]] = [0.55, -0.55, 0.00]
        kp[0, KEYPOINT_INDEX["k_rear_hub_left"]] = [-0.60, 0.30, -0.05]
        kp[0, KEYPOINT_INDEX["k_rear_hub_right"]] = [-0.60, 0.30, 0.05]
        kp[0, KEYPOINT_INDEX["k_rear_wheel_ground"]] = [-0.60, 0.65, 0.00]
        kp[0, KEYPOINT_INDEX["k_front_hub_left"]] = [0.80, 0.30, -0.05]
        kp[0, KEYPOINT_INDEX["k_front_hub_right"]] = [0.80, 0.30, 0.05]
        kp[0, KEYPOINT_INDEX["k_front_wheel_ground"]] = [0.80, 0.65, 0.00]
        kp[0, KEYPOINT_INDEX["k_left_pedal"]] = [0.00, 0.20, -0.20]
        kp[0, KEYPOINT_INDEX["k_right_pedal"]] = [0.00, -0.20, 0.20]

        crank = bicycle_crank_angle(kp)
        self.assertAlmostEqual(float(np.rad2deg(crank[0])), 0.0, places=1)

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

    def test_skeleton_part_groups_cover_all_edges(self) -> None:
        validate_skeleton_part_groups()
        grouped = [edge for edges in SKELETON_PART_GROUPS.values() for edge in edges]
        self.assertEqual(len(grouped), len(BICYCLE_SKELETON_NAMES))
        self.assertEqual(set(grouped), set(BICYCLE_SKELETON_NAMES))

    def test_camera_view_reorient_flips_image_up(self) -> None:
        pts = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        from data_generation_pipeline_tools.visualize_bicycle_pose3d import reorient_for_display, view_angles_for_reorient

        out = reorient_for_display(pts, "camera_view")
        np.testing.assert_allclose(out[0], [0.1, -0.3, 0.2], atol=1e-6)
        elev, azim = view_angles_for_reorient("camera_view")
        self.assertEqual(elev, 0.0)
        self.assertEqual(azim, -90.0)

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

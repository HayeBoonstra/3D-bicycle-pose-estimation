from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES
from data_generation_pipeline_tools.visualize_raw_annotations import visualize_clip


class VisualizeRawAnnotationsTests(unittest.TestCase):
    def test_visualize_clip_writes_panel_image_without_rendered_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_dir = root / "clip_test"
            ann_dir = clip_dir / "per_frame_annotations"
            ann_dir.mkdir(parents=True)

            points_2d = []
            points_3d = []
            for idx, name in enumerate(BICYCLE_KEYPOINT_NAMES):
                x = 100.0 + idx * 8.0
                y = 120.0 + (idx % 5) * 12.0
                points_2d.append({"name": name, "x": x, "y": y, "v": 2})
                points_3d.append([idx * 0.03, (idx % 4) * 0.04, 2.0 + (idx % 3) * 0.05])

            annotation = {
                "clip_id": clip_dir.name,
                "frame": 0,
                "frame_index": 0,
                "image_width": 640,
                "image_height": 480,
                "keypoints": points_2d,
                "gt_bbox_xywh": [90.0, 100.0, 200.0, 120.0],
            }
            (ann_dir / "keypoints_2d_frame_0000.json").write_text(json.dumps(annotation), encoding="utf-8")

            row3d = {
                "clip_id": clip_dir.name,
                "frame_index": 0,
                "joint_names": BICYCLE_KEYPOINT_NAMES,
                "kps_camera": points_3d,
                "kps_world": (np.asarray(points_3d) + np.asarray([1.0, 0.0, 0.0])).tolist(),
            }
            (clip_dir / "keypoints_3d.jsonl").write_text(json.dumps(row3d) + "\n", encoding="utf-8")

            out = root / "vis"
            written = visualize_clip(
                clip_dir,
                out,
                coord_frame="bicycle",
                frame_step=1,
                max_frames=None,
                output_width=900,
                show_names=False,
            )
            self.assertEqual(written, 1)
            self.assertTrue((out / clip_dir.name / "frame_0000.png").exists())


if __name__ == "__main__":
    unittest.main()

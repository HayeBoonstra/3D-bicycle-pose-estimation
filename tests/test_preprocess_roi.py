from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from pipeline.preprocess_roi import crop_and_resize


class RoiTransformTests(unittest.TestCase):
    def test_roi_transform_is_invertible(self):
        image = Image.new("RGB", (640, 480), color=(0, 0, 0))
        crop, transform = crop_and_resize(image, [100, 50, 500, 400], (256, 192))
        self.assertEqual(crop.size, (256, 192))

        points_img = np.asarray([[120.0, 80.0], [300.0, 200.0], [450.0, 350.0]], dtype=np.float32)
        points_roi = transform.image_to_roi(points_img)
        points_back = transform.roi_to_image(points_roi)
        self.assertTrue(np.allclose(points_img, points_back, atol=1e-4))


if __name__ == "__main__":
    unittest.main()


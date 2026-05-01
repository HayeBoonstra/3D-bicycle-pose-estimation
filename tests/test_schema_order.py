from __future__ import annotations

import unittest

from pipeline.schema import BICYCLE_KEYPOINT_NAMES


class SchemaOrderTests(unittest.TestCase):
    def test_expected_keypoint_count(self):
        self.assertEqual(len(BICYCLE_KEYPOINT_NAMES), 18)
        self.assertEqual(BICYCLE_KEYPOINT_NAMES[0], "k_bottom_bracket")
        self.assertEqual(BICYCLE_KEYPOINT_NAMES[-1], "k_right_pedal")


if __name__ == "__main__":
    unittest.main()


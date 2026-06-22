"""Tests for balanced scene sampling."""

from __future__ import annotations

import unittest

from scene_registry import SceneEntry, sample_scenes_balanced, scene_counts_for_registry


def _entries() -> list[SceneEntry]:
    return [
        SceneEntry(id="a", blend="a.blend", bike="b", rider="r"),
        SceneEntry(id="b", blend="b.blend", bike="b", rider="r"),
        SceneEntry(id="c", blend="c.blend", bike="b", rider="r"),
    ]


class TestSceneBalance(unittest.TestCase):
    def test_evens_out_dominant_scene(self) -> None:
        entries = _entries()
        prior = {"a": 10, "b": 10, "c": 1}
        planned = sample_scenes_balanced(entries, 6, seed=0, prior_counts=prior)
        projected = scene_counts_for_registry(entries, prior)
        for entry, _ in planned:
            projected[entry.id] += 1
        # All six picks go to the lone underrepresented scene before touching a/b.
        self.assertEqual(projected["c"], 7)
        self.assertEqual(projected["a"], 10)
        self.assertEqual(projected["b"], 10)
        self.assertEqual([entry.id for entry, _ in planned], ["c"] * 6)

    def test_reproducible_for_fixed_seed(self) -> None:
        entries = _entries()
        prior = {"a": 5, "b": 2, "c": 0}
        first = sample_scenes_balanced(entries, 4, seed=99, prior_counts=prior)
        second = sample_scenes_balanced(entries, 4, seed=99, prior_counts=prior)
        self.assertEqual([e.id for e, _ in first], [e.id for e, _ in second])


if __name__ == "__main__":
    unittest.main()

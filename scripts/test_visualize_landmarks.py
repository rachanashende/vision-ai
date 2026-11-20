"""Offline helper checks: python -B scripts/test_visualize_landmarks.py."""

from types import SimpleNamespace
import unittest

import numpy as np

from visualize_landmarks import draw_group, sample_indices


class VisualizationTests(unittest.TestCase):
    def test_six_even_samples_include_endpoints(self):
        self.assertEqual(sample_indices(26), [0, 5, 10, 15, 20, 25])
        self.assertEqual(sample_indices(6), list(range(6)))
        with self.assertRaises(ValueError):
            sample_indices(5)

    def test_missing_and_offscreen_landmarks_are_safe(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_group(image, [], [], (0, 255, 0))
        self.assertFalse(image.any())
        invisible = SimpleNamespace(x=float("nan"), y=0.5)
        draw_group(image, [invisible], [], (0, 255, 0))
        self.assertFalse(image.any())
        visible = SimpleNamespace(x=0.3, y=0.4)
        connection = SimpleNamespace(start=0, end=1)
        draw_group(image, [visible, SimpleNamespace(x=0.6, y=0.7)],
                   [connection], (0, 255, 0))
        self.assertTrue(image.any())


if __name__ == "__main__":
    unittest.main()

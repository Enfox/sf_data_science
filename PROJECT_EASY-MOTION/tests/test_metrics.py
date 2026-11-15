"""Небольшие проверки математики. Модель и видео для них не нужны."""
import unittest

import numpy as np
from easy_motion.metrics import align_pose, compare_pose, cosine_distance, weighted_distance


class TestMetrics(unittest.TestCase):
    def test_cosine_known_values(self):
        self.assertAlmostEqual(cosine_distance([1, 0], [0, 1]), 1)
        self.assertAlmostEqual(cosine_distance([1, 2], [2, 4]), 0)
        self.assertAlmostEqual(cosine_distance([1, 0], [-1, 0]), 2)
        with self.assertRaises(ValueError):
            cosine_distance([0, 0], [1, 1])

    def test_weighted_distance(self):
        first, second = [[0, 0], [0, 0]], [[1, 2], [3, 4]]
        self.assertAlmostEqual(weighted_distance(first, second, [1, 3]), 6)
        self.assertAlmostEqual(weighted_distance(first, second, [10, 30]), 6)
        with self.assertRaises(ValueError):
            weighted_distance(first, second, [0, 0])

    def test_affine_known_transform(self):
        points = np.array([[0, 0], [2, 0], [0, 3], [4, 4]], dtype=float)
        expected = points @ np.array([[2, 0.3], [-0.2, 1.5]]) + [10, 20]
        aligned, _ = align_pose(points, expected)
        np.testing.assert_allclose(aligned, expected, atol=1e-10)
        with self.assertRaises(ValueError):
            align_pose(np.array([[0, 0], [1, 1], [2, 2]]), np.array([[1, 1], [2, 2], [3, 3]]))

    def test_missing_and_low_confidence(self):
        missing = {"points": None}
        result = compare_pose(missing, missing)
        self.assertEqual(result["status"], "missing_person")
        self.assertTrue(np.isnan(result["weighted_l1_px"]))
        low = {"points": np.ones((17, 2)), "point_scores": np.ones(17)}
        self.assertEqual(compare_pose(low, low)["status"], "insufficient_points")


if __name__ == "__main__":
    unittest.main()

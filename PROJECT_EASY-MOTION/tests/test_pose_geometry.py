"""Проверяем защиту от слияния ног без загрузки весов нейросети."""
import unittest

import numpy as np
from easy_motion_models.pose import legs_are_ambiguous


class TestPoseGeometry(unittest.TestCase):
    def pose(self):
        points = np.zeros((17, 2))
        points[[5, 6]] = [[40, 20], [80, 20]]
        points[[11, 12]] = [[45, 100], [75, 100]]
        points[[13, 14]] = [[35, 155], [85, 155]]
        points[[15, 16]] = [[25, 210], [95, 210]]
        return points

    def test_merged_legs_are_rejected(self):
        points = self.pose()
        self.assertFalse(legs_are_ambiguous(points))
        points[14] = points[13]
        points[16] = points[15]
        self.assertTrue(legs_are_ambiguous(points))

    def test_knee_at_hip_is_rejected(self):
        points = self.pose()
        points[13] = points[11] + [1, 1]
        self.assertTrue(legs_are_ambiguous(points))


if __name__ == '__main__':
    unittest.main()

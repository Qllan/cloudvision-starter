"""Логические тесты (не требуют скачивания нейросети)."""
import unittest
import numpy as np

from run_ai import cloud_percent_from_mask
from cloudvision.rules import should_delete


class CloudMaskTests(unittest.TestCase):
    def test_no_clouds(self):
        self.assertEqual(cloud_percent_from_mask(np.array([[0, 3], [0, 3]])), 0.0)

    def test_thick_and_thin(self):
        self.assertEqual(cloud_percent_from_mask(np.array([[1, 2], [0, 3]])), 50.0)

    def test_strict_threshold(self):
        self.assertFalse(should_delete(75.0))
        self.assertTrue(should_delete(75.01))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from experiment_cloud_classes import class_stats, metrics


class TestClassPolicy(unittest.TestCase):
    def test_class_stats(self):
        mask = np.array([[0,1],[2,3]], dtype=np.uint8)
        result = class_stats(mask)
        self.assertEqual(result['thick_percent'], 25)
        self.assertEqual(result['thin_percent'], 25)
        self.assertEqual(result['both_percent'], 50)
        self.assertEqual(result['shadow_percent'], 25)

    def test_iou(self):
        a=np.array([[True,True],[False,False]])
        b=np.array([[True,False],[True,False]])
        self.assertAlmostEqual(metrics(a,b)['iou'], 1/3)

    def test_empty_masks(self):
        z=np.zeros((2, 2), dtype=bool)
        self.assertEqual(metrics(z,z)['iou'], 1.)

    def test_invalid_classes(self):
        with self.assertRaises(ValueError):
            class_stats(np.array([[0,99]],dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()

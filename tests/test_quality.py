"""Проверка того, как считается точность облачности."""
import unittest
import numpy as np
from evaluate_ai import scores


class EvaluationTests(unittest.TestCase):
    def test_perfect_mask(self):
        predicted = np.array([[1, 2], [0, 3]], dtype=np.uint8)
        manual = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        result = scores(predicted, manual)
        self.assertEqual(result["cloud_iou"], 1.0)
        self.assertEqual(result["model_percent"], 50.0)
        self.assertEqual(result["manual_percent"], 50.0)

    def test_false_clouds(self):
        predicted = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        manual = np.zeros((2, 2), dtype=np.uint8)
        result = scores(predicted, manual)
        self.assertEqual(result["cloud_iou"], 0.0)
        self.assertEqual(result["absolute_error_pp"], 50.0)

    def test_empty_cloud_image(self):
        predicted = np.zeros((2, 2), dtype=np.uint8)
        manual = np.zeros((2, 2), dtype=np.uint8)
        self.assertEqual(scores(predicted, manual)["cloud_iou"], 1.0)

    def test_threshold(self):
        predicted = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        manual = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        self.assertFalse(scores(predicted, manual)["same_75_percent_decision"])


if __name__ == "__main__":
    unittest.main()

"""Проверки геометрии фрагментов и подсчёта качества."""
import unittest
import numpy as np
from experiment_tiling import regions_2x2, calculate_scores


class TestTiling(unittest.TestCase):
    def test_four_regions_cover_entire_image(self):
        for width, height in [(3280, 2464), (5, 3), (8, 8)]:
            pieces = regions_2x2(width, height)
            self.assertEqual(len(pieces), 4)
            count = np.zeros((height, width), dtype=np.uint8)
            for x0, y0, x1, y1 in pieces:
                self.assertTrue(0 <= x0 < x1 <= width)
                self.assertTrue(0 <= y0 < y1 <= height)
                count[y0:y1, x0:x1] += 1
            self.assertTrue(np.all(count >= 1))

    def test_perfect_mask(self):
        mask = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        manual = np.array([[False, True], [False, True]])
        result = calculate_scores(mask, manual)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["model_percent"], 50.0)

    def test_all_empty_mask(self):
        zero = np.zeros((2, 3), dtype=np.uint8)
        score = calculate_scores(zero, zero > 0)
        self.assertEqual(score["iou"], 1.0)
        self.assertEqual(score["abs_error_pp"], 0)


if __name__ == "__main__":
    unittest.main()

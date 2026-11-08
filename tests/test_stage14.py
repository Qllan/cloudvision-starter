"""Pure logic tests, no expensive model execution."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from run_competition_demo import action_for, cloud_percent, review_reason, safe_delete_input


class Stage14Tests(unittest.TestCase):
    def test_75_is_keep(self):
        self.assertEqual(action_for(75.0), 'СОХРАНИТЬ')

    def test_over_75_is_reject(self):
        self.assertEqual(action_for(75.00001), 'ОТСЕЯТЬ')

    def test_mask_percentage(self):
        mask = Image.fromarray(np.array([[255, 255], [255, 0]], dtype=np.uint8), 'L')
        self.assertEqual(cloud_percent(mask), 75.0)

    def test_review_near_threshold(self):
        self.assertTrue(review_reason(65.0))
        self.assertFalse(review_reason(50.0, 51.0))
        self.assertTrue(review_reason(50.0, 80.0))

    def test_no_arbitrary_delete_path(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(safe_delete_input(Path(folder)))

    def test_invalid_percent(self):
        for value in (-1, 101, float('nan')):
            with self.assertRaises(ValueError):
                action_for(value)


if __name__ == '__main__':
    unittest.main()

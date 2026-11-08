import unittest
import numpy as np
from review_tiling import colorize_classes, COLORS

class ColorizeTests(unittest.TestCase):
    def test_colorize(self):
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = colorize_classes(mask)
        self.assertEqual(result.shape, (2, 2, 3))
        np.testing.assert_array_equal(result[1, 0], COLORS[2])
    def test_unknown_class(self):
        with self.assertRaises(ValueError):
            colorize_classes(np.array([[4]], dtype=np.uint8))

if __name__ == '__main__':
    unittest.main()

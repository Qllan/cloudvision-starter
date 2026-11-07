"""Запуск: python -m unittest discover -s tests -v"""

import unittest

from cloudvision.rules import should_delete


class RulesTests(unittest.TestCase):
    def test_below_threshold_is_kept(self):
        self.assertFalse(should_delete(10))

    def test_exactly_75_is_kept(self):
        self.assertFalse(should_delete(75.0))

    def test_above_75_is_deleted(self):
        self.assertTrue(should_delete(75.01))

    def test_100_is_deleted(self):
        self.assertTrue(should_delete(100))

    def test_invalid_values(self):
        for value in (-1, 101, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    should_delete(value)


if __name__ == "__main__":
    unittest.main()

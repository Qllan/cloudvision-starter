import unittest
import tempfile
from pathlib import Path
from PIL import Image
from label_with_ai_fast import find_ai_mask, mask_to_binary, cloud_percent, apply_stroke


class FastLabelerTests(unittest.TestCase):
    def test_find_stage14_mask_exact_match(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / 'shot_1_policy_mask.png').touch()
            (folder / 'shot_11_policy_mask.png').touch()
            self.assertEqual(find_ai_mask(Path('shot_1.jpg'), folder).name, 'shot_1_policy_mask.png')

    def test_binary_full_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cloud.png'
            Image.frombytes('L', (2, 2), bytes([0, 255, 255, 0])).save(path)
            mask = mask_to_binary(path, (2, 2))
            self.assertEqual(mask.getextrema(), (0, 255))
            self.assertEqual(cloud_percent(mask), 50.0)

    def test_shadow_not_cloud(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'four_classes.png'
            Image.frombytes('L', (2, 2), bytes([0, 1, 2, 3])).save(path)
            mask = mask_to_binary(path, (2, 2))
            self.assertEqual(list(mask.tobytes()), [0, 255, 255, 0])

    def test_edit_keeps_original_dimensions(self):
        mask = Image.new('L', (300, 200), 0)
        apply_stroke(mask, [(20, 20), (40, 35)], (100, 67), 4, True)
        self.assertEqual(mask.size, (300, 200))
        self.assertGreater(cloud_percent(mask), 0)
        apply_stroke(mask, [(20, 20), (40, 35)], (100, 67), 4, False)
        self.assertLess(cloud_percent(mask), 1)

    def test_wrong_mask_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'color.png'
            Image.new('RGB', (10, 10), (255, 0, 0)).save(p)
            with self.assertRaises(ValueError):
                mask_to_binary(p, (10, 10))

if __name__ == '__main__':
    unittest.main()

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "process_satellite_images.py"
spec = importlib.util.spec_from_file_location("cloudvision_processing", SCRIPT)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class ProcessingTests(unittest.TestCase):
    def test_strict_threshold(self):
        self.assertTrue(module.decide("a.jpg", 75.0, "a.png").keep)
        self.assertFalse(module.decide("a.jpg", 75.001, "a.png").keep)
        self.assertTrue(module.decide("a.jpg", 0.0, "a.png").keep)
        self.assertFalse(module.decide("a.jpg", 100.0, "a.png").keep)

    def test_two_cloud_classes_and_no_shadow(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            arr = np.array([[0, 1], [2, 3]], dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(path)
            self.assertAlmostEqual(module.percentage_from_mask(path, (2, 2)), 50.0)

    def test_exact_75(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            arr = np.array([[1, 1], [2, 0]], dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(path)
            p = module.percentage_from_mask(path, (2, 2))
            self.assertEqual(p, 75.0)
            self.assertTrue(module.decide("x.jpg", p, "x.png").keep)

    def test_reject_colored_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            Image.new("RGB", (2, 2), "red").save(path)
            with self.assertRaises(ValueError):
                module.percentage_from_mask(path, (2, 2))

    def test_reject_incorrect_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            Image.new("L", (2, 2)).save(path)
            with self.assertRaises(ValueError):
                module.percentage_from_mask(path, (3, 2))

    def test_reject_unknown_class(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            Image.new("L", (2, 2), 255).save(path)
            with self.assertRaises(ValueError):
                module.percentage_from_mask(path, (2, 2))

    def test_preview_keeps_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "a.jpg"
            original.write_bytes(b"test")
            out = root / "saved"
            d = module.decide("a.jpg", 99.0, "mask.png")
            module.apply_decision(d, original, out, "preview")
            self.assertTrue(original.exists())
            self.assertFalse(out.exists())

    def test_copy_does_not_delete_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "a.jpg"
            original.write_bytes(b"test")
            out = root / "saved"
            d = module.decide("a.jpg", 10.0, "mask.png")
            module.apply_decision(d, original, out, "copy")
            self.assertTrue(original.exists())
            self.assertTrue((out / "a.jpg").exists())

    def test_delete_only_work_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data" / "work_input").mkdir(parents=True)
            self.assertTrue(module.safe_to_delete(root / "data" / "work_input", root))
            (root / "data" / "input").mkdir()
            self.assertFalse(module.safe_to_delete(root / "data" / "input", root))

    def test_output_report_names(self):
        d1 = module.decide("keep.jpg", 75.0, "keep_tiled_mask.png")
        d2 = module.decide("reject.jpg", 76.0, "reject_tiled_mask.png")
        self.assertEqual(module.output_row(d1, "copy")["сохранённый_файл"], "keep.jpg")
        self.assertEqual(module.output_row(d2, "delete")["удалённый_файл"], "reject.jpg")
        self.assertEqual(module.output_row(d2, "preview")["удалённый_файл"], "")

if __name__ == "__main__":
    unittest.main()

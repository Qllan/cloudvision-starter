"""No-model unit tests for GPU-independent batch orchestration."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from cloudvision.batch import discover_images, run_batch


class BatchV4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.output = self.root / "output"
        (self.images / "a").mkdir(parents=True)
        (self.images / "b").mkdir(parents=True)
        self.one = self.images / "a" / "same.jpg"
        self.two = self.images / "b" / "same.jpg"
        Image.new("RGB", (8, 8), (10, 10, 10)).save(self.one)
        Image.new("RGB", (8, 8), (200, 200, 200)).save(self.two)

    @staticmethod
    def predict(image):
        result = Image.new("L", image.size, 0)
        brush = ImageDraw.Draw(result)
        if image.getpixel((0, 0))[0] < 50:
            brush.rectangle((0, 0, 5, 7), fill=255)  # 48 / 64 = exactly 75%
        else:
            brush.rectangle((0, 0, 7, 7), fill=255)  # 100% > 75
        return result

    def test_recursive_paths_resume_and_threshold(self):
        original = self.one.read_bytes()
        stats = run_batch(self.images, self.output, "model-A", self.predict, log=lambda _: None)
        self.assertEqual(stats, {"found": 2, "processed": 2, "skipped": 0, "failed": 0})
        self.assertTrue((self.output / "masks/a/same_policy_mask.png").exists())
        self.assertTrue((self.output / "masks/b/same_policy_mask.png").exists())
        with (self.output / "operator_preview.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([row["решение_по_75%"] for row in rows], ["СОХРАНИТЬ", "ОТСЕЯТЬ"])
        self.assertEqual(self.one.read_bytes(), original)
        again = run_batch(self.images, self.output, "model-A", self.predict, log=lambda _: None)
        self.assertEqual(again["processed"], 0)
        self.assertEqual(again["skipped"], 2)
        changed_model = run_batch(self.images, self.output, "model-B", self.predict, log=lambda _: None)
        self.assertEqual(changed_model["processed"], 2)

    def test_missing_cached_mask_reprocessed(self):
        run_batch(self.images, self.output, "model-A", self.predict, log=lambda _: None)
        (self.output / "masks/a/same_policy_mask.png").unlink()
        result = run_batch(self.images, self.output, "model-A", self.predict, log=lambda _: None)
        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["skipped"], 1)

    def test_corrupt_image_does_not_block_others(self):
        (self.images / "a" / "invalid.jpg").write_bytes(b"not a jpeg")
        stats = run_batch(self.images, self.output, "model-A", self.predict, log=lambda _: None)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["processed"], 2)

    def test_rejects_output_inside_input(self):
        with self.assertRaises(ValueError):
            run_batch(self.images, self.images / "output", "model-A", self.predict)

    def test_discovery_filters_nonimages(self):
        (self.images / "README.txt").write_text("hello", encoding="utf-8")
        self.assertEqual(len(discover_images(self.images)), 2)

    def test_invalid_mask_is_not_saved(self):
        def bad_predict(image):
            return Image.new("RGB", image.size, "white")
        result = run_batch(self.images, self.output, "model-A", bad_predict, log=lambda _: None)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["processed"], 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for verified v4 mask checks, including the blind TEST exclusion."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from audit_v4_labels import audit_labels


class AuditV4LabelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.images = root / "images"
        self.images.mkdir()
        self.labels = root / "verified"
        self.labels.mkdir()
        self.csv = root / "selected.csv"
        self.pred = root / "predictions.csv"
        self.images_by_split = {}
        with self.csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["relative_path", "scene_group", "split"])
            for day, part in [(1, "train"), (2, "val"), (3, "test")]:
                name = f"2026-09-{day:02d}_10-00-00_SXC3-250_1.jpg"
                Image.new("RGB", (10, 10), (40, 50, 60)).save(self.images / name)
                writer.writerow([name, f"2026-09-{day:02d}_sxc3-250", part])
                self.images_by_split[part] = name
        with self.pred.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["исходный_файл", "облачность_%"])
            writer.writerow([self.images_by_split["train"], "90"])
            writer.writerow([self.images_by_split["val"], "10"])

    def test_missing_labels_and_test_untouched(self):
        Image.new("L", (10, 10), 0).save(
            self.labels / self.images_by_split["train"].replace(".jpg", ".png"))
        rows = audit_labels(self.images, self.csv, self.labels, self.pred)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["status"] for r in rows], ["OK", "MISSING"])
        self.assertEqual(rows[0]["verified_percent"], "0.000")
        self.assertEqual(rows[0]["decision_changed"], "YES")
        self.assertEqual(rows[0]["needs_recheck"], "YES")

    def test_invalid_non_binary_mask(self):
        Image.new("L", (10, 10), 127).save(
            self.labels / self.images_by_split["train"].replace(".jpg", ".png"))
        rows = audit_labels(self.images, self.csv, self.labels, self.pred)
        self.assertTrue(rows[0]["status"].startswith("INVALID"))


if __name__ == "__main__":
    unittest.main()

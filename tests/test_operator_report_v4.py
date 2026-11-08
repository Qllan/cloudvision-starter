"""Offline operator-report checks for CloudVision AI v4 (without torch)."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_operator_report_v4 import build_report, read_batch_report


class OperatorReportV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "photos"
        self.batch = self.root / "batch"
        self.output = self.batch / "operator_review"
        (self.images / "nested").mkdir(parents=True)
        (self.batch / "masks/nested").mkdir(parents=True)
        self.names = [
            "nested/2025-09-16_12-00-00_SXC3-237_1.jpg",
            "nested/2025-09-16_12-01-00_SXC3-237_1.jpg",
            "nested/2025-09-17_12-00-00_SXC3-239_1.jpg",
        ]
        self.masks = []
        percents = [75, 80, 0]
        for name, percent in zip(self.names, percents):
            Image.new("RGB", (20, 20), (100, 140, 190)).save(self.images / name)
            mask_rel = Path("masks") / Path(name).parent / (Path(name).stem + "_policy_mask.png")
            mask = Image.new("L", (20, 20), 0)
            mask.putdata([255] * (percent * 4) + [0] * (400 - percent * 4))
            mask.save(self.batch / mask_rel)
            self.masks.append(mask_rel.as_posix())
        self.percentages = percents
        self.write_batch_csv()

    def write_batch_csv(self):
        with (self.batch / "operator_preview.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["исходный_файл", "облачность_%", "решение_по_75%", "маска", "фактическое_действие"])
            for name, mask, percent in zip(self.names, self.masks, self.percentages):
                writer.writerow([name, f"{percent:.3f}",
                                 "ОТСЕЯТЬ" if percent > 75 else "СОХРАНИТЬ",
                                 mask, "ТОЛЬКО ПРЕДПРОСМОТР"])

    def test_build_html_and_csv_prioritizes_rejection_and_near_threshold(self):
        original_bytes = (self.images / self.names[0]).read_bytes()
        original_mask_bytes = (self.batch / self.masks[0]).read_bytes()
        stats = build_report(self.images, self.batch, self.output)
        self.assertEqual(stats, {"total": 3, "kept": 2, "rejected": 1,
                                 "review": 3, "previewed": 3})
        with (self.output / "operator_review.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["решение_по_75%"], "ОТСЕЯТЬ")
        self.assertEqual(rows[0]["облачность_%"], "80.000")
        self.assertEqual(rows[1]["облачность_%"], "75.000")
        self.assertEqual(rows[1]["решение_по_75%"], "СОХРАНИТЬ")
        self.assertEqual(rows[0]["кадров_в_серии"], "2")
        self.assertEqual(rows[2]["проверить"], "ДА")
        self.assertTrue((self.output / "index.html").exists())
        self.assertEqual((self.images / self.names[0]).read_bytes(), original_bytes)
        self.assertEqual((self.batch / self.masks[0]).read_bytes(), original_mask_bytes)
        for row in rows:
            self.assertTrue((self.output / row["превью"]).is_file())

    def test_preview_limit_still_exports_every_row(self):
        stats = build_report(self.images, self.batch, self.output, preview_limit=1)
        self.assertEqual(stats["previewed"], 1)
        with (self.output / "operator_review.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(bool(row["превью"]) for row in rows), 1)

    def test_wrong_percent_is_rejected(self):
        self.percentages[0] = 61
        self.write_batch_csv()
        with self.assertRaisesRegex(ValueError, "расходятся"):
            read_batch_report(self.images, self.batch)

    def test_wrong_action_rejected(self):
        self.percentages[0] = 76
        self.write_batch_csv()
        with self.assertRaises(ValueError):
            read_batch_report(self.images, self.batch)

    def test_missing_mask_and_traversal_rejected(self):
        (self.batch / self.masks[0]).unlink()
        with self.assertRaises(ValueError):
            read_batch_report(self.images, self.batch)
        self.names[0] = "../secret.jpg"
        self.write_batch_csv()
        with self.assertRaisesRegex(ValueError, "Небезопасный путь"):
            read_batch_report(self.images, self.batch)

    def test_nonbinary_mask_rejected(self):
        mask_path = self.batch / self.masks[1]
        Image.new("L", (20, 20), 127).save(mask_path)
        with self.assertRaisesRegex(ValueError, "бинарной"):
            read_batch_report(self.images, self.batch)

    def test_outputs_cannot_be_written_into_originals_or_masks(self):
        with self.assertRaises(ValueError):
            build_report(self.images, self.batch, self.images / "reports")
        with self.assertRaises(ValueError):
            build_report(self.images, self.batch, self.batch / "masks" / "operator")


if __name__ == "__main__":
    unittest.main()

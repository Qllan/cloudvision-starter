"""Offline operator-report checks for CloudVision AI v4 (without torch)."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_operator_report_v4 import build_report, read_batch_report
from cloudvision.operator_queue import review_queue


class OperatorReportV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "photos"
        self.batch = self.root / "batch"
        self.output = self.batch / "operator_review"
        self.verified = self.root / "pilot_verified"
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
                                 "review": 3, "manual_pending": 3,
                                 "manual_done": 0, "previewed": 3})
        with (self.output / "operator_review.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["решение_по_75%"], "ОТСЕЯТЬ")
        self.assertTrue(all(r["статус_проверки"] == "ТРЕБУЕТСЯ РУЧНАЯ ПРОВЕРКА"
                            for r in rows))
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

    def test_rounded_csv_at_threshold_respects_exact_mask(self):
        # 750001/1000000 = 75.0001%; CSV rounds to "75.000".
        name = self.names[0]
        Image.new("RGB", (1000, 1000), "navy").save(self.images / name)
        mask = Image.new("L", (1000, 1000), 0)
        mask.putdata([255] * 750001 + [0] * 249999)
        mask.save(self.batch / self.masks[0])
        self.percentages[0] = 75.000
        # Keep the true (>75%) decision despite rounding in the CSV.
        path = self.batch / "operator_preview.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["исходный_файл", "облачность_%", "решение_по_75%", "маска", "фактическое_действие"])
            for i, (rel, mask_rel, pct) in enumerate(
                    zip(self.names, self.masks, self.percentages)):
                decision = "ОТСЕЯТЬ" if i == 0 else (
                    "ОТСЕЯТЬ" if pct > 75 else "СОХРАНИТЬ")
                writer.writerow([rel, f"{pct:.3f}", decision, mask_rel,
                                 "ТОЛЬКО ПРЕДПРОСМОТР"])
        rows = read_batch_report(self.images, self.batch)
        self.assertEqual(rows[0]["decision"], "ОТСЕЯТЬ")

    def test_manual_saved_mask_changes_status_and_exposes_disagreement(self):
        self.verified.mkdir()
        label = self.verified / Path(self.names[1]).with_suffix(".png")
        label.parent.mkdir(parents=True)
        Image.new("L", (20, 20), 0).save(label)
        stats = build_report(self.images, self.batch, self.output, verified=self.verified)
        self.assertEqual(stats["manual_done"], 1)
        self.assertEqual(stats["manual_pending"], 2)
        with (self.output / "operator_review.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        reviewed = next(r for r in rows if r["исходный_файл"] == self.names[1])
        self.assertEqual(reviewed["статус_проверки"], "ПРОВЕРЕНО ОПЕРАТОРОМ")
        self.assertEqual(reviewed["ручная_облачность_%"], "0.000")
        self.assertEqual(reviewed["решение_по_75%"], "ОТСЕЯТЬ")
        self.assertEqual(reviewed["ручное_решение_по_75%"], "СОХРАНИТЬ")
        self.assertEqual(reviewed["расхождение_пп"], "80.000")
        self.assertEqual(reviewed["IoU_с_ручной_маской"], "0.00000")

    def test_invalid_manual_mask_fails_before_any_report_changes(self):
        self.verified.mkdir()
        label = self.verified / Path(self.names[0]).with_suffix(".png")
        label.parent.mkdir(parents=True)
        Image.new("L", (20, 20), 128).save(label)
        with self.assertRaisesRegex(ValueError, "0/255"):
            build_report(self.images, self.batch, self.output, verified=self.verified)
        self.assertFalse((self.output / "index.html").exists())

    def test_review_queue_preserves_priority_and_rejects_traversal(self):
        build_report(self.images, self.batch, self.output)
        selected = review_queue(self.output / "operator_review.csv", self.images)
        self.assertEqual([path.name for path in selected],
                         [Path(self.names[i]).name for i in (1, 0, 2)])
        queue = self.output / "unsafe.csv"
        queue.write_text("исходный_файл\n../secret.jpg\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            review_queue(queue, self.images)

    def test_outputs_cannot_be_written_into_originals_or_masks(self):
        with self.assertRaises(ValueError):
            build_report(self.images, self.batch, self.images / "reports")
        with self.assertRaises(ValueError):
            build_report(self.images, self.batch, self.batch / "masks" / "operator")


if __name__ == "__main__":
    unittest.main()

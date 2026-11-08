"""Regression tests for the independent, blinded TEST protocol (no CUDA needed)."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudvision.v4_holdout import (
    aggregate_scores, check_frozen, freeze_checkpoints,
    mask_scores, prepared_test_pairs, test_manifest_names,
)


class HoldoutV4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "photos"
        self.labels = self.root / "blind_reference"
        self.images.mkdir()
        self.labels.mkdir()
        self.manifest = self.root / "selection.csv"
        self.v3 = self.root / "old_v3.pt"
        self.v4 = self.root / "new_v4.pt"
        self.frozen = self.root / "reports" / "frozen_models.json"
        self.v3.write_bytes(b"synthetic v3 checkpoint (hash only)")
        self.v4.write_bytes(b"synthetic v4 checkpoint (hash only)")
        self.test_name = "2026-08-31_12-00-00_SXC3-250_1.jpg"
        with self.manifest.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relative_path", "scene_group", "split"])
            w.writerow(["2026-08-30_11-00-00_SXC3-250_1.jpg", "2026-08-30_sxc3-250", "train"])
            w.writerow([self.test_name, "2026-08-31_sxc3-250", "test"])

    def test_freeze_does_not_need_or_open_test_images(self):
        self.assertFalse((self.images / self.test_name).exists())
        self.assertEqual(test_manifest_names(self.manifest), [self.test_name])
        snap = freeze_checkpoints(self.manifest, self.v3, self.v4, self.frozen)
        self.assertEqual(snap["test_count"], 1)
        check_frozen(self.manifest, self.v3, self.v4, self.frozen)
        with self.assertRaises(FileExistsError):
            freeze_checkpoints(self.manifest, self.v3, self.v4, self.frozen)

    def test_frozen_checkpoint_and_manifest_cannot_change(self):
        freeze_checkpoints(self.manifest, self.v3, self.v4, self.frozen)
        self.v4.write_bytes(b"modified checkpoint")
        with self.assertRaisesRegex(ValueError, "отличается"):
            check_frozen(self.manifest, self.v3, self.v4, self.frozen)
        self.v4.write_bytes(b"synthetic v4 checkpoint (hash only)")
        self.manifest.write_text(self.manifest.read_text(encoding="utf-8-sig") + "\n",
                                 encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "изменён"):
            check_frozen(self.manifest, self.v3, self.v4, self.frozen)

    def test_evaluation_requires_complete_strict_reference(self):
        Image.new("RGB", (16, 12), (30, 30, 30)).save(self.images / self.test_name)
        with self.assertRaisesRegex(FileNotFoundError, "Не хватает TEST-масок"):
            prepared_test_pairs(self.manifest, self.images, self.labels)
        path = self.labels / self.test_name.replace(".jpg", ".png")
        Image.new("L", (16, 12), 255).save(path)
        pairs = prepared_test_pairs(self.manifest, self.images, self.labels)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], self.test_name)
        Image.new("L", (8, 6), 255).save(path)
        with self.assertRaisesRegex(ValueError, "Неверный режим или размер"):
            prepared_test_pairs(self.manifest, self.images, self.labels)
        Image.new("L", (16, 12), 127).save(path)
        with self.assertRaisesRegex(ValueError, "Не бинарная"):
            prepared_test_pairs(self.manifest, self.images, self.labels)

    def test_precise_75_percent_threshold_and_empty_iou(self):
        ref = np.zeros((4, 4), dtype=bool)
        ref.flat[:12] = True  # exactly 75%, must KEEP
        row = mask_scores(ref, ref)
        self.assertEqual(row["predicted_percent"], 75.0)
        self.assertEqual(row["reference_action"], "KEEP")
        self.assertEqual(row["correct_action"], 1)
        other = ref.copy()
        other.flat[12] = True
        row = mask_scores(other, ref)
        self.assertEqual(row["predicted_action"], "REJECT")
        self.assertEqual(row["correct_action"], 0)
        self.assertEqual(mask_scores(np.zeros((4, 4), bool), np.zeros((4, 4), bool))["iou"], 1)
        with self.assertRaises(ValueError):
            mask_scores(ref.astype(np.uint8), ref)

    def test_aggregate_misses_and_false_rejects(self):
        keep = np.zeros((4, 4), bool)
        reject = np.ones((4, 4), bool)
        rows = [mask_scores(reject, keep), mask_scores(keep, reject),
                mask_scores(keep, keep)]
        summary = aggregate_scores(rows)
        self.assertEqual(summary["test_count"], 3)
        self.assertEqual(summary["correct_actions"], 1)
        self.assertEqual(summary["wrong_rejects"], 1)
        self.assertEqual(summary["missed_rejects"], 1)


if __name__ == "__main__":
    unittest.main()

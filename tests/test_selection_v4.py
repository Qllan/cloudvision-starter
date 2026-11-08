"""Unit tests for diverse scene selection and leak-safe v4 labeling."""
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from cloudvision.batch import run_batch
from cloudvision.selection import read_manifest, scene_group, select_samples, write_manifest


class SelectionV4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for day in range(1, 21):
            for index in range(2):
                path = self.images / f"2025-09-{day:02d}_12-0{index}-00_SXC3-240_{index}.jpg"
                im = np.zeros((32, 32, 3), dtype=np.uint8)
                im[:, :, :] = [day * 10, index * 40, 220 - day * 7]
                Image.fromarray(im).save(path)
        self.csv = self.root / "sample.csv"

    def test_grouping_nearby_frames(self):
        one = scene_group(Path("2025-09-16_23-34-58_SXC3-237_1.jpg"))
        two = scene_group(Path("2025-09-16_23-35-41_SXC3-237_1.jpg"))
        self.assertEqual(one, two)

    def test_reproducible_split_without_leak(self):
        rows = select_samples(self.images, count=15, seed=17)
        self.assertEqual(rows, select_samples(self.images, count=15, seed=17))
        self.assertEqual(len(rows), 15)
        self.assertEqual(len({r["relative_path"] for r in rows}), 15)
        self.assertEqual(len({r["scene_group"] for r in rows}), 15)
        self.assertEqual(set(r["split"] for r in rows), {"train", "val", "test"})
        write_manifest(self.csv, rows)
        filtered = read_manifest(self.csv, self.images, {"train", "val"})
        self.assertEqual(len(filtered), sum(row["split"] != "test" for row in rows))
        with self.assertRaises(FileExistsError):
            write_manifest(self.csv, rows)

    def test_batch_uses_only_train_val(self):
        rows = select_samples(self.images, count=12, seed=4)
        write_manifest(self.csv, rows)
        def predict(im):
            return Image.new("L", im.size, 0)
        summary = run_batch(self.images, self.root / "report", "model-A", predict,
                            manifest=self.csv, splits={"train", "val"},
                            log=lambda msg: None)
        self.assertEqual(summary["processed"], sum(row["split"] != "test" for row in rows))
        self.assertEqual(summary["failed"], 0)
        with (self.root / "report/operator_preview.csv").open(encoding="utf-8-sig", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), summary["processed"])

    def test_manifest_rejects_traversal(self):
        self.csv.write_text("relative_path,scene_group,split\n../secret.jpg,file:secret,train\n",
                            encoding="utf-8")
        with self.assertRaises(ValueError):
            read_manifest(self.csv, self.images)

    def test_manifest_rejects_group_leak(self):
        group = "2025-09-01_sxc3-240"
        self.csv.write_text(
            "relative_path,scene_group,split\n"
            f"2025-09-01_12-00-00_SXC3-240_0.jpg,{group},train\n"
            f"2025-09-01_12-01-00_SXC3-240_1.jpg,{group},test\n",
            encoding="utf-8")
        with self.assertRaises(ValueError):
            read_manifest(self.csv, self.images)

    def test_exclude_already_seen(self):
        seen = {"2025-09-01_12-00-00_SXC3-240_0.jpg",
                "2025-09-01_12-01-00_SXC3-240_1.jpg"}
        rows = select_samples(self.images, count=10, seed=42, excluded=seen)
        self.assertTrue(all(r["relative_path"] not in seen for r in rows))

    def test_no_fewer_than_five_scene_groups(self):
        subset = self.root / "few"
        subset.mkdir()
        for day in range(1, 5):
            Image.new("RGB", (10, 10)).save(
                subset / f"2025-09-{day:02d}_12-00-00_SXC3-1_0.jpg")
        with self.assertRaises(ValueError):
            select_samples(subset, count=10)


if __name__ == "__main__":
    unittest.main()

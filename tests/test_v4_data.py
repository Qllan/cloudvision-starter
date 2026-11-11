"""Isolated v4 dataset tests; no torch, GPU, satellite dataset or TEST pixels."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cloudvision.v4_data import prepare_examples


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class V4DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.new_images = self.root / "new"
        self.new_labels = self.root / "verified"
        self.old_images = self.root / "old"
        self.old_labels = self.root / "old_labels"
        for folder in (self.new_images, self.new_labels, self.old_images, self.old_labels):
            folder.mkdir()
        self.manifest = self.root / "selection.csv"
        self.old_split = self.root / "old_split.csv"

        self.rows = []
        for day, split in [(1, "train"), (2, "val"), (3, "test")]:
            name = f"2026-09-{day:02d}_10-00-00_SXC3-250_1.jpg"
            self.rows.append({"relative_path": name,
                              "scene_group": f"2026-09-{day:02d}_sxc3-250",
                              "split": split})
            # TEST has no image file on purpose. Preparing training must not open it.
            if split != "test":
                Image.new("RGB", (16, 12)).save(self.new_images / name)
                Image.new("L", (16, 12), 255).save(
                    self.new_labels / name.replace(".jpg", ".png"))
        write_csv(self.manifest, ["relative_path", "scene_group", "split"], self.rows)
        self.old_rows = []
        for day, split in [(4, "train"), (5, "val"), (6, "test")]:
            name = f"2025-09-{day:02d}_10-00-00_SXC3-100_1.jpg"
            label = f"{Path(name).stem}_label.png"
            self.old_rows.append({"image": name, "label": label,
                                  "group": f"2025-09-{day:02d}_SXC3-100",
                                  "split": split})
            if split == "train":
                Image.new("RGB", (16, 12)).save(self.old_images / name)
                Image.new("L", (16, 12), 0).save(self.old_labels / label)
        write_csv(self.old_split, ["image", "label", "group", "split"], self.old_rows)

    def build(self, with_old=True):
        return prepare_examples(
            self.manifest, self.new_images, self.new_labels,
            self.old_split if with_old else None,
            self.old_images if with_old else None,
            self.old_labels if with_old else None,
        )

    def test_train_val_and_old_train_only(self):
        items = self.build()
        self.assertEqual(len(items["train"]), 2)
        self.assertEqual(len(items["val"]), 1)
        self.assertEqual([r.source for r in items["train"]], ["new", "old"])
        self.assertEqual(items["val"][0].source, "new")
        self.assertFalse(any("2026-09-03" in r.image_name for p in items.values() for r in p))

    def test_new_only_works_without_old_folders(self):
        items = self.build(with_old=False)
        self.assertEqual(len(items["train"]), 1)

    def test_missing_manual_mask_is_error(self):
        self.new_labels.joinpath(self.rows[0]["relative_path"].replace(".jpg", ".png")).unlink()
        with self.assertRaises(FileNotFoundError):
            self.build(with_old=False)

    def test_nonbinary_mask_rejected(self):
        path = self.new_labels / self.rows[0]["relative_path"].replace(".jpg", ".png")
        Image.new("L", (16, 12), 128).save(path)
        with self.assertRaises(ValueError):
            self.build(with_old=False)

    def test_wrong_size_rejected(self):
        path = self.new_labels / self.rows[0]["relative_path"].replace(".jpg", ".png")
        Image.new("L", (8, 12), 0).save(path)
        with self.assertRaises(ValueError):
            self.build(with_old=False)

    def test_overlap_old_new_scene_fails(self):
        self.old_rows[0]["image"] = self.rows[1]["relative_path"]
        self.old_rows[0]["group"] = self.rows[1]["scene_group"]
        write_csv(self.old_split, ["image", "label", "group", "split"], self.old_rows)
        with self.assertRaisesRegex(ValueError, "Найдены группы сцен"):
            self.build()

    def test_old_split_leak_fails(self):
        self.old_rows[1]["image"] = "2025-09-04_12-00-00_SXC3-100_2.jpg"
        self.old_rows[1]["group"] = "2025-09-04_SXC3-100"
        write_csv(self.old_split, ["image", "label", "group", "split"], self.old_rows)
        with self.assertRaisesRegex(ValueError, "Утечка"):
            self.build()

    def test_old_manual_mask_can_be_downscaled(self):
        """Legacy mask may be lower-res while preserving scene geometry."""
        label = self.old_labels / self.old_rows[0]["label"]
        Image.new("L", (8, 6), 255).save(label)  # original is 16 x 12
        result = self.build()
        self.assertEqual(result["train"][-1].source, "old")
        self.assertEqual(len(result["train"]), 2)

    def test_old_manual_mask_bad_aspect_ratio_is_rejected(self):
        label = self.old_labels / self.old_rows[0]["label"]
        Image.new("L", (8, 8), 255).save(label)
        with self.assertRaisesRegex(ValueError, "пропорции"):
            self.build()

    def test_new_manual_mask_must_remain_native_resolution(self):
        label = self.new_labels / self.rows[0]["relative_path"].replace(".jpg", ".png")
        Image.new("L", (8, 6), 255).save(label)
        with self.assertRaisesRegex(ValueError, "совпадать"):
            self.build(with_old=False)

    def test_old_val_test_not_required_on_disk(self):
        items = self.build()
        self.assertEqual(len(items["train"]), 2)  # only old TRAIN file exists
        self.assertEqual(len(items["val"]), 1)


if __name__ == "__main__":
    unittest.main()

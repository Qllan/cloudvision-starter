"""Local-only v5 audit tests; no PyTorch, GPU or TEST image pixels."""
import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cloudvision.selection import scene_group
from cloudvision.v5_data import prepare_v5_plan
from select_v5_holdout import choose_holdout, excluded_groups


def write_csv(path: Path, columns: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def filename(date: str, camera: str = "SXC3-240", seq: str = "1") -> str:
    return f"{date}_12-00-00_{camera}_{seq}.jpg"


class V5PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.new = self.root / "new"
        self.old = self.root / "old"
        self.new_masks = self.root / "new_masks"
        self.old_masks = self.root / "old_masks"
        self.pilot_masks = self.root / "pilot_masks"
        for root in (self.new, self.old, self.new_masks, self.old_masks, self.pilot_masks):
            root.mkdir()
        self.new_manifest = self.root / "v4_selection.csv"
        self.old_split = self.root / "old_split.csv"
        self.pilot_csv = self.root / "operator_review.csv"
        self.train_v4 = filename("2026-05-01")
        self.val_v4 = filename("2026-05-02")
        self.test_v4 = filename("2026-05-03")
        self.old_train = filename("2025-09-16")
        self.old_val = filename("2025-09-17")
        self.old_test = filename("2025-09-18")
        self.pilot_unique = filename("2025-09-19")
        self.pilot_on_old_val = filename("2025-09-17", seq="2")
        self.new_rows = [
            {"relative_path": name, "scene_group": scene_group(Path(name)), "split": split}
            for name, split in ((self.train_v4, "train"),
                                (self.val_v4, "val"), (self.test_v4, "test"))
        ]
        write_csv(self.new_manifest, ["relative_path", "scene_group", "split"], self.new_rows)
        self.old_rows = [
            {"image": name, "label": Path(name).stem + "_label.png",
             "group": scene_group(Path(name)).upper(), "split": split}
            for name, split in ((self.old_train, "train"),
                                (self.old_val, "val"), (self.old_test, "test"))
        ]
        write_csv(self.old_split, ["image", "label", "group", "split"], self.old_rows)
        for name in (self.train_v4, self.val_v4):
            Image.new("RGB", (16, 12), "navy").save(self.new / name)
            Image.new("L", (16, 12), 255).save(self.new_masks / (Path(name).stem + ".png"))
        Image.new("RGB", (16, 12), "purple").save(self.old / self.old_train)
        Image.new("L", (8, 6), 255).save(self.old_masks / (Path(self.old_train).stem + "_label.png"))
        shutil.copy2(self.old / self.old_train, self.new / self.old_train)
        for name in (self.old_train, self.pilot_unique):
            if name == self.pilot_unique:
                Image.new("RGB", (16, 12), "green").save(self.new / name)
            Image.new("L", (16, 12), 0).save(
                self.pilot_masks / (Path(name).stem + ".png"))
        self.pilot_rows = [
            {"исходный_файл": name, "статус_проверки": "ПРОВЕРЕНО ОПЕРАТОРОМ"}
            for name in (self.old_train, self.pilot_unique, self.pilot_on_old_val)
        ]
        write_csv(self.pilot_csv, ["исходный_файл", "статус_проверки"], self.pilot_rows)

    def build(self):
        return prepare_v5_plan(
            self.new_manifest, self.new, self.new_masks,
            self.old_split, self.old, self.old_masks,
            self.pilot_csv, self.pilot_masks,
        )

    def test_pilot_duplicate_replaces_old_train_without_double_counting(self):
        plan, excluded = self.build()
        self.assertEqual(len(plan), 4)  # v4 TRAIN + pilot old replacement + pilot new + v4 VAL
        self.assertEqual(sum(item.split == "train" for item in plan), 3)
        self.assertEqual(sum(item.split == "val" for item in plan), 1)
        self.assertEqual(sum(item.source == "pilot" for item in plan), 2)
        self.assertFalse(any(item.source == "old" for item in plan))
        self.assertEqual({row["reason"] for row in excluded},
                         {"REPLACES_OLD_TRAIN_LABEL", "SCENE_IN_OLD_OR_V4_VAL_TEST"})
        # OLD VAL/TEST and v4 TEST don't exist as pixel files; no attempt to open them.
        self.assertFalse((self.old / self.old_val).exists())
        self.assertFalse((self.old / self.old_test).exists())
        self.assertFalse((self.new / self.test_v4).exists())

    def test_missing_pilot_mask_rejected(self):
        (self.pilot_masks / (Path(self.pilot_unique).stem + ".png")).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.build()

    def test_pilot_without_manual_verification_rejected(self):
        self.pilot_rows[1]["статус_проверки"] = "ТРЕБУЕТСЯ РУЧНАЯ ПРОВЕРКА"
        write_csv(self.pilot_csv, ["исходный_файл", "статус_проверки"], self.pilot_rows)
        with self.assertRaisesRegex(ValueError, "не проверен"):
            self.build()

    def test_pilot_same_name_different_pixels_rejected(self):
        Image.new("RGB", (16, 12), "yellow").save(self.new / self.old_train)
        with self.assertRaisesRegex(ValueError, "разные байты"):
            self.build()

    def test_holdout_selection_excludes_all_seen_groups_without_pixels(self):
        # Extra candidate scenes: one per date/camera; no image open is necessary.
        for day in range(1, 15):
            path = self.new / filename(f"2026-07-{day:02d}")
            path.write_bytes(b"not even a valid JPG")
        excluded = excluded_groups(self.new_manifest, self.old_split, self.pilot_csv)
        chosen = choose_holdout(self.new, excluded, count=8, seed=77)
        self.assertEqual(len(chosen), 8)
        self.assertEqual(len({r["scene_group"] for r in chosen}), 8)
        self.assertTrue(all(row["scene_group"] not in excluded for row in chosen))
        self.assertTrue(all(row["split"] == "test" for row in chosen))
        self.assertEqual(chosen, choose_holdout(self.new, excluded, count=8, seed=77))

    def test_holdout_excludes_previous_preview_groups(self):
        previous = self.root / "previous_preview.csv"
        selected = filename("2026-07-01")
        write_csv(previous, ["исходный_файл"], [{"исходный_файл": selected}])
        excluded = excluded_groups(self.new_manifest, self.old_split, self.pilot_csv, previous)
        self.assertIn(scene_group(Path(selected)), excluded)

    def test_holdout_requires_distinct_new_scenes(self):
        excluded = excluded_groups(self.new_manifest, self.old_split, self.pilot_csv)
        with self.assertRaisesRegex(ValueError, "Доступно"):
            choose_holdout(self.new, excluded, 6)


if __name__ == "__main__":
    unittest.main()

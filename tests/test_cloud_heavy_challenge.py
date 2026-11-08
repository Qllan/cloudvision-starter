"""Offline tests for visually curated cloud-heavy challenge selection.

All photos are synthetic and no AI / torch is imported.
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cloudvision.selection import scene_group
from prepare_cloud_heavy_challenge import browse, freeze, make_pool, protected_groups


def csv_file(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def name(day: int, seq: int = 1):
    return f"2026-10-{day:02d}_12-00-00_SXC3-240_{seq}.jpg"


class CloudHeavyChallengeTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.base = Path(t.name)
        self.images = self.base / "photos"
        self.images.mkdir()
        self.old = self.base / "old.csv"
        self.v4 = self.base / "v4.csv"
        self.pilot = self.base / "pilot.csv"
        self.newtest = self.base / "v5.csv"
        self.previous = self.base / "previous.csv"
        self.pooldir = self.base / "gallery"
        self.manifest = self.base / "challenge" / "manifest.csv"
        self.selected = self.base / "selected_cloud_heavy.txt"
        # These 5 scenes must never enter the challenge.
        csv_file(self.old, ["image", "split"], [{"image": name(1), "split": "train"},
                                                 {"image": name(2), "split": "test"}])
        csv_file(self.v4, ["relative_path", "split"], [
            {"relative_path": name(3), "split": "val"}])
        csv_file(self.pilot, ["исходный_файл"], [
            {"исходный_файл": name(4)}])
        csv_file(self.newtest, ["relative_path", "scene_group", "split"], [
            {"relative_path": name(5), "scene_group": scene_group(Path(name(5))),
             "split": "test"}])
        csv_file(self.previous, ["исходный_файл"], [
            {"исходный_файл": name(6)}])
        for day in range(1, 16):
            for seq in range(1, 3):
                Image.new("RGB", (18, 12), (day * 8, 30, 90)).save(
                    self.images / name(day, seq))

    def blocked(self):
        return protected_groups(self.v4, self.old, self.pilot,
                                self.newtest, self.previous)

    def test_all_prior_splits_and_preview_are_excluded(self):
        blocked = self.blocked()
        self.assertEqual(len(blocked), 6)
        pool = make_pool(self.images, blocked, count=100, per_group=2, seed=3)
        self.assertEqual(len(pool), 18)  # 9 unseen groups, two images each
        self.assertTrue(all(row["scene_group"] not in blocked for row in pool))
        self.assertEqual(make_pool(self.images, blocked, count=100, per_group=2, seed=3),
                         pool)

    def test_gallery_and_freeze_selected_targeted_manifest(self):
        counts = browse(self.images, self.blocked(), count=12, per_group=2,
                        seed=2, out=self.pooldir)
        self.assertEqual(counts["total"], 12)
        html = (self.pooldir / "index.html").read_text(encoding="utf-8")
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertIn('type="checkbox"', html)
        self.assertIn("selected_cloud_heavy.txt", html)
        with (self.pooldir / "candidate_pool.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        one_per_group = list({row["scene_group"]: row for row in rows}.values())
        self.assertGreaterEqual(len(one_per_group), 5)
        chosen = one_per_group[:5]
        self.selected.write_text("\n".join(row["relative_path"] for row in chosen) + "\n",
                                 encoding="utf-8")
        stats = freeze(self.selected, self.pooldir / "candidate_pool.csv",
                       self.images, self.blocked(), self.manifest)
        self.assertEqual(stats["selected"], 5)
        with self.manifest.open(encoding="utf-8-sig", newline="") as file:
            manifest = list(csv.DictReader(file))
        self.assertTrue(all(row["split"] == "test" for row in manifest))
        self.assertEqual(len({row["scene_group"] for row in manifest}), 5)
        self.assertIn("TARGETED_CLOUD_HEAVY_CHALLENGE_NOT_RANDOM",
                      (self.manifest.parent / "protocol.json").read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            freeze(self.selected, self.pooldir / "candidate_pool.csv",
                   self.images, self.blocked(), self.manifest)

    def test_same_camera_date_group_is_not_independent(self):
        browse(self.images, self.blocked(), count=30, per_group=2,
               seed=3, out=self.pooldir)
        with (self.pooldir / "candidate_pool.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        same = [row for row in rows if row["scene_group"] == rows[0]["scene_group"]]
        distinct = [row for row in rows if row["scene_group"] != rows[0]["scene_group"]]
        self.assertEqual(len(same), 2)
        chosen = same + [row for row in distinct if row["scene_group"] not in
                          {r["scene_group"] for r in same}][:3]
        self.selected.write_text("\n".join(row["relative_path"] for row in chosen),
                                 encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "две фотографии одной серии"):
            freeze(self.selected, self.pooldir / "candidate_pool.csv",
                   self.images, self.blocked(), self.manifest)
        self.assertFalse(self.manifest.exists())

    def test_manually_injected_old_test_scene_is_rejected(self):
        browse(self.images, self.blocked(), count=12, per_group=2,
               seed=2, out=self.pooldir)
        csv_path = self.pooldir / "candidate_pool.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        rows[0] = {"relative_path": name(2), "scene_group": scene_group(Path(name(2)))}
        csv_file(csv_path, ["relative_path", "scene_group"], rows)
        self.selected.write_text("\n".join(r["relative_path"] for r in rows[:5]),
                                 encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "небезопасная группа"):
            freeze(self.selected, csv_path, self.images, self.blocked(), self.manifest)

    def test_already_used_test_manifest_must_have_consistent_group(self):
        csv_file(self.newtest, ["relative_path", "scene_group", "split"], [
            {"relative_path": name(5), "scene_group": "fake_group", "split": "test"}])
        with self.assertRaisesRegex(ValueError, "Неверная группа"):
            self.blocked()


if __name__ == "__main__":
    unittest.main()

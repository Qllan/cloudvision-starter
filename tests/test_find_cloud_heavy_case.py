"""Tests for read-only satellite image and human-mask locator."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from find_cloud_heavy_case import (
    create_preview, find_manual_mask, find_photos, search_and_preview,
)


class FindCloudHeavyCaseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.images = self.root / "photos"
        self.labels = self.root / "verified"
        self.out = self.root / "reports"
        self.name = "2025-11-30_22-18-23_SXC3-2334_1.jpg"
        (self.images / "camera").mkdir(parents=True)
        (self.labels / "camera").mkdir(parents=True)
        self.photo = self.images / "camera" / self.name
        self.mask = self.labels / "camera" / (Path(self.name).stem + ".png")
        Image.new("RGB", (20, 12), (40, 70, 110)).save(self.photo)
        mask = Image.new("L", (20, 12), 0)
        for x in range(18):
            for y in range(12):
                mask.putpixel((x, y), 255)
        mask.save(self.mask)

    def test_find_exact_and_partial_name(self):
        self.assertEqual(find_photos(self.images, self.name.upper()), [self.photo])
        self.assertEqual(find_photos(self.images, "SXC3-2334"), [self.photo])
        self.assertEqual(find_manual_mask(self.photo, self.images, self.labels), self.mask)

    def test_creates_three_panels_90_percent_without_modifying_sources(self):
        old_photo = self.photo.read_bytes()
        old_mask = self.mask.read_bytes()
        result = search_and_preview(
            self.images, self.labels, self.name, self.out, width=200)
        self.assertEqual(result["matches"], 1)
        self.assertAlmostEqual(result["percent"], 90.0)
        with Image.open(result["preview"]) as rendered:
            self.assertEqual(rendered.size, (20 * 3, 12 + 36))
        self.assertEqual(self.photo.read_bytes(), old_photo)
        self.assertEqual(self.mask.read_bytes(), old_mask)
        with self.assertRaisesRegex(FileExistsError, "перезапись запрещена"):
            search_and_preview(self.images, self.labels, self.name, self.out)

    def test_multiple_matches_avoid_guessing_preview(self):
        second = self.images / "camera" / self.name.replace("_1.jpg", "_2.jpg")
        Image.new("RGB", (20, 12)).save(second)
        result = search_and_preview(
            self.images, self.labels, "SXC3-2334", self.out)
        self.assertEqual(result["matches"], 2)
        self.assertIsNone(result["preview"])
        self.assertFalse(self.out.exists())

    def test_mask_must_match_original_resolution_and_binary(self):
        Image.new("L", (18, 12), 0).save(self.mask)
        with self.assertRaisesRegex(ValueError, "совпадать"):
            create_preview(self.photo, self.mask, self.out / "preview.jpg")
        Image.new("L", (20, 12), 120).save(self.mask)
        with self.assertRaisesRegex(ValueError, "0/255"):
            create_preview(self.photo, self.mask, self.out / "preview.jpg")
        self.assertFalse(self.out.exists())

    def test_protect_source_and_label_paths(self):
        with self.assertRaisesRegex(ValueError, "не должна быть внутри"):
            search_and_preview(
                self.images, self.labels, self.name, self.images / "preview")
        with self.assertRaisesRegex(ValueError, "не должна быть внутри"):
            search_and_preview(
                self.images, self.labels, self.name, self.labels / "preview")


if __name__ == "__main__":
    unittest.main()

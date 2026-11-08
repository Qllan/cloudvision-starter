"""CPU/Pillow-only tests for frozen cloud-heavy four-panel diagnostics."""
from __future__ import annotations

import argparse
import csv
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cloudvision.diagnostic_panels import binary_fraction, render_four_panels
from cloudvision.selection import scene_group
from compare_cloud_heavy_models import run, select_case


class CloudHeavyFourPanelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rgb = Image.new("RGB", (8, 4), "#2861a2")
        self.human = Image.new("L", self.rgb.size, 0)
        for y in range(4):
            for x in range(6):
                self.human.putpixel((x, y), 255)  # 24 / 32 => 75% exactly
        self.v4 = Image.new("L", self.rgb.size, 0)
        self.v5 = Image.new("L", self.rgb.size, 255)

    def test_exact_threshold_and_four_aligned_panels(self):
        jpeg, scores = render_four_panels(self.rgb, self.human, self.v4, self.v5,
                                          panel_width=400)
        self.assertEqual(scores, {"human": 75.0, "v4": 0.0, "v5": 100.0})
        with Image.open(io.BytesIO(jpeg)) as displayed:
            self.assertEqual(displayed.size, (32, 68))
        self.assertEqual(binary_fraction(self.human, self.rgb.size), 75.0)

    def test_wrong_mask_shape_or_grayscale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "match RGB"):
            render_four_panels(self.rgb, self.human, Image.new("L", (6, 4), 255), self.v5)
        with self.assertRaisesRegex(ValueError, "binary 0/255"):
            render_four_panels(self.rgb, self.human, Image.new("L", (8, 4), 100), self.v5)
        with self.assertRaisesRegex(ValueError, "must be L"):
            render_four_panels(self.rgb, self.human.convert("RGB"), self.v4, self.v5)
        with self.assertRaisesRegex(ValueError, "panel_width"):
            render_four_panels(self.rgb, self.human, self.v4, self.v5, panel_width=100)

    def _prepare_case(self):
        images = self.root / "images"
        labels = self.root / "labels"
        images.mkdir()
        labels.mkdir()
        filename = "2025-11-30_22-18-23_SXC3-2334_1.jpg"
        image = images / filename
        mask = labels / filename.replace(".jpg", ".png")
        self.rgb.save(image)
        self.human.save(mask)
        manifest = self.root / "manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["relative_path", "scene_group", "split"])
            writer.writerow([filename, scene_group(Path(filename)), "test"])
        return images, labels, image, mask, manifest

    def test_only_challenge_manifest_images_are_selectable(self):
        images, labels, image, mask, manifest = self._prepare_case()
        found, chosen_mask = select_case(manifest, images, labels, "SXC3-2334")
        self.assertEqual(found, image.resolve())
        self.assertEqual(chosen_mask, mask)
        with self.assertRaisesRegex(FileNotFoundError, "No image matched"):
            select_case(manifest, images, labels, "another_photo")
        image2 = images / "2025-11-30_22-18-25_SXC3-2334_2.jpg"
        self.rgb.save(image2)
        with manifest.open("a", encoding="utf-8") as file:
            file.write(f"{image2.name},{scene_group(Path(image2.name))},test\n")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            select_case(manifest, images, labels, "SXC3-2334")

    def test_frozen_file_required_before_any_output_or_torch(self):
        images, labels, image, mask, manifest = self._prepare_case()
        model_dir = self.root / "models"
        model_dir.mkdir()
        model4, model5 = model_dir / "v4.pt", model_dir / "v5.pt"
        model4.write_bytes(b"old")
        model5.write_bytes(b"new")
        out = self.root / "review"
        params = argparse.Namespace(manifest=manifest, input=images, labels=labels,
                                    query="SXC3-2334", v4=model4, v5=model5,
                                    frozen=self.root / "evaluation" / "frozen_models.json", out=out,
                                    device="cpu", panel_width=420)
        with self.assertRaisesRegex(FileNotFoundError, "frozen_models.json"):
            run(params)
        self.assertFalse(out.exists())

    def test_output_never_inside_ground_truth_or_model_folder(self):
        images, labels, image, mask, manifest = self._prepare_case()
        model_dir = self.root / "models"
        model_dir.mkdir()
        model4, model5 = model_dir / "v4.pt", model_dir / "v5.pt"
        model4.write_bytes(b"old")
        model5.write_bytes(b"new")
        args = argparse.Namespace(manifest=manifest, input=images, labels=labels,
                                  query="SXC3-2334", v4=model4, v5=model5,
                                  frozen=self.root / "evaluation" / "frozen_models.json",
                                  out=labels / "preview", device="cpu", panel_width=420)
        with self.assertRaisesRegex(ValueError, "output must be separate"):
            run(args)


if __name__ == "__main__":
    unittest.main()

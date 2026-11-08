import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from audit_dataset import evaluate_annotation, analyze_dataset


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.images.mkdir()
        self.labels.mkdir()
        Image.new("RGB", (200, 100), "blue").save(self.images / "sample.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def label(self, values, mode="L"):
        Image.fromarray(np.array(values, dtype=np.uint8), mode=mode).save(self.labels / "sample_label.png")

    def test_valid_binary_mask(self):
        self.label([[0, 255], [255, 0]])
        result = evaluate_annotation(self.images / "sample.jpg", self.labels / "sample_label.png")
        # Видимое поле для разметки 2x2 — неправильное соотношение сторон (1:1 против 2:1).
        self.assertEqual(result["status"], "ERROR")

    def test_valid_binary_mask_matching_aspect(self):
        self.label([[0, 255, 255, 0], [0, 255, 255, 0]])
        result = evaluate_annotation(self.images / "sample.jpg", self.labels / "sample_label.png")
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["manual_cloud_percent"], "50.000")

    def test_empty_review(self):
        self.label([[0, 0, 0, 0], [0, 0, 0, 0]])
        result = evaluate_annotation(self.images / "sample.jpg", self.labels / "sample_label.png")
        self.assertEqual(result["status"], "REVIEW")

    def test_not_binary_rejected(self):
        self.label([[0, 127, 255, 0], [0, 255, 0, 0]])
        result = evaluate_annotation(self.images / "sample.jpg", self.labels / "sample_label.png")
        self.assertEqual(result["status"], "ERROR")

    def test_unlabelled_count(self):
        records, unlabelled, unused = analyze_dataset(self.images, self.labels)
        self.assertEqual((len(records), len(unlabelled), unused), (0, 1, 0))


if __name__ == "__main__":
    unittest.main()

"""Offline regression tests for the CloudVision AI competition report generator."""
from __future__ import annotations

import unittest

from build_competition_report import validate_holdout, validate_pilot, sparkline


class CompetitionReportTests(unittest.TestCase):
    def setUp(self):
        self.a = {
            "model": "v4-policy", "image": "scene.jpg",
            "predicted_percent": "75.0", "reference_percent": "75.0",
            "absolute_error_pp": "0.0", "iou": "1.0",
            "predicted_action": "KEEP", "reference_action": "KEEP",
            "correct_action": "1",
        }
        self.b = dict(self.a, model="v5-pilot")
        self.summary_a = {
            "model": "v4-policy", "test_count": "1", "mae_pp": "0.0",
            "mean_iou": "1.0", "correct_actions": "1",
            "reference_rejects": "0", "wrong_rejects": "0", "missed_rejects": "0",
        }
        self.summary_b = dict(self.summary_a, model="v5-pilot")

    def test_exact_75_means_keep(self):
        groups, images = validate_holdout(
            [self.summary_a, self.summary_b], [self.a, self.b],
            {"v4-policy", "v5-pilot"})
        self.assertEqual(groups["v4-policy"]["correct_actions"], "1")
        self.assertEqual(len(images["v5-pilot"]), 1)

    def test_different_test_reference_is_rejected(self):
        changed = dict(self.b, reference_percent="0", absolute_error_pp="75.0")
        altered = dict(self.summary_b, mae_pp="75.0")
        with self.assertRaisesRegex(ValueError, "Разные ручные эталоны"):
            validate_holdout(
                [self.summary_a, altered], [self.a, changed],
                {"v4-policy", "v5-pilot"})

    def test_wrong_aggregate_is_rejected(self):
        altered = dict(self.summary_b, wrong_rejects="1")
        with self.assertRaisesRegex(ValueError, "Несогласованный"):
            validate_holdout(
                [self.summary_a, altered], [self.a, self.b],
                {"v4-policy", "v5-pilot"})

    def test_pilot_reject_requires_human_review(self):
        case = {
            "исходный_файл": "sat.jpg", "облачность_%": "74",
            "решение_по_75%": "СОХРАНИТЬ",
            "ручная_облачность_%": "85",
            "ручное_решение_по_75%": "ОТСЕЯТЬ",
            "статус_проверки": "ПРОВЕРЕНО ОПЕРАТОРОМ",
            "расхождение_пп": "11", "IoU_с_ручной_маской": "0.7",
        }
        self.assertEqual(validate_pilot([case])["missed"], 1)
        case["статус_проверки"] = "ТРЕБУЕТСЯ РУЧНАЯ ПРОВЕРКА"
        with self.assertRaisesRegex(ValueError, "без ручного подтверждения"):
            validate_pilot([case])

    def test_svg_is_self_contained(self):
        chart = sparkline([3.6, 2.3, 2.9])
        self.assertIn("<svg", chart)
        self.assertIn("<polyline", chart)
        self.assertIn("Эпоха 2", chart)


if __name__ == "__main__":
    unittest.main()

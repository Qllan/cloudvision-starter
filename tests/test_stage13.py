import unittest
from evaluate_v3_test import actions, decision_stats


class Stage13Tests(unittest.TestCase):
    def test_strict_75(self):
        self.assertEqual(actions(75.0), 'KEEP')
        self.assertEqual(actions(75.0001), 'REJECT')
        self.assertEqual(actions(0), 'KEEP')
        self.assertEqual(actions(100), 'REJECT')

    def test_counts(self):
        rows = [
            {'reference_percent': 80, 'predicted_percent': 90},
            {'reference_percent': 85, 'predicted_percent': 20},
            {'reference_percent': 10, 'predicted_percent': 95},
            {'reference_percent': 10, 'predicted_percent': 10},
        ]
        stats = decision_stats(rows)
        self.assertEqual((stats['correct_reject'], stats['missed_reject'], stats['wrong_reject'], stats['correct_keep']), (1, 1, 1, 1))
        self.assertEqual(stats['balanced_accuracy'], 0.5)

    def test_no_rejects_recall_undefined(self):
        s = decision_stats([{'reference_percent': 25, 'predicted_percent': 25}])
        self.assertIsNone(s['reject_recall'])
        self.assertIsNone(s['balanced_accuracy'])


if __name__ == '__main__':
    unittest.main()

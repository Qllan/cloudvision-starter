import unittest

from train_cloudvision_v3 import area_key, confusion, policy_key


class Stage12Tests(unittest.TestCase):
    def test_all_keep_not_balanced(self):
        rows = [
            {'reference_action':'KEEP', 'predicted_action':'KEEP', 'predicted_percent': 30},
            {'reference_action':'KEEP', 'predicted_action':'KEEP', 'predicted_percent': 50},
            {'reference_action':'REJECT', 'predicted_action':'KEEP', 'predicted_percent': 60},
        ]
        s = confusion(rows)
        self.assertAlmostEqual(s['accuracy'], 2/3)
        self.assertAlmostEqual(s['balanced_accuracy'], 0.5)
        self.assertEqual(s['missed_reject'], 1)

    def test_full_detection(self):
        rows = [
            {'reference_action':'KEEP', 'predicted_action':'KEEP', 'predicted_percent': 20},
            {'reference_action':'REJECT', 'predicted_action':'REJECT', 'predicted_percent': 80},
        ]
        self.assertEqual(confusion(rows)['balanced_accuracy'], 1)

    def test_policy_prefers_detecting_rejects(self):
        m1 = {'mae_pp': 10, 'mean_iou': 0.55}
        m2 = {'mae_pp': 18, 'mean_iou': 0.56}
        rows1 = [
            {'reference_action':'KEEP','predicted_action':'KEEP','predicted_percent': 20},
            {'reference_action':'REJECT','predicted_action':'KEEP','predicted_percent': 60},
        ]
        rows2 = [
            {'reference_action':'KEEP','predicted_action':'KEEP','predicted_percent': 20},
            {'reference_action':'REJECT','predicted_action':'REJECT','predicted_percent': 80},
        ]
        self.assertLess(policy_key(m2, rows2), policy_key(m1, rows1))
        self.assertLess(area_key(m1, rows1), area_key(m2, rows2))

    def test_no_reject_reference_raises(self):
        with self.assertRaises(ValueError):
            confusion([{'reference_action':'KEEP','predicted_action':'KEEP'}])


if __name__ == '__main__':
    unittest.main()

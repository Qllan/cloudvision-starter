import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_cloudvision_v2 import cloud_loss, is_degenerate, pick_key


class Stage11Tests(unittest.TestCase):
    def test_finite_loss_and_gradient(self):
        logits = torch.zeros((2, 1, 8, 8), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0, :, :3] = 1
        valid = torch.ones_like(logits)
        valid[:, :, :1] = 0
        loss = cloud_loss(logits, targets, valid)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_detects_all_cloud_prediction(self):
        self.assertTrue(is_degenerate([{'predicted_percent': 100.0},
                                      {'predicted_percent': 99.2}]))
        self.assertFalse(is_degenerate([{'predicted_percent': 20.},
                                       {'predicted_percent': 60.}]))

    def test_non_collapse_preferred(self):
        bad = {'mae_pp': 10, 'mean_iou': .6}
        good = {'mae_pp': 15, 'mean_iou': .5}
        self.assertGreater(pick_key(bad, [{'predicted_percent': 100.}]),
                           pick_key(good, [{'predicted_percent': 15.}]))


if __name__ == '__main__':
    unittest.main()

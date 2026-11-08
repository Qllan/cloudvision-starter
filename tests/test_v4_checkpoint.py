"""Validate independent selection of v4 MAE and IoU checkpoint epochs."""
import unittest

from cloudvision.v4_checkpoint import checkpoint_key


class V4CheckpointSelectionTests(unittest.TestCase):
    def test_area_objective_favors_smaller_mae(self):
        a = {"mae_pp": 5.0, "mean_iou": .65, "decision_accuracy": 1.0}
        b = {"mae_pp": 4.0, "mean_iou": .60, "decision_accuracy": 1.0}
        self.assertLess(checkpoint_key(b, False, "area"),
                        checkpoint_key(a, False, "area"))
        self.assertLess(checkpoint_key(a, False, "mask"),
                        checkpoint_key(b, False, "mask"))

    def test_mask_objective_favors_iou(self):
        a = {"mae_pp": 3.0, "mean_iou": .70, "decision_accuracy": 1.0}
        b = {"mae_pp": 4.0, "mean_iou": .80, "decision_accuracy": .9}
        self.assertLess(checkpoint_key(b, False, "mask"),
                        checkpoint_key(a, False, "mask"))
        self.assertLess(checkpoint_key(a, False, "area"),
                        checkpoint_key(b, False, "area"))

    def test_degenerate_checkpoint_loses_to_nondegenerate(self):
        degenerate = {"mae_pp": 0.0, "mean_iou": 1.0, "decision_accuracy": 1.0}
        useful = {"mae_pp": 12.0, "mean_iou": .4, "decision_accuracy": .5}
        for objective in ("area", "mask"):
            self.assertLess(checkpoint_key(useful, False, objective),
                            checkpoint_key(degenerate, True, objective))

    def test_tiebreakers_and_bad_values(self):
        tied_mae_good_accuracy = {"mae_pp": 5.0, "mean_iou": .6, "decision_accuracy": 1.0}
        tied_mae_bad_accuracy = {"mae_pp": 5.0, "mean_iou": .7, "decision_accuracy": .8}
        self.assertLess(checkpoint_key(tied_mae_good_accuracy, False, "area"),
                        checkpoint_key(tied_mae_bad_accuracy, False, "area"))
        with self.assertRaises(ValueError):
            checkpoint_key(tied_mae_good_accuracy, False, "unknown")
        with self.assertRaises(ValueError):
            checkpoint_key({"mae_pp": float("nan"), "mean_iou": .5,
                            "decision_accuracy": .5}, False, "mask")


if __name__ == "__main__":
    unittest.main()

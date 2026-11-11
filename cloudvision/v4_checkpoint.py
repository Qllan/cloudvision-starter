"""Pure functions for selecting independently optimized CloudVision v4 checkpoints.

Validation metrics are computed on exactly the same validation split;
the final original-resolution validation comparison is reported separately.
"""


def checkpoint_key(metrics: dict, degenerate: bool, objective: str) -> tuple:
    """Return lexicographic key (smaller is better) for area or mask quality."""
    if objective not in {"area", "mask"}:
        raise ValueError("Unknown checkpoint objective")
    error = float(metrics["mae_pp"])
    iou = float(metrics["mean_iou"])
    accuracy = float(metrics["decision_accuracy"])
    for number in (error, iou, accuracy):
        if not (float("-inf") < number < float("inf")):
            raise ValueError("Validation metric must be finite")
    if error < 0 or not 0 <= iou <= 1 or not 0 <= accuracy <= 1:
        raise ValueError("Invalid validation metric range")
    if objective == "area":
        return (int(degenerate), error, -accuracy, -iou)
    return (int(degenerate), -iou, error, -accuracy)

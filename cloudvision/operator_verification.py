"""Human verification status for pilot only; never treat AI masks as ground truth.

A saved, native-resolution binary mask is considered an operator review artifact.
These assisted reviews are NOT independent TEST and must stay out of historical
validation/test metrics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cloudvision.rules import should_delete

PENDING = "ТРЕБУЕТСЯ РУЧНАЯ ПРОВЕРКА"
VERIFIED = "ПРОВЕРЕНО ОПЕРАТОРОМ"


def manual_review(pilot_row: dict, verified_root: Path) -> dict:
    relative = pilot_row["relative"]
    source = verified_root.resolve()
    label = source / relative.with_suffix(".png")
    empty = {
        "status": PENDING,
        "manual_percent": None,
        "manual_decision": "",
        "difference_pp": None,
        "mask_iou": None,
    }
    if not label.exists():
        return empty
    if label.is_symlink() or not label.is_file() or not label.resolve().is_relative_to(source):
        raise ValueError(f"Небезопасная ручная маска: {label}")
    with Image.open(pilot_row["mask"]) as pred_file, Image.open(label) as hand_file:
        if hand_file.mode != "L" or hand_file.size != pred_file.size:
            raise ValueError(f"Ручная маска не L или не совпадает с исходным размером: {label}")
        histogram = hand_file.histogram()
        if any(count for value, count in enumerate(histogram) if value not in (0, 255)):
            raise ValueError(f"Ручная маска должна содержать только 0/255: {label}")
        pred = np.asarray(pred_file, dtype=np.uint8) == 255
        hand = np.asarray(hand_file, dtype=np.uint8) == 255
    percent = 100 * np.count_nonzero(hand) / hand.size
    intersection = np.count_nonzero(pred & hand)
    union = np.count_nonzero(pred | hand)
    return {
        "status": VERIFIED,
        "manual_percent": float(percent),
        "manual_decision": "ОТСЕЯТЬ" if should_delete(percent) else "СОХРАНИТЬ",
        "difference_pp": float(abs(pilot_row["percent"] - percent)),
        "mask_iou": float(intersection / union) if union else 1.0,
    }

"""Точное правило отсечения изображений из условий конкурса СПУТНИКС."""

import math


def should_delete(cloud_percent: float) -> bool:
    """Удалять только при облачности строго выше 75%."""
    if not math.isfinite(cloud_percent) or not 0 <= cloud_percent <= 100:
        raise ValueError("Процент облачности должен быть от 0 до 100.")
    return cloud_percent > 75.0

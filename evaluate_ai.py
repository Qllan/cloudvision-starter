"""Оценка качества модели по разметке пользователя; запускается после label_clouds.py."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MASKS = ROOT / "reports" / "masks"
LABELS = ROOT / "reports" / "labels"
OUTPUT = ROOT / "reports" / "evaluation.csv"


def scores(predicted_class_mask: np.ndarray, manual_binary_mask: np.ndarray) -> dict:
    """Сравниваем бинарные маски облаков (плотные + тонкие)."""
    cloud_predicted = (predicted_class_mask == 1) | (predicted_class_mask == 2)
    cloud_manual = manual_binary_mask > 0
    if cloud_predicted.shape != cloud_manual.shape:
        raise ValueError("Размеры масок должны совпадать")
    if cloud_predicted.size == 0:
        raise ValueError("Пустые маски")
    intersection = np.logical_and(cloud_predicted, cloud_manual).sum()
    union = np.logical_or(cloud_predicted, cloud_manual).sum()
    iou = float(intersection / union) if union else 1.0
    pred_pct = 100 * float(cloud_predicted.mean())
    manual_pct = 100 * float(cloud_manual.mean())
    return {
        "model_percent": pred_pct,
        "manual_percent": manual_pct,
        "absolute_error_pp": abs(pred_pct - manual_pct),
        "cloud_iou": iou,
        "same_75_percent_decision": (pred_pct > 75) == (manual_pct > 75),
    }


def main():
    labels = sorted(LABELS.glob("*_label.png")) if LABELS.exists() else []
    if not labels:
        print("Нет ручных разметок. Сначала запусти label_clouds.py и сохрани хотя бы одну маску.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for label_path in labels:
        stem = label_path.stem.removesuffix("_label")
        prediction_path = MASKS / f"{stem}_mask.png"
        if not prediction_path.exists():
            print(f"Нет прогноза для {stem}, пропускаем.")
            continue
        with Image.open(label_path) as label_image:
            manual = np.asarray(label_image.convert("L")) > 127
            target_size = label_image.size
        with Image.open(prediction_path) as prediction_image:
            prediction = np.asarray(prediction_image.resize(target_size, Image.Resampling.NEAREST))

        result = scores(prediction, manual)
        records.append({"file": stem, **result})
        print(f"{stem}: ИИ {result['model_percent']:.1f}% | разметка {result['manual_percent']:.1f}% "
              f"| ошибка {result['absolute_error_pp']:.1f} п.п. | IoU {result['cloud_iou']:.2f}")

    if not records:
        print("Не найдено пар разметка/прогноз.")
        return
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    mean_iou = np.mean([r["cloud_iou"] for r in records])
    mean_error = np.mean([r["absolute_error_pp"] for r in records])
    print(f"\nОценено: {len(records)}; средний IoU: {mean_iou:.3f}; "
          f"средняя абсолютная ошибка: {mean_error:.1f} процентных пункта.")
    print(f"Отчёт: {OUTPUT.relative_to(ROOT)}")
    print("Внимание: оценка зависит от качества ручной разметки!")


if __name__ == "__main__":
    main()

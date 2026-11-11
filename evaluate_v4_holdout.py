"""CloudVision AI v4 — frozen, one-time independent TEST comparison.

Protocol:
  1) --freeze BEFORE examining TEST images: hash and lock v3/v4 checkpoints.
  2) Annotate the five TEST images manually, blind to ALL model predictions.
  3) --evaluate AFTER all reference masks exist. Never tune on TEST results.

No source images, labels, or checkpoint files are modified. Reports only.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from cloudvision.v4_holdout import (
    aggregate_scores, check_frozen, freeze_checkpoints, mask_scores,
    prepared_test_pairs,
)
from cloudvision.inference import load_checkpoint, predict_mask

ROOT = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Нет строк для отчёта")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(manifest: Path, images: Path, labels: Path, v3: Path, v4: Path,
             output: Path, device_name: str) -> None:
    frozen = output / "frozen_models.json"
    if (output / "summary.csv").exists() or (output / "per_image.csv").exists():
        raise FileExistsError(
            f"Отчёт независимого TEST уже существует в {output}; "
            "не используй TEST для повторного выбора модели")
    if output.resolve().is_relative_to(images.resolve()) or output.resolve().is_relative_to(labels.resolve()):
        raise ValueError("Отчёт нельзя сохранять в папке изображений или ручных масок")
    if labels.resolve() == (ROOT / "data/verified_v4").resolve():
        raise ValueError("Для независимого TEST нужны отдельные ручные маски data/test_verified_v4")

    check_frozen(manifest, v3, v4, frozen)
    pairs = prepared_test_pairs(manifest, images, labels)
    # Only after ALL references pass validation, load models and open TEST pixels.
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA не найдена")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model3, size3, _ = load_checkpoint(v3, "policy", device)
    model4, size4, meta4 = load_checkpoint(v4, "policy", device)
    if meta4.get("version") != 4:
        raise ValueError("Выбранная v4-модель не имеет маркера version=4")
    models = {"v3-policy": (model3, size3), "v4-policy-frozen": (model4, size4)}
    per_image = []
    aggregates = {}
    for model_name, (model, size) in models.items():
        rows = []
        for relative, image_path, label_path in pairs:
            with Image.open(image_path) as photo:
                original = ImageOps.exif_transpose(photo).convert("RGB")
            with Image.open(label_path) as hand_mask:
                reference = np.asarray(hand_mask, dtype=np.uint8) == 255
            prediction = np.asarray(predict_mask(original, model, size, device),
                                    dtype=np.uint8) == 255
            row = {"model": model_name, "image": relative}
            row.update(mask_scores(prediction, reference))
            rows.append(row)
        per_image.extend(rows)
        summary = {"model": model_name}
        summary.update(aggregate_scores(rows))
        aggregates[model_name] = summary

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_image.csv", per_image)
    _write_csv(output / "summary.csv", list(aggregates.values()))
    print("Независимая финальная проверка TEST (в исходном разрешении):")
    for summary in aggregates.values():
        print(f"{summary['model']}: MAE {summary['mae_pp']:.2f} п.п. | "
              f"IoU {summary['mean_iou']:.3f} | "
              f"правильных решений {summary['correct_actions']}/{summary['test_count']} | "
              f"ложных отсевов {summary['wrong_rejects']} | "
              f"пропущенных отсевов {summary['missed_rejects']}")
    print("Один TEST-отчёт:", output / "summary.csv")
    print("Подробности:", output / "per_image.csv")
    print("Фиксация моделей:", frozen)
    print("Это оценка на пяти отложенных снимках, НЕ гарантия точности на всех 4207.")
    print("Исходные фотографии, ручные маски и checkpoint не менялись.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Зафиксировать выбранную v4 и однократно проверить v3/v4 на независимом TEST")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true",
                        help="Закрепить веса/выборку контрольными суммами ДО TEST-разметки")
    action.add_argument("--evaluate", action="store_true",
                        help="Один раз оценить обе модели, только ПОСЛЕ слепой ручной TEST-разметки")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/selection_v4/selection.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/new_raw")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/test_verified_v4")
    parser.add_argument("--v3", type=Path, default=ROOT / "models/cloudvision_unet_v3_policy.pt")
    parser.add_argument("--v4", type=Path, default=ROOT / "models/cloudvision_unet_v4_policy.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final_test_v4")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        if args.freeze:
            if args.out.resolve().is_relative_to(args.images.resolve()) or args.out.resolve().is_relative_to(args.labels.resolve()):
                raise ValueError("Папка отчётов не должна быть внутри исходных изображений или ручных масок")
            snapshot = freeze_checkpoints(
                args.manifest, args.v3, args.v4, args.out / "frozen_models.json")
            print("Checkpoint и выборка зафиксированы. TEST-изображения не открывались.")
            print(f"TEST-снимков в CSV: {snapshot['test_count']}")
            print("Фиксация:", args.out / "frozen_models.json")
            print("Теперь вручную разметь TEST без AI-подсказок.")
        else:
            evaluate(args.manifest, args.images, args.labels, args.v3, args.v4,
                     args.out, args.device)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

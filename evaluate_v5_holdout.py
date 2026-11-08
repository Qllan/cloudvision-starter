"""Evaluate frozen v4 vs v5 once on a NEW blind TEST, no training/tuning.

The command --freeze hashes the checkpoints/manifest before labeling;
--evaluate requires every TEST reference mask and exactly the same hashes.
The original JPEG/PNG and network weights are never changed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from cloudvision.v4_holdout import aggregate_scores, mask_scores, prepared_test_pairs
from cloudvision.v5_holdout import freeze_v5, verify_v5_freeze
from cloudvision.inference import load_checkpoint, predict_mask

ROOT = Path(__file__).resolve().parent


def write_one_time(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("Пустой отчёт TEST")
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    root = args.input.resolve()
    label_root = args.labels.resolve()
    out = args.out.resolve()
    if (out == root or out.is_relative_to(root) or
            out == label_root or out.is_relative_to(label_root)):
        raise ValueError("TEST отчёты должны сохраняться отдельно от исходников и эталонных масок")
    if (out / "summary.csv").exists() or (out / "per_image.csv").exists():
        raise FileExistsError("Независимый TEST уже оценён. Не переподбирай модель по TEST!")
    verify_v5_freeze(args.manifest, args.v4, args.v5, out / "frozen_models.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device)
    # All 24 masks are checked first; no predictions until data are complete.
    pairs = prepared_test_pairs(args.manifest, args.input, args.labels)
    model4, size4, meta4 = load_checkpoint(args.v4, "policy", device)
    model5, size5, meta5 = load_checkpoint(args.v5, "policy", device)
    if meta4.get("version") != 4 or meta5.get("version") != 5:
        raise ValueError("Неожиданные версии checkpoint v4/v5")
    per_image = []
    summary = []
    for name, model, size in (("v4-policy", model4, size4), ("v5-pilot", model5, size5)):
        rows = []
        for rel, photo, mask in pairs:
            with Image.open(photo) as source:
                original = ImageOps.exif_transpose(source).convert("RGB")
            with Image.open(mask) as human:
                target = np.asarray(human, dtype=np.uint8) == 255
            predicted = np.asarray(predict_mask(original, model, size, device),
                                   dtype=np.uint8) == 255
            row = {"model": name, "image": rel}
            row.update(mask_scores(predicted, target))
            rows.append(row)
        per_image.extend(rows)
        entry = {"model": name}
        entry.update(aggregate_scores(rows))
        summary.append(entry)
    out.mkdir(parents=True, exist_ok=True)
    write_one_time(out / "per_image.csv", per_image)
    write_one_time(out / "summary.csv", summary)
    for row in summary:
        print(f"{row['model']}: MAE={row['mae_pp']:.2f} п.п.; "
              f"IoU={row['mean_iou']:.3f}; "
              f"решений={row['correct_actions']}/{row['test_count']}; "
              f"пропущенных отсевов={row['missed_rejects']}; "
              f"ложных отсевов={row['wrong_rejects']}")
    print("Отложенный независимый TEST:", out / "summary.csv")
    print("Результаты не следует использовать для подбора параметров этой же v5.")


def main():
    p = argparse.ArgumentParser(description="Один независимый TEST v4 vs v5 на новых кадрах")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="Закрепить v4/v5 до ручной TEST-разметки")
    group.add_argument("--evaluate", action="store_true", help="Однократно сравнить после ручной разметки")
    p.add_argument("--manifest", type=Path, default=ROOT / "reports/v5/holdout_manifest.csv")
    p.add_argument("--input", type=Path, default=ROOT / "data/new_raw")
    p.add_argument("--labels", type=Path, default=ROOT / "data/v5_holdout_verified")
    p.add_argument("--v4", type=Path, default=ROOT / "models/cloudvision_unet_v4_policy.pt")
    p.add_argument("--v5", type=Path, default=ROOT / "models/cloudvision_unet_v5_pilot.pt")
    p.add_argument("--out", type=Path, default=ROOT / "reports/v5/final_holdout")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    try:
        if args.freeze:
            out = args.out.resolve()
            if (out == args.input.resolve() or out.is_relative_to(args.input.resolve())
                    or out == args.labels.resolve() or out.is_relative_to(args.labels.resolve())):
                raise ValueError("Папка --out не должна быть внутри папок исходников/масок")
            snapshot = freeze_v5(args.manifest, args.v4, args.v5, args.out / "frozen_models.json")
            print(f"Новый TEST заморожен: {snapshot['test_count']} снимков.")
            print("Checkpoint и CSV защищены SHA-256. Пиксели TEST НЕ открывались.")
            print("Теперь размечай эти снимки вручную без подсказки AI.")
            print("Фиксация:", args.out / "frozen_models.json")
        else:
            evaluate(args)
    except (ValueError, OSError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()

r"""CloudVision AI — исследование вклада тонких облаков в ошибки.

Запуск (Windows, из корня проекта):
    .\.venv\Scripts\python.exe experiment_cloud_classes.py

Считывает только существующие маски этапа 5 и ручные маски;
ничего не удаляет и не меняет.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
PREDICTED = ROOT / "reports" / "tiling_experiment" / "masks"
LABELS = ROOT / "reports" / "labels"
INPUT = ROOT / "data" / "input"
DEST = ROOT / "reports" / "class_policy_experiment"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def mask_from_file(path: Path) -> np.ndarray:
    # В исходной программе маска индексная (0..3), цветные превью использовать нельзя.
    with Image.open(path) as img:
        if img.mode in ("L", "P"):
            arr = np.asarray(img.convert("L"), dtype=np.uint8)
        elif img.mode in ("RGB", "RGBA"):
            arr = np.asarray(img.getchannel("R"), dtype=np.uint8)
        else:
            arr = np.asarray(img.convert("L"), dtype=np.uint8)
    if arr.ndim != 2 or not np.isin(arr, [0, 1, 2, 3]).all():
        raise ValueError(f"{path.name}: ожидались классы 0, 1, 2, 3; не цветная картинка")
    return arr


def class_stats(mask: np.ndarray) -> dict:
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Нужна двумерная непустая маска")
    if not np.isin(mask, [0, 1, 2, 3]).all():
        raise ValueError("Ожидались индексы классов 0..3")
    return {
        "thick_percent": 100 * float(np.mean(mask == 1)),
        "thin_percent": 100 * float(np.mean(mask == 2)),
        "shadow_percent": 100 * float(np.mean(mask == 3)),
        "both_percent": 100 * float(np.mean(np.isin(mask, [1, 2]))),
    }


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    if pred.shape != truth.shape:
        raise ValueError("Разные размеры")
    a, b = pred.astype(bool), truth.astype(bool)
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return {
        "percent": 100 * float(np.mean(a)),
        "error_pp": abs(100 * float(np.mean(a)) - 100 * float(np.mean(b))),
        "iou": float(intersection / union) if union else 1.0,
    }


def original_for_stem(stem: str) -> Path | None:
    for ext in EXTENSIONS:
        path = INPUT / (stem + ext)
        if path.exists():
            return path
    return None


def comparison_preview(original: Path, predicted: np.ndarray, target: Path):
    with Image.open(original) as img:
        src = img.convert("RGB")
    # Умеренный размер превью, безопасный для ПК средней мощности
    src.thumbnail((900, 700))
    reduced = np.asarray(Image.fromarray(predicted, "L").resize(src.size, Image.Resampling.NEAREST))
    full = np.isin(reduced, [1, 2])
    strict = reduced == 1
    base = np.asarray(src)
    views = []
    for cloud, color in ((full, np.array([255, 90, 55])), (strict, np.array([34, 190, 100]))):
        arr = base.copy()
        arr[cloud] = (0.55 * arr[cloud] + 0.45 * color).astype(np.uint8)
        views.append(Image.fromarray(arr))
    w, h = src.size
    final = Image.new("RGB", (w * 3, h + 42), "white")
    final.paste(src, (0, 42))
    final.paste(views[0], (w, 42))
    final.paste(views[1], (2 * w, 42))
    draw = ImageDraw.Draw(final)
    for x, title in ((12, "ORIGINAL"), (w + 12, "THICK + THIN (current)"), (2 * w + 12, "THICK ONLY (experiment)")):
        draw.text((x, 14), title, fill="black")
    target.parent.mkdir(parents=True, exist_ok=True)
    final.save(target, quality=90)


def main():
    files = sorted(PREDICTED.glob("*_tiled_mask.png"))
    if not files:
        raise SystemExit("Не найдены исходные маски. Сначала выполни experiment_tiling.py --limit 3")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "previews").mkdir(exist_ok=True)
    rows = []
    for file in files:
        stem = file.name.removesuffix("_tiled_mask.png")
        mask = mask_from_file(file)
        part = class_stats(mask)
        item = {"file": stem, **{k: round(v, 3) for k, v in part.items()},
                "manual_percent": "", "both_error_pp": "", "both_iou": "",
                "thick_error_pp": "", "thick_iou": ""}
        lab_path = LABELS / f"{stem}_label.png"
        if lab_path.exists():
            with Image.open(lab_path) as im:
                truth = (np.asarray(im.convert("L")) > 127)
            nearest = np.asarray(Image.fromarray(mask).resize((truth.shape[1], truth.shape[0]), Image.Resampling.NEAREST))
            both = metrics(np.isin(nearest, [1, 2]), truth)
            thick = metrics(nearest == 1, truth)
            item.update(manual_percent=round(100 * float(np.mean(truth)), 3),
                        both_error_pp=round(both["error_pp"], 3), both_iou=round(both["iou"], 4),
                        thick_error_pp=round(thick["error_pp"], 3), thick_iou=round(thick["iou"], 4))
        orig = original_for_stem(stem)
        if orig:
            comparison_preview(orig, mask, DEST / "previews" / f"{stem}_comparison.jpg")
        rows.append(item)
        print(f"{stem}: плотные {part['thick_percent']:.1f}%, тонкие {part['thin_percent']:.1f}%, обе категории {part['both_percent']:.1f}%")
    csv_path = DEST / "results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    annotated = [r for r in rows if r["manual_percent"] != ""]
    if annotated:
        for policy, key in (("Плотные + тонкие", "both_error_pp"), ("Только плотные", "thick_error_pp")):
            avg = sum(r[key] for r in annotated) / len(annotated)
            print(f"Средняя ошибка, {policy}: {avg:.2f} п.п. на {len(annotated)} размеченных снимках")
    print("Отчёт:", csv_path)
    print("Визуализации:", DEST / "previews")
    print("Это только диагностическое сравнение. Тонкие облака нельзя исключать без проверки качества!")
    print("Оригиналы и старые маски не меняются; файлы не удаляются.")


if __name__ == "__main__":
    main()

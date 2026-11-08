r"""CloudVision AI — проверка готовности ручной разметки к обучению.

Запуск из корня проекта: .\.venv\Scripts\python.exe audit_dataset.py
НЕ запускает нейросеть, НЕ меняет оригиналы и маски.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
FIELDNAMES = ["image", "label", "image_size", "label_size", "manual_cloud_percent", "status", "note"]


def evaluate_annotation(image_path: Path, label_path: Path) -> dict:
    """Проверка одной пары оригинал / бинарная маска (0 или 255)."""
    record = {
        "image": image_path.name, "label": label_path.name,
        "image_size": "", "label_size": "", "manual_cloud_percent": "",
        "status": "OK", "note": "",
    }
    try:
        with Image.open(image_path) as image:
            iw, ih = image.size
        with Image.open(label_path) as image:
            lw, lh = image.size
            mode = image.mode
            pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception as exc:
        record["status"] = "ERROR"
        record["note"] = f"Ошибка чтения: {exc}"
        return record

    record["image_size"] = f"{iw}x{ih}"
    record["label_size"] = f"{lw}x{lh}"
    problems = []
    if min(iw, ih, lw, lh) <= 0:
        problems.append("Пустое изображение")
    elif abs(iw / ih - lw / lh) > 0.02:
        problems.append("Соотношение сторон снимка и маски отличается")
    if mode != "L":
        problems.append(f"Маска должна быть grayscale L, сейчас {mode}")
    classes = np.unique(pixels)
    if not np.isin(classes, (0, 255)).all():
        problems.append(f"Найдены значения кроме 0 и 255: {classes[:10].tolist()}")
    cloud_percent = 100 * float(np.count_nonzero(pixels == 255)) / pixels.size
    record["manual_cloud_percent"] = f"{cloud_percent:.3f}"
    if problems:
        record["status"] = "ERROR"
        record["note"] = "; ".join(problems)
    elif cloud_percent == 0:
        record["status"] = "REVIEW"
        record["note"] = "Пустая маска: проверь, что облаков действительно нет и разметка сохранена"
    elif cloud_percent > 99.5:
        record["status"] = "REVIEW"
        record["note"] = "Почти полностью облачный снимок: проверь полноту разметки"
    return record


def create_review_image(image_path: Path, label_path: Path, output_path: Path) -> None:
    """Визуальная проверка: исходный снимок слева, ручная маска справа."""
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    with Image.open(label_path) as label:
        binary = label.convert("L")
    original.thumbnail((600, 480), Image.Resampling.LANCZOS)
    mask = binary.resize(original.size, Image.Resampling.NEAREST).point(lambda v: 255 if v > 127 else 0)
    overlay = Image.new("RGB", original.size, (0, 240, 85))
    translucent = Image.blend(original, overlay, 0.40)
    marked = Image.composite(translucent, original, mask)
    canvas = Image.new("RGB", (2 * original.width, original.height + 30), "white")
    canvas.paste(original, (0, 30))
    canvas.paste(marked, (original.width, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), "ORIGINAL", fill="black")
    draw.text((original.width + 6, 8), "YOUR LABELS (GREEN)", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def analyze_dataset(images_dir: Path, labels_dir: Path) -> tuple[list[dict], list[str], int]:
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    records = []
    unlabelled = []
    found = set()
    for image in images:
        label = labels_dir / f"{image.stem}_label.png"
        if label.is_file():
            records.append(evaluate_annotation(image, label))
            found.add(label.name)
        else:
            unlabelled.append(image.name)
    unused = len([p for p in labels_dir.glob("*_label.png") if p.name not in found]) if labels_dir.is_dir() else 0
    return records, unlabelled, unused


def main():
    parser = argparse.ArgumentParser(description="Проверить разметку облаков перед обучением")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "input")
    parser.add_argument("--labels", type=Path, default=ROOT / "reports" / "labels")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "dataset_audit")
    parser.add_argument("--preview-limit", type=int, default=12)
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Не найдена папка снимков: {args.input}")
    if args.preview_limit < 0:
        parser.error("--preview-limit должен быть >= 0")
    records, unlabelled, unused = analyze_dataset(args.input, args.labels)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "labels_audit.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    statuses = Counter(r["status"] for r in records)
    valid = [r for r in records if r["status"] in ("OK", "REVIEW")]
    percents = [float(r["manual_cloud_percent"]) for r in valid]
    high = sum(p > 75 for p in percents)
    low = sum(p <= 75 for p in percents)
    print("\nCLOUDVISION AI — ПРОВЕРКА РАЗМЕТКИ")
    print(f"Всего снимков: {len(unlabelled) + len(records)}")
    print(f"Есть ручная маска: {len(records)}; без разметки: {len(unlabelled)}")
    print(f"Корректные/требующие проверки маски: {len(valid)}; ошибки: {statuses['ERROR']}")
    print(f"Из размеченных: до 75% включительно — {low}; выше 75% — {high}")
    print(f"Маски без соответствующего снимка: {unused}")
    if statuses['REVIEW']:
        print(f"Проверь вручную {statuses['REVIEW']} маски со статусом REVIEW")
    if high == 0:
        print("[!] Нет размеченных примеров >75% — правило отсева пока нельзя проверить на реальных облаках.")
    if len(valid) < 10:
        print("[!] Для первого исследования собери хотя бы 10-12 разнообразных ПОЛНОСТЬЮ размеченных снимков.")
    print("[!] Для оценки качества понадобится отдельная тестовая группа снимков, не используемая при обучении.")
    print(f"Отчёт: {path}")
    created = 0
    for row in records:
        if created >= args.preview_limit:
            break
        if row["status"] == "ERROR":
            continue
        img = args.input / row["image"]
        lbl = args.labels / row["label"]
        try:
            create_review_image(img, lbl, args.out / "previews" / f"{img.stem}_review.jpg")
            created += 1
        except Exception as exc:
            print(f"Не удалось создать превью для {img.name}: {exc}")
    print(f"Проверочных изображений создано: {created}")
    print("Исходники и ручные маски не изменены.")


if __name__ == "__main__":
    main()

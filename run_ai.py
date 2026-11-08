"""CloudVision AI — первый тест настоящей модели сегментации облаков.

Запуск из корневой папки проекта:
    python run_ai.py --limit 3
    python run_ai.py --all

Важно: это БЕЗОПАСНЫЙ предварительный просмотр. Скрипт НЕ удаляет
оригиналы и не объявляет прогнозы проверенной истиной.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from cloudvision.rules import should_delete


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "input"
REPORT_DIR = ROOT / "reports"
PREVIEW_DIR = REPORT_DIR / "previews"
MASK_DIR = REPORT_DIR / "masks"
MODEL_ID = "uga-gaim/CLIPSeg-CloudSEN12Plus-FFT"
PROMPTS = ["clear", "thick cloud", "thin cloud", "cloud shadow"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def cloud_percent_from_mask(mask: np.ndarray) -> float:
    """Классы 1 и 2 — облака; 0 — ясно, 3 — тень облака."""
    if mask.size == 0:
        raise ValueError("Пустая маска облачности")
    return float(100.0 * np.count_nonzero((mask == 1) | (mask == 2)) / mask.size)


def save_preview(image: Image.Image, predicted_mask: np.ndarray, output_path: Path) -> None:
    """Сохраняет уменьшенную картинку с красными облачными участками."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    preview.thumbnail((800, 800))
    cloud = ((predicted_mask == 1) | (predicted_mask == 2)).astype(np.uint8) * 110
    alpha = Image.fromarray(cloud, mode="L")
    alpha = alpha.resize(preview.size, Image.Resampling.NEAREST)
    red_layer = Image.new("RGB", preview.size, (255, 50, 70))
    result = Image.composite(red_layer, preview, alpha)
    result.save(output_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudVision AI: анализ облачности")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=3,
                       help="Сколько снимков проверить (по умолчанию 3)")
    group.add_argument("--all", action="store_true", help="Проверить все снимки")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        raise SystemExit("Не найдена папка data/input")
    image_paths = sorted(p for p in INPUT_DIR.iterdir()
                         if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not args.all:
        if args.limit < 1:
            parser.error("--limit должен быть положительным числом")
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise SystemExit("Нет изображений в data/input")

    # Тяжёлые библиотеки загружаем только при реальном запуске модели.
    import torch
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    print("Загружаем предобученную модель; при первом запуске потребуется интернет.")
    print("Модель обучена на Sentinel-2, поэтому результаты для СПУТНИКС ещё не проверены.")
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    model.to("cpu")  # Работает даже без дискретной видеокарты.

    results = []
    for number, path in enumerate(image_paths, start=1):
        print(f"\n[{number}/{len(image_paths)}] {path.name}", flush=True)
        try:
            with Image.open(path) as src:
                image = src.convert("RGB")

            inputs = processor(
                text=PROMPTS,
                images=[image] * len(PROMPTS),
                return_tensors="pt",
                padding=True,
            )
            with torch.inference_mode():
                logits = model(**inputs).logits

            # По описанию модели: logits (4, H, W); argmax по 4 классам.
            if logits.ndim != 3 or logits.shape[0] != len(PROMPTS):
                raise RuntimeError(f"Неожиданная форма результата: {tuple(logits.shape)}")
            predicted_mask = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
            # Сохраняем классы нейросети как отдельное изображение,
            # чтобы позже сравнить с ручной разметкой.
            Image.fromarray(predicted_mask).save(MASK_DIR / f"{path.stem}_mask.png")
            cloud_pct = cloud_percent_from_mask(predicted_mask)
            reject = should_delete(cloud_pct)
            status = "К УДАЛЕНИЮ" if reject else "К СОХРАНЕНИЮ"
            preview_name = f"{path.stem}_cloud_preview.jpg"
            save_preview(image, predicted_mask, PREVIEW_DIR / preview_name)
            print(f"  Прогноз облачности: {cloud_pct:.2f}% | {status}")
            results.append({
                "file": path.name,
                "cloud_percent": f"{cloud_pct:.2f}",
                "decision_preview": status,
                "preview_file": preview_name,
                "error": "",
            })
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")
            results.append({
                "file": path.name,
                "cloud_percent": "",
                "decision_preview": "ОШИБКА (НЕ ОБРАБОТАН)",
                "preview_file": "",
                "error": str(exc),
            })

    report_file = REPORT_DIR / "cloud_report_preview.csv"
    with report_file.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=[
            "file", "cloud_percent", "decision_preview", "preview_file", "error"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nГотово. Отчёт: {report_file.relative_to(ROOT)}")
    print(f"Картинки с выделенными облаками: {PREVIEW_DIR.relative_to(ROOT)}")
    print(f"Маски модели: {MASK_DIR.relative_to(ROOT)}")
    print("ВАЖНО: исходные изображения НЕ удалены и НЕ перенесены.")


if __name__ == "__main__":
    main()

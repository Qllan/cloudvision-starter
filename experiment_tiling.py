"""CloudVision AI — сравнение базовой модели и обработки снимка 2×2 фрагментами.

Запуск из корня проекта (Windows 11):
  .\\.venv\\Scripts\\python.exe experiment_tiling.py --limit 3

Оригиналы, разметка и маски предыдущих этапов НЕ изменяются.
Это эксперимент, а не обученная специально для СПУТНИКС модель.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "input"
LABELS = ROOT / "reports" / "labels"
BASELINE = ROOT / "reports" / "masks"
OUT = ROOT / "reports" / "tiling_experiment"
MODEL_ID = "uga-gaim/CLIPSeg-CloudSEN12Plus-FFT"
PROMPTS = ["clear", "thick cloud", "thin cloud", "cloud shadow"]
EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def find_original(stem: str) -> Path | None:
    for ext in EXTENSIONS:
        candidate = INPUT / (stem + ext)
        if candidate.is_file():
            return candidate
    return None


def regions_2x2(width: int, height: int, overlap_ratio: float = 0.08) -> list[tuple[int, int, int, int]]:
    """Четыре прямоугольника (x0, y0, x1, y1) с перекрытием на стыках."""
    if width < 2 or height < 2:
        raise ValueError("Слишком маленькое изображение")
    if not (0 <= overlap_ratio < 0.5):
        raise ValueError("overlap_ratio должен быть от 0 до 0.5")
    mx, my = width // 2, height // 2
    ox = min(mx - 1, round(width * overlap_ratio / 2))
    oy = min(my - 1, round(height * overlap_ratio / 2))
    xs = ((0, mx + ox), (mx - ox, width))
    ys = ((0, my + oy), (my - oy, height))
    return [(x0, y0, x1, y1) for y0, y1 in ys for x0, x1 in xs]


def calculate_scores(predicted: np.ndarray, manual: np.ndarray) -> dict:
    """Предсказание классов 0..3 и бинарная разметка одной формы."""
    if predicted.shape != manual.shape or predicted.size == 0:
        raise ValueError("Размеры масок не совпадают или маски пусты")
    cloud = (predicted == 1) | (predicted == 2)
    manually_cloudy = manual.astype(bool)
    both = np.count_nonzero(cloud & manually_cloudy)
    union = np.count_nonzero(cloud | manually_cloudy)
    return {
        "model_percent": 100.0 * np.mean(cloud),
        "manual_percent": 100.0 * np.mean(manually_cloudy),
        "abs_error_pp": abs(100.0 * np.mean(cloud) - 100.0 * np.mean(manually_cloudy)),
        "iou": (both / union) if union else 1.0,
    }


def make_comparison(image: Image.Image, predicted: np.ndarray, manual: np.ndarray, target: Path) -> None:
    """Сравнение двух половин. Зелёный — TP, красный — FP, синий — FN."""
    h, w = manual.shape
    base = image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    cloud = (predicted == 1) | (predicted == 2)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[cloud & manual] = (30, 210, 70, 145)
    overlay[cloud & ~manual] = (240, 45, 55, 150)
    overlay[~cloud & manual] = (40, 100, 250, 160)
    mixed = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(overlay, "RGBA")).convert("RGB")
    result = Image.new("RGB", (w * 2, h + 36), "white")
    result.paste(base, (0, 36))
    result.paste(mixed, (w, 36))
    draw = ImageDraw.Draw(result)
    draw.text((10, 11), "ORIGINAL", fill="black")
    draw.text((w + 10, 11), "GREEN correct | RED false | BLUE missed", fill="black")
    target.parent.mkdir(parents=True, exist_ok=True)
    result.save(target, quality=90)


def infer_tile_logits(image: Image.Image, model, processor, torch) -> np.ndarray:
    """Получаем четыре карты оценок классов для одного фрагмента."""
    import torch.nn.functional as F

    
    inputs = processor(
        images=[image] * len(PROMPTS),
        text=PROMPTS,
        return_tensors="pt",
        padding=True,
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        if logits.ndim != 3 or logits.shape[0] != len(PROMPTS):
            raise RuntimeError(f"Неожиданная форма модели: {tuple(logits.shape)}")
        resized = F.interpolate(logits[None], size=(image.height, image.width), mode="bilinear", align_corners=False)[0]
    return resized.cpu().numpy().astype(np.float32)


def infer_tiled(image: Image.Image, model, processor, torch) -> np.ndarray:
    """2×2 с перекрытием и усреднением оценок в зоне перекрытия."""
    width, height = image.size
    sums = np.zeros((4, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint8)
    rectangles = regions_2x2(width, height)
    for number, (x0, y0, x1, y1) in enumerate(rectangles, start=1):
        print(f"    Фрагмент {number}/4: {x0}:{x1}, {y0}:{y1}", flush=True)
        crop = image.crop((x0, y0, x1, y1))
        logits = infer_tile_logits(crop, model, processor, torch)
        sums[:, y0:y1, x0:x1] += logits
        counts[y0:y1, x0:x1] += 1
    if np.any(counts == 0):
        raise RuntimeError("Часть изображения осталась без предсказаний")
    sums /= counts[None, :, :]
    return np.argmax(sums, axis=0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент CLIPSeg: целый снимок против 2×2")
    parser.add_argument("--limit", type=int, default=3, help="Сколько размеченных снимков обработать")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit должен быть больше нуля")
    labels = sorted(LABELS.glob("*_label.png")) if LABELS.exists() else []
    pairs = []
    for label_path in labels:
        stem = label_path.stem.removesuffix("_label")
        source = find_original(stem)
        if source is not None:
            pairs.append((stem, source, label_path))
    if not pairs:
        raise SystemExit("Нет пар оригинальный снимок + ручная разметка. Сначала запусти label_clouds.py")
    pairs = pairs[: args.limit]
    print(f"Размеченных изображений: {len(pairs)}. Предыдущие маски и оригиналы не изменятся.")
    print("Загружаем ту же модель CLIPSeg, что в предыдущем эксперименте...")
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).to("cpu")
    model.eval()
    torch.set_grad_enabled(False)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    (OUT / "comparisons").mkdir(parents=True, exist_ok=True)

    records = []
    for k, (stem, path, label_path) in enumerate(pairs, 1):
        print(f"\n[{k}/{len(pairs)}] {path.name}", flush=True)
        with Image.open(path) as source_file:
            source = source_file.convert("RGB")
        with Image.open(label_path) as lab:
            manual = np.array(lab.convert("L")) > 127
            label_size = lab.size
        predicted_tiled = infer_tiled(source, model, processor, torch)
        mask_path = OUT / "masks" / f"{stem}_tiled_mask.png"
        Image.fromarray(predicted_tiled, mode="L").save(mask_path)
        tiled_small = np.asarray(Image.fromarray(predicted_tiled).resize(label_size, Image.Resampling.NEAREST))
        newer = calculate_scores(tiled_small, manual)
        make_comparison(source, tiled_small, manual, OUT / "comparisons" / f"{stem}_tiled_comparison.jpg")
        record = {
            "file": path.name,
            "manual_percent": round(newer["manual_percent"], 3),
            "tiled_percent": round(newer["model_percent"], 3),
            "tiled_abs_error_pp": round(newer["abs_error_pp"], 3),
            "tiled_iou": round(newer["iou"], 4),
            "baseline_percent": "",
            "baseline_abs_error_pp": "",
            "baseline_iou": "",
        }
        old_path = BASELINE / f"{stem}_mask.png"
        if old_path.exists():
            with Image.open(old_path) as old:
                old_small = np.asarray(old.resize(label_size, Image.Resampling.NEAREST))
            prev = calculate_scores(old_small, manual)
            record.update({
                "baseline_percent": round(prev["model_percent"], 3),
                "baseline_abs_error_pp": round(prev["abs_error_pp"], 3),
                "baseline_iou": round(prev["iou"], 4),
            })
            print(f"  Было: облачность {prev['model_percent']:.1f}% / IoU {prev['iou']:.3f}")
        print(f"  Стало: облачность {newer['model_percent']:.1f}% / IoU {newer['iou']:.3f}")
        records.append(record)

    out_csv = OUT / "comparison_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nГотово! Сравнение показателей: {out_csv.relative_to(ROOT)}")
    print(f"Цветные сравнения: {(OUT / 'comparisons').relative_to(ROOT)}")
    print("Это эксперимент. Оригиналы не изменены, удаление файлов выключено.")
    print("Важно: точность зависит от качества ручной разметки, 3 снимков недостаточно для окончательных выводов.")


if __name__ == "__main__":
    main()

r"""CloudVision AI: compare model cloud mask with a manually drawn label.

Run from your existing project root on Windows:
    .\.venv\Scripts\python.exe compare_cloud_masks.py

The script never modifies original satellite images or label masks.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'data' / 'input'
MASKS = ROOT / 'reports' / 'masks'
LABELS = ROOT / 'reports' / 'labels'
OUTPUT = ROOT / 'reports' / 'comparisons'
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')


def find_source(stem: str):
    for ext in EXTENSIONS:
        candidate = INPUT / (stem + ext)
        if candidate.is_file():
            return candidate
    return None


def render_comparison(source_path: Path, prediction_path: Path, label_path: Path, output_path: Path):
    with Image.open(label_path) as image:
        manual = np.asarray(image.convert('L')) > 127
        size = image.size
    with Image.open(prediction_path) as image:
        predicted = np.asarray(image.resize(size, Image.Resampling.NEAREST))
    with Image.open(source_path) as image:
        source = image.convert('RGB').resize(size, Image.Resampling.LANCZOS)

    predicted_cloud = (predicted == 1) | (predicted == 2)
    true_positive = predicted_cloud & manual
    false_positive = predicted_cloud & ~manual
    false_negative = ~predicted_cloud & manual

    # RGBA overlay: green = correct cloud, red = incorrectly marked cloud,
    # blue = missed cloud. Pixels correctly classified as clear are uncolored.
    tint = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    tint[true_positive] = [35, 200, 75, 145]
    tint[false_positive] = [240, 50, 55, 150]
    tint[false_negative] = [50, 110, 255, 155]
    overlay = Image.fromarray(tint, 'RGBA')
    mixed = Image.alpha_composite(source.convert('RGBA'), overlay).convert('RGB')

    # Two aligned panels, so it is easy to compare with the raw satellite shot.
    panel = Image.new('RGB', (size[0] * 2, size[1] + 50), 'white')
    panel.paste(source, (0, 50))
    panel.paste(mixed, (size[0], 50))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 13), 'ORIGINAL IMAGE', fill='black')
    draw.text((size[0] + 12, 13), 'GREEN: correct | RED: false cloud | BLUE: missed', fill='black')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path, quality=90)

    union = np.count_nonzero(predicted_cloud | manual)
    iou = np.count_nonzero(true_positive) / union if union else 1.0
    return {
        'manual_percent': manual.mean() * 100,
        'model_percent': predicted_cloud.mean() * 100,
        'iou': iou,
    }


def main():
    if not LABELS.exists():
        print('Не найдена папка reports/labels. Сначала сохрани ручную разметку.')
        return
    found = 0
    for label in sorted(LABELS.glob('*_label.png')):
        stem = label.stem.removesuffix('_label')
        source = find_source(stem)
        prediction = MASKS / f'{stem}_mask.png'
        if source is None or not prediction.exists():
            print('Пропуск:', stem, '— нет исходного снимка или предсказания')
            continue
        output = OUTPUT / f'{stem}_comparison.jpg'
        scores = render_comparison(source, prediction, label, output)
        found += 1
        print(f'{stem}: ИИ {scores["model_percent"]:.1f}% | ручная {scores["manual_percent"]:.1f}% '
              f'| IoU {scores["iou"]:.3f}')
        if scores['manual_percent'] == 0:
            print('  ПРОВЕРЬ: ручная маска полностью пустая. Это верно только если на снимке нет облаков.')
        print('  Сравнение:', output.relative_to(ROOT))
    if not found:
        print('Не удалось найти пары оригинал/маска ИИ/ручная маска.')
    else:
        print(f'Готово. Создано сравнений: {found}. Оригиналы не изменены.')


if __name__ == '__main__':
    main()

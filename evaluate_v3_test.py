"""Stage 13: evaluate two frozen U-Net v3 checkpoints on the held-out TEST split.

No training, no parameter selection, no deletion, no copying source images.
Run once to report generalization, then use a truly new dataset for future tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_cloudvision import CloudDataset, TinyUNet, evaluate, save_example
from train_cloudvision_v2 import read_split

ROOT = Path(__file__).resolve().parent


def actions(percent: float) -> str:
    """Competition rule: above 75 means REJECT; exactly 75 means KEEP."""
    return 'REJECT' if percent > 75.0 else 'KEEP'


def decision_stats(rows: list[dict]) -> dict:
    """Confusion matrix and per-class recall. Undefined recall is None."""
    tp = sum(actions(float(row['reference_percent'])) == 'REJECT' and
             actions(float(row['predicted_percent'])) == 'REJECT' for row in rows)
    fn = sum(actions(float(row['reference_percent'])) == 'REJECT' and
             actions(float(row['predicted_percent'])) == 'KEEP' for row in rows)
    fp = sum(actions(float(row['reference_percent'])) == 'KEEP' and
             actions(float(row['predicted_percent'])) == 'REJECT' for row in rows)
    tn = sum(actions(float(row['reference_percent'])) == 'KEEP' and
             actions(float(row['predicted_percent'])) == 'KEEP' for row in rows)
    n = tp + fn + fp + tn
    reject_recall = tp / (tp + fn) if tp + fn else None
    keep_recall = tn / (tn + fp) if tn + fp else None
    balanced = (reject_recall + keep_recall) / 2 if reject_recall is not None and keep_recall is not None else None
    return {
        'n': n,
        'actually_reject': tp + fn,
        'actually_keep': tn + fp,
        'correct_reject': tp,
        'missed_reject': fn,
        'wrong_reject': fp,
        'correct_keep': tn,
        'accuracy': (tp + tn) / n if n else None,
        'reject_recall': reject_recall,
        'keep_recall': keep_recall,
        'balanced_accuracy': balanced,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('Пустой отчёт не записывается')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensure_no_scene_overlap(parts):
    sets = {split: {r.get('group', '').strip() for r in rows} for split, rows in parts.items()}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        shared = sets[a] & sets[b]
        if shared:
            raise ValueError(f'Ошибка: группы сцен пересекаются между {a} и {b}: {sorted(shared)[:5]}')


def load_v3(path: Path, device: torch.device, expected_size: int | None = None):
    if not path.is_file():
        raise FileNotFoundError(f'Нет модели: {path}')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint or 'size' not in checkpoint:
        raise ValueError(f'Неизвестный формат контрольной точки: {path}')
    size = int(checkpoint['size'])
    if expected_size is not None and size != expected_size:
        raise ValueError('Модели имеют разный размер входа: корректное сравнение требует одинакового размера')
    model = TinyUNet().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description='TEST after freezing Area & Policy. Does not delete images.')
    parser.add_argument('--split', type=Path, default=ROOT / 'reports/training/dataset_split.csv')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/input')
    parser.add_argument('--labels', type=Path, default=ROOT / 'reports/labels')
    parser.add_argument('--area', type=Path, default=ROOT / 'models/cloudvision_unet_v3_area.pt')
    parser.add_argument('--policy', type=Path, default=ROOT / 'models/cloudvision_unet_v3_policy.pt')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/test_v3')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--previews', type=int, default=6)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.previews < 0:
        parser.error('batch-size должен быть >0; previews >=0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA не найдена. Используй --device cpu')
    if not args.split.is_file():
        parser.error('Нет dataset_split.csv; сначала выполни split_dataset.py')
    parts = read_split(args.split)
    if not all(parts[k] for k in ('train', 'val', 'test')):
        parser.error('TRAIN/VAL/TEST должны содержать хотя бы один снимок')
    ensure_no_scene_overlap(parts)
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    device = torch.device(args.device)
    area, area_ckpt = load_v3(args.area, device)
    policy, policy_ckpt = load_v3(args.policy, device, int(area_ckpt['size']))
    if area_ckpt.get('selection') != 'area' or policy_ckpt.get('selection') != 'policy':
        raise ValueError('Перепутаны модели area/policy или выбраны не те контрольные точки')
    size = int(area_ckpt['size'])
    dataset = CloudDataset(parts['test'], args.images, args.labels, size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    summary_rows = []
    print('CloudVision AI — этап 13: отложенная TEST-группа')
    print(f'Всего тестовых снимков: {len(dataset)}. Обучение и удаление отключены.')
    print('ВНИМАНИЕ: TEST ранее использовалась для отчёта v1, это не полностью новый набор.')
    for label, model, ckpt in [('area', area, area_ckpt), ('policy', policy, policy_ckpt)]:
        metrics, rows = evaluate(model, loader, device, parts['test'])
        if len(rows) != len(dataset):
            raise RuntimeError('Не все тестовые снимки обработаны')
        write_csv(args.out / f'test_predictions_{label}.csv', rows)
        stats = decision_stats(rows)
        summary_rows.append({
            'model': label,
            'epoch': ckpt.get('best_epoch', ''),
            'test_mae_pp': round(metrics['mae_pp'], 3),
            'test_iou': round(metrics['mean_iou'], 4),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
        })
        for idx in range(min(len(dataset), args.previews)):
            filename = Path(parts['test'][idx]['image']).stem + '_comparison.jpg'
            save_example(model, dataset, idx, device, args.out / f'previews_{label}' / filename)
        print(f"{label.upper()}: MAE {metrics['mae_pp']:.2f} п.п.; IoU {metrics['mean_iou']:.3f}; "
              f"верно {stats['correct_reject'] + stats['correct_keep']}/{stats['n']}; "
              f"пропущен отсев {stats['missed_reject']}; ложный отсев {stats['wrong_reject']}")
    write_csv(args.out / 'test_comparison.csv', summary_rows)
    with (args.out / 'test_summary.json').open('w', encoding='utf-8') as output:
        json.dump(summary_rows, output, ensure_ascii=False, indent=2)
    print('Сравнение:', args.out / 'test_comparison.csv')
    print('TEST использовалась только для оценки. Не выбирай модель по этому результату.')
    print('Файлы изображений и разметка не изменены. Ничего не удалялось.')


if __name__ == '__main__':
    main()

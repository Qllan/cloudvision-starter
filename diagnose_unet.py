"""Inspect current U-Net checkpoint on train/validation only (never test).

Place beside train_cloudvision.py. Does not modify any images or models.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_cloudvision import CloudDataset, TinyUNet, evaluate

ROOT = Path(__file__).resolve().parent


def describe(values):
    if not values:
        return 'нет данных'
    return (f'от {min(values):.2f}% до {max(values):.2f}% | '
            f'среднее {sum(values)/len(values):.2f}%')


def main():
    p = argparse.ArgumentParser(description='Диагностика предсказаний U-Net (без обучения)')
    p.add_argument('--model', type=Path, default=ROOT / 'models/cloudvision_unet.pt')
    p.add_argument('--split', type=Path, default=ROOT / 'reports/training/dataset_split.csv')
    p.add_argument('--images', type=Path, default=ROOT / 'data/input')
    p.add_argument('--labels', type=Path, default=ROOT / 'reports/labels')
    args = p.parse_args()
    if not args.model.is_file() or not args.split.is_file():
        p.error('Не найдена модель или разделение данных. Проверь папки models и reports/training.')
    with args.split.open('r', encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    ckpt = torch.load(args.model, map_location='cpu', weights_only=True)
    size = int(ckpt.get('size', 192))
    model = TinyUNet()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    torch.set_num_threads(4)
    print('CloudVision AI — диагностика исходной U-Net')
    print('Контрольная точка:', args.model)
    print('Эпоха модели:', ckpt.get('best_epoch', 'не указана'))
    print('Размер изображения для модели:', size)
    for part in ('train', 'val'):
        part_rows = [r for r in rows if r['split'] == part]
        if not part_rows:
            print(f'Нет {part}-снимков')
            continue
        ds = CloudDataset(part_rows, args.images, args.labels, size=size, augment=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        metrics, predictions = evaluate(model, loader, torch.device('cpu'), part_rows)
        predicted = [v['predicted_percent'] for v in predictions]
        true = [v['reference_percent'] for v in predictions]
        collapsed = (all(v >= 99 for v in predicted)
                     or all(v <= 1 for v in predicted))
        print(f'\n{part.upper()} ({len(part_rows)} изображений)')
        print('  Ручная облачность:', describe(true))
        print('  Прогноз U-Net:', describe(predicted))
        print(f"  MAE: {metrics['mae_pp']:.2f} п.п. | IoU: {metrics['mean_iou']:.3f} | "
              f"точность порога 75%: {metrics['decision_accuracy']:.1%}")
        if collapsed:
            print('  ВНИМАНИЕ: маска почти всегда вся одного класса — коллапс предсказаний!')
        print('  Первые 5 снимков:')
        for entry in predictions[:5]:
            print(f"    {entry['image']}: ручная {entry['reference_percent']:.1f}%, "
                  f"модель {entry['predicted_percent']:.1f}%")
    print('\nТестовая группа НЕ открывалась. Ни один снимок не изменён.')


if __name__ == '__main__':
    main()

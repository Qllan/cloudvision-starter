"""Second training experiment: validation-based checkpoint selection for CloudVision AI.

Requires the previous `train_cloudvision.py` in the same directory.
The *test* split is never inspected or evaluated here. No originals are deleted.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_cloudvision import CloudDataset, TinyUNet, evaluate, save_example

ROOT = Path(__file__).resolve().parent


def cloud_loss(logits, target, valid):
    """Pixel BCE + modest Dice term + soft cloud-area penalty.

    All terms ignore the black padding introduced by letterboxing.
    The area penalty is differentiable; it is NOT a final hard threshold.
    """
    valid_sum = valid.sum(dim=(1, 2, 3)).clamp_min(1)
    pixels = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none'
    )
    bce = ((pixels * valid).sum(dim=(1, 2, 3)) / valid_sum).mean()
    prob = torch.sigmoid(logits) * valid
    actual = target * valid
    dice_loss = (1 - (2 * (prob * actual).sum(dim=(1, 2, 3)) + 1) /
                 (prob.sum(dim=(1, 2, 3)) + actual.sum(dim=(1, 2, 3)) + 1)).mean()
    area_error = ((prob.sum(dim=(1, 2, 3)) / valid_sum) -
                  (actual.sum(dim=(1, 2, 3)) / valid_sum)).abs().mean()
    return bce + 0.20 * dice_loss + 0.30 * area_error


def read_split(path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {part: [r for r in rows if r['split'] == part]
            for part in ('train', 'val', 'test')}


def is_degenerate(predictions):
    if not predictions:
        return True
    values = [r['predicted_percent'] for r in predictions]
    return all(x >= 99.0 for x in values) or all(x <= 1.0 for x in values)


def pick_key(metrics, predictions):
    """Prefer nondegenerate models; within them minimize validation MAE.

    This is only a validation criterion, not a claim of production readiness.
    """
    return (int(is_degenerate(predictions)), metrics['mae_pp'], -metrics['mean_iou'])


def train(args):
    if not args.split.is_file():
        raise FileNotFoundError(f'Нет файла разделения: {args.split}. Запусти split_dataset.py.')
    if args.size < 64 or args.size % 4 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError('--size должен быть >=64 и кратным 4; batch-size и epochs >=1')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA недоступна. Используй --device cpu')
    parts = read_split(args.split)
    if any(not parts[k] for k in ('train', 'val', 'test')):
        raise ValueError('В разделении должны присутствовать train, val, test')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    train_ds = CloudDataset(parts['train'], args.images, args.labels, args.size, augment=True)
    val_ds = CloudDataset(parts['val'], args.images, args.labels, args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = TinyUNet().to(device)
    # Start slightly below 50% rather than defaulting to 100% cloud coverage.
    nn.init.constant_(model.out.bias, -0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    history, best_key, best_epoch = [], None, None
    print('CloudVision AI — эксперимент U-Net v2')
    print('TRAIN:', len(train_ds), 'VAL:', len(val_ds),
          '| TEST зарезервирован:', len(parts['test']))
    print('На этом этапе тестовая группа НЕ используется для выбора модели.')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for image, target, valid in train_loader:
            image = image.to(device)
            target = target.to(device)
            valid = valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = cloud_loss(logits, target, valid)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * image.size(0)
        val_metrics, val_rows = evaluate(model, val_loader, device, parts['val'])
        val_percent = [x['predicted_percent'] for x in val_rows]
        degenerate = is_degenerate(val_rows)
        epoch_row = {
            'epoch': epoch,
            'train_loss': round(total_loss / len(train_ds), 5),
            'val_mae_pp': round(val_metrics['mae_pp'], 3),
            'val_iou': round(val_metrics['mean_iou'], 4),
            'val_decision_accuracy': round(val_metrics['decision_accuracy'], 4),
            'val_min_percent': round(min(val_percent), 2),
            'val_max_percent': round(max(val_percent), 2),
            'all_one_class': int(degenerate),
        }
        history.append(epoch_row)
        print(f"Эпоха {epoch:02d}/{args.epochs}: "
              f"train loss={epoch_row['train_loss']:.3f} | "
              f"VAL MAE={epoch_row['val_mae_pp']:.2f} п.п. | "
              f"IoU={epoch_row['val_iou']:.3f} | "
              f"порог75={val_metrics['decision_accuracy']:.1%} | "
              f"прогнозы={min(val_percent):.0f}–{max(val_percent):.0f}%"
              + (' [ОДИН КЛАСС!]' if degenerate else ''))
        current_key = pick_key(val_metrics, val_rows)
        if best_key is None or current_key < best_key:
            best_key = current_key
            best_epoch = epoch
            torch.save({
                'state_dict': model.state_dict(), 'size': args.size,
                'seed': args.seed, 'best_epoch': epoch, 'threshold': 0.5,
                'selection': 'validation area MAE (nondegenerate preferred)',
            }, args.model)
    with (args.out / 'training_history.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    selected_metrics, selected_rows = evaluate(model, val_loader, device, parts['val'])
    with (args.out / 'validation_predictions.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    for i in range(min(len(val_ds), 5)):
        path = args.out / 'validation_previews' / (Path(parts['val'][i]['image']).stem + '_comparison.jpg')
        save_example(model, val_ds, i, device, path)
    print('\nЛУЧШАЯ ЭПОХА ПО ВАЛИДАЦИИ:', best_epoch)
    print(f"Val MAE={selected_metrics['mae_pp']:.2f} п.п.; "
          f"IoU={selected_metrics['mean_iou']:.3f}; "
          f"правильные решения 75%={selected_metrics['decision_accuracy']:.1%}")
    if is_degenerate(selected_rows):
        print('ВНИМАНИЕ: даже лучшая модель выделяет почти всё/ничего. НЕ применяй для сортировки.')
    print('История обучения:', args.out / 'training_history.csv')
    print('Прогнозы по VAL:', args.out / 'validation_predictions.csv')
    print('Модель v2:', args.model)
    print('Исходная U-Net не заменена; тестовую группу не открывали; файлы не удалялись.')
    return selected_metrics, selected_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=Path, default=ROOT / 'reports/training/dataset_split.csv')
    ap.add_argument('--images', type=Path, default=ROOT / 'data/input')
    ap.add_argument('--labels', type=Path, default=ROOT / 'reports/labels')
    ap.add_argument('--out', type=Path, default=ROOT / 'reports/training_v2')
    ap.add_argument('--model', type=Path, default=ROOT / 'models/cloudvision_unet_v2.pt')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--size', type=int, default=192)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()

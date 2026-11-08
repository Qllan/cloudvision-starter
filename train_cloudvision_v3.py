"""Stage 12: compare area-accurate and 75%-decision-accurate U-Net checkpoints.

Requires train_cloudvision.py and train_cloudvision_v2.py from previous stages.
Uses TRAIN for gradients and VAL for selection. Never reads TEST images.
Does not sort or delete any source images.
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
from train_cloudvision_v2 import cloud_loss, read_split, is_degenerate

ROOT = Path(__file__).resolve().parent


def confusion(rows):
    """Counts and balanced accuracy for rule cloud_percent >75 = REJECT."""
    tp = sum(r['reference_action'] == 'REJECT' and r['predicted_action'] == 'REJECT' for r in rows)
    fn = sum(r['reference_action'] == 'REJECT' and r['predicted_action'] == 'KEEP' for r in rows)
    fp = sum(r['reference_action'] == 'KEEP' and r['predicted_action'] == 'REJECT' for r in rows)
    tn = sum(r['reference_action'] == 'KEEP' and r['predicted_action'] == 'KEEP' for r in rows)
    # The two classes must exist in VAL for balanced selection.
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError('VAL должна содержать снимки как выше 75%, так и до 75% включительно')
    reject_recall = tp / (tp + fn)
    keep_recall = tn / (tn + fp)
    return {
        'correct_reject': tp, 'missed_reject': fn,
        'wrong_reject': fp, 'correct_keep': tn,
        'reject_recall': reject_recall, 'keep_recall': keep_recall,
        'balanced_accuracy': (reject_recall + keep_recall) / 2,
        'accuracy': (tp + tn) / len(rows),
    }


def area_key(metrics, rows):
    """Lower is better. Optimize area percentage, as v2 did."""
    return (int(is_degenerate(rows)), metrics['mae_pp'], -metrics['mean_iou'])


def policy_key(metrics, rows):
    """Lower is better. Do not reward the trivial 'KEEP everything' policy."""
    c = confusion(rows)
    return (int(is_degenerate(rows)), -c['balanced_accuracy'],
            -c['accuracy'], metrics['mae_pp'], -metrics['mean_iou'])


def checkpoint_payload(model, args, epoch, selection):
    return {
        'state_dict': model.state_dict(), 'size': args.size, 'seed': args.seed,
        'best_epoch': epoch, 'threshold': 0.5, 'selection': selection,
    }


def write_rows(dest, rows):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    if not args.split.is_file():
        raise FileNotFoundError(f'Нет {args.split}. Сначала запусти split_dataset.py')
    if args.size < 64 or args.size % 4 != 0 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError('size >=64 и кратен 4, epochs >=1, batch-size >=1')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA недоступна. Используй CPU')
    parts = read_split(args.split)
    if not all(parts[p] for p in ('train', 'val', 'test')):
        raise ValueError('Должны быть train, val и test')
    all_groups = [{r.get('group') for r in parts[p]} for p in ('train', 'val', 'test')]
    if any(all_groups[i] & all_groups[j] for i, j in ((0,1),(0,2),(1,2))):
        raise ValueError('Одна группа сцен обнаружена в разных частях выборки!')
    val_positive = sum(float(r['manual_cloud_percent']) > 75 for r in parts['val'])
    if val_positive == 0 or val_positive == len(parts['val']):
        raise ValueError('Для выбора модели VAL должна содержать оба класса решения 75%')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    device = torch.device(args.device)
    train_ds = CloudDataset(parts['train'], args.images, args.labels, args.size, augment=True)
    val_ds = CloudDataset(parts['val'], args.images, args.labels, args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TinyUNet().to(device)
    nn.init.constant_(model.out.bias, -0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    paths = {
        'area': args.models / 'cloudvision_unet_v3_area.pt',
        'policy': args.models / 'cloudvision_unet_v3_policy.pt',
    }
    best = {'area': (None, None), 'policy': (None, None)}
    history = []
    print('CloudVision AI / этап 12: два способа выбирать лучшую U-Net')
    print(f"TRAIN {len(parts['train'])} | VAL {len(parts['val'])} "
          f"(отклонить по разметке {val_positive}) | TEST {len(parts['test'])} — НЕ используется")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        for x, y, valid in train_loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = cloud_loss(model(x), y, valid)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * x.size(0)
        metrics, rows = evaluate(model, val_loader, device, parts['val'])
        c = confusion(rows)
        history.append({
            'epoch': epoch,
            'train_loss': round(train_sum / len(train_ds), 5),
            'val_mae_pp': round(metrics['mae_pp'], 3),
            'val_iou': round(metrics['mean_iou'], 4),
            'val_balanced_accuracy': round(c['balanced_accuracy'], 4),
            'val_accuracy': round(c['accuracy'], 4),
            'correct_reject': c['correct_reject'],
            'missed_reject': c['missed_reject'],
            'wrong_reject': c['wrong_reject'],
            'correct_keep': c['correct_keep'],
            'all_one_class': int(is_degenerate(rows)),
        })
        print(f"Эпоха {epoch:02d}/{args.epochs}: MAE={metrics['mae_pp']:.2f} п.п. "
              f"| IoU={metrics['mean_iou']:.3f} | решений={c['accuracy']:.1%} "
              f"| баланс={c['balanced_accuracy']:.1%} "
              f"| отклонено правильно {c['correct_reject']}/{c['correct_reject']+c['missed_reject']} "
              f"| ложный отсев {c['wrong_reject']}")
        for label, key in (('area', area_key(metrics, rows)), ('policy', policy_key(metrics, rows))):
            old_key, _ = best[label]
            if old_key is None or key < old_key:
                torch.save(checkpoint_payload(model, args, epoch, label), paths[label])
                best[label] = (key, epoch)

    write_rows(args.out / 'training_history.csv', history)
    report = []
    for label, path in paths.items():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        metrics, rows = evaluate(model, val_loader, device, parts['val'])
        c = confusion(rows)
        write_rows(args.out / f'validation_predictions_{label}.csv', rows)
        report.append({
            'choice': label, 'epoch': checkpoint['best_epoch'],
            'val_mae_pp': round(metrics['mae_pp'], 3),
            'val_iou': round(metrics['mean_iou'], 4),
            'val_accuracy': round(c['accuracy'], 4),
            'val_balanced_accuracy': round(c['balanced_accuracy'], 4),
            'correct_reject': c['correct_reject'],
            'missed_reject': c['missed_reject'],
            'wrong_reject': c['wrong_reject'],
            'correct_keep': c['correct_keep'],
            'checkpoint': str(path),
        })
        for idx in range(min(3, len(val_ds))):
            dest = args.out / f'previews_{label}' / (Path(parts['val'][idx]['image']).stem + '_comparison.jpg')
            save_example(model, val_ds, idx, device, dest)
    write_rows(args.out / 'model_comparison.csv', report)
    print('\nСравнение двух моделей на VAL:')
    for r in report:
        print(f"{r['choice']:6s} эпоха {r['epoch']}: MAE {r['val_mae_pp']} п.п.; "
              f"верных решений {r['val_accuracy']:.1%}; правильный отсев "
              f"{r['correct_reject']}/{r['correct_reject']+r['missed_reject']}")
    print('Отчёт:', args.out / 'model_comparison.csv')
    print('Тестовые снимки не открывались; оригиналы не изменены; ничего не удалялось.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=Path, default=ROOT / 'reports/training/dataset_split.csv')
    ap.add_argument('--images', type=Path, default=ROOT / 'data/input')
    ap.add_argument('--labels', type=Path, default=ROOT / 'reports/labels')
    ap.add_argument('--models', type=Path, default=ROOT / 'models')
    ap.add_argument('--out', type=Path, default=ROOT / 'reports/training_v3')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--size', type=int, default=192)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--seed', type=int, default=42)
    train(ap.parse_args())


if __name__ == '__main__':
    main()

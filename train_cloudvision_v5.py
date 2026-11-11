"""CloudVision v5 pilot fine-tuning with a new independent holdout reserved.

Warm-starts frozen U-Net v4 Policy and uses only verified training labels:
old TRAIN, v4 TRAIN, and AI-assisted manually reviewed PILOT. Keeps the v4 VAL
for checkpoint selection. The NEW v5 holdout is checked by name/group only,
never opened, predicted, or used to optimize v5.

Existing models, mask files and original images are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from cloudvision.v4_checkpoint import checkpoint_key
from cloudvision.v5_data import prepare_v5_plan, validate_frozen_v5_plan
from train_cloudvision import TinyUNet, evaluate, save_example
from train_cloudvision_v2 import cloud_loss, is_degenerate
from train_cloudvision_v4 import V4Dataset, _write_csv, evaluate_originals

ROOT = Path(__file__).resolve().parent


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size и patience должны быть >=1")
    if args.size < 64 or args.size % 4:
        raise ValueError("--size должен быть >=64 и кратен четырём")
    if not 0 < args.lr < 0.01:
        raise ValueError("Укажи --lr между 0 и 0.01")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна")
    if args.model.resolve() == args.start_checkpoint.resolve():
        raise ValueError("Нельзя перезаписать исходную v4")
    if args.model.name.casefold().startswith("cloudvision_unet_v3") or args.model.name.casefold().startswith("cloudvision_unet_v4"):
        raise ValueError("Имя нового checkpoint должно относиться к v5, а не к старой модели")
    if args.model.is_file() or (args.out / "training_history.csv").exists():
        raise FileExistsError("Результат v5 уже существует: используй новое имя модели и папку отчётов")
    protected = [args.images.resolve(), args.new_masks.resolve(), args.old_images.resolve(),
                 args.old_masks.resolve(), args.pilot_masks.resolve()]
    for candidate in [args.model.resolve(), args.out.resolve()]:
        if any(candidate == original or candidate.is_relative_to(original)
               for original in protected):
            raise ValueError(f"Запрещено записывать результаты внутрь исходных данных: {candidate}")

    pairs, exclusions = prepare_v5_plan(
        args.new_manifest, args.images, args.new_masks,
        args.old_split, args.old_images, args.old_masks,
        args.pilot_report, args.pilot_masks)
    validate_frozen_v5_plan(args.preflight, pairs, args.holdout)

    counts = Counter((r.split, r.source) for r in pairs)
    train_pairs = [r for r in pairs if r.split == "train"]
    val_pairs = [r for r in pairs if r.split == "val"]
    if not any(r.source == "pilot" for r in train_pairs):
        raise ValueError("В новой обучающей выборке отсутствуют исправленные пилотные кадры")
    if len(val_pairs) < 5:
        raise ValueError("Недостаточно примеров VAL")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.start_checkpoint, map_location=device, weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("selection") != "policy"
            or checkpoint.get("version") != 4 or "state_dict" not in checkpoint):
        raise ValueError("Для v5 нужна сохранённая v4 Policy, версия 4, а не v3")
    model = TinyUNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    train_ds = V4Dataset(train_pairs, args.size, augment=True)
    val_ds = V4Dataset(val_pairs, args.size, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    source_weights = {"pilot": 4.0, "v4": 2.0, "old": 1.0}
    sampler = WeightedRandomSampler(
        [source_weights[row.source] for row in train_pairs],
        num_samples=len(train_pairs), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Baseline is the exact same v4 checkpoint we warm-started from.
    base_metrics, base_rows = evaluate_originals(
        model, val_pairs, int(checkpoint["size"]), device)
    args.out.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "baseline_v4_validation.csv", base_rows)

    print("CloudVision AI v5 — дообучение из v4 (только TRAIN + PILOT)")
    print(f"GPU: {device.type.upper()} | size={args.size} | epochs={args.epochs}")
    print(f"TRAIN {len(train_ds)}: старых {counts[('train','old')]}, "
          f"v4 TRAIN {counts[('train','v4')]}, PILOT {counts[('train','pilot')]}")
    print(f"VAL {len(val_ds)}: неизменённая v4 VAL; новый v5 TEST не открывается")
    print(f"v4 на VAL: MAE={base_metrics['mae_pp']:.2f} п.п., "
          f"IoU={base_metrics['mean_iou']:.3f}, "
          f"решения={base_metrics['decision_accuracy']:.1%}")
    if exclusions:
        print(f"Пилотные замены / исключения по группам: {len(exclusions)}")
    history = []
    best_key = None
    best_epoch = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y, valid in train_loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = cloud_loss(model(x), y, valid)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
        metrics, rows = evaluate(model, val_loader, device)
        degenerate = is_degenerate(rows)
        key = checkpoint_key(metrics, degenerate, "area")
        history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / len(train_ds), 5),
            "val_mae_pp": round(metrics["mae_pp"], 3),
            "val_iou": round(metrics["mean_iou"], 4),
            "val_decision_accuracy": round(metrics["decision_accuracy"], 4),
            "degenerate": int(degenerate),
        })
        _write_csv(args.out / "training_history.csv", history)
        print(f"Эпоха {epoch:02d}/{args.epochs}: "
              f"loss={history[-1]['train_loss']:.4f} | "
              f"VAL MAE={metrics['mae_pp']:.2f} п.п. | "
              f"IoU={metrics['mean_iou']:.3f} | "
              f"решения={metrics['decision_accuracy']:.1%}"
              + (" [ВЫРОЖДЕННЫЙ ПРОГНОЗ]" if degenerate else ""), flush=True)
        if best_key is None or key < best_key:
            best_key, best_epoch, stale = key, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "size": args.size, "seed": args.seed,
                "best_epoch": epoch, "threshold": 0.5,
                "version": 5, "selection": "policy",
                "selection_metric": "min VAL MAE, nondegenerate preferred",
                "warm_start": args.start_checkpoint.name,
                "pilot_assisted": True,
                "train_count": len(train_pairs),
                "val_count": len(val_pairs),
            }, args.model)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Ранняя остановка: {args.patience} эпох без улучшения")
                break

    best = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    metrics, rows = evaluate_originals(model, val_pairs, args.size, device)
    _write_csv(args.out / "validation_predictions_v5.csv", rows)
    _write_csv(args.out / "model_comparison.csv", [
        {"model": "v4-policy", "best_epoch": checkpoint.get("best_epoch", ""),
         "val_mae_pp": round(base_metrics["mae_pp"], 3),
         "val_iou": round(base_metrics["mean_iou"], 4),
         "val_decision_accuracy": round(base_metrics["decision_accuracy"], 4)},
        {"model": "v5-pilot", "best_epoch": best_epoch,
         "val_mae_pp": round(metrics["mae_pp"], 3),
         "val_iou": round(metrics["mean_iou"], 4),
         "val_decision_accuracy": round(metrics["decision_accuracy"], 4)},
    ])
    model.eval()
    for index in range(min(4, len(val_ds))):
        dest = args.out / "validation_previews" / (
            Path(val_pairs[index].relative_path).stem + "_comparison.jpg")
        save_example(model, val_ds, index, device, dest)
    print(f"Сохранена v5: эпоха {best_epoch} | "
          f"MAE={metrics['mae_pp']:.2f} п.п. | IoU={metrics['mean_iou']:.3f}")
    print("Новый checkpoint:", args.model)
    print("Сравнение с v4:", args.out / "model_comparison.csv")
    print("ЭТО VAL, а не новый независимый TEST.")
    print("Исходники, v3/v4 и прошлые TEST не изменялись.")


def main():
    p = argparse.ArgumentParser(description="U-Net v5: осторожное GPU дообучение с пилотными масками")
    p.add_argument("--images", type=Path, default=ROOT / "data/new_raw")
    p.add_argument("--new-manifest", type=Path, default=ROOT / "reports/selection_v4/selection.csv")
    p.add_argument("--new-masks", type=Path, default=ROOT / "data/verified_v4")
    p.add_argument("--old-split", type=Path, default=ROOT / "reports/training/dataset_split.csv")
    p.add_argument("--old-images", type=Path, default=ROOT / "data/input")
    p.add_argument("--old-masks", type=Path, default=ROOT / "reports/labels")
    p.add_argument("--pilot-report", type=Path,
                   default=ROOT / "reports/competition_pilot_v4/operator_review/operator_review.csv")
    p.add_argument("--pilot-masks", type=Path, default=ROOT / "data/pilot_verified_v4")
    p.add_argument("--preflight", type=Path, default=ROOT / "reports/v5/preflight/train_manifest.csv")
    p.add_argument("--holdout", type=Path, default=ROOT / "reports/v5/holdout_manifest.csv")
    p.add_argument("--start-checkpoint", type=Path,
                   default=ROOT / "models/cloudvision_unet_v4_policy.pt")
    p.add_argument("--model", type=Path, default=ROOT / "models/cloudvision_unet_v5_pilot.pt")
    p.add_argument("--out", type=Path, default=ROOT / "reports/training_v5")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.00008)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    try:
        train(a)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()

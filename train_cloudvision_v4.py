"""CloudVision AI v4: CUDA fine-tuning of the v3 U-Net on verified cloud masks.

Uses only manually checked new TRAIN/VAL masks and optionally historical TRAIN.
Does not open new TEST images or use old VAL/TEST for gradients.
Writes a *new* v4 checkpoint; never alters the v3 model or source photos.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from cloudvision.v4_data import Example, prepare_examples
from train_cloudvision import TinyUNet, evaluate, letterbox_rgb_and_mask, save_example
from train_cloudvision_v2 import cloud_loss, is_degenerate
from run_competition_demo import predict_mask

ROOT = Path(__file__).resolve().parent


class V4Dataset(Dataset):
    """Read original files on demand, with matching aspect-preserving masks."""
    def __init__(self, rows: list[Example], size: int, augment: bool = False):
        self.rows = rows
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row.image_path) as photo, Image.open(row.label_path) as mask:
            image = ImageOps.exif_transpose(photo).convert("RGB")
            original, target, valid = letterbox_rgb_and_mask(image, mask, self.size)
        x = np.asarray(original, dtype=np.float32).copy() / 255.0
        y = (np.asarray(target, dtype=np.uint8) >= 128).astype(np.float32)
        v = (np.asarray(valid, dtype=np.uint8) > 0).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                x, y, v = np.flip(x, axis=1), np.flip(y, axis=1), np.flip(v, axis=1)
            if random.random() < 0.5:
                x, y, v = np.flip(x, axis=0), np.flip(y, axis=0), np.flip(v, axis=0)
            # Mild brightness/contrast changes reduce sensitivity to new terrain.
            if random.random() < 0.6:
                contrast = random.uniform(0.88, 1.12)
                brightness = random.uniform(0.90, 1.10)
                x = np.clip((x - 0.5) * contrast + 0.5, 0, 1)
                x = np.clip(x * brightness, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(y[None])),
                torch.from_numpy(np.ascontiguousarray(v[None])))


def evaluate_originals(model: TinyUNet, examples: list[Example], size: int,
                       device: torch.device) -> tuple[dict[str, float], list[dict]]:
    """Evaluate at original satellite resolution, exactly like operator inference."""
    model.eval()
    rows = []
    for example in examples:
        with Image.open(example.image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(example.label_path) as verified:
            reference = np.asarray(verified, dtype=np.uint8) == 255
        predicted = np.asarray(predict_mask(image, model, size, device),
                               dtype=np.uint8) == 255
        total = reference.size
        reference_percent = 100 * float(np.count_nonzero(reference)) / total
        predicted_percent = 100 * float(np.count_nonzero(predicted)) / total
        intersection = np.count_nonzero(predicted & reference)
        union = np.count_nonzero(predicted | reference)
        iou = (intersection / union) if union else 1.0
        rows.append({
            "image": example.image_name,
            "predicted_percent": round(predicted_percent, 3),
            "reference_percent": round(reference_percent, 3),
            "absolute_error_pp": round(abs(predicted_percent - reference_percent), 3),
            "iou": round(float(iou), 5),
            "reference_action": "REJECT" if reference_percent > 75 else "KEEP",
            "predicted_action": "REJECT" if predicted_percent > 75 else "KEEP",
        })
    return {
        "mae_pp": sum(r["absolute_error_pp"] for r in rows) / len(rows),
        "mean_iou": sum(r["iou"] for r in rows) / len(rows),
        "decision_accuracy": sum(r["reference_action"] == r["predicted_action"]
                                 for r in rows) / len(rows),
    }, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_arguments(args) -> None:
    if args.size < 64 or args.size % 4 or args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("size должен быть >=64 и кратен 4; epochs, batch-size и patience >=1")
    if not 0 < args.lr <= 0.01:
        raise ValueError("Укажи learning rate между 0 и 0.01")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна в текущем PyTorch")
    if args.model.resolve() == args.start_checkpoint.resolve():
        raise ValueError("Модель v4 должна сохраняться отдельно от старой v3!")
    if args.model.is_file() and not args.overwrite:
        raise FileExistsError(f"Модель уже существует: {args.model}. Укажи другой --model или --overwrite")
    if (args.out / "training_history.csv").exists() and not args.overwrite:
        raise FileExistsError(f"История обучения уже существует: {args.out}. Укажи другой --out или --overwrite")
    sources = [args.images.resolve(), args.verified.resolve()]
    if not args.no_old_train:
        sources.extend([args.old_images.resolve(), args.old_masks.resolve()])
    for path in (args.model.resolve(), args.out.resolve()):
        if any(path == src or path.is_relative_to(src) for src in sources):
            raise ValueError(f"Нельзя записывать модель или отчёты внутрь папок исходных данных: {path}")


def train(args) -> dict[str, float]:
    _validate_arguments(args)
    parts = prepare_examples(
        args.manifest, args.images, args.verified,
        old_split=None if args.no_old_train else args.old_split,
        old_images=None if args.no_old_train else args.old_images,
        old_masks=None if args.no_old_train else args.old_masks,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    model = TinyUNet().to(device)
    if not args.start_checkpoint.is_file():
        raise FileNotFoundError(f"Не найдена модель v3: {args.start_checkpoint}")
    checkpoint = torch.load(args.start_checkpoint, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("selection") != "policy" or "state_dict" not in checkpoint:
        raise ValueError("Нужна исходная контрольная точка U-Net v3 Policy")
    model.load_state_dict(checkpoint["state_dict"])

    train_ds = V4Dataset(parts["train"], args.size, augment=True)
    val_ds = V4Dataset(parts["val"], args.size, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    old_size = int(checkpoint["size"])
    old_metrics, old_rows = evaluate_originals(model, parts["val"], old_size, device)

    # Upweight new TRAIN scenes during fine-tuning to learn new terrain colors.
    weights = [3.0 if sample.source == "new" else 1.0 for sample in parts["train"]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "baseline_v3_validation_predictions.csv", old_rows)
    history = []
    best_key = None
    best_epoch = 0
    stale_epochs = 0

    from collections import Counter
    counts = Counter(s.source for s in parts["train"])
    print(f"CloudVision AI v4 | {device.type.upper()} | размер {args.size} | "
          f"веса v3, эпоха {checkpoint.get('best_epoch','?')}")
    print(f"TRAIN {len(train_ds)} (новых: {counts['new']}, старых: {counts['old']}); "
          f"VAL {len(val_ds)}; TEST — НЕ открывается")
    print(f"v3 на новой VAL (исходное разрешение): MAE {old_metrics['mae_pp']:.2f} п.п. | "
          f"IoU {old_metrics['mean_iou']:.3f} | "
          f"точность решений {old_metrics['decision_accuracy']:.1%}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        accumulated = 0.0
        for x, y, valid in train_loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = cloud_loss(model(x), y, valid)
            loss.backward()
            optimizer.step()
            accumulated += float(loss.item()) * x.size(0)
        metrics, rows = evaluate(model, val_loader, device)
        for row, example in zip(rows, parts["val"]):
            row["image"] = example.image_name
        # Prefer a meaningful cloud-area estimate; break MAE ties by the 75% rule.
        key = (int(is_degenerate(rows)), metrics["mae_pp"],
               -metrics["decision_accuracy"], -metrics["mean_iou"])
        history.append({
            "epoch": epoch,
            "train_loss": round(accumulated / len(train_ds), 5),
            "val_mae_pp": round(metrics["mae_pp"], 3),
            "val_iou": round(metrics["mean_iou"], 4),
            "val_decision_accuracy": round(metrics["decision_accuracy"], 4),
            "degenerate": int(is_degenerate(rows)),
        })
        _write_csv(args.out / "training_history.csv", history)
        print(f"Эпоха {epoch:02d}/{args.epochs}: loss={history[-1]['train_loss']:.4f} | "
              f"VAL MAE={metrics['mae_pp']:.2f} п.п. | IoU={metrics['mean_iou']:.3f} | "
              f"решения={metrics['decision_accuracy']:.1%}"
              + (" [ВЫРОЖДЕННЫЙ ПРОГНОЗ]" if is_degenerate(rows) else ""), flush=True)
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "size": args.size,
                "seed": args.seed,
                "best_epoch": epoch,
                "threshold": 0.5,
                "selection": "policy",  # compatible with existing competition inference
                "version": 4,
                "selection_metric": "validation MAE; decision accuracy tie-break",
                "warm_start": str(args.start_checkpoint.name),
                "new_train_count": counts["new"],
                "old_train_count": counts["old"],
            }, args.model)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Ранняя остановка: {args.patience} эпох без улучшения на VAL")
                break

    best = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    metrics, rows = evaluate_originals(model, parts["val"], args.size, device)
    _write_csv(args.out / "validation_predictions_v4.csv", rows)
    _write_csv(args.out / "model_comparison.csv", [
        {"model": "v3-policy", "best_epoch": checkpoint.get("best_epoch", ""),
         "val_mae_pp": round(old_metrics["mae_pp"], 3),
         "val_iou": round(old_metrics["mean_iou"], 4),
         "val_decision_accuracy": round(old_metrics["decision_accuracy"], 4)},
        {"model": "v4-policy", "best_epoch": best_epoch,
         "val_mae_pp": round(metrics["mae_pp"], 3),
         "val_iou": round(metrics["mean_iou"], 4),
         "val_decision_accuracy": round(metrics["decision_accuracy"], 4)},
    ])
    model.eval()
    for index in range(min(4, len(val_ds))):
        example = parts["val"][index]
        dest = args.out / "validation_previews" / (Path(example.image_name).stem + "_comparison.jpg")
        save_example(model, val_ds, index, device, dest)
    print(f"Лучшая эпоха v4: {best_epoch} | MAE {metrics['mae_pp']:.2f} п.п. | "
          f"IoU {metrics['mean_iou']:.3f} | решения {metrics['decision_accuracy']:.1%}")
    print("Модель:", args.model)
    print("Сравнение:", args.out / "model_comparison.csv")
    print("TEST не читался; старый v3 checkpoint не изменялся.")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Обучение U-Net v4 на вручную проверенных снимках СПУТНИКС")
    p.add_argument("--manifest", type=Path, default=ROOT / "reports/selection_v4/selection.csv")
    p.add_argument("--images", type=Path, default=ROOT / "data/new_raw")
    p.add_argument("--verified", type=Path, default=ROOT / "data/verified_v4")
    p.add_argument("--old-split", type=Path, default=ROOT / "reports/training/dataset_split.csv")
    p.add_argument("--old-images", type=Path, default=ROOT / "data/input")
    p.add_argument("--old-masks", type=Path, default=ROOT / "reports/labels")
    p.add_argument("--no-old-train", action="store_true",
                   help="Учить только по 35 новым TRAIN (без старых изображений)")
    p.add_argument("--start-checkpoint", type=Path,
                   default=ROOT / "models/cloudvision_unet_v3_policy.pt")
    p.add_argument("--model", type=Path, default=ROOT / "models/cloudvision_unet_v4_policy.pt")
    p.add_argument("--out", type=Path, default=ROOT / "reports/training_v4")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.00015)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--overwrite", action="store_true",
                   help="Явно разрешить заменить предыдущий результат v4 (НЕ старую v3)")
    args = p.parse_args()
    try:
        train(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()

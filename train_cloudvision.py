"""Train a small U-Net on manual Sputniks cloud masks, on CPU by default.

First run split_dataset.py. This is an experimental model, not production-ready.
This script ONLY reads originals; it writes model/report files into new folders.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent


class ConvPair(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class TinyUNet(nn.Module):
    """Compact binary segmentation network that can train without a GPU."""
    def __init__(self):
        super().__init__()
        self.enc1 = ConvPair(3, 8)
        self.enc2 = ConvPair(8, 16)
        self.bottleneck = ConvPair(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = ConvPair(48, 16)
        self.dec1 = ConvPair(24, 8)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.bottleneck(self.pool(b))
        up_b = nn.functional.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat([up_b, b], dim=1))
        up_a = nn.functional.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False)
        e = self.dec1(torch.cat([up_a, a], dim=1))
        return self.out(e)


def letterbox_rgb_and_mask(photo: Image.Image, mask: Image.Image, size: int):
    """Maintain aspect ratio, and mark padding as invalid so it is ignored."""
    width, height = photo.size
    ratio = min(size / width, size / height)
    neww, newh = max(1, round(width * ratio)), max(1, round(height * ratio))
    neww, newh = min(neww, size), min(newh, size)
    x, y = (size - neww) // 2, (size - newh) // 2
    a = Image.new("RGB", (size, size), (0, 0, 0))
    b = Image.new("L", (size, size), 0)
    valid = Image.new("L", (size, size), 0)
    a.paste(photo.convert("RGB").resize((neww, newh), Image.Resampling.BILINEAR), (x, y))
    b.paste(mask.convert("L").resize((neww, newh), Image.Resampling.NEAREST), (x, y))
    valid.paste(Image.new("L", (neww, newh), 255), (x, y))
    return a, b, valid


class CloudDataset(Dataset):
    def __init__(self, rows: list[dict], images: Path, labels: Path, size=192, augment=False):
        self.rows = rows
        self.images, self.labels = images, labels
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        with Image.open(self.images / r["image"]) as im, Image.open(self.labels / r["label"]) as mask:
            original, target, valid = letterbox_rgb_and_mask(im, mask, self.size)
        x = np.asarray(original, dtype=np.float32).copy() / 255.0
        y = (np.asarray(target, dtype=np.uint8) >= 128).astype(np.float32)
        v = (np.asarray(valid, dtype=np.uint8) > 0).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                x, y, v = np.flip(x, 1), np.flip(y, 1), np.flip(v, 1)
            if random.random() < 0.5:
                x, y, v = np.flip(x, 0), np.flip(y, 0), np.flip(v, 0)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(y[None])),
                torch.from_numpy(np.ascontiguousarray(v[None])))


def segmentation_loss(logits, target, valid):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    p = logits.sigmoid() * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum(dim=(1, 2, 3)) + 1) / (
        p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


def evaluate(model, loader, device, rows=None):
    model.eval()
    loss_sum = 0.0
    outputs = []
    idx = 0
    with torch.no_grad():
        for x, y, valid in loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            logits = model(x)
            loss_sum += segmentation_loss(logits, y, valid).item() * x.size(0)
            predicted = (logits.sigmoid() >= 0.5).float() * valid
            actual = y * valid
            for k in range(x.size(0)):
                val = valid[k].sum().item()
                pred_percent = 100 * predicted[k].sum().item() / max(1, val)
                true_percent = 100 * actual[k].sum().item() / max(1, val)
                intersect = (predicted[k] * actual[k]).sum().item()
                union = ((predicted[k] + actual[k]) > 0).float().sum().item()
                iou = (intersect / union) if union else 1.0
                info = {"predicted_percent": round(pred_percent, 3),
                        "reference_percent": round(true_percent, 3),
                        "absolute_error_pp": round(abs(pred_percent - true_percent), 3),
                        "iou": round(iou, 5),
                        "reference_action": "REJECT" if true_percent > 75 else "KEEP",
                        "predicted_action": "REJECT" if pred_percent > 75 else "KEEP"}
                if rows is not None:
                    info["image"] = rows[idx]["image"]
                outputs.append(info)
                idx += 1
    mae = sum(v["absolute_error_pp"] for v in outputs) / max(1, len(outputs))
    iou = sum(v["iou"] for v in outputs) / max(1, len(outputs))
    decisions = sum(v["reference_action"] == v["predicted_action"] for v in outputs) / max(1, len(outputs))
    return {"loss": loss_sum / max(1, len(loader.dataset)), "mae_pp": mae,
            "mean_iou": iou, "decision_accuracy": decisions}, outputs


def save_example(model, dataset, row_index, device, dest):
    x, y, valid = dataset[row_index]
    with torch.no_grad():
        pred = (model(x[None].to(device)).sigmoid().cpu()[0, 0].numpy() >= 0.5)
    orig = (x.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    refmask = y[0].numpy() >= 0.5
    def overlay(mask, color):
        out = orig.copy()
        out[mask] = ((out[mask].astype(np.float32) * 0.5) + np.array(color) * 0.5).astype(np.uint8)
        return Image.fromarray(out)
    a = Image.fromarray(orig)
    b = overlay(refmask, [0, 240, 80])
    c = overlay(pred & (valid[0].numpy() > 0), [240, 40, 50])
    sheet = Image.new("RGB", (dataset.size * 3, dataset.size + 30), "white")
    for i, item in enumerate((a, b, c)):
        sheet.paste(item, (i * dataset.size, 30))
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(("ORIGINAL", "MANUAL", "AI PREDICTION")):
        draw.text((i * dataset.size + 5, 8), name, fill="black")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=ROOT / "reports/training/dataset_split.csv")
    ap.add_argument("--images", type=Path, default=ROOT / "data/input")
    ap.add_argument("--labels", type=Path, default=ROOT / "reports/labels")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/training")
    ap.add_argument("--model", type=Path, default=ROOT / "models/cloudvision_unet.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.split.exists():
        ap.error("Нет файла разделения. Сначала запусти split_dataset.py")
    if args.size < 64 or args.size % 4 or args.batch_size < 1 or args.epochs < 1:
        ap.error("Нужно --size >=64, кратный 4; --epochs >=1; --batch-size >=1")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA недоступна, используй --device cpu")
    torch.set_num_threads(min(os.cpu_count() or 2, 4))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    with args.split.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    partitions = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if any(len(v) < 1 for v in partitions.values()):
        ap.error("В разбиении отсутствует обучающая, проверочная или тестовая группа")
    train_ds = CloudDataset(partitions["train"], args.images, args.labels, args.size, augment=True)
    val_ds = CloudDataset(partitions["val"], args.images, args.labels, args.size)
    test_ds = CloudDataset(partitions["test"], args.images, args.labels, args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = TinyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_loss = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    print("\nCloudVision AI — обучение маленькой U-Net (эксперимент)")
    print("Тренировочных:", len(train_ds), "проверочных:", len(val_ds),
          "независимых тестовых:", len(test_ds), "| устройство:", device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, valid in train_loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = segmentation_loss(logits, y, valid)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        val_metrics, _ = evaluate(model, val_loader, device)
        train_loss = total_loss / len(train_ds)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_loss": round(val_metrics["loss"], 5),
                        "val_mae_pp": round(val_metrics["mae_pp"], 3),
                        "val_mean_iou": round(val_metrics["mean_iou"], 4)})
        print(f"Эпоха {epoch:02d}/{args.epochs}: train loss={train_loss:.3f}, "
              f"val loss={val_metrics['loss']:.3f}, val MAE={val_metrics['mae_pp']:.2f} п.п.")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({"state_dict": model.state_dict(), "size": args.size,
                        "seed": args.seed, "best_epoch": epoch}, args.model)
    with (args.out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    # Evaluate on held-out test ONCE after selecting model by validation loss only.
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    test_summary, per_image = evaluate(model, test_loader, device, partitions["test"])
    with (args.out / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    for idx in range(min(len(test_ds), 6)):
        save_example(model, test_ds, idx, device, args.out / "test_previews" /
                     (Path(partitions["test"][idx]["image"]).stem + "_comparison.jpg"))
    print("\nРЕЗУЛЬТАТ НА НЕЗАВИСИМЫХ СНИМКАХ")
    print(f"MAE: {test_summary['mae_pp']:.2f} процентных пункта")
    print(f"Средний IoU: {test_summary['mean_iou']:.3f}")
    print(f"Правильные решения по 75%: {test_summary['decision_accuracy']:.1%}")
    print("Лучшая эпоха:", checkpoint["best_epoch"])
    print("Отчёты:", args.out)
    print("Модель:", args.model)
    print("ВНИМАНИЕ: модель экспериментальная, исходники не менялись, сортировка не запускалась.")

if __name__ == "__main__":
    main()

"""Checkpoint loading and binary cloud-mask inference (no file mutations).

This module intentionally contains no copy/delete operations and no CLI. It is
shared by the batch preview, frozen holdout evaluations, operator diagnostics,
and the competition demo. Preprocessing and threshold reproduce the original
frozen U-Net prediction path exactly; do not change for historical TEST runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_cloudvision import TinyUNet, letterbox_rgb_and_mask


def load_checkpoint(path: Path, expected_selection: str, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Не найдена обученная модель: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"state_dict", "size", "selection"}.issubset(checkpoint):
        raise ValueError(f"Неподдерживаемый файл модели: {path}")
    if checkpoint["selection"] != expected_selection:
        raise ValueError(f"Ожидалась модель '{expected_selection}', а получена '{checkpoint['selection']}'")
    size = int(checkpoint["size"])
    if size < 64 or size % 4 != 0:
        raise ValueError("Некорректный размер входа модели")
    model = TinyUNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, size, checkpoint


def predict_mask(image: Image.Image, model: torch.nn.Module, size: int,
                 device: torch.device) -> Image.Image:
    """Same letterbox + 0..1 RGB preprocessing as training; inverse crop/resize."""
    dummy = Image.new("L", image.size, 0)
    framed, _, valid = letterbox_rgb_and_mask(image, dummy, size)
    x = np.asarray(framed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        predicted = (torch.sigmoid(logits[0, 0]).cpu().numpy() >= 0.5)
    valid_np = (np.asarray(valid) > 0)
    ys, xs = np.where(valid_np)
    if not len(xs):
        raise ValueError("Маска валидной области оказалась пустой")
    cropped = predicted[ys.min():ys.max()+1, xs.min():xs.max()+1]
    binary = Image.fromarray((cropped * 255).astype(np.uint8), "L")
    return binary.resize(image.size, Image.Resampling.NEAREST)

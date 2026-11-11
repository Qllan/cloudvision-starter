"""One-way integrity and label checks for the independent v4 TEST evaluation.

This module does not load PyTorch, run models, or inspect TEST RGB pixels when
freezing checkpoints. TEST originals are opened only to validate complete
human-drawn masks during the final evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cloudvision.selection import read_manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_names(path: Path) -> list[str]:
    """Read only manifest text; never read TEST image contents during freeze."""
    if not path.is_file():
        raise FileNotFoundError(f"Нет списка снимков: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        required = {"relative_path", "scene_group", "split"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError("Нужен исходный selection.csv с split и scene_group")
        selected = [row["relative_path"] for row in rows if row["split"] == "test"]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("TEST должен быть непустым и без повторений")
    return selected


def freeze_checkpoints(manifest: Path, v3: Path, v4: Path, dest: Path) -> dict:
    """Freeze exact checkpoint bytes and selected manifest before seeing TEST."""
    names = test_manifest_names(manifest)
    if not v3.is_file() or not v4.is_file():
        raise FileNotFoundError("Для заморозки требуются оба checkpoint v3 и v4")
    if v3.resolve() == v4.resolve():
        raise ValueError("v3 и v4 должны быть разными файлами")
    if dest.is_file():
        raise FileExistsError(f"Зафиксированные модели уже есть: {dest}")
    if (dest.parent / "summary.csv").exists() or (dest.parent / "per_image.csv").exists():
        raise FileExistsError("В папке уже есть итоговые результаты TEST")
    data = {
        "schema": 1,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "one fixed independent TEST comparison, not model selection",
        "manifest_sha256": file_sha256(manifest),
        "test_count": len(names),
        "v3": {"name": v3.name, "sha256": file_sha256(v3), "bytes": v3.stat().st_size},
        "v4": {"name": v4.name, "sha256": file_sha256(v4), "bytes": v4.stat().st_size},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return data


def check_frozen(manifest: Path, v3: Path, v4: Path, freeze_file: Path) -> dict:
    if not freeze_file.is_file():
        raise FileNotFoundError("Сначала зафиксируй модели с --freeze, до ручной TEST-разметки")
    with freeze_file.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != 1:
        raise ValueError("Неподдерживаемый формат фиксации моделей")
    if data["manifest_sha256"] != file_sha256(manifest):
        raise ValueError("Список selection.csv изменён после фиксации моделей")
    if data["test_count"] != len(test_manifest_names(manifest)):
        raise ValueError("Число TEST-снимков изменилось")
    for name, path in (("v3", v3), ("v4", v4)):
        record = data[name]
        if (record["name"] != path.name or record["sha256"] != file_sha256(path)
                or record["bytes"] != path.stat().st_size):
            raise ValueError(f"Модель {name} отличается от зафиксированной. TEST не запускать!")
    return data


def prepared_test_pairs(manifest: Path, images: Path, labels: Path) -> list[tuple[str, Path, Path]]:
    """All reference masks must be strictly full-size 0/255 before inference."""
    if labels.resolve() == images.resolve() or labels.resolve().is_relative_to(images.resolve()):
        raise ValueError("Ручные TEST-маски должны лежать отдельно от исходных фотографий")
    # read_manifest checks all split/group assignments and reads TEST filenames,
    # but no model is loaded here. File pixels are only read after labeling.
    selected = read_manifest(manifest, images, {"test"})
    root = images.resolve()
    verified_root = labels.resolve()
    result = []
    missing = []
    for image in selected:
        rel = image.relative_to(root)
        mask_file = verified_root / rel.with_suffix(".png")
        if not mask_file.is_file():
            missing.append(rel.as_posix())
        else:
            result.append((rel.as_posix(), image, mask_file))
    if missing:
        raise FileNotFoundError("Не хватает TEST-масок:\n" + "\n".join(missing))
    for rel, image, mask_file in result:
        if mask_file.is_symlink():
            raise ValueError(f"TEST-маска — символическая ссылка: {mask_file}")
        with Image.open(image) as source:
            size = ImageOps.exif_transpose(source).size
        with Image.open(mask_file) as mask:
            if mask.mode != "L" or mask.size != size:
                raise ValueError(f"Неверный режим или размер ручной TEST-маски: {mask_file}")
            if any(count for value, count in enumerate(mask.histogram()) if value not in (0, 255)):
                raise ValueError(f"Не бинарная ручная TEST-маска: {mask_file}")
    return result


def mask_scores(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    if prediction.shape != ground_truth.shape or not prediction.size:
        raise ValueError("Разные размеры или пустые маски")
    if prediction.dtype != np.bool_ or ground_truth.dtype != np.bool_:
        raise ValueError("Ожидаются бинарные numpy boolean masks")
    total = prediction.size
    predicted_percent = 100.0 * np.count_nonzero(prediction) / total
    reference_percent = 100.0 * np.count_nonzero(ground_truth) / total
    intersect = np.count_nonzero(prediction & ground_truth)
    union = np.count_nonzero(prediction | ground_truth)
    iou = 1.0 if union == 0 else float(intersect / union)
    predicted_action = "REJECT" if predicted_percent > 75 else "KEEP"
    reference_action = "REJECT" if reference_percent > 75 else "KEEP"
    return {
        "predicted_percent": round(predicted_percent, 3),
        "reference_percent": round(reference_percent, 3),
        "absolute_error_pp": round(abs(predicted_percent - reference_percent), 3),
        "iou": round(iou, 5),
        "predicted_action": predicted_action,
        "reference_action": reference_action,
        "correct_action": int(predicted_action == reference_action),
    }


def aggregate_scores(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Пустой TEST")
    n = len(rows)
    return {
        "test_count": n,
        "mae_pp": round(sum(r["absolute_error_pp"] for r in rows) / n, 3),
        "mean_iou": round(sum(r["iou"] for r in rows) / n, 5),
        "correct_actions": sum(r["correct_action"] for r in rows),
        "decision_accuracy": round(sum(r["correct_action"] for r in rows) / n, 4),
        "reference_rejects": sum(r["reference_action"] == "REJECT" for r in rows),
        "predicted_rejects": sum(r["predicted_action"] == "REJECT" for r in rows),
        "wrong_rejects": sum(r["reference_action"] == "KEEP" and r["predicted_action"] == "REJECT" for r in rows),
        "missed_rejects": sum(r["reference_action"] == "REJECT" and r["predicted_action"] == "KEEP" for r in rows),
    }

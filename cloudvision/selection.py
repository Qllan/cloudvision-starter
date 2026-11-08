"""Reproducible scene-aware RGB sample selection for manual cloud annotation.

Selects a small and visually varied subset without loading the whole dataset
into RAM or using model predictions. No image is changed or copied.
"""
from __future__ import annotations

import csv
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cloudvision.batch import discover_images

SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ("relative_path", "scene_group", "split")


def scene_group(relative: Path) -> str:
    """Group consecutive captures from the same day/camera together."""
    stem = relative.stem
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_[^_]+_(SXC[^_]+)", stem, re.IGNORECASE)
    if match:
        return (match.group(1) + "_" + match.group(2)).casefold()
    # Unknown file naming: conservatively join captures sharing a folder.
    if relative.parent != Path("."):
        return "folder:" + relative.parent.as_posix().casefold()
    return "file:" + stem.casefold()


def read_manifest(manifest: Path, root: Path, splits: set[str] | None = None) -> list[Path]:
    """Validate file selection, prevent traversal, return existing safe paths."""
    if splits is not None and (not splits or not splits <= set(SPLITS)):
        raise ValueError(f"Недопустимые части выборки: {splits}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Не найден CSV со списком снимков: {manifest}")
    root = root.resolve()
    selected = []
    seen = set()
    groups = {}
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(MANIFEST_FIELDS) <= set(reader.fieldnames):
            raise ValueError(f"CSV должен содержать поля: {MANIFEST_FIELDS}")
        for row in reader:
            rel_text = row["relative_path"]
            group = row["scene_group"]
            part = row["split"]
            rel = Path(rel_text)
            if (not rel_text or rel.is_absolute() or ".." in rel.parts or "\\" in rel_text
                    or rel_text.startswith("/") or ":" in rel_text or part not in SPLITS or not group):
                raise ValueError(f"Некорректная строка в CSV: {row}")
            if rel.as_posix().casefold() in seen:
                raise ValueError(f"Повторное изображение: {rel_text}")
            seen.add(rel.as_posix().casefold())
            if scene_group(rel) != group:
                raise ValueError(f"Несовпадение группы сцены: {rel_text}")
            if group in groups and groups[group] != part:
                raise ValueError(f"Утечка данных: группа {group} указана в разных частях")
            groups[group] = part
            if splits is not None and part not in splits:
                continue
            path = root / rel
            if not path.is_file() or path.is_symlink() or path.resolve().is_relative_to(root) is False:
                raise ValueError(f"Файл отсутствует или небезопасный путь: {rel_text}")
            selected.append(path.resolve())
    if not selected:
        raise ValueError("CSV не содержит снимков для выбранных частей")
    return selected


def rgb_features(path: Path) -> np.ndarray:
    """Tiny 48x48 thumbnail features: colors, brightness, texture, haze."""
    with Image.open(path) as im:
        im.draft("RGB", (128, 128))
        thumb = ImageOps.exif_transpose(im).convert("RGB")
        thumb.thumbnail((48, 48), Image.Resampling.BILINEAR)
        thumb = thumb.resize((48, 48), Image.Resampling.BILINEAR)
        hsv = np.asarray(thumb.convert("HSV"), dtype=np.uint8)
        rgb = np.asarray(thumb, dtype=np.float32) / 255.0
    h, s, v = [hsv[:, :, i].astype(np.float32) / 255.0 for i in range(3)]
    gradients = np.abs(np.diff(v, axis=0)).mean() + np.abs(np.diff(v, axis=1)).mean()
    parts = [
        np.histogram(h, bins=8, range=(0, 1))[0] / h.size,
        np.histogram(s, bins=6, range=(0, 1))[0] / s.size,
        np.histogram(v, bins=6, range=(0, 1))[0] / v.size,
        rgb.mean(axis=(0, 1)),
        rgb.std(axis=(0, 1)),
        np.array([
            float(((v > .72) & (s < .28)).mean()),
            float(((v < .25)).mean()),
            float(((h > .49) & (h < .72) & (s > .35)).mean()),
            float(gradients),
        ]),
    ]
    return np.concatenate(parts).astype(np.float32)


def _candidates(groups: dict[str, list[Path]], count: int, seed: int) -> tuple[list[tuple[str, Path]], int]:
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    if len(keys) < 5:
        raise ValueError("Меньше пяти независимых групп сцен; нельзя создать честную проверку")
    max_per_group = max(1, math.ceil(count / len(keys)))
    target = min(sum(len(v) for v in groups.values()), max(250, count * 8))
    candidates = []
    shuffled = {key: rng.sample(groups[key], len(groups[key])) for key in keys}
    for i in range(max_per_group):
        for key in keys:
            if i < len(shuffled[key]):
                candidates.append((key, shuffled[key][i]))
            if len(candidates) >= target:
                break
        if len(candidates) >= target:
            break
    # If there are many groups, use only one candidate per group; if not,
    # allow additional images per group but never split the group across sets.
    return candidates, max_per_group


def _diverse_indices(features: np.ndarray, candidate_groups: list[str], count: int,
                     cap: int) -> list[int]:
    # Normalize RGB features so each contributes to selection.
    x = (features - features.mean(axis=0)) / np.maximum(.075, features.std(axis=0))
    picked = []
    counts = defaultdict(int)
    distances = np.full(len(x), np.inf, dtype=np.float64)
    first = int(np.linalg.norm(x - x.mean(axis=0), axis=1).argmin())
    for _ in range(count):
        valid = np.array([counts[g] < cap and i not in picked
                          for i, g in enumerate(candidate_groups)])
        if not valid.any():
            break
        if not picked:
            index = first
        else:
            index = int(np.argmax(np.where(valid, distances, -1)))
        picked.append(index)
        counts[candidate_groups[index]] += 1
        distances = np.minimum(distances, ((x - x[index]) ** 2).sum(axis=1))
    return picked


def _assign_splits(items: list[tuple[str, Path]], seed: int) -> dict[str, str]:
    groups = defaultdict(list)
    for group, path in items:
        groups[group].append(path)
    if len(groups) < 5:
        raise ValueError("Недостаточно независимых групп сцен для train/val/test")
    targets = {"train": len(items) * .7, "val": len(items) * .2, "test": len(items) * .1}
    counts = {split: 0 for split in SPLITS}
    names = sorted(groups)
    random.Random(seed + 11).shuffle(names)
    names.sort(key=lambda name: -len(groups[name]))
    result = {}
    for group in names:
        chosen = max(SPLITS, key=lambda split: (targets[split] - counts[split]) / targets[split])
        result[group] = chosen
        counts[chosen] += len(groups[group])
    if any(v == 0 for v in counts.values()):
        raise ValueError("Не удалось выделить train, val и test")
    return result


def select_samples(input_dir: Path, count: int = 50, seed: int = 42,
                   excluded: set[str] | None = None) -> list[dict[str, str]]:
    if count < 10:
        raise ValueError("Для независимых групп выбери не меньше 10 снимков")
    source = input_dir.resolve()
    groups = defaultdict(list)
    exclude = {x.casefold() for x in (excluded or set())}
    for path in discover_images(source):
        rel = path.relative_to(source)
        if rel.as_posix().casefold() in exclude:
            continue
        groups[scene_group(rel)].append(path)
    if sum(map(len, groups.values())) < count:
        raise ValueError("Недостаточно снимков для указанного размера выборки")
    candidates, cap = _candidates(groups, count, seed)
    good, descriptors = [], []
    for group, path in candidates:
        try:
            descriptors.append(rgb_features(path))
            good.append((group, path))
        except (OSError, ValueError):
            continue
    if len(good) < count:
        raise ValueError(f"Слишком мало читаемых снимков: {len(good)}")
    indices = _diverse_indices(np.stack(descriptors), [g for g, _ in good], count, cap)
    if len(indices) != count:
        raise ValueError("Не получилось выбрать нужное число разнообразных снимков")
    chosen = [good[i] for i in indices]
    splits = _assign_splits(chosen, seed)
    result = [{"relative_path": path.relative_to(source).as_posix(),
               "scene_group": group, "split": splits[group]} for group, path in chosen]
    return sorted(result, key=lambda r: (SPLITS.index(r["split"]), r["relative_path"]))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise FileExistsError(f"Список уже существует: {path}. Не перезаписывай зафиксированную выборку")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

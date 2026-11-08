"""CloudVision v5: deterministic, read-only audit of pilot TRAIN candidates.

Create a complete training manifest from historical TRAIN, v4 TRAIN and
manually reviewed PILOT. Keep the existing v4 VAL unchanged. Never open
historical/new TEST image pixels or add VAL/TEST samples to TRAIN.

PILOT labels were AI-assisted, not independent reference test labels.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from cloudvision.selection import scene_group
from cloudvision.v4_data import _read_csv, _relative_path, _validate_mask


@dataclass(frozen=True)
class TrainPair:
    relative_path: str
    image_path: Path
    mask_path: Path
    group: str
    split: str
    source: str

    @property
    def image_name(self) -> str:
        """Match the existing CloudVision train/evaluate dataset interface."""
        return self.relative_path


def _resolve(path: Path, base: Path) -> Path:
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base.resolve()) or not resolved.is_file() or path.is_symlink():
        raise ValueError(f"Отсутствующий или небезопасный путь: {path}")
    return resolved


def _same_bytes(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    if left.stat().st_size != right.stat().st_size:
        return False
    def sha(path: Path) -> bytes:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.digest()
    return sha(left) == sha(right)


def _pair(rel_text: str, image_root: Path, mask_root: Path, mask_rel: Path,
          split: str, source: str, *, allow_scaled: bool = False) -> TrainPair:
    rel = _relative_path(rel_text)
    image = _resolve(rel, image_root)
    mask = _resolve(mask_rel, mask_root)
    _validate_mask(image, mask, allow_scaled=allow_scaled)
    return TrainPair(rel.as_posix(), image, mask, scene_group(rel), split, source)


def prepare_v5_plan(new_manifest: Path, new_images: Path, new_masks: Path,
                    old_split: Path, old_images: Path, old_masks: Path,
                    pilot_report: Path, pilot_masks: Path) -> tuple[list[TrainPair], list[dict]]:
    """Prioritize verified PILOT masks for exact duplicate TRAIN images.

    All groups in OLD/NEW VAL+TEST are protected. A PILOT scene that
    overlaps a protected group is excluded in full (not silently relabeled).
    Same-name duplicate images across old/new roots must have identical bytes.
    """
    new_rows = _read_csv(new_manifest, {"relative_path", "scene_group", "split"})
    old_rows = _read_csv(old_split, {"image", "label", "group", "split"})
    pilot_rows = _read_csv(pilot_report, {"исходный_файл", "статус_проверки"})
    if not pilot_rows:
        raise ValueError("Пилотный операторский отчёт пуст")
    new_groups: dict[str, str] = {}
    protected = set()
    result: dict[str, TrainPair] = {}
    val: list[TrainPair] = []
    reserved = set()

    for row in new_rows:
        rel = _relative_path(row["relative_path"])
        key = rel.as_posix().casefold()
        if key in reserved:
            raise ValueError(f"Дубликат в v4 selection.csv: {rel}")
        reserved.add(key)
        group = scene_group(rel)
        if group != row["scene_group"].casefold():
            raise ValueError(f"Несовпадение группы v4: {rel}")
        part = row["split"]
        if part not in {"train", "val", "test"}:
            raise ValueError(f"Неверный split в selection.csv: {part}")
        previous = new_groups.setdefault(group, part)
        if previous != part:
            raise ValueError(f"Утечка между v4 splits: {group}")
        if part == "test":
            protected.add(group)
            continue  # TEST pixels never accessed
        if part == "val":
            protected.add(group)
        label_rel = rel.with_suffix(".png")
        sample = _pair(rel.as_posix(), new_images, new_masks, label_rel, part, "v4")
        if part == "val":
            val.append(sample)
        else:
            result[key] = sample

    old_groups: dict[str, str] = {}
    old_reserved = set()
    for row in old_rows:
        rel = _relative_path(row["image"])
        key = rel.as_posix().casefold()
        if key in old_reserved:
            raise ValueError(f"Дубликат в старом split: {rel}")
        old_reserved.add(key)
        group = scene_group(rel)
        if group != row["group"].casefold():
            raise ValueError(f"Неверная старая группа сцены: {rel}")
        part = row["split"]
        if part not in {"train", "val", "test"}:
            raise ValueError(f"Неверный старый split: {part}")
        prior = old_groups.setdefault(group, part)
        if prior != part:
            raise ValueError(f"Старая сцена попала в несколько splits: {group}")
        if part != "train":
            protected.add(group)
    if set(new_groups).intersection(old_groups):
        raise ValueError("Старые и v4 группы пересекаются: сначала разберись с разделением")

    for row in old_rows:
        if row["split"] != "train":
            continue
        rel = _relative_path(row["image"])
        key = rel.as_posix().casefold()
        sample = _pair(rel.as_posix(), old_images, old_masks,
                       _relative_path(row["label"]), "train", "old", allow_scaled=True)
        if key in result:
            if not _same_bytes(result[key].image_path, sample.image_path):
                raise ValueError(f"Одинаковое имя, но разные оригиналы: {rel}")
            raise ValueError(f"Дублированные TRAIN между old и v4: {rel}")
        result[key] = sample

    # A PILOT group that shares OLD or v4 VAL/TEST is excluded in full.
    excluded: list[dict] = []
    seen_pilot = set()
    pilot_entries = []
    for row in pilot_rows:
        rel = _relative_path(row["исходный_файл"])
        key = rel.as_posix().casefold()
        if key in seen_pilot:
            raise ValueError(f"Повторный пилотный кадр: {rel}")
        seen_pilot.add(key)
        if row["статус_проверки"] != "ПРОВЕРЕНО ОПЕРАТОРОМ":
            raise ValueError(f"Пилот ещё не проверен вручную: {rel}")
        group = scene_group(rel)
        pilot_entries.append((key, rel, group))

    if not val or not result:
        raise ValueError("Требуются непустые TRAIN и VAL от v4")
    for key, rel, group in pilot_entries:
        if group in protected:
            excluded.append({"image": rel.as_posix(), "reason": "SCENE_IN_OLD_OR_V4_VAL_TEST"})
            continue
        candidate = _pair(rel.as_posix(), new_images, pilot_masks,
                          rel.with_suffix(".png"), "train", "pilot")
        if key in result:
            if not _same_bytes(candidate.image_path, result[key].image_path):
                raise ValueError(f"Одинаковые имена, разные байты: {rel}")
            excluded.append({"image": rel.as_posix(), "reason": "REPLACES_OLD_TRAIN_LABEL"
                             if result[key].source == "old" else "REPLACES_V4_TRAIN_LABEL"})
        result[key] = candidate

    train = list(result.values())
    if set(item.group for item in train).intersection(item.group for item in val):
        raise ValueError("Сцены TRAIN и VAL пересекаются после добавления пилота")
    if any(item.group in protected for item in train):
        raise ValueError("В TRAIN присутствует защищённая сцена")
    return sorted(train + val, key=lambda item: (item.split, item.source, item.relative_path)), excluded



def validate_frozen_v5_plan(plan_csv: Path, expected: list[TrainPair],
                            holdout_csv: Path) -> None:
    """Only use a preflight CSV that exactly matches revalidated original files.

    The new holdout is read only as a TEXT manifest; its photos/masks are not
    inspected or evaluated during training.
    """
    rows = _read_csv(plan_csv, {"split", "source", "relative_path", "scene_group",
                                "image_path", "mask_path"})
    def fields(row: TrainPair) -> tuple[str, ...]:
        return (row.split, row.source, row.relative_path, row.group,
                str(row.image_path), str(row.mask_path))
    expected_set = {fields(item) for item in expected}
    actual_set = {tuple(row[field] for field in (
        "split", "source", "relative_path", "scene_group",
        "image_path", "mask_path")) for row in rows}
    if len(rows) != len(expected_set) or actual_set != expected_set:
        raise ValueError("Предварительный план TRAIN изменился; создай новый отчёт preflight "
                         "в другой папке до обучения")
    holdout_rows = _read_csv(holdout_csv, {"relative_path", "scene_group", "split"})
    if not holdout_rows:
        raise ValueError("Новая отложенная выборка пуста")
    holdout_names = set()
    holdout_groups = set()
    for item in holdout_rows:
        rel = _relative_path(item["relative_path"])
        if item["split"] != "test" or scene_group(rel) != item["scene_group"].casefold():
            raise ValueError(f"Некорректный новый TEST-манифест: {rel}")
        if rel.as_posix().casefold() in holdout_names:
            raise ValueError(f"Повторный TEST-кадр: {rel}")
        holdout_names.add(rel.as_posix().casefold())
        holdout_groups.add(scene_group(rel))
    train_val_groups = {pair.group for pair in expected}
    if holdout_groups.intersection(train_val_groups):
        raise ValueError("Новая отложенная группа пересекается с TRAIN/VAL")
    if holdout_names.intersection(pair.relative_path.casefold() for pair in expected):
        raise ValueError("Новый TEST содержит TRAIN/VAL изображение")

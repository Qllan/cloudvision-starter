"""Safe dataset preparation for v4 fine-tuning; never loads TEST pixels.

New train/val images come from a fixed CSV and hand-corrected binary masks.
Optionally adds only OLD TRAIN samples (not old val/test) and rejects any
scene-group overlap between old and new data.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from cloudvision.selection import read_manifest, scene_group

PARTS = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    image_path: Path
    label_path: Path
    image_name: str
    group: str
    split: str
    source: str


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Некорректные колонки в {path}; требуются {sorted(required)}")
        return list(reader)


def _relative_path(text: str) -> Path:
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or text.startswith("/") or ":" in text or "\\" in text:
        raise ValueError(f"Небезопасный путь в CSV: {text}")
    return path


def _validate_mask(image: Path, label: Path, *, allow_scaled: bool = False) -> None:
    """Check verified labels; legacy labels may be proportionally downsampled.

    Historical manual masks were drawn at a smaller resolution, as allowed by
    audit_dataset.py and the original training pipeline. New v4 masks must
    still match source image dimensions exactly.
    """
    if not image.is_file() or image.is_symlink():
        raise FileNotFoundError(f"Отсутствует изображение: {image}")
    if not label.is_file() or label.is_symlink():
        raise FileNotFoundError(f"Нет ручной маски (AI-маска не подходит): {label}")
    try:
        with Image.open(image) as im:
            with ImageOps.exif_transpose(im) as correct:
                size = correct.size
        with Image.open(label) as mask:
            if mask.mode != "L":
                raise ValueError(f"Маска должна быть в режиме L: {label}; сейчас {mask.mode}")
            mw, mh = mask.size
            iw, ih = size
            if min(iw, ih, mw, mh) <= 0:
                raise ValueError(f"Пустой размер изображения/маски: {label}")
            if allow_scaled:
                if abs(iw / ih - mw / mh) > 0.02:
                    raise ValueError(
                        f"Старая маска имеет неверные пропорции: {label}, "
                        f"изображение {size}, маска {mask.size}")
            elif mask.size != size:
                raise ValueError(
                    f"Новая проверенная маска должна совпадать с исходником: "
                    f"{label}; ожидается {size}, получено {mask.size}")
            histogram = mask.histogram()
            if any(v for i, v in enumerate(histogram) if i not in (0, 255)):
                raise ValueError(f"Маска должна содержать только 0 и 255: {label}")
    except OSError as error:
        raise ValueError(f"Ошибка чтения файла {image.name}: {error}") from error


def prepare_examples(
    manifest: Path, new_images: Path, verified_masks: Path,
    old_split: Path | None = None, old_images: Path | None = None,
    old_masks: Path | None = None,
) -> dict[str, list[Example]]:
    """Build train/val lists and reject collisions; never examine TEST images.

    If old data is requested, only old TRAIN is added. All old split groups
    (including old VAL/TEST) are checked against new group identifiers to keep
    validation/test provenance transparent. This is a conservative guard.
    """
    source = new_images.resolve()
    corrected = verified_masks.resolve()
    # Validate uniqueness and split-group consistency without opening TEST pixels.
    read_manifest(manifest, source, {"train", "val"})
    selected = _read_csv(manifest, {"relative_path", "scene_group", "split"})
    new_groups = {row["scene_group"] for row in selected}
    parts = {"train": [], "val": []}
    for row in selected:
        if row["split"] == "test":
            continue
        rel = _relative_path(row["relative_path"])
        img = source / rel
        label = corrected / rel.with_suffix(".png")
        _validate_mask(img, label)
        parts[row["split"]].append(Example(
            image_path=img, label_path=label, image_name=rel.as_posix(),
            group=row["scene_group"], split=row["split"], source="new"))
    if not parts["train"] or not parts["val"]:
        raise ValueError("Новые TRAIN и VAL должны содержать ручные маски")
    if old_split is not None:
        if old_images is None or old_masks is None:
            raise ValueError("Для старых данных нужны --old-images и --old-masks")
        old_rows = _read_csv(old_split, {"image", "label", "split", "group"})
        old_group_parts = {}
        for row in old_rows:
            part = row["split"]
            if part not in PARTS:
                raise ValueError(f"Неизвестная часть старой выборки: {part}")
            rel = _relative_path(row["image"])
            group = scene_group(rel)
            if group != row["group"].casefold():
                raise ValueError(f"Группа старого снимка не совпадает с именем: {rel}")
            if group in old_group_parts and old_group_parts[group] != part:
                raise ValueError(f"Утечка между старыми TRAIN/VAL/TEST: {group}")
            old_group_parts[group] = part
        overlap = new_groups.intersection(old_group_parts)
        if overlap:
            raise ValueError(
                f"Найдены группы сцен одновременно в старой и новой выборке: "
                f"{', '.join(sorted(overlap)[:5])}. Проверь независимость разбиения.")
        old_image_root = old_images.resolve()
        old_label_root = old_masks.resolve()
        for row in old_rows:
            if row["split"] != "train":
                continue
            image_rel = _relative_path(row["image"])
            label_rel = _relative_path(row["label"])
            img = old_image_root / image_rel
            mask = old_label_root / label_rel
            _validate_mask(img, mask, allow_scaled=True)
            parts["train"].append(Example(
                image_path=img, label_path=mask,
                image_name=image_rel.as_posix(), group=scene_group(image_rel),
                split="train", source="old"))
    return parts

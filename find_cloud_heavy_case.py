"""Find a satellite original and its *human-verified* cloud mask.

Read-only for source RGB and verified PNG. Creates only an optional, small
three-panel JPG preview in a separate reports/ folder. No AI inference,
retraining, checkpoint changes, TEST reevaluation or photo deletion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps


def _safe_print(*values: object) -> None:
    """Avoid UnicodeEncodeError on legacy Windows consoles using cp1252."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    message = " ".join(str(value) for value in values)
    safe = message.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)

ROOT = Path(__file__).resolve().parent
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CASE = "2025-11-30_22-18-23_SXC3-2334_1"


def find_photos(root: Path, query: str) -> list[Path]:
    """Case-insensitive exact filename OR substring, excluding symlinks."""
    if not root.is_dir():
        raise FileNotFoundError(f"Папка исходных фотографий не найдена: {root}")
    if not query.strip():
        raise ValueError("Запрос не может быть пустым")
    needle = query.strip().casefold()
    photos = []
    for path in root.rglob("*"):
        if (path.is_file() and not path.is_symlink()
                and path.suffix.casefold() in PHOTO_EXTS
                and needle in path.name.casefold()):
            photos.append(path)
    return sorted(photos, key=lambda p: p.as_posix().casefold())


def find_manual_mask(photo: Path, image_root: Path, label_root: Path) -> Path | None:
    """Use a mirrored relative path; never guess another photo's mask."""
    relative = photo.relative_to(image_root)
    mask = label_root / relative.with_suffix(".png")
    if (not mask.is_file() or mask.is_symlink()
            or not mask.resolve().is_relative_to(label_root.resolve())):
        return None
    return mask


def create_preview(photo: Path, mask_path: Path, output: Path, *,
                   width: int = 480) -> float:
    """Validate manual binary mask and render RGB | mask | overlay."""
    if output.exists():
        raise FileExistsError(f"Превью уже существует, перезапись запрещена: {output}")
    if width < 120 or width > 1600:
        raise ValueError("Ширина панели должна быть от 120 до 1600")
    with Image.open(photo) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(mask_path) as source_mask:
        if source_mask.mode != "L" or source_mask.size != original.size:
            raise ValueError("Ручная PNG-маска должна быть L и совпадать с исходным разрешением RGB")
        counts = source_mask.histogram()
        if any(count for i, count in enumerate(counts) if i not in (0, 255)):
            raise ValueError("Ручная маска должна содержать только 0/255")
        percent = 100.0 * counts[255] / (original.width * original.height)
        mask = source_mask.copy()
    original.thumbnail((width, 550), Image.Resampling.LANCZOS)
    small_mask = mask.resize(original.size, Image.Resampling.NEAREST)
    mask_rgb = small_mask.convert("RGB")
    tint = Image.new("RGB", original.size, (240, 60, 115))
    overlay = Image.composite(
        Image.blend(original, tint, 0.52), original, small_mask)
    canvas = Image.new("RGB", (original.width * 3, original.height + 36), "white")
    drawer = ImageDraw.Draw(canvas)
    for idx, (title, panel) in enumerate(
            (("ORIGINAL RGB", original), ("HUMAN MASK (WHITE = CLOUD)", mask_rgb),
             ("HUMAN OVERLAY (PINK)", overlay))):
        x = idx * original.width
        drawer.text((x + 7, 9), title, fill="black")
        canvas.paste(panel, (x, 36))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=90)
    return percent


def search_and_preview(image_root: Path, labels: Path, query: str,
                       out: Path, width: int = 480) -> dict:
    source = image_root.resolve()
    label_root = labels.resolve()
    destination = out.resolve()
    for protected in (source, label_root):
        if destination == protected or destination.is_relative_to(protected):
            raise ValueError("Выходная папка не должна быть внутри снимков или ручных масок")
    matches = find_photos(source, query)
    _safe_print(f"Найдено оригиналов: {len(matches)}")
    for path in matches:
        _safe_print("RGB:", path)
        mask = find_manual_mask(path, source, label_root)
        _safe_print("Ручная маска:", mask if mask else "НЕ НАЙДЕНА")
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError("Снимок не найден: проверь имя, либо укажи --query SXC3-2334")
        _safe_print("Совпадений несколько. Уточни --query до полного имени одного снимка.")
        return {"matches": len(matches), "preview": None}
    mask = find_manual_mask(matches[0], source, label_root)
    if mask is None:
        _safe_print("PNG-маска отсутствует; проверь папку --labels. Превью не создавалось.")
        return {"matches": 1, "preview": None}
    output = destination / f"{matches[0].stem}_original_manual_preview.jpg"
    percent = create_preview(matches[0], mask, output, width=width)
    _safe_print(f"Ручная облачность: {percent:.3f}%")
    _safe_print("Правило >75%:", "ОТСЕЯТЬ" if percent > 75 else "СОХРАНИТЬ")
    _safe_print("Изображение для отправки:", output)
    _safe_print("Все исходные RGB, PNG и checkpoints остались без изменений.")
    return {"matches": 1, "preview": output, "percent": percent}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Найти спутниковый снимок и независимую ручную маску; сделать превью")
    parser.add_argument("--query", default=DEFAULT_CASE,
                        help="Полное имя JPEG или его часть; по умолчанию самый спорный challenge-кадр")
    parser.add_argument("--input", type=Path, default=ROOT / "data/new_raw")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/cloud_heavy_verified")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/cloud_heavy_case_review")
    parser.add_argument("--width", type=int, default=480)
    args = parser.parse_args()
    try:
        search_and_preview(args.input, args.labels, args.query, args.out, args.width)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

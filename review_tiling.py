r"""CloudVision AI: readable coloured masks for the stage-5 tiled experiment.

Windows 11 (run from the project root):
    .\.venv\Scripts\python.exe review_tiling.py

Reads only original images and predicted masks; writes new images into
reports/tiling_experiment/color_previews. No originals are modified or deleted.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "input"
MASKS = ROOT / "reports" / "tiling_experiment" / "masks"
OUTPUT = ROOT / "reports" / "tiling_experiment" / "color_previews"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")

# Class index -> display color. These colors are for DISPLAY, not for AI prediction.
COLORS = np.array([
    (32, 38, 51),    # 0: clear ground/water
    (234, 57, 79),    # 1: thick cloud
    (255, 188, 44),   # 2: thin cloud
    (50, 146, 231),   # 3: cloud shadow
], dtype=np.uint8)
LABELS = ["CLEAR (0)", "THICK CLOUD (1)", "THIN CLOUD (2)", "SHADOW (3)"]


def find_original(stem: str):
    for extension in EXTENSIONS:
        candidate = INPUT / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def colorize_classes(classes: np.ndarray) -> np.ndarray:
    if classes.ndim != 2 or classes.size == 0:
        raise ValueError("Mask must be nonempty and 2-D")
    if classes.min() < 0 or classes.max() >= len(COLORS):
        raise ValueError("Expected class indices 0, 1, 2 or 3")
    return COLORS[classes]


def make_preview(mask_path: Path, image_path: Path, target: Path):
    with Image.open(mask_path) as img:
        classes = np.asarray(img.convert("L"), dtype=np.uint8)
    palette = colorize_classes(classes)
    color_mask = Image.fromarray(palette, mode="RGB")
    with Image.open(image_path) as src:
        original = src.convert("RGB").resize(color_mask.size, Image.Resampling.LANCZOS)

    # Overlay on original only at cloud/shadow pixels, leaving clear areas intact.
    alpha = np.where(classes == 0, 0, 135).astype(np.uint8)
    overlay = color_mask.convert("RGBA")
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    blended = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")

    max_width = 1100
    if original.width > max_width:
        max_height = max(1, round(original.height * max_width / original.width))
        dim = (max_width, max_height)
        original = original.resize(dim, Image.Resampling.LANCZOS)
        blended = blended.resize(dim, Image.Resampling.LANCZOS)
        color_mask = color_mask.resize(dim, Image.Resampling.NEAREST)

    w, h = original.size
    header_height = 68
    result = Image.new("RGB", (3 * w, h + header_height), (255, 255, 255))
    result.paste(original, (0, header_height))
    result.paste(color_mask, (w, header_height))
    result.paste(blended, (2 * w, header_height))
    pen = ImageDraw.Draw(result)
    pen.text((12, 8), "ORIGINAL", fill=(0, 0, 0))
    pen.text((w + 12, 8), "COLOR CLASSES", fill=(0, 0, 0))
    pen.text((2 * w + 12, 8), "OVERLAY", fill=(0, 0, 0))
    gap = max(112, w // 4)
    for j, (name, rgb) in enumerate(zip(LABELS, COLORS)):
        x = 12 + j * gap
        pen.rectangle((x, 34, x + 15, 49), fill=tuple(int(k) for k in rgb))
        pen.text((x + 20, 35), name, fill=(0, 0, 0))
    target.parent.mkdir(parents=True, exist_ok=True)
    result.save(target, quality=92)
    return 100 * np.mean((classes == 1) | (classes == 2))


def main():
    if not MASKS.is_dir():
        raise SystemExit("Masks not found: run experiment_tiling.py --limit 3 first")
    found = sorted(MASKS.glob("*_tiled_mask.png"))
    if not found:
        raise SystemExit("No tiled masks found in reports/tiling_experiment/masks")
    processed = 0
    for mask_path in found:
        stem = mask_path.name.removesuffix("_tiled_mask.png")
        original = find_original(stem)
        if original is None:
            print(f"SKIP: No original found for {stem}")
            continue
        destination = OUTPUT / f"{stem}_color_review.jpg"
        percent = make_preview(mask_path, original, destination)
        print(f"{stem}: estimated cloudiness {percent:.2f}% -> {destination.relative_to(ROOT)}")
        processed += 1
    print(f"Done. Created {processed} readable previews. Nothing was deleted.")


if __name__ == "__main__":
    main()

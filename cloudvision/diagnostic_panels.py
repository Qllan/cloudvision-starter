"""Pillow-only visualization helpers for frozen cloud-heavy diagnostics.

Four aligned panels: original RGB and manual/v4/v5 cloud overlays.
Does not perform inference, create masks, alter sources, or write CSV.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

PANEL_NAMES = ("ORIGINAL RGB", "HUMAN CLOUDS", "U-NET V4 CLOUDS", "U-NET V5 CLOUDS")
PINK = (245, 70, 118)


def binary_fraction(mask: Image.Image, size: tuple[int, int]) -> float:
    """Strict full-resolution L/0-255 cloud fraction, never a preview estimate."""
    if mask.mode != "L" or mask.size != size:
        raise ValueError("Cloud mask must be L and match RGB original resolution")
    hist = mask.histogram()
    if any(count for value, count in enumerate(hist) if value not in (0, 255)):
        raise ValueError("Cloud mask must be binary 0/255")
    return 100.0 * hist[255] / (size[0] * size[1])


def render_four_panels(original: Image.Image, human: Image.Image,
                       v4: Image.Image, v5: Image.Image,
                       *, panel_width: int = 420) -> tuple[bytes, dict[str, float]]:
    """Return JPEG bytes and exact % values; no filesystem writes.

    All masks are first checked at the *original* image resolution.
    Masks are then resized with NEAREST for display only.
    """
    if original.mode != "RGB":
        raise ValueError("Original must be RGB")
    if panel_width < 150 or panel_width > 1400:
        raise ValueError("panel_width must be between 150 and 1400")
    values = {
        "human": binary_fraction(human, original.size),
        "v4": binary_fraction(v4, original.size),
        "v5": binary_fraction(v5, original.size),
    }
    image = original.copy()
    image.thumbnail((panel_width, 650), Image.Resampling.LANCZOS)
    w, h = image.size
    header = 64
    sheet = Image.new("RGB", (4 * w, h + header), "white")
    draw = ImageDraw.Draw(sheet)
    sheet.paste(image, (0, header))

    for index, (key, mask) in enumerate((("human", human), ("v4", v4), ("v5", v5)), 1):
        display_mask = mask.resize((w, h), Image.Resampling.NEAREST)
        pink = Image.new("RGB", (w, h), PINK)
        tinted = Image.blend(image, pink, alpha=0.50)
        overlaid = Image.composite(tinted, image, display_mask)
        sheet.paste(overlaid, (index * w, header))
        decision = "REJECT" if values[key] > 75.0 else "KEEP"
        draw.text((index * w + 7, 35),
                  f"{values[key]:.2f}%  |  {decision}  (>75% reject)",
                  fill="#343d4a")

    for index, title in enumerate(PANEL_NAMES):
        draw.text((index * w + 7, 11), title, fill="#101828")

    out = io.BytesIO()
    sheet.save(out, format="JPEG", quality=90)
    return out.getvalue(), values

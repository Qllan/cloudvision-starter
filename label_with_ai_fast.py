"""CloudVision AI: responsive, AI-assisted cloud mask editor for Windows 11.

Uses Tkinter Canvas for instant brush feedback. Original-resolution binary masks
are saved under data/manual_masks_ai; source images/AI masks are never changed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageOps, ImageTk

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = ROOT / "data" / "input"
DEFAULT_AI_MASKS = ROOT / "reports" / "competition_demo" / "masks"
DEFAULT_OUTPUT = ROOT / "data" / "manual_masks_ai"
DEFAULT_LOG = ROOT / "reports" / "label_assisted" / "label_log.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def find_ai_mask(image_path: Path, folder: Path, input_root: Path | None = None) -> Path | None:
    """Search *exact*, known mask names; avoid matching another satellite shot."""
    stem = image_path.stem
    # Mirror input subfolders so repeated filenames cannot collide.
    target_folder = folder / image_path.relative_to(input_root).parent if input_root else folder
    candidates = (
        f"{stem}_policy_mask.png",  # stage 14, run_competition_demo.py
        f"{stem}_mask.png",
        f"{stem}.png",
        f"{image_path.name}.png",
    )
    for name in candidates:
        path = target_folder / name
        if path.is_file():
            return path
    return None


def mask_to_binary(mask_path: Path, expected_size: tuple[int, int]) -> Image.Image:
    """Read raw binary or 4-class label mask. Never accept preview color images.

    Classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow.
    Shadows are NOT clouds. Binary grayscale masks can use 0/1 or 0/255.
    """
    with Image.open(mask_path) as source:
        if source.mode not in ("1", "L", "P"):
            raise ValueError(f"{mask_path.name}: нужна чёрно-белая маска, не цветное превью ({source.mode})")
        mask = source.convert("L")
    if mask.size != expected_size:
        raise ValueError(
            f"{mask_path.name}: размер маски {mask.size}, снимка {expected_size}. "
            "Автоматически растягивать несовпадающую маску небезопасно."
        )
    extrema = mask.getextrema()
    if extrema == (0, 0):
        return mask
    max_value = extrema[1]
    if max_value <= 3:
        # 0 clear; 1 thick; 2 thin; 3 shadow
        return mask.point(lambda p: 255 if p in (1, 2) else 0, mode="L")
    if all(count == 0 for value, count in enumerate(mask.histogram()) if value not in (0, 255)):
        return mask
    raise ValueError(f"{mask_path.name}: маска должна иметь значения 0/255 или классы 0/1/2/3")


def cloud_percent(mask: Image.Image) -> float:
    """Calculate cloud fraction of a binary L image (0/255)."""
    counts = mask.histogram()
    return 100.0 * counts[255] / (mask.width * mask.height)


def apply_stroke(
    mask: Image.Image,
    points: list[tuple[float, float]],
    view_size: tuple[int, int],
    radius: int,
    add: bool,
) -> None:
    """Draw stroke at native resolution using display-space points."""
    if not points:
        return
    view_w, view_h = view_size
    scale_x = mask.width / view_w
    scale_y = mask.height / view_h
    original_points = [
        (round(x * scale_x), round(y * scale_y)) for x, y in points
    ]
    r = max(1, round(radius * max(scale_x, scale_y)))
    ink = 255 if add else 0
    pen = ImageDraw.Draw(mask)
    if len(original_points) > 1:
        pen.line(original_points, fill=ink, width=2 * r, joint="curve")
    for x, y in (original_points[0], original_points[-1]):
        pen.ellipse((x - r, y - r, x + r, y + r), fill=ink)


class FastLabelEditor:
    def __init__(self, root: tk.Tk, images: list[Path], masks_dir: Path,
                 out_dir: Path, log_path: Path, input_root: Path | None = None):
        self.root = root
        self.images = images
        self.input_root = input_root.resolve() if input_root else None
        self.masks_dir = masks_dir
        self.out_dir = out_dir
        self.log_path = log_path
        self.index = 0
        self.radius = 16  # screen pixels; original resolution scales automatically
        self.overlay_visible = True
        self.modified = False
        self.stroke: list[tuple[float, float]] = []
        self.stroke_add = True
        self.temp_items: list[int] = []

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.root.title("CloudVision AI — быстрая проверка масок")
        self.root.configure(bg="#eaf0f6")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        toolbar = tk.Frame(root, bg="#eaf0f6")
        toolbar.pack(fill="x", padx=10, pady=7)
        tk.Button(toolbar, text="← Предыдущий (B)", command=self.previous).pack(side="left", padx=3)
        tk.Button(toolbar, text="Сохранить (S)", command=self.save).pack(side="left", padx=3)
        tk.Button(toolbar, text="Следующий (N) →", command=self.next).pack(side="left", padx=3)
        tk.Button(toolbar, text="Сброс к ИИ (R)", command=self.reset).pack(side="left", padx=3)
        tk.Button(toolbar, text="Маска вкл/выкл (O)", command=self.toggle_overlay).pack(side="left", padx=3)
        tk.Button(toolbar, text="Кисть −", command=lambda: self.change_brush(-5)).pack(side="left", padx=3)
        tk.Button(toolbar, text="Кисть +", command=lambda: self.change_brush(5)).pack(side="left", padx=3)

        self.info = tk.StringVar()
        tk.Label(root, textvariable=self.info, bg="#eaf0f6", anchor="w", font=("Segoe UI", 10)).pack(
            fill="x", padx=14, pady=(0, 5)
        )
        self.canvas = tk.Canvas(root, bg="#1b1f29", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=10, pady=2)
        self.tip = tk.StringVar(value="ЛКМ — добавить облако | ПКМ — стереть | S — сохранить | N/B — кадры | +/- — кисть")
        tk.Label(root, textvariable=self.tip, bg="#eaf0f6", anchor="w").pack(fill="x", padx=14, pady=7)

        self.canvas.bind("<ButtonPress-1>", lambda e: self.begin_stroke(e, True))
        self.canvas.bind("<ButtonPress-3>", lambda e: self.begin_stroke(e, False))
        self.canvas.bind("<B1-Motion>", self.extend_stroke)
        self.canvas.bind("<B3-Motion>", self.extend_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<ButtonRelease-3>", self.end_stroke)
        for key, callback in (
            ("<KeyPress-s>", self.save), ("<KeyPress-n>", self.next),
            ("<KeyPress-b>", self.previous), ("<KeyPress-r>", self.reset),
            ("<KeyPress-o>", self.toggle_overlay),
            ("<KeyPress-plus>", lambda: self.change_brush(5)),
            ("<KeyPress-minus>", lambda: self.change_brush(-5)),
            ("<KeyPress-equal>", lambda: self.change_brush(5)),
            ("<KeyPress-q>", self.close),
        ):
            root.bind(key, lambda event, action=callback: action())

        self.load_current()

    @property
    def current_image(self) -> Path:
        return self.images[self.index]

    @property
    def current_output(self) -> Path:
        relative = self.current_image.relative_to(self.input_root) if self.input_root else self.current_image
        return self.out_dir / relative.parent / f"{relative.stem}.png"

    def load_current(self) -> None:
        with Image.open(self.current_image) as source:
            self.original = ImageOps.exif_transpose(source).convert("RGB")
        w, h = self.original.size
        available_w = max(480, min(1250, self.root.winfo_screenwidth() - 75))
        available_h = max(350, min(900, self.root.winfo_screenheight() - 205))
        self.base_view = self.original.copy()
        self.base_view.thumbnail((available_w, available_h), Image.Resampling.BILINEAR)
        self.view_size = self.base_view.size
        self.canvas.config(width=self.view_size[0], height=self.view_size[1])

        self.ai_path = find_ai_mask(self.current_image, self.masks_dir, self.input_root)
        if self.ai_path:
            self.seed_mask = mask_to_binary(self.ai_path, (w, h))
            info_seed = f"ИИ: {cloud_percent(self.seed_mask):.1f}%"
        else:
            self.seed_mask = Image.new("L", (w, h), 0)
            info_seed = "⚠ ПРЕДСКАЗАНИЕ ИИ ОТСУТСТВУЕТ (пустая маска — не прогноз)"

        if self.current_output.is_file():
            self.mask = mask_to_binary(self.current_output, (w, h))
            info_seed += " | загружена ранее исправленная маска"
        else:
            self.mask = self.seed_mask.copy()
        self.seed_info = info_seed
        self.modified = False
        self.stroke.clear()
        self.refresh_overlay()

    def refresh_overlay(self):
        """Only on load / finished stroke, NOT for every mouse movement."""
        if self.overlay_visible:
            little_mask = self.mask.resize(self.view_size, Image.Resampling.NEAREST)
            alpha = little_mask.point(lambda v: 105 if v >= 128 else 0)
            colored = Image.new("RGBA", self.view_size, (0, 235, 95, 0))
            colored.putalpha(alpha)
            result = Image.alpha_composite(self.base_view.convert("RGBA"), colored).convert("RGB")
        else:
            result = self.base_view
        self.tk_image = ImageTk.PhotoImage(result)
        if getattr(self, "canvas_photo", None) is None:
            self.canvas_photo = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        else:
            self.canvas.itemconfigure(self.canvas_photo, image=self.tk_image)
        self.canvas.tag_lower(self.canvas_photo)
        self.info.set(
            f"{self.index + 1}/{len(self.images)} | {self.current_image.name} | "
            f"Облачность после исправлений: {cloud_percent(self.mask):.2f}% | "
            f"Кисть: {self.radius}px | {self.seed_info}"
            + (" | НЕ СОХРАНЕНО" if self.modified else "")
        )

    def clamp(self, x, y):
        return (
            max(0, min(self.view_size[0] - 1, int(x))),
            max(0, min(self.view_size[1] - 1, int(y))),
        )

    def begin_stroke(self, event, add: bool):
        self.stroke_add = add
        self.stroke = [self.clamp(event.x, event.y)]
        self.draw_temporary(self.stroke[0], self.stroke[0])

    def draw_temporary(self, a, b):
        color = "#37ef83" if self.stroke_add else "#ff4f65"
        item = self.canvas.create_line(
            a[0], a[1], b[0], b[1],
            width=self.radius * 2, fill=color,
            capstyle=tk.ROUND, joinstyle=tk.ROUND,
        )
        self.temp_items.append(item)

    def extend_stroke(self, event):
        if not self.stroke:
            return
        p = self.clamp(event.x, event.y)
        if p == self.stroke[-1]:
            return
        self.draw_temporary(self.stroke[-1], p)
        self.stroke.append(p)

    def end_stroke(self, event):
        if not self.stroke:
            return
        p = self.clamp(event.x, event.y)
        if self.stroke[-1] != p:
            self.stroke.append(p)
        apply_stroke(self.mask, self.stroke, self.view_size, self.radius, self.stroke_add)
        self.stroke = []
        for item in self.temp_items:
            self.canvas.delete(item)
        self.temp_items.clear()
        self.modified = True
        self.refresh_overlay()

    def _confirm_changes(self) -> bool:
        if not self.modified:
            return True
        answer = messagebox.askyesnocancel(
            "Несохранённые изменения",
            "Сохранить исправленную маску перед переходом?",
            parent=self.root,
        )
        if answer is None:
            return False
        if answer:
            self.save()
        return True

    def save(self):
        if self.ai_path is None and not self.modified:
            if not messagebox.askyesno("Нет прогноза ИИ",
                "На этом снимке отсутствует маска ИИ. Сохранить пустую маску как ручную разметку?",
                parent=self.root):
                return
        self.current_output.parent.mkdir(parents=True, exist_ok=True)
        if self.stroke:
            # On-screen painting should be committed before saving.
            apply_stroke(self.mask, self.stroke, self.view_size, self.radius, self.stroke_add)
            self.stroke = []
            self.canvas.delete("stroke")
            for item in self.temp_items:
                self.canvas.delete(item)
            self.temp_items.clear()
        self.mask.save(self.current_output, format="PNG")
        new_log = not self.log_path.is_file() or self.log_path.stat().st_size == 0
        with self.log_path.open("a", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            if new_log:
                writer.writerow(["image_name", "mask_file", "corrected_cloud_percent", "ai_cloud_percent"])
            writer.writerow([
                self.current_image.name, str(self.current_output),
                f"{cloud_percent(self.mask):.3f}", f"{cloud_percent(self.seed_mask):.3f}",
            ])
        self.modified = False
        self.refresh_overlay()
        self.tip.set(f"Сохранено: {self.current_output.name} — {cloud_percent(self.mask):.2f}%")
        print(f"[OK] {self.current_image.name}: {cloud_percent(self.mask):.2f}% -> {self.current_output}")

    def next(self):
        if self.index < len(self.images) - 1 and self._confirm_changes():
            self.index += 1
            self.load_current()

    def previous(self):
        if self.index > 0 and self._confirm_changes():
            self.index -= 1
            self.load_current()

    def reset(self):
        if messagebox.askyesno("Сброс к ИИ", "Заменить текущие исправления на исходную маску ИИ?", parent=self.root):
            self.mask = self.seed_mask.copy()
            self.modified = True
            self.refresh_overlay()

    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        self.refresh_overlay()

    def change_brush(self, delta):
        self.radius = max(2, min(160, self.radius + delta))
        self.info.set(self.info.get().split(" | Кисть:")[0] + f" | Кисть: {self.radius}px")

    def close(self):
        if self._confirm_changes():
            self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudVision AI: быстрый редактор облачных масок")
    parser.add_argument("--input", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--ai-masks", type=Path, default=DEFAULT_AI_MASKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=str, default="", help="Начать со снимка, имя которого содержит текст")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Нет папки с изображениями: {args.input}")
    images = sorted(x for x in args.input.rglob("*") if x.is_file() and not x.is_symlink() and x.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        parser.error(f"Нет изображений в {args.input}")
    if not args.ai_masks.is_dir():
        parser.error(f"Нет папки масок ИИ: {args.ai_masks}. Сначала запусти run_competition_demo.py")
    root = tk.Tk()
    editor = FastLabelEditor(root, images, args.ai_masks, args.out, DEFAULT_LOG, input_root=args.input.resolve())
    if args.start:
        match = next((i for i, p in enumerate(images) if args.start.lower() in p.name.lower()), None)
        if match is None:
            root.destroy()
            parser.error("Снимок для --start не найден")
        editor.index = match
        editor.load_current()
    print(f"Снимков: {len(images)}. Исправления -> {args.out}")
    print("ИИ-подсказки взяты из stage14; модель заново не запускается.")
    root.mainloop()


if __name__ == "__main__":
    main()

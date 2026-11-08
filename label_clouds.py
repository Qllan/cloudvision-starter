"""Мини-редактор ручной разметки облаков (Windows, Tkinter + Pillow).

ЛКМ — закрасить облака, ПКМ — стереть, S — сохранить.
Размечай только облака, а не белые скалы, песок, снег и море.
"""
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk, ImageDraw

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "input"
MODEL_MASKS = ROOT / "reports" / "masks"
LABELS = ROOT / "reports" / "labels"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


class CloudLabeler:
    def __init__(self, root, image_paths):
        self.root = root
        self.paths = image_paths
        self.index = 0
        self.brush = 14
        self.last_xy = None
        self.dirty = False
        self.photo = None
        root.title("CloudVision AI — ручная проверка облаков")
        root.geometry("1040x850")

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.info = tk.Label(top, font=("Segoe UI", 11), justify="left", anchor="w")
        self.info.pack(fill="x")
        self.canvas = tk.Canvas(root, bg="#202020", cursor="crosshair")
        self.canvas.pack(expand=True, fill="both", padx=8, pady=4)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.start(e, 255))
        self.canvas.bind("<B1-Motion>", lambda e: self.paint(e, 255))
        self.canvas.bind("<ButtonPress-3>", lambda e: self.start(e, 0))
        self.canvas.bind("<B3-Motion>", lambda e: self.paint(e, 0))
        self.canvas.bind("<ButtonRelease-1>", self.end)
        self.canvas.bind("<ButtonRelease-3>", self.end)

        bottom = tk.Frame(root)
        bottom.pack(fill="x", padx=8, pady=6)
        for title, callback in [("← Предыдущий", lambda: self.go(-1)),
                                ("Сохранить (S)", self.save),
                                ("Следующий →", lambda: self.go(1)),
                                ("Кисть −", lambda: self.change_brush(-5)),
                                ("Кисть +", lambda: self.change_brush(5)),
                                ("Стереть всю разметку", self.clear)]:
            tk.Button(bottom, text=title, command=callback, padx=7).pack(side="left", padx=3)
        self.status = tk.Label(root, anchor="w", justify="left")
        self.status.pack(fill="x", padx=10, pady=(0, 10))
        root.bind("s", lambda e: self.save())
        root.bind("<Left>", lambda e: self.go(-1))
        root.bind("<Right>", lambda e: self.go(1))
        root.bind("<bracketleft>", lambda e: self.change_brush(-5))
        root.bind("<bracketright>", lambda e: self.change_brush(5))
        self.load()

    def load(self):
        path = self.paths[self.index]
        with Image.open(path) as img:
            self.base = img.convert("RGB")
        self.base.thumbnail((980, 690), Image.Resampling.LANCZOS)
        self.annotation_path = LABELS / f"{path.stem}_label.png"
        if self.annotation_path.exists():
            with Image.open(self.annotation_path) as previous:
                self.mask = previous.convert("L").resize(self.base.size, Image.Resampling.NEAREST)
        else:
            self.mask = Image.new("L", self.base.size, 0)
        self.dirty = False
        self.last_xy = None
        self.render()

    def render(self):
        self.render_overlay()
        self.update_status()

    def render_overlay(self):
        # Зелёным показываем только ручную разметку, чтобы ИИ не влиял на оценку.
        base = self.base.convert("RGBA")
        overlay = Image.new("RGBA", self.base.size, (0, 235, 90, 105))
        transparent = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        composed = Image.alpha_composite(base, Image.composite(overlay, transparent, self.mask))
        self.photo = ImageTk.PhotoImage(composed.convert("RGB"))
        self.canvas.config(width=self.base.width, height=self.base.height,
                           scrollregion=(0, 0, self.base.width, self.base.height))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        name = self.paths[self.index].name
        self.info.config(text=f"Снимок {self.index+1}/{len(self.paths)}: {name}\n"
                         "ЛЕВАЯ кнопка — выделить облака зелёным; ПРАВАЯ — стереть. "
                         "S — сохранить; стрелки — листать.")

    def update_status(self):
        # Вычисляем долю отмеченных облаков только по запросу,
        # а не на каждом движении мыши.
        from PIL import ImageStat
        coverage = ImageStat.Stat(self.mask).mean[0] * 100.0 / 255.0
        status = "Не сохранено" if self.dirty else "Сохранено/без изменений"
        self.status.config(text=f"Кисть: {self.brush}px | "
                           f"Ручная облачность: {coverage:.1f}% | {status}")

    def draw_at(self, xy, color):
        x, y = xy
        if not (0 <= x < self.base.width and 0 <= y < self.base.height):
            return
        draw = ImageDraw.Draw(self.mask)
        if self.last_xy is not None:
            draw.line([self.last_xy, xy], fill=color, width=2*self.brush)
        draw.ellipse((x-self.brush, y-self.brush, x+self.brush, y+self.brush), fill=color)
        self.last_xy = xy
        self.dirty = True
        self.render_overlay()

    def start(self, event, color):
        self.last_xy = None
        self.paint(event, color)

    def paint(self, event, color):
        xy = (int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y)))
        self.draw_at(xy, color)

    def end(self, _event):
        self.last_xy = None
        self.update_status()

    def change_brush(self, step):
        self.brush = max(3, min(100, self.brush + step))
        self.render()

    def clear(self):
        if messagebox.askyesno("Очистить?", "Очистить ручную разметку этого снимка?"):
            self.mask = Image.new("L", self.base.size, 0)
            self.dirty = True
            self.render()

    def save(self):
        LABELS.mkdir(parents=True, exist_ok=True)
        self.mask.save(self.annotation_path)
        self.dirty = False
        self.render()
        self.status.config(text=f"Разметка сохранена: {self.annotation_path.relative_to(ROOT)}")

    def go(self, step):
        index = self.index + step
        if not (0 <= index < len(self.paths)):
            return
        if self.dirty:
            answer = messagebox.askyesnocancel("Сохранить изменения?", "Сохранить разметку перед переходом?")
            if answer is None:
                return
            if answer:
                self.save()
        self.index = index
        self.load()


def main():
    if not MODEL_MASKS.exists():
        print("Сначала запусти обновлённый run_ai.py — он создаст reports/masks.")
        return
    paths = []
    for mask_path in sorted(MODEL_MASKS.glob("*_mask.png")):
        stem = mask_path.stem.removesuffix("_mask")
        matches = [INPUT / (stem + ext) for ext in EXTENSIONS]
        original = next((path for path in matches if path.exists()), None)
        if original is not None:
            paths.append(original)
    if not paths:
        print("Не найдено исходных снимков, соответствующих маскам модели.")
        return
    root = tk.Tk()
    CloudLabeler(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()

from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


INPUT_DIR = Path("data/input")
PRED_MASK_DIR = Path("reports/competition_demo/masks")
OUTPUT_MASK_DIR = Path("data/manual_masks_ai")
LOG_DIR = Path("reports/label_assisted")
LOG_FILE = LOG_DIR / "label_log.csv"

OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def find_predicted_mask(image_path: Path):
    stem = image_path.stem
    name = image_path.name

    candidates = [
        PRED_MASK_DIR / f"{stem}.png",
        PRED_MASK_DIR / f"{stem}_mask.png",
        PRED_MASK_DIR / f"{name}.png",
        PRED_MASK_DIR / f"{name}_mask.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    # запасной вариант: поиск по stem
    for path in PRED_MASK_DIR.glob("*.png"):
        if stem in path.stem:
            return path

    return None


def load_image_rgb(path: Path):
    image = Image.open(path).convert("RGB")
    return np.array(image)


def load_mask(path: Path, size_hw):
    h, w = size_hw
    if path is None or not path.exists():
        return np.zeros((h, w), dtype=np.uint8)

    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        # если маска закодирована как 0/1/2/3 или 0/255
        if arr.max() <= 3:
            mask = (arr > 0).astype(np.uint8)
        else:
            mask = (arr > 127).astype(np.uint8)
    else:
        # если почему-то RGB, считаем облаком всё ненулевое
        mask = (arr.mean(axis=2) > 30).astype(np.uint8)

    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
    pil_mask = pil_mask.resize((w, h), resample=Image.NEAREST)
    return (np.array(pil_mask) > 127).astype(np.uint8)


def save_mask(path: Path, mask: np.ndarray):
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    img.save(path)


def cloud_percent(mask: np.ndarray):
    return float(mask.mean() * 100.0)


class AssistedLabelEditor:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.index = 0
        self.brush_size = 20
        self.mouse_down = False
        self.draw_value = 1
        self.show_overlay = True

        self.image = None
        self.seed_mask = None
        self.mask = None
        self.current_image_path = None

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.ax_left, self.ax_right = self.axes

        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.load_current()

    def current_output_path(self):
        return OUTPUT_MASK_DIR / f"{self.current_image_path.stem}.png"

    def load_current(self):
        self.current_image_path = self.image_paths[self.index]
        self.image = load_image_rgb(self.current_image_path)
        h, w = self.image.shape[:2]

        pred_path = find_predicted_mask(self.current_image_path)
        self.seed_mask = load_mask(pred_path, (h, w))

        # если уже есть исправленная маска, открываем её
        out_path = self.current_output_path()
        if out_path.exists():
            self.mask = load_mask(out_path, (h, w))
        else:
            self.mask = self.seed_mask.copy()

        self.redraw()

    def redraw(self):
        self.ax_left.clear()
        self.ax_right.clear()

        self.ax_left.imshow(self.image)
        self.ax_left.set_title(
            f"ORIGINAL\n{self.current_image_path.name}\n"
            f"Кисть: {self.brush_size}px"
        )
        self.ax_left.axis("off")

        self.ax_right.imshow(self.image)
        if self.show_overlay:
            overlay = np.zeros((*self.mask.shape, 4), dtype=np.float32)
            overlay[..., 1] = 1.0   # green
            overlay[..., 3] = self.mask * 0.35
            self.ax_right.imshow(overlay)

        self.ax_right.set_title(
            "AI-SEEDED MASK (EDITABLE)\n"
            f"Текущая облачность: {cloud_percent(self.mask):.2f}%"
        )
        self.ax_right.axis("off")

        self.fig.suptitle(
            "ЛКМ = добавить облако | ПКМ = стереть | S = сохранить | "
            "N = следующий | B = предыдущий | R = сброс к AI | "
            "O = overlay | +/- = кисть | Q = выход",
            fontsize=10
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def paint(self, x, y, value):
        h, w = self.mask.shape
        x = int(round(x))
        y = int(round(y))

        yy, xx = np.ogrid[:h, :w]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= self.brush_size ** 2
        self.mask[circle] = value

    def on_mouse_press(self, event):
        if event.inaxes != self.ax_right or event.xdata is None or event.ydata is None:
            return

        self.mouse_down = True

        if event.button == 1:      # left click
            self.draw_value = 1
        elif event.button == 3:    # right click
            self.draw_value = 0
        else:
            return

        self.paint(event.xdata, event.ydata, self.draw_value)
        self.redraw()

    def on_mouse_release(self, event):
        self.mouse_down = False

    def on_mouse_move(self, event):
        if not self.mouse_down:
            return
        if event.inaxes != self.ax_right or event.xdata is None or event.ydata is None:
            return

        self.paint(event.xdata, event.ydata, self.draw_value)
        self.redraw()

    def save_current(self):
        out_path = self.current_output_path()
        save_mask(out_path, self.mask)

        file_exists = LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "image_name",
                    "saved_mask",
                    "cloud_percent",
                    "seed_cloud_percent",
                ])
            writer.writerow([
                self.current_image_path.name,
                str(out_path),
                round(cloud_percent(self.mask), 3),
                round(cloud_percent(self.seed_mask), 3),
            ])

        print(f"[OK] Сохранено: {out_path}")
        print(f"     Облачность (исправленная): {cloud_percent(self.mask):.2f}%")

    def next_image(self):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.load_current()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def on_key_press(self, event):
        key = (event.key or "").lower()

        if key == "s":
            self.save_current()
        elif key == "n":
            self.next_image()
        elif key == "b":
            self.prev_image()
        elif key == "r":
            self.mask = self.seed_mask.copy()
            self.redraw()
            print("[INFO] Маска сброшена к исходному предсказанию AI.")
        elif key == "o":
            self.show_overlay = not self.show_overlay
            self.redraw()
        elif key in ["+", "="]:
            self.brush_size = min(self.brush_size + 5, 200)
            self.redraw()
        elif key == "-":
            self.brush_size = max(self.brush_size - 5, 1)
            self.redraw()
        elif key == "q":
            plt.close(self.fig)


def main():
    if not INPUT_DIR.exists():
        print(f"Папка не найдена: {INPUT_DIR}")
        return

    if not PRED_MASK_DIR.exists():
        print(f"Папка с AI-масками не найдена: {PRED_MASK_DIR}")
        print("Сначала запусти run_competition_demo.py")
        return

    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        print("В папке data/input нет изображений.")
        return

    print("CloudVision AI — полуавтоматическая разметка")
    print(f"Снимков найдено: {len(image_paths)}")
    print(f"AI-маски читаются из: {PRED_MASK_DIR}")
    print(f"Исправленные маски будут сохранены в: {OUTPUT_MASK_DIR}")
    print("Горячие клавиши:")
    print("  S = сохранить")
    print("  N = следующий снимок")
    print("  B = предыдущий снимок")
    print("  R = вернуть исходную AI-маску")
    print("  O = показать/скрыть overlay")
    print("  + / - = изменить размер кисти")
    print("  Q = выход")

    editor = AssistedLabelEditor(image_paths)
    plt.show()


if __name__ == "__main__":
    main()
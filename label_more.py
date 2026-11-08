"""Открыть ручную разметку для дополнительных снимков СПУТНИКС.

По умолчанию показывает только снимки, для которых ещё нет разметки.
--all: открывает все, включая ранее размеченные.
"""
import argparse
import tkinter as tk

from label_clouds import CloudLabeler, INPUT, LABELS, EXTENSIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    images = sorted(p for p in INPUT.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not args.all:
        images = [p for p in images if not (LABELS / f"{p.stem}_label.png").exists()]
    if not images:
        print("Новых изображений без разметки нет. Для проверки старых запусти: python label_more.py --all")
        return
    print(f"Открываю {len(images)} снимков для ручной разметки")
    print("Облака закрашивай ПО ВСЕЙ ПЛОЩАДИ, а не только линиями.")
    print("Если облаков нет, проверь это и сохрани пустую маску клавишей S.")
    print("Старайся добавить разные сюжеты, включая очень облачные кадры.")
    root = tk.Tk()
    CloudLabeler(root, images)
    root.mainloop()


if __name__ == "__main__":
    main()

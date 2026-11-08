"""Copy selected photos into data/work_input for a safe deletion demonstration.

Refuses to overwrite existing files. Never removes originals.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=5)
    a = p.parse_args()
    if a.limit < 1:
        p.error('Нужно указать --limit не меньше 1')
    source = ROOT / 'data' / 'input'
    work = ROOT / 'data' / 'work_input'
    if not source.is_dir():
        p.error(f'Нет папки {source}')
    files = sorted(x for x in source.iterdir() if x.is_file() and x.suffix.lower() in EXT and not x.is_symlink())[:a.limit]
    if not files:
        p.error('В папке снимков нет файлов')
    if any((work / f.name).exists() for f in files):
        p.error('Некоторые рабочие копии уже существуют. Ничего не перезаписываем')
    work.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, work / f.name)
        print('Создана рабочая копия:', f.name)
    print('Оригиналы в data/input не изменены')


if __name__ == '__main__':
    main()

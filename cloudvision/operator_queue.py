"""Read the already-generated operator review CSV for assisted PILOT labeling.

Does not re-run a model or access the independent TEST split. The queue preserves
report priority order so near-threshold frames appear first, but all frames
remain available for operator inspection.
"""
from __future__ import annotations

import csv
from pathlib import Path

from cloudvision.batch import EXTENSIONS


def review_queue(csv_path: Path, input_root: Path) -> list[Path]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Нет операторского CSV: {csv_path}")
    root = input_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Нет папки снимков: {root}")
    selected = []
    seen = set()
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "исходный_файл" not in reader.fieldnames:
            raise ValueError("В операторском CSV нет колонки исходный_файл")
        for row in reader:
            value = row["исходный_файл"]
            if not value or "\\" in value or ":" in value or value.startswith("/"):
                raise ValueError(f"Недопустимый путь снимка: {value}")
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in EXTENSIONS:
                raise ValueError(f"Недопустимый путь снимка: {value}")
            key = relative.as_posix().casefold()
            if key in seen:
                raise ValueError(f"Повторный снимок в очереди: {value}")
            seen.add(key)
            path = root / relative
            if (not path.is_file() or path.is_symlink()
                    or not path.resolve().is_relative_to(root)):
                raise ValueError(f"Снимок отсутствует или небезопасен: {value}")
            selected.append(path.resolve())
    if not selected:
        raise ValueError("Очередь ручной проверки пуста")
    return selected

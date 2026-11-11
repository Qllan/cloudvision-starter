"""Select 50 diverse SPUTNIX images without using the neural network.

A separate holdout (test) set is created and must not be used for tuning.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from cloudvision.selection import select_samples, write_manifest

ROOT = Path(__file__).resolve().parent


def load_seen(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "исходный_файл" not in reader.fieldnames:
            raise ValueError(f"Нет колонки исходный_файл в {path}")
        return {row["исходный_файл"].replace("\\", "/") for row in reader}


def main() -> None:
    p = argparse.ArgumentParser(description="Выбрать разнообразные снимки для разметки v4")
    p.add_argument("--input", type=Path, default=ROOT / "data/new_raw")
    p.add_argument("--out", type=Path, default=ROOT / "reports/selection_v4/selection.csv")
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-report", type=Path,
                   default=ROOT / "reports/batch_v4_new/operator_preview.csv",
                   help="Исключить ранее просмотренные снимки из новой выборки")
    a = p.parse_args()
    try:
        excluded = load_seen(a.exclude_report)
        rows = select_samples(a.input, a.count, a.seed, excluded=excluded)
        write_manifest(a.out, rows)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        p.error(str(error))
    counts = Counter(row["split"] for row in rows)
    print(f"Выбрано: {len(rows)} снимков. TRAIN={counts['train']}, "
          f"VAL={counts['val']}, TEST={counts['test']}")
    print(f"Исключено ранее просмотренных снимков: {len(excluded)}")
    print("Список:", a.out)
    print("TEST не использовать для обучения и настройки модели.")
    print("Исходные изображения не копировались и не менялись.")


if __name__ == "__main__":
    main()

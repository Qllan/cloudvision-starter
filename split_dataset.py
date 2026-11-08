"""CloudVision AI: reproducible group-wise train/validation/test split.

Input: reports/dataset_audit/labels_audit.csv, data/input, reports/labels.
Only rows with status OK are included by default. REVIEW is excluded until verified.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def scene_group(filename: str) -> str:
    # Example: 2025-09-18_12-14-01_SXC3-234_1.jpg
    # Keep all images from the same day and camera/scene together.
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"Не удалось прочитать дату/камеру из имени: {filename}")
    return f"{parts[0]}_{parts[2]}"


def split_rows(rows: list[dict], seed: int = 42) -> list[dict]:
    """Split at group level, favoring similar positive proportions and sizes."""
    groups = defaultdict(list)
    for row in rows:
        groups[scene_group(row["image"])].append(row)
    keys = sorted(groups)
    if len(keys) < 5:
        raise ValueError("Нужно не менее 5 групп сцен для раздельных выборок")
    n = len(rows)
    total_high = sum(float(r["manual_cloud_percent"]) > 75 for r in rows)
    desired = {"train": 0.70, "val": 0.15, "test": 0.15}
    best = None
    rng = random.Random(seed)
    for _ in range(3000):
        shuffled = keys.copy()
        rng.shuffle(shuffled)
        bins = {"train": [], "val": [], "test": []}
        counts = {x: 0 for x in bins}
        # Entire groups are assigned together; random order yields candidate splits.
        for key in shuffled:
            if counts["train"] < round(0.70 * n):
                dest = "train"
            elif counts["val"] < round(0.15 * n):
                dest = "val"
            else:
                dest = "test"
            bins[dest].append(key)
            counts[dest] += len(groups[key])
        if not all(bins.values()):
            continue
        highs = {k: sum(float(r["manual_cloud_percent"]) > 75 for group in groupkeys for r in groups[group])
                 for k, groupkeys in bins.items()}
        # Avoid a split with zero heavily clouded images, whenever possible.
        score = sum(abs(counts[k] - n * desired[k]) for k in bins)
        score += sum(abs(highs[k] - total_high * desired[k]) * 2 for k in bins)
        if total_high >= 3:
            score += sum(100 for k in bins if highs[k] == 0)
        if best is None or score < best[0]:
            best = (score, bins)
    assert best is not None
    lookup = {group: part for part, groups_in_part in best[1].items() for group in groups_in_part}
    result = []
    for r in rows:
        entry = dict(r)
        entry["group"] = scene_group(r["image"])
        entry["split"] = lookup[entry["group"]]
        result.append(entry)
    return sorted(result, key=lambda r: (r["split"], r["image"]))


def generate(audit: Path, images: Path, labels: Path, output: Path, seed: int = 42):
    if not audit.exists():
        raise FileNotFoundError(f"Сначала запусти audit_dataset.py: {audit}")
    with audit.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    kept = []
    excluded = Counter()
    for row in records:
        if row["status"] != "OK":
            excluded[row["status"]] += 1
            continue
        if not (images / row["image"]).is_file() or not (labels / row["label"]).is_file():
            excluded["FILE_MISSING"] += 1
            continue
        kept.append(row)
    assigned = split_rows(kept, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "label", "manual_cloud_percent", "group", "split"]
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in assigned])
    print("\nCloudVision AI — разделение данных")
    print("Использовано размеченных снимков:", len(assigned))
    print("Исключено:", dict(excluded))
    for s in ("train", "val", "test"):
        subset = [r for r in assigned if r["split"] == s]
        print(f"{s}: {len(subset)} изображений, из них облачность >75%: ",
              sum(float(r["manual_cloud_percent"]) > 75 for r in subset),
              f"; групп: {len(set(r['group'] for r in subset))}")
    print("Файл разбиения:", output)
    print("Проверочная и тестовая группы не используются для обучения.")
    return assigned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, default=ROOT / "reports/dataset_audit/labels_audit.csv")
    p.add_argument("--images", type=Path, default=ROOT / "data/input")
    p.add_argument("--labels", type=Path, default=ROOT / "reports/labels")
    p.add_argument("--output", type=Path, default=ROOT / "reports/training/dataset_split.csv")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    generate(a.audit, a.images, a.labels, a.output, a.seed)

if __name__ == "__main__":
    main()

"""Select a NEW v5 blind holdout from unseen scene groups, without AI inference.

Uniformly sample one RGB filename per camera/date scene group. This selection
DOES NOT inspect image pixels and is independent of cloud predictions.
It is NOT guaranteed to contain >75%-cloud photos; do not manipulate the
sample after seeing predictions to inflate apparent accuracy.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from cloudvision.batch import discover_images
from cloudvision.selection import scene_group, write_manifest
from cloudvision.v4_data import _read_csv, _relative_path

ROOT = Path(__file__).resolve().parent


def excluded_groups(new_manifest: Path, old_split: Path,
                    pilot_report: Path, previous_preview: Path | None = None) -> set[str]:
    groups = set()
    for row in _read_csv(new_manifest, {"relative_path", "split"}):
        groups.add(scene_group(_relative_path(row["relative_path"])))
    for row in _read_csv(old_split, {"image", "split"}):
        groups.add(scene_group(_relative_path(row["image"])))
    for row in _read_csv(pilot_report, {"исходный_файл"}):
        groups.add(scene_group(_relative_path(row["исходный_файл"])))
    if previous_preview is not None and previous_preview.is_file():
        for row in _read_csv(previous_preview, {"исходный_файл"}):
            groups.add(scene_group(_relative_path(row["исходный_файл"])))
    return groups


def choose_holdout(images: Path, excluded: set[str], count: int, seed: int = 2026) -> list[dict]:
    """Group-uniform sample of filenames, one per group, no pixel data loaded."""
    if count < 5:
        raise ValueError("Для проверки выбери не меньше пяти новых сцен")
    source = images.resolve()
    scenes: dict[str, list[Path]] = defaultdict(list)
    for image in discover_images(source):
        relative = image.relative_to(source)
        group = scene_group(relative)
        if group not in excluded:
            scenes[group].append(relative)
    if len(scenes) < count:
        raise ValueError(
            f"Доступно {len(scenes)} новых групп, нужно {count}. "
            "Не используй уже просмотренные TRAIN/VAL/TEST.")
    rng = random.Random(seed)
    names = rng.sample(sorted(scenes), count)
    chosen = []
    for name in names:
        relative = rng.choice(sorted(scenes[name]))
        chosen.append({"relative_path": relative.as_posix(),
                       "scene_group": name, "split": "test"})
    return sorted(chosen, key=lambda r: r["relative_path"])


def main() -> None:
    p = argparse.ArgumentParser(description="Новая отложенная выборка v5 без просмотра пикселей")
    p.add_argument("--images", type=Path, default=ROOT / "data/new_raw")
    p.add_argument("--v4-selection", type=Path, default=ROOT / "reports/selection_v4/selection.csv")
    p.add_argument("--old-split", type=Path, default=ROOT / "reports/training/dataset_split.csv")
    p.add_argument("--pilot-report", type=Path,
                   default=ROOT / "reports/competition_pilot_v4/operator_review/operator_review.csv")
    p.add_argument("--previous-preview", type=Path,
                   default=ROOT / "reports/batch_v4_new/operator_preview.csv")
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, default=ROOT / "reports/v5/holdout_manifest.csv")
    a = p.parse_args()
    try:
        if a.out.exists():
            raise FileExistsError(f"Выборка уже зафиксирована: {a.out}")
        excluded = excluded_groups(
            a.v4_selection, a.old_split, a.pilot_report, a.previous_preview)
        chosen = choose_holdout(a.images, excluded, a.count, a.seed)
        write_manifest(a.out, chosen)
    except (OSError, ValueError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))
    print(f"Новая отложенная выборка: {len(chosen)} кадров из разных дат/камер")
    print(f"Исключены группы старой TRAIN/VAL/TEST, v4 TRAIN/VAL/TEST, PILOT"
          f" и раннего предпросмотра: {len(excluded)}")
    print("Манифест:", a.out)
    print("Нейросеть и пиксели выбранных снимков не открывались.")
    print("Это случайная выборка ГРУПП сцен, она может не содержать сильно облачных фото.")
    print("Не используй снимки из неё для обучения, подбора эпох и порога.")


if __name__ == "__main__":
    main()

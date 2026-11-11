"""Read-only preparation of CloudVision v5 TRAIN and unchanged v4 VAL.

Run BEFORE training. This script never edits originals, model checkpoints,
existing masks, previous VAL/TEST manifests, or PILOT operator reports.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from cloudvision.v5_data import prepare_v5_plan

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Безопасный аудит TRAIN для U-Net v5")
    ap.add_argument("--images", type=Path, default=ROOT / "data/new_raw")
    ap.add_argument("--new-manifest", type=Path, default=ROOT / "reports/selection_v4/selection.csv")
    ap.add_argument("--new-masks", type=Path, default=ROOT / "data/verified_v4")
    ap.add_argument("--old-split", type=Path, default=ROOT / "reports/training/dataset_split.csv")
    ap.add_argument("--old-images", type=Path, default=ROOT / "data/input")
    ap.add_argument("--old-masks", type=Path, default=ROOT / "reports/labels")
    ap.add_argument("--pilot-report", type=Path,
                    default=ROOT / "reports/competition_pilot_v4/operator_review/operator_review.csv")
    ap.add_argument("--pilot-masks", type=Path, default=ROOT / "data/pilot_verified_v4")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/v5/preflight")
    args = ap.parse_args()
    inputs = [args.images.resolve(), args.new_masks.resolve(), args.old_images.resolve(),
              args.old_masks.resolve(), args.pilot_masks.resolve()]
    dest = args.out.resolve()
    if any(dest == folder or dest.is_relative_to(folder) or folder.is_relative_to(dest)
           for folder in inputs):
        ap.error("Папка --out не должна пересекаться с папками изображений или масок")
    try:
        rows, exclusions = prepare_v5_plan(
            args.new_manifest, args.images, args.new_masks, args.old_split,
            args.old_images, args.old_masks, args.pilot_report, args.pilot_masks)
        if (dest / "train_manifest.csv").exists() or (dest / "summary.json").exists():
            raise FileExistsError(
                f"План в {dest} уже создан. Для нового эксперимента используй другую --out; "
                "готовый план нельзя незаметно заменить.")
        counter = Counter((r.split, r.source) for r in rows)
        replaced = sum(e["reason"].startswith("REPLACES_") for e in exclusions)
        excluded = sum(e["reason"] == "SCENE_IN_OLD_OR_V4_VAL_TEST" for e in exclusions)
        if counter[("train", "pilot")] == 0:
            raise ValueError("Ни один пилотный снимок не пригоден для нового TRAIN")
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "train_manifest.csv").open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["split", "source", "relative_path", "scene_group",
                             "image_path", "mask_path"])
            for row in rows:
                writer.writerow([row.split, row.source, row.relative_path, row.group,
                                 str(row.image_path), str(row.mask_path)])
        with (dest / "excluded_pilot.csv").open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image", "reason"])
            writer.writeheader()
            writer.writerows(exclusions)
        summary = {
            "train": sum(r.split == "train" for r in rows),
            "val": sum(r.split == "val" for r in rows),
            "train_new_v4": counter[("train", "v4")],
            "train_old": counter[("train", "old")],
            "train_pilot": counter[("train", "pilot")],
            "pilot_replaced_existing_train": replaced,
            "pilot_excluded_protected_scenes": excluded,
            "independent_test_untouched": True,
        }
        with (dest / "summary.json").open("x", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print("CloudVision v5 — подготовка TRAIN/VAL без обучения")
        print(f"TRAIN {summary['train']} | старых {summary['train_old']} | "
              f"v4 {summary['train_new_v4']} | PILOT {summary['train_pilot']}")
        print(f"VAL {summary['val']} (прежняя v4 VAL, без изменения)")
        print(f"PILOT заменили прежние TRAIN-дубликаты: {replaced}; "
              f"не допускаются по пересечению VAL/TEST: {excluded}")
        print("Манифест:", dest / "train_manifest.csv")
        print("Исключения:", dest / "excluded_pilot.csv")
        print("Веса и TEST не изменены. PILOT был AI-assisted, это не независимая оценка.")
    except (ValueError, OSError, FileNotFoundError, FileExistsError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()

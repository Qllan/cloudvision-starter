"""Audit manually saved v4 cloud labels without modifying them or TEST data."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from cloudvision.selection import read_manifest

ROOT = Path(__file__).resolve().parent
FIELDS = (
    "relative_path", "split", "status", "ai_percent", "verified_percent",
    "absolute_error_pp", "ai_action", "verified_action", "decision_changed",
    "needs_recheck", "recheck_reason", "mask_path",
)


def _read_predictions(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        if not rows.fieldnames or not {"исходный_файл", "облачность_%"}.issubset(rows.fieldnames):
            raise ValueError("В отчёте нейросети не найдены нужные колонки")
        predictions = {}
        for row in rows:
            key = row["исходный_файл"].replace("\\", "/")
            if key in predictions:
                raise ValueError(f"Дублирующийся прогноз: {key}")
            value = float(row["облачность_%"])
            if not 0 <= value <= 100:
                raise ValueError(f"Некорректная облачность {value}: {key}")
            predictions[key] = value
    return predictions


def audit_labels(image_dir: Path, manifest: Path, corrected_dir: Path,
                 predictions_csv: Path) -> list[dict[str, str]]:
    # Validates manifest paths, group consistency and existence of RGB images.
    read_manifest(manifest, image_dir, {"train", "val"})
    predictions = _read_predictions(predictions_csv)
    root = image_dir.resolve()
    corrected = corrected_dir.resolve()
    output = []
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                continue
            rel_text = row["relative_path"]
            rel = Path(rel_text)
            path = corrected / rel.with_suffix(".png")
            item = {column: "" for column in FIELDS}
            item.update(relative_path=rel_text, split=row["split"], mask_path=str(path))
            item["status"] = "MISSING"
            ai_percent = predictions.get(rel_text)
            if ai_percent is not None:
                item["ai_percent"] = f"{ai_percent:.3f}"
                item["ai_action"] = "REJECT" if ai_percent > 75 else "KEEP"
            if not path.is_file():
                output.append(item)
                continue
            try:
                with Image.open(root / rel) as original:
                    expected = ImageOps.exif_transpose(original).size
                with Image.open(path) as mask:
                    if mask.mode != "L" or mask.size != expected:
                        raise ValueError(f"Неверный режим/размер: {mask.mode}, {mask.size}; ожидается L, {expected}")
                    histogram = mask.histogram()
                    if any(value for i, value in enumerate(histogram) if i not in (0, 255)):
                        raise ValueError("Маска не бинарная 0/255")
                    percent = histogram[255] * 100.0 / (mask.width * mask.height)
            except (OSError, ValueError) as error:
                item["status"] = "INVALID: " + str(error)
                output.append(item)
                continue
            item["status"] = "OK"
            item["verified_percent"] = f"{percent:.3f}"
            item["verified_action"] = "REJECT" if percent > 75 else "KEEP"
            reasons = []
            if percent <= 0.1 or percent >= 99.9:
                reasons.append("крайняя облачность: перепроверить")
            if abs(percent - 75) <= 5:
                reasons.append("около порога 75%")
            if ai_percent is not None:
                diff = abs(ai_percent - percent)
                item["absolute_error_pp"] = f"{diff:.3f}"
                if diff >= 20:
                    reasons.append("большое расхождение с ИИ")
                changed = item["ai_action"] != item["verified_action"]
                item["decision_changed"] = "YES" if changed else "NO"
                if changed:
                    reasons.append("изменилось решение 75%")
            item["needs_recheck"] = "YES" if reasons else "NO"
            item["recheck_reason"] = "; ".join(reasons)
            output.append(item)
    return output


def save_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Проверить исправленные маски train/val для CloudVision AI v4")
    p.add_argument("--input", type=Path, default=ROOT / "data/new_raw")
    p.add_argument("--manifest", type=Path, default=ROOT / "reports/selection_v4/selection.csv")
    p.add_argument("--verified", type=Path, default=ROOT / "data/verified_v4")
    p.add_argument("--predictions", type=Path, default=ROOT / "reports/batch_v4_selected/operator_preview.csv")
    p.add_argument("--out", type=Path, default=ROOT / "reports/label_audit_v4/audit.csv")
    a = p.parse_args()
    try:
        rows = audit_labels(a.input, a.manifest, a.verified, a.predictions)
        save_report(a.out, rows)
    except (OSError, ValueError) as error:
        p.error(str(error))
    counts = Counter(r["status"] for r in rows)
    marked = sum(r["needs_recheck"] == "YES" for r in rows)
    changed = sum(r["decision_changed"] == "YES" for r in rows)
    print(f"TRAIN/VAL: {len(rows)} | сохранено корректных: {counts['OK']} | "
          f"нет масок: {counts['MISSING']} | некорректных: {len(rows)-counts['OK']-counts['MISSING']}")
    print(f"Стоит перепроверить: {marked} | сменилось решение по 75%: {changed}")
    missing = [r["relative_path"] for r in rows if r["status"] != "OK"]
    if missing:
        print("Ещё не готово:")
        for name in missing:
            print("  - " + name)
    print("Полный отчёт:", a.out)
    print("TEST не проверялся, изображения и маски не изменены.")


if __name__ == "__main__":
    main()

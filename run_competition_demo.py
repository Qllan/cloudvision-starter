"""CloudVision AI: safe competition demo with the frozen U-Net v3 Policy model.

No original photos are deleted. `--mode delete` only removes rejected copies from
PROJECT/data/work_input, and only after an explicit flag. Default is preview.
Requires train_cloudvision.py (TinyUNet, letterbox_rgb_and_mask) from stage 10.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from cloudvision.inference import load_checkpoint, predict_mask

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "input"
WORK_INPUT = ROOT / "data" / "work_input"
DEFAULT_ACCEPTED = ROOT / "data" / "accepted_demo"
DEFAULT_OUT = ROOT / "reports" / "competition_demo"
DEFAULT_MODEL = ROOT / "models" / "cloudvision_unet_v3_policy.pt"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
THRESHOLD = 75.0
FIELDS = (
    "исходный_файл", "прогноз_облачности_%", "решение_по_75%",
    "фактическое_действие", "сохранённый_файл", "удалённый_файл",
    "модель", "эпоха", "area_облачность_%", "нужна_проверка",
    "причина_проверки", "файл_маски"
)


def action_for(percent: float) -> str:
    if not np.isfinite(percent) or percent < 0 or percent > 100:
        raise ValueError(f"Некорректный процент облачности: {percent}")
    return "ОТСЕЯТЬ" if percent > THRESHOLD else "СОХРАНИТЬ"


def safe_delete_input(path: Path) -> bool:
    """No deletion outside the dedicated working-copy directory."""
    return not path.is_symlink() and path.resolve() == WORK_INPUT.resolve() and path.is_dir()


def cloud_percent(mask: Image.Image) -> float:
    a = np.asarray(mask, dtype=np.uint8)
    if not a.size or not np.isin(a, (0, 255)).all():
        raise ValueError("Ожидалась бинарная маска 0/255")
    return 100.0 * float(np.count_nonzero(a)) / a.size


def review_reason(policy_percent: float, area_percent: float | None = None) -> str:
    reasons = []
    if abs(policy_percent - THRESHOLD) <= 15:
        reasons.append("рядом с порогом 75%")
    if area_percent is not None and abs(policy_percent - area_percent) > 20:
        reasons.append("модели расходятся более чем на 20 п.п.")
    return "; ".join(reasons)


def create_preview(image: Image.Image, mask: Image.Image, percent: float,
                   decision: str, dest: Path):
    thumb = image.copy()
    thumb.thumbnail((640, 500), Image.Resampling.LANCZOS)
    thumb_mask = mask.resize(thumb.size, Image.Resampling.NEAREST)
    original = np.asarray(thumb.convert("RGB"), dtype=np.uint8)
    overlay = original.copy()
    chosen = np.asarray(thumb_mask, dtype=np.uint8) >= 128
    overlay[chosen] = (0.60 * overlay[chosen].astype(np.float32) +
                       0.40 * np.array([245, 50, 80], np.float32)).astype(np.uint8)
    w, h = thumb.size
    sheet = Image.new("RGB", (w * 2, h + 48), "white")
    sheet.paste(thumb, (0, 48))
    sheet.paste(Image.fromarray(overlay), (w, 48))
    draw = ImageDraw.Draw(sheet)
    draw.text((7, 9), "ORIGINAL", fill="black")
    draw.text((w + 7, 9), f"CLOUD: {percent:.2f}% | {decision}", fill="black")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=89)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    if args.limit < 0 or args.preview_limit < 0:
        raise ValueError("limit и preview-limit не могут быть отрицательными")
    source = args.input.resolve()
    dest = args.accepted.resolve()
    report_dir = args.out.resolve()
    if not args.input.is_dir():
        raise FileNotFoundError(f"Папка снимков не найдена: {args.input}")
    if args.mode == "delete":
        if not (args.confirm_delete_copies and safe_delete_input(args.input)):
            raise ValueError("Удаление разрешено ТОЛЬКО из data/work_input с --confirm-delete-copies")
    if args.mode != "preview":
        if dest == source or dest.is_relative_to(source) or source.is_relative_to(dest):
            raise ValueError("Папки исходных снимков и сохранённых копий не должны вложенно пересекаться")
    images = sorted(p for p in args.input.iterdir()
                    if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValueError("В папке нет подходящих спутниковых изображений")
    device = torch.device("cpu")
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    policy, size, meta = load_checkpoint(args.model, "policy", device)
    area = None
    if args.area_model is not None:
        area, area_size, _ = load_checkpoint(args.area_model, "area", device)
        if area_size != size:
            raise ValueError("У моделей Policy и Area разные размеры входа")
    print(f"CloudVision AI — конкурсная демонстрация | Модель Policy, эпоха {meta.get('best_epoch','?')}")
    print("Порог: облачность >75% — ОТСЕЯТЬ; 75% или меньше — СОХРАНИТЬ")
    print(f"Режим: {args.mode} | снимков: {len(images)}")
    if args.mode == "preview":
        print("Без копирования и удаления исходных файлов.")

    planned: list[dict] = []
    for index, image_path in enumerate(images, 1):
        with Image.open(image_path) as f:
            original = ImageOps.exif_transpose(f).convert("RGB")
        mask = predict_mask(original, policy, size, device)
        pct = cloud_percent(mask)
        area_pct = None
        if area is not None:
            area_pct = cloud_percent(predict_mask(original, area, size, device))
        action = action_for(pct)
        reason = review_reason(pct, area_pct)
        mask_file = f"{image_path.stem}_policy_mask.png"
        mask_dest = report_dir / "masks" / mask_file
        mask_dest.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_dest)
        if args.preview_limit and index <= args.preview_limit:
            create_preview(original, mask, pct, action,
                           report_dir / "previews" / f"{image_path.stem}_preview.jpg")
        planned.append({
            "исходный_файл": image_path.name,
            "прогноз_облачности_%": f"{pct:.3f}",
            "решение_по_75%": action,
            "фактическое_действие": "НЕ ВЫПОЛНЕНО (ПЛАН)",
            "сохранённый_файл": "", "удалённый_файл": "",
            "модель": "U-Net v3 Policy", "эпоха": meta.get("best_epoch", ""),
            "area_облачность_%": "" if area_pct is None else f"{area_pct:.3f}",
            "нужна_проверка": "ДА" if reason else "НЕТ",
            "причина_проверки": reason, "файл_маски": str(mask_dest),
        })
        print(f"[{index}/{len(images)}] {image_path.name}: {pct:.2f}% | {action}" +
              (f" | ПРОВЕРИТЬ: {reason}" if reason else ""))
    write_csv(report_dir / "plan.csv", planned)

    if args.mode == "preview":
        actual = []
        for row in planned:
            result = dict(row)
            result["фактическое_действие"] = "ТОЛЬКО ПРОСМОТР"
            actual.append(result)
        write_csv(report_dir / "operator_report.csv", actual)
    else:
        # Check all conflicts BEFORE modifying any source image.
        for item in planned:
            if item["решение_по_75%"] == "СОХРАНИТЬ":
                target = dest / item["исходный_файл"]
                if target.exists():
                    raise FileExistsError(f"Копия уже существует: {target}; ничего не изменено")
        actual = []
        for row in planned:
            result = dict(row)
            src = args.input / row["исходный_файл"]
            if row["решение_по_75%"] == "СОХРАНИТЬ":
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / src.name)
                result["сохранённый_файл"] = str(dest / src.name)
                result["фактическое_действие"] = "СКОПИРОВАН"
            elif args.mode == "delete":
                src.unlink()
                result["удалённый_файл"] = row["исходный_файл"]
                result["фактическое_действие"] = "УДАЛЁН ИЗ WORK_INPUT"
            else:
                result["фактическое_действие"] = "ОТСЕЯН БЕЗ УДАЛЕНИЯ"
            actual.append(result)
            write_csv(report_dir / "operator_report.csv", actual)

    kept = sum(item["решение_по_75%"] == "СОХРАНИТЬ" for item in planned)
    rejected = len(planned) - kept
    flagged = sum(item["нужна_проверка"] == "ДА" for item in planned)
    print(f"\nОбработано {len(planned)} | Сохранить {kept} | Отсеять {rejected} | Проверить {flagged}")
    print("План:", report_dir / "plan.csv")
    print("Отчёт оператору:", report_dir / "operator_report.csv")
    if args.mode != "delete":
        print("ОРИГИНАЛЫ НЕ УДАЛЕНЫ.")
    else:
        print("Удалены только отклонённые РАБОЧИЕ КОПИИ из data/work_input")


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudVision AI — демонстрация по кейсу СПУТНИКС")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--area-model", type=Path, default=None,
                        help="Необязательная модель Area, чтобы показывать расхождение прогнозов")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--limit", type=int, default=0, help="0 — все снимки")
    parser.add_argument("--preview-limit", type=int, default=12, help="Сколько иллюстраций сохранять")
    parser.add_argument("--mode", choices=("preview", "copy", "delete"), default="preview")
    parser.add_argument("--confirm-delete-copies", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError) as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()

"""CloudVision AI — конкурсный конвейер сортировки спутниковых JPEG.

Использует маски уже рассчитанной модели CLIPSeg (0=поверхность, 1/2=облака,
3=тень). По умолчанию ТОЛЬКО показывает решения, ничего не удаляет.
Запускай из корня проекта; дополнительные зависимости — из предыдущих этапов.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent
INPUT = PROJECT / "data" / "input"
WORK_INPUT = PROJECT / "data" / "work_input"
MASKS = PROJECT / "reports" / "tiling_experiment" / "masks"
OUTPUT = PROJECT / "reports" / "processing"
ACCEPTED = PROJECT / "data" / "accepted"
THRESHOLD = 75.0
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Decision:
    filename: str
    percent: float
    keep: bool
    mask_file: str


def percentage_from_mask(mask_path: Path, expected_size: tuple[int, int]) -> float:
    """Процент площади классов 1 (плотные) и 2 (тонкие). Тень 3 не облако."""
    with Image.open(mask_path) as im:
        if im.mode != "L":
            raise ValueError(f"{mask_path}: нужна индексная PNG-маска в режиме L, не цветной превью")
        if im.size != expected_size:
            raise ValueError(f"{mask_path}: маска {im.size}, снимок {expected_size}; размеры не совпадают")
        arr = np.asarray(im, dtype=np.uint8)
    if arr.size == 0 or not np.isin(arr, (0, 1, 2, 3)).all():
        raise ValueError(f"{mask_path}: допустимы только классы 0,1,2,3")
    return 100.0 * float(np.count_nonzero((arr == 1) | (arr == 2))) / arr.size


def decide(filename: str, cloud_percent: float, mask_name: str) -> Decision:
    if not 0.0 <= cloud_percent <= 100.0:
        raise ValueError("Процент облачности должен быть от 0 до 100")
    # СТРОГО выше 75% — к удалению; ровно 75% сохраняем.
    return Decision(filename, cloud_percent, cloud_percent <= THRESHOLD, mask_name)


def safe_to_delete(input_dir: Path, project: Path) -> bool:
    """Удалять разрешено исключительно из специальной папки с РАБОЧИМИ КОПИЯМИ."""
    approved = project.resolve() / "data" / "work_input"
    return input_dir.resolve() == approved and not input_dir.is_symlink() and approved.is_dir()


def mask_for(image_path: Path, mask_dir: Path) -> Path:
    return mask_dir / f"{image_path.stem}_tiled_mask.png"


def load_inference():
    """Ленивая загрузка модели: модель нужна только при --infer-missing."""
    try:
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        from experiment_tiling import infer_tiled, MODEL_ID
    except ImportError as ex:
        raise RuntimeError("Не найдены torch/transformers или experiment_tiling.py из этапа 5") from ex
    print("Загружаем CLIPSeg для снимков без готовой маски...")
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).to("cpu")
    model.eval()
    return model, processor, torch, infer_tiled


def get_mask(image_path: Path, mask_dir: Path, inference=None) -> Path | None:
    path = mask_for(image_path, mask_dir)
    if path.is_file():
        return path
    if inference is None:
        return None
    model, processor, torch, infer_tiled = inference
    print(f"  ИИ анализирует: {image_path.name}")
    with Image.open(image_path) as img:
        source = img.convert("RGB")
    predicted = infer_tiled(source, model, processor, torch)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(predicted, mode="L").save(path)
    return path


def output_row(decision: Decision, mode: str) -> dict[str, str]:
    if decision.keep:
        return {
            "исходный_файл": decision.filename,
            "облачность_процент": f"{decision.percent:.3f}",
            "сохранённый_файл": decision.filename if mode != "preview" else "",
            "удалённый_файл": "",
            "решение": "СОХРАНЕН" if mode != "preview" else "К СОХРАНЕНИЮ",
            "маска": decision.mask_file,
        }
    return {
        "исходный_файл": decision.filename,
        "облачность_процент": f"{decision.percent:.3f}",
        "сохранённый_файл": "",
        "удалённый_файл": decision.filename if mode == "delete" else "",
        "решение": "УДАЛЕН" if mode == "delete" else "К УДАЛЕНИЮ (НЕ УДАЛЯЛСЯ)",
        "маска": decision.mask_file,
    }


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["исходный_файл", "облачность_процент", "сохранённый_файл", "удалённый_файл", "решение", "маска"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_decision(decision: Decision, original: Path, accepted: Path, mode: str) -> None:
    if mode == "preview":
        return
    if decision.keep:
        accepted.mkdir(parents=True, exist_ok=True)
        target = accepted / original.name
        if target.exists():
            raise FileExistsError(f"Файл {target} уже существует; перезаписи не будет")
        shutil.copy2(original, target)
    elif mode == "delete":
        original.unlink()  # Только рабочие копии, только с явным подтверждением.


def main() -> None:
    p = argparse.ArgumentParser(description="CloudVision AI: сохранить до 75%, удалить выше 75%")
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--masks", type=Path, default=MASKS)
    p.add_argument("--report-dir", type=Path, default=OUTPUT)
    p.add_argument("--accepted", type=Path, default=ACCEPTED)
    p.add_argument("--limit", type=int, default=0, help="0 — все файлы")
    p.add_argument("--infer-missing", action="store_true", help="Запускать CLIPSeg, если маски ещё нет")
    p.add_argument("--mode", choices=("preview", "copy", "delete"), default="preview")
    p.add_argument("--confirm-delete", action="store_true", help="Явное согласие удалить РАБОЧИЕ КОПИИ")
    args = p.parse_args()
    if args.limit < 0:
        p.error("--limit не может быть отрицательным")
    input_dir = args.input.resolve()
    mask_dir = args.masks.resolve()
    accepted_dir = args.accepted.resolve()
    report_dir = args.report_dir.resolve()
    if not input_dir.is_dir():
        p.error(f"Папка снимков не найдена: {input_dir}")
    if args.mode == "delete":
        if not args.confirm_delete or not safe_to_delete(args.input, PROJECT):
            p.error("Удаление возможно только с --confirm-delete и --input data/work_input (КОПИИ снимков!)")
    if args.mode != "preview":
        if accepted_dir == input_dir or accepted_dir.is_relative_to(input_dir):
            p.error("Папка сохранения не должна находиться внутри папки снимков")
    files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in EXTS)
    if args.limit:
        files = files[:args.limit]
    if not files:
        p.error("Нет изображений в выбранной папке")
    inference = None
    actions: list[tuple[Path, Decision]] = []
    ignored: list[str] = []
    for f in files:
        mask = mask_for(f, mask_dir)
        if not mask.exists() and args.infer_missing and inference is None:
            inference = load_inference()
        mask = get_mask(f, mask_dir, inference)
        if mask is None:
            print(f"[ПРОПУЩЕНО] {f.name}: нет маски (добавь --infer-missing)")
            ignored.append(f.name)
            continue
        with Image.open(f) as image:
            actual_size = image.size
        percent = percentage_from_mask(mask, actual_size)
        decision = decide(f.name, percent, mask.name)
        actions.append((f, decision))
        print(f"{f.name}: {percent:.2f}% => {'СОХРАНИТЬ' if decision.keep else 'ОТСЕЯТЬ'}")
    if not actions:
        print("Нет снимков с масками. Ничего не изменено.")
        return
    # До действий сохраняем полный предполагаемый план.
    plan = [output_row(d, "preview") for _, d in actions]
    write_report(report_dir / "planned_actions.csv", plan)
    # Защита от частичной обработки, если файл назначения уже существует.
    if args.mode != "preview":
        for source, decision in actions:
            if decision.keep and (accepted_dir / source.name).exists():
                p.error(f"Сохранённый файл уже существует: {accepted_dir / source.name}. Ничего не изменено")
    completed = []
    try:
        for source, decision in actions:
            apply_decision(decision, source, accepted_dir, args.mode)
            completed.append(output_row(decision, args.mode))
            # Обновляем отчёт по фактически завершённым действиям.
            write_report(report_dir / "processing_report.csv", completed)
    finally:
        if completed:
            write_report(report_dir / "processing_report.csv", completed)
    kept = sum(d.keep for _, d in actions)
    rejected = len(actions) - kept
    print("\nОТЧЁТ ОПЕРАТОРУ:")
    print(f"Обработано: {len(completed)}; к сохранению: {kept}; к отсеву: {rejected}; без масок: {len(ignored)}")
    for row in completed:
        print(f"  {row['исходный_файл']} | {row['облачность_процент']}% | {row['решение']}")
    if args.mode != "delete":
        print("ФАЙЛЫ НЕ УДАЛЯЛИСЬ.")
    print(f"Отчёт: {report_dir / 'processing_report.csv'}")


if __name__ == "__main__":
    main()

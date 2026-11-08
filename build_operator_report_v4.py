"""Static operator review for CloudVision AI v4 preview batches.

Reads existing photos, binary PNG masks, and operator_preview.csv.
Creates self-contained local HTML and preview thumbnails, plus a prioritized CSV.
No model inference, no copies/deletions of original photographs.
"""
from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageOps

from cloudvision.rules import should_delete
from cloudvision.selection import scene_group
from cloudvision.operator_verification import PENDING, VERIFIED, manual_review

ROOT = Path(__file__).resolve().parent
EXPECTED = {"исходный_файл", "облачность_%", "решение_по_75%", "маска", "фактическое_действие"}
FIELDS = ("исходный_файл", "облачность_%", "решение_по_75%", "статус_проверки",
          "ручная_облачность_%", "ручное_решение_по_75%", "расхождение_пп",
          "IoU_с_ручной_маской", "проверить", "причины_проверки",
          "группа_сцены", "кадров_в_серии", "превью")


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or value.startswith("/") or "\\" in value or ":" in value or ".." in path.parts:
        raise ValueError(f"Небезопасный путь в CSV: {value}")
    return path


def _safe_input(base: Path, relative: Path) -> Path:
    path = base / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(base):
        raise ValueError(f"Файл отсутствует или путь небезопасен: {relative}")
    return path


def read_batch_report(images: Path, batch: Path) -> list[dict]:
    """Validate full CSV/mask consistency before creating any report."""
    images = images.resolve()
    batch = batch.resolve()
    csv_path = batch / "operator_preview.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Нет отчёта пакетной обработки: {csv_path}")
    if images == batch or images.is_relative_to(batch) or batch.is_relative_to(images):
        raise ValueError("Папки изображений и пакетного отчёта должны быть отдельными")
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not EXPECTED.issubset(reader.fieldnames):
            raise ValueError(f"CSV должен содержать колонки: {sorted(EXPECTED)}")
        raw = list(reader)
    if not raw:
        raise ValueError("Пустой operator_preview.csv")
    seen = set()
    rows = []
    groups = Counter()
    for item in raw:
        relative = _safe_relative(item["исходный_файл"])
        key = relative.as_posix().casefold()
        if key in seen:
            raise ValueError(f"Повторный снимок в отчёте: {relative}")
        seen.add(key)
        photo = _safe_input(images, relative)
        mask_rel = _safe_relative(item["маска"])
        expected_mask = Path("masks") / relative.parent / f"{relative.stem}_policy_mask.png"
        if mask_rel != expected_mask:
            raise ValueError(f"Неверное имя маски для {relative}: {mask_rel}")
        mask_path = _safe_input(batch, mask_rel)
        try:
            percent = float(item["облачность_%"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Некорректный процент облачности: {relative}") from exc
        if not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError(f"Недопустимое значение облачности: {relative}")
        if item["фактическое_действие"] != "ТОЛЬКО ПРЕДПРОСМОТР":
            raise ValueError("Ожидался исключительно безопасный режим предпросмотра")
        with Image.open(photo) as original:
            size = ImageOps.exif_transpose(original).size
        with Image.open(mask_path) as mask:
            if mask.mode != "L" or mask.size != size:
                raise ValueError(f"Неверный размер/режим PNG-маски: {mask_path}")
            counts = mask.histogram()
            if any(count for v, count in enumerate(counts) if v not in (0, 255)):
                raise ValueError(f"Маска должна быть бинарной 0/255: {mask_path}")
            actual_percent = 100 * counts[255] / (size[0] * size[1])
            if abs(actual_percent - percent) > 0.01:
                raise ValueError(f"Маска и процент в CSV расходятся: {relative}")
        # The CSV percentage is rounded to 3 decimals. A mask covering
        # 75.0001% may display as 75.000 but must still be REJECTED.
        decision = "ОТСЕЯТЬ" if should_delete(actual_percent) else "СОХРАНИТЬ"
        if item["решение_по_75%"] != decision:
            raise ValueError(f"Решение в CSV не соответствует точному правилу >75%: {relative}")
        group = scene_group(relative)
        groups[group] += 1
        notes = []
        if decision == "ОТСЕЯТЬ":
            notes.append("Предложен отсев: подтвердить маску перед действием")
        if abs(percent - 75) <= 15:
            notes.append("Близко к порогу 75%")
        if percent < 1:
            notes.append("Почти 0%: проверить тонкие облака")
        rows.append({
            "relative": relative, "photo": photo, "mask": mask_path,
            "percent": percent, "decision": decision, "group": group,
            "reasons": notes,
        })
    for row in rows:
        row["series_size"] = groups[row["group"]]
    return rows


def _preview(photo_path: Path, mask_path: Path, dest: Path, width: int = 460) -> None:
    """Render small ORIGINAL + AI OVERLAY, not an asserted ground-truth mask."""
    with Image.open(photo_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
        original.thumbnail((width, 330), Image.Resampling.LANCZOS)
    with Image.open(mask_path) as source_mask:
        small_mask = source_mask.resize(original.size, Image.Resampling.NEAREST)
    tint = Image.new("RGB", original.size, (238, 64, 115))
    colored = Image.blend(original, tint, 0.5)
    overlay = Image.composite(colored, original, small_mask)
    canvas = Image.new("RGB", (original.width * 2, original.height + 30), "white")
    canvas.paste(original, (0, 30))
    canvas.paste(overlay, (original.width, 30))
    painter = ImageDraw.Draw(canvas)
    painter.text((8, 8), "ORIGINAL", fill="black")
    painter.text((original.width + 8, 8), "AI MASK (PINK)", fill="black")
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=86)


def _url(relative: Path) -> str:
    return "/".join(quote(part, safe="") for part in relative.parts)


def _card(item: dict) -> str:
    rel = item["relative"]
    name = html.escape(rel.as_posix(), quote=True)
    preview = html.escape(_url(item["preview"]), quote=True)
    level = ("review" if item["manual"]["status"] == PENDING else
             ("reject" if item["decision"] == "ОТСЕЯТЬ" else "keep"))
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in item["reasons"])
    notes = f"<ul>{reasons}</ul>" if reasons else "<p>Автоматических флагов нет; это не подтверждает правильность маски.</p>"
    group = html.escape(item["group"])
    status = html.escape(item["manual"]["status"])
    manual_percent = item["manual"]["manual_percent"]
    comparison = (f'<p>Ручная облачность: {manual_percent:.2f}% | '
                  f'Ручное решение: {html.escape(item["manual"]["manual_decision"])} | '
                  f'IoU: {item["manual"]["mask_iou"]:.3f}</p>'
                  if manual_percent is not None else
                  "<p>Ручного эталона нет: решение модели не подтверждено.</p>")
    return (
        f'<article class="card {level}"><img src="{preview}" alt="Исходное RGB слева и маска ИИ справа: {name}" loading="lazy">'
        f'<div class="body"><h3>{name}</h3><p><strong>{item["percent"]:.2f}%</strong> '
        f'<span class="decision">{item["decision"]}</span></p>'
        f'<p><strong>Статус: {status}</strong></p>{comparison}'
        f'<p>Группа: {group}; кадров этой серии: {item["series_size"]}</p>{notes}</div></article>'
    )


def _page(items: list[dict]) -> str:
    total = len(items)
    keep = sum(item["decision"] == "СОХРАНИТЬ" for item in items)
    rejects = total - keep
    flagged = sum(bool(item["reasons"]) for item in items)
    pending = sum(item["manual"]["status"] == PENDING for item in items)
    done = total - pending
    cards = "\n".join(_card(item) for item in items)
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CloudVision AI v4 — операторский предпросмотр</title>
<style>
:root{{font-family:system-ui,"Segoe UI",sans-serif;color:#1d293b;background:#f5f7fa}}
body{{max-width:1250px;margin:1rem auto;padding:0 1rem}}
h1{{font-size:1.65rem}}p{{line-height:1.45}}
.summary{{display:flex;gap:.65rem;flex-wrap:wrap;margin:1rem 0}}
.summary b{{background:white;padding:.75rem 1.1rem;border-radius:.65rem;border:1px solid #dce3ed}}
.notice{{padding:1rem;background:white;border-left:4px solid #d59d3e;border-radius:.4rem}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));gap:1rem}}
.card{{background:white;border-radius:.7rem;overflow:hidden;border:1px solid #dce3ed;box-shadow:0 2px 9px #1d293b12}}
.card img{{display:block;width:100%;height:auto;background:#dce3ed}}
.body{{padding:.8rem 1rem}}
h3{{font-size:1rem;overflow-wrap:anywhere}}
.decision{{font-weight:600;margin-left:.4rem}}
.card.review{{border-color:#d59d3e}}.card.reject{{border-color:#b44b60}}
ul{{padding-left:1.1rem}}li{{margin:.25rem 0}}
footer{{margin:1.5rem 0;color:#586174}}
</style></head><body>
<h1>🛰️ CloudVision AI v4 — отчёт для оператора</h1>
<div class="summary"><b>Всего: {total}</b><b>Прогноз: СОХРАНИТЬ {keep}</b><b>Прогноз: ОТСЕЯТЬ {rejects}</b><b>Вручную проверено: {done}</b><b>Ожидает проверки: {pending}</b><b>Приоритет по эвристике: {flagged}</b></div>
<div class="notice"><strong>Режим: только предпросмотр.</strong> Ни один исходный снимок не удалён и не перенесён.
Решение сети — только рекомендация. Все снимки без сохранённой ручной маски
имеют статус «ТРЕБУЕТСЯ РУЧНАЯ ПРОВЕРКА», даже если далеко от порога.
Правило: строго более 75% — предложить отсев. Близость к порогу (±15 п.п.)
и почти нулевой прогноз повышают приоритет, но не гарантируют выявление
ложных облаков на суше или воде. Группы показывают близкие по дате/камере кадры.</div>
<p>Сначала показаны отклоняемые или спорные кадры. Слева оригинал, справа розовая маска ИИ.</p>
<main class="gallery">{cards}</main>
<footer>Оценки нейросети являются рекомендациями для оператора. Проверка не измеряет точность без ручной эталонной разметки.</footer>
</body></html>
"""


def build_report(images: Path, batch: Path, out: Path, preview_limit: int = 0,
                 verified: Path | None = None) -> dict[str, int]:
    if preview_limit < 0:
        raise ValueError("preview-limit не может быть отрицательным")
    images = images.resolve()
    batch = batch.resolve()
    out = out.resolve()
    if out == images or out.is_relative_to(images) or images.is_relative_to(out):
        raise ValueError("Папка отчёта не должна пересекаться с исходными снимками")
    if out == batch or batch.is_relative_to(out):
        raise ValueError("Отдельный отчёт должен быть в подпапке batch либо другой папке")
    if out == batch / "masks" or out.is_relative_to(batch / "masks") or (batch / "masks").is_relative_to(out):
        raise ValueError("Отчёт нельзя сохранять поверх исходных масок")
    if verified is None:
        verified = ROOT / "data/pilot_verified_v4"
    verified = verified.resolve()
    if (verified == images or verified == batch or verified == out
            or verified.is_relative_to(images) or verified.is_relative_to(batch)
            or verified.is_relative_to(out)):
        raise ValueError("Ручные маски должны быть в отдельной папке от оригиналов и отчётов")
    rows = read_batch_report(images, batch)
    for item in rows:
        item["manual"] = manual_review(item, verified)
    rows.sort(key=lambda item: (
        0 if item["manual"]["status"] == PENDING else 1,
        0 if item["decision"] == "ОТСЕЯТЬ" else (1 if item["reasons"] else 2),
        abs(item["percent"] - 75),
        item["relative"].as_posix(),
    ))
    out.mkdir(parents=True, exist_ok=True)
    for item in rows:
        relative_preview = Path("previews") / item["relative"].parent / f"{item['relative'].stem}_review.jpg"
        item["preview"] = relative_preview
    for item in rows if not preview_limit else rows[:preview_limit]:
        _preview(item["photo"], item["mask"], out / item["preview"])
    # If a limit is specified, HTML shows only actually rendered previews.
    visible = rows if not preview_limit else rows[:preview_limit]
    visible_names = {item["relative"].as_posix() for item in visible}
    with (out / "operator_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow({
                "исходный_файл": item["relative"].as_posix(),
                "облачность_%": f"{item['percent']:.3f}",
                "решение_по_75%": item["decision"],
                "статус_проверки": item["manual"]["status"],
                "ручная_облачность_%": (
                    f"{item['manual']['manual_percent']:.3f}"
                    if item["manual"]["manual_percent"] is not None else ""),
                "ручное_решение_по_75%": item["manual"]["manual_decision"],
                "расхождение_пп": (
                    f"{item['manual']['difference_pp']:.3f}"
                    if item["manual"]["difference_pp"] is not None else ""),
                "IoU_с_ручной_маской": (
                    f"{item['manual']['mask_iou']:.5f}"
                    if item["manual"]["mask_iou"] is not None else ""),
                "проверить": "ДА" if item["reasons"] else "НЕТ",
                "причины_проверки": "; ".join(item["reasons"]),
                "группа_сцены": item["group"],
                "кадров_в_серии": item["series_size"],
                "превью": item["preview"].as_posix() if item["relative"].as_posix() in visible_names else "",
            })
    (out / "index.html").write_text(_page(visible), encoding="utf-8")
    return {
        "total": len(rows),
        "kept": sum(r["decision"] == "СОХРАНИТЬ" for r in rows),
        "rejected": sum(r["decision"] == "ОТСЕЯТЬ" for r in rows),
        "review": sum(bool(r["reasons"]) for r in rows),
        "manual_pending": sum(r["manual"]["status"] == PENDING for r in rows),
        "manual_done": sum(r["manual"]["status"] == VERIFIED for r in rows),
        "previewed": len(visible),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Отчёт оператора с изображениями, масками и ручными флагами")
    parser.add_argument("--input", type=Path, default=ROOT / "data/new_raw")
    parser.add_argument("--batch", type=Path, default=ROOT / "reports/competition_pilot_v4")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verified", type=Path, default=ROOT / "data/pilot_verified_v4",
                        help="Отдельные ручные маски пилота; отсутствующие кадры останутся непроверенными")
    parser.add_argument("--preview-limit", type=int, default=60,
                        help="До 60 превью по умолчанию; 0 — превью для всех строк, CSV содержит все строки")
    args = parser.parse_args()
    dest = args.out or (args.batch / "operator_review")
    try:
        summary = build_report(args.input, args.batch, dest, args.preview_limit,
                               verified=args.verified)
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.error(str(error))
    print(f"Проверено: {summary['total']} | СОХРАНИТЬ {summary['kept']} | "
          f"ОТСЕЯТЬ {summary['rejected']} | приоритетных по эвристике {summary['review']} | "
          f"ручных масок {summary['manual_done']} | без ручной проверки {summary['manual_pending']}")
    print("HTML:", dest / "index.html")
    print("CSV:", dest / "operator_review.csv")
    print("Оригиналы и маски не менялись. Это только предпросмотр.")


if __name__ == "__main__":
    main()

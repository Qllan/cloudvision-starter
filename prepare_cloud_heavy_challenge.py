"""Human-selected cloud-heavy challenge from NEW scenes only (not a random TEST).

browse: choose an AI-free RGB candidate pool and make an offline thumbnail gallery.
freeze: read a plain-text list of selected filenames, validate independent scene
groups, freeze a *targeted* manifest; never replace existing manifests.
No inference, no model training, no deletion/copying of original satellite RGB.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from cloudvision.batch import discover_images
from cloudvision.selection import scene_group, write_manifest
from cloudvision.v4_data import _read_csv, _relative_path
from select_v5_holdout import excluded_groups

ROOT = Path(__file__).resolve().parent


def protected_groups(v4_selection: Path, old_split: Path, pilot_report: Path,
                     v5_holdout: Path, previous_preview: Path | None = None) -> set[str]:
    """Exclude every previous TRAIN, VAL, TEST, PILOT and preview scene."""
    groups = excluded_groups(v4_selection, old_split, pilot_report, previous_preview)
    new_test = _read_csv(v5_holdout, {"relative_path", "scene_group", "split"})
    for row in new_test:
        rel = _relative_path(row["relative_path"])
        if row["split"] != "test" or row["scene_group"].casefold() != scene_group(rel):
            raise ValueError(f"Неверная группа/часть в v5 holdout: {rel}")
        groups.add(scene_group(rel))
    return groups


def make_pool(images: Path, blocked: set[str], count: int = 100,
              per_group: int = 2, seed: int = 2026) -> list[dict[str, str]]:
    """Sample unseen groups before looking at pixels; up to two frames/group."""
    if count < 5 or per_group < 1 or per_group > 4:
        raise ValueError("Нужно --count >=5 и 1 <= --per-group <= 4")
    root = images.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Нет исходных снимков: {root}")
    groups = defaultdict(list)
    for image in discover_images(root):
        if image.is_symlink():
            continue
        rel = image.relative_to(root)
        group = scene_group(rel)
        if group not in blocked:
            groups[group].append(rel)
    if len(groups) < 5:
        raise ValueError(f"Осталось только {len(groups)} новых групп; недостаточно")
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    sampled = {key: rng.sample(sorted(groups[key]), min(per_group, len(groups[key])))
               for key in keys}
    chosen = []
    for rank in range(per_group):
        for key in keys:
            if rank < len(sampled[key]):
                rel = sampled[key][rank]
                chosen.append({"relative_path": rel.as_posix(), "scene_group": key})
                if len(chosen) >= count:
                    return chosen
    return chosen


def _jpeg_data(path: Path) -> str:
    with Image.open(path) as source:
        source.draft("RGB", (480, 340))
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((380, 250), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _gallery(rows: list[dict[str, str]], root: Path) -> str:
    cards = []
    for index, row in enumerate(rows):
        relative = row["relative_path"]
        safe = html.escape(relative, quote=True)
        group = html.escape(row["scene_group"])
        img = _jpeg_data(root / _relative_path(relative))
        cards.append(
            f'<article><label><input class="pick" type="checkbox" value="{safe}"> '
            f'<strong>Выбрать кадр {index + 1}</strong></label>'
            f'<img alt="{safe}" loading="lazy" src="data:image/jpeg;base64,{img}">'
            f'<small>{safe}</small><small>Группа: {group}</small></article>'
        )
    return """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>CloudVision — отбор сильно облачных сцен</title>
<style>body{font:15px system-ui,sans-serif;margin:20px auto;max-width:1440px;padding:0 16px;color:#1b2739;background:#f5f8fc}
h1{font-size:24px}.notice{max-width:1000px;line-height:1.5}.toolbar{position:sticky;top:0;padding:14px;background:#fff;border:1px solid #bbb;border-radius:9px;z-index:10;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
button{background:#284dc2;color:white;border:0;border-radius:6px;padding:10px;cursor:pointer;font-size:15px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:14px}
article{background:white;border:1px solid #dde3ed;border-radius:9px;padding:10px;min-width:0}
img{width:100%;height:210px;object-fit:contain;background:#e5ebf5;display:block;margin:8px 0}
small{display:block;overflow-wrap:anywhere;color:#49556b}label{cursor:pointer}</style></head><body>
<h1>🛰️ CloudVision AI — визуальный отбор сильно облачных сцен</h1>
<p class="notice">Просматривай только исходный RGB (без AI-маски). Отмечай фотографии,
где большая часть кадра предположительно закрыта облаками. <b>Не путай снег,
соль, горный рельеф и отражения на воде с облаками.</b> Это целевая,
не случайная выборка; окончательная доля облаков будет известна лишь
после ручной разметки. Для чистоты проверки выбирай максимум один кадр
на одну группу дата/камера.</p>
<div class="toolbar"><b>Выбрано: <span id="total">0</span></b>
<button id="save">Скачать selected_cloud_heavy.txt</button>
<small>Сохрани TXT в корень проекта; ничего не загружается на сервер.</small></div>
<main>""" + "\n".join(cards) + """</main>
<script>
const boxes=[...document.querySelectorAll(".pick")];
function count(){document.getElementById("total").textContent=String(boxes.filter(b=>b.checked).length)}
boxes.forEach(b=>b.addEventListener("change",count));
document.getElementById("save").addEventListener("click",()=>{
 const rows=boxes.filter(b=>b.checked).map(b=>b.value);
 if(rows.length<5){alert("Выбери хотя бы 5 отдельных сцен.");return}
 const blob=new Blob([rows.join("\\n")+"\\n"],{type:"text/plain;charset=utf-8"});
 const url=URL.createObjectURL(blob), a=document.createElement("a");
 a.href=url;a.download="selected_cloud_heavy.txt";a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
});
</script></body></html>"""


def browse(images: Path, blocked: set[str], count: int, per_group: int,
           seed: int, out: Path) -> dict[str, int]:
    root, out = images.resolve(), out.resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Галерею нельзя сохранять внутри оригинальных снимков")
    if (out / "index.html").exists() or (out / "candidate_pool.csv").exists():
        raise FileExistsError("Галерея уже создана: укажи другое --out вместо перезаписи")
    pool = make_pool(root, blocked, count, per_group, seed)
    # Render all thumbnails before writing either artifact; fail without half a gallery.
    page = _gallery(pool, root)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate_pool.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["relative_path", "scene_group"])
        writer.writeheader()
        writer.writerows(pool)
    (out / "index.html").write_text(page, encoding="utf-8")
    return {"total": len(pool), "groups": len(set(p["scene_group"] for p in pool))}


def freeze(selected: Path, pool_csv: Path, images: Path, blocked: set[str],
           dest: Path) -> dict[str, int]:
    if dest.exists() or (dest.parent / "protocol.json").exists():
        raise FileExistsError("Целевой манифест/протокол уже существует; не перезаписывай проверку")
    root = images.resolve()
    if dest.resolve() == root or dest.resolve().is_relative_to(root):
        raise ValueError("Манифест нельзя сохранять среди исходных фото")
    rows = _read_csv(pool_csv, {"relative_path", "scene_group"})
    available = {}
    for row in rows:
        rel = _relative_path(row["relative_path"])
        key = rel.as_posix().casefold()
        group = scene_group(rel)
        if key in available or row["scene_group"].casefold() != group or group in blocked:
            raise ValueError(f"Повтор/небезопасная группа в candidate_pool.csv: {rel}")
        available[key] = (rel, group)
    if not selected.is_file():
        raise FileNotFoundError(f"Нет списка выбранных кадров: {selected}")
    names = [value.strip() for value in selected.read_text(encoding="utf-8-sig").splitlines()
             if value.strip()]
    if len(names) < 5:
        raise ValueError("Для проверки выбери хотя бы пять новых снимков")
    chosen, seen_names, seen_groups = [], set(), set()
    for value in names:
        rel = _relative_path(value)
        key = rel.as_posix().casefold()
        if key not in available or key in seen_names:
            raise ValueError(f"Снимок не входит в первоначальный пул либо повторяется: {value}")
        _, group = available[key]
        if group in seen_groups:
            raise ValueError(f"В целевой проверке нельзя брать две фотографии одной серии: {group}")
        image = root / rel
        if not image.is_file() or image.is_symlink() or not image.resolve().is_relative_to(root):
            raise ValueError(f"Нет исходного снимка или небезопасный путь: {rel}")
        seen_names.add(key)
        seen_groups.add(group)
        chosen.append({"relative_path": rel.as_posix(), "scene_group": group, "split": "test"})
    write_manifest(dest, sorted(chosen, key=lambda r: r["relative_path"]))
    protocol = {
        "type": "TARGETED_CLOUD_HEAVY_CHALLENGE_NOT_RANDOM",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "human visual screening of unseen RGB images without AI predictions",
        "selected_count": len(chosen),
        "source_pool_count": len(rows),
        "unique_scene_groups": len(seen_groups),
        "reference_masks": "blind human annotation after model freeze",
        "threshold": "strictly greater than 75 percent; exactly 75 percent KEEP",
        "no_original_deletion": True,
    }
    with (dest.parent / "protocol.json").open("x", encoding="utf-8") as file:
        json.dump(protocol, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return {"selected": len(chosen), "groups": len(seen_groups)}


def main() -> None:
    p = argparse.ArgumentParser(description="Целевая независимая проверка сильной облачности")
    p.add_argument("stage", choices=("browse", "freeze"))
    p.add_argument("--images", type=Path, default=ROOT / "data/new_raw")
    p.add_argument("--v4-selection", type=Path, default=ROOT / "reports/selection_v4/selection.csv")
    p.add_argument("--old-split", type=Path, default=ROOT / "reports/training/dataset_split.csv")
    p.add_argument("--pilot-report", type=Path, default=ROOT / "reports/competition_pilot_v4/operator_review/operator_review.csv")
    p.add_argument("--holdout", type=Path, default=ROOT / "reports/v5/holdout_manifest.csv")
    p.add_argument("--previous-preview", type=Path, default=ROOT / "reports/batch_v4_new/operator_preview.csv")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--seed", type=int, default=2075)
    p.add_argument("--browse-out", type=Path, default=ROOT / "reports/v5/cloud_heavy_pool")
    p.add_argument("--selected", type=Path, default=ROOT / "selected_cloud_heavy.txt")
    p.add_argument("--out", type=Path, default=ROOT / "reports/v5/cloud_heavy_challenge/challenge_manifest.csv")
    args = p.parse_args()
    try:
        blocked = protected_groups(args.v4_selection, args.old_split,
                                   args.pilot_report, args.holdout,
                                   args.previous_preview)
        if args.stage == "browse":
            result = browse(args.images, blocked, args.count, args.per_group,
                            args.seed, args.browse_out)
            print(f"Сформирована галерея из {result['total']} снимков "
                  f"({result['groups']} новых групп), ни одна модель не запускалась.")
            print("Открой:", args.browse_out / "index.html")
            print("Выбери визуально облачные кадры, скачай selected_cloud_heavy.txt")
            print("Если из пулa не удалось отобрать 5 различных сцен, "
                  "отложенный набор не нужно искусственно дополнять старыми TEST.")
        else:
            result = freeze(args.selected, args.browse_out / "candidate_pool.csv",
                            args.images, blocked, args.out)
            print(f"Целевой challenge заморожен: {result['selected']} кадров, "
                  f"{result['groups']} независимых групп.")
            print("Manifest:", args.out)
            print("Protocol:", args.out.parent / "protocol.json")
            print("Теперь зафиксируй SHA-256 v4/v5 с evaluate_v5_holdout.py --freeze "
                  "--manifest ... --out ... ДО слепой разметки.")
    except (OSError, ValueError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()

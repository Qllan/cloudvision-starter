"""CloudVision AI: self-contained offline competition evidence report.

Read ONLY existing metrics/verified operator CSVs. No model inference, source
image mutation, data uploads, or modification of previously frozen TEST reports.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV {path} не содержит колонок: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV пуст: {path}")
    return rows


def number(row: dict, key: str, low: float | None = None, high: float | None = None) -> float:
    try:
        result = float(row[key])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Некорректная метрика {key}: {row}") from exc
    if not math.isfinite(result) or low is not None and result < low or high is not None and result > high:
        raise ValueError(f"Число вне допустимого диапазона в {key}: {row}")
    return result


def integer(row: dict, key: str) -> int:
    value = number(row, key, 0)
    if not value.is_integer():
        raise ValueError(f"Ожидалось целое значение {key}")
    return int(value)


def f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def validate_holdout(summary: list[dict], per_image: list[dict], labels: set[str]):
    by_model = {row["model"]: row for row in summary}
    if set(by_model) != labels or len(summary) != len(labels):
        raise ValueError(f"Ожидались модели {sorted(labels)} в summary.csv")
    grouped = defaultdict(list)
    for row in per_image:
        grouped[row["model"]].append(row)
    if set(grouped) != labels:
        raise ValueError("Модели per_image.csv и summary.csv не совпадают")
    pair_reference: dict[str, tuple[str, str]] = {}
    for model in labels:
        rows = grouped[model]
        if len(rows) != integer(by_model[model], "test_count"):
            raise ValueError(f"Количество TEST для {model} не совпадает с summary")
        if len(set(row["image"] for row in rows)) != len(rows):
            raise ValueError(f"Дубликаты снимков в независимом TEST {model}")
        errors, ious = [], []
        correct, reference_rejects, false_rejects, missed_rejects = 0, 0, 0, 0
        for row in rows:
            reference = number(row, "reference_percent", 0, 100)
            prediction = number(row, "predicted_percent", 0, 100)
            error = number(row, "absolute_error_pp", 0, 100)
            if abs(error - abs(reference - prediction)) > 0.003:
                raise ValueError(f"Несогласованная MAE снимка {row['image']}")
            iou = number(row, "iou", 0, 1)
            if row["reference_action"] not in {"KEEP", "REJECT"} or row["predicted_action"] not in {"KEEP", "REJECT"}:
                raise ValueError("Неверные названия решений в TEST")
            # Rounded percentages may not identify the side of the >75 boundary
            # when the true pixel ratio is 75.0001% but CSV says 75.000.
            if abs(reference - 75) >= 0.002 and (row["reference_action"] == "REJECT") != (reference > 75):
                raise ValueError(f"Нарушение правила >75%: {row['image']}")
            if abs(prediction - 75) >= 0.002 and (row["predicted_action"] == "REJECT") != (prediction > 75):
                raise ValueError(f"Нарушение правила >75%: {row['image']}")
            match = row["predicted_action"] == row["reference_action"]
            if integer(row, "correct_action") != int(match):
                raise ValueError(f"Несогласованное correct_action у {row['image']}")
            reference_key = (row["reference_percent"], row["reference_action"])
            if row["image"] in pair_reference and pair_reference[row["image"]] != reference_key:
                raise ValueError(f"Разные ручные эталоны у двух моделей: {row['image']}")
            pair_reference[row["image"]] = reference_key
            correct += int(match)
            reference_rejects += int(row["reference_action"] == "REJECT")
            false_rejects += int(row["predicted_action"] == "REJECT" and row["reference_action"] == "KEEP")
            missed_rejects += int(row["predicted_action"] == "KEEP" and row["reference_action"] == "REJECT")
            errors.append(error)
            ious.append(iou)
        s = by_model[model]
        actual = {
            "correct_actions": correct, "reference_rejects": reference_rejects,
            "wrong_rejects": false_rejects, "missed_rejects": missed_rejects,
        }
        for key, value in actual.items():
            if integer(s, key) != value:
                raise ValueError(f"Несогласованный отчёт {model}: {key}")
        if abs(number(s, "mae_pp", 0, 100) - sum(errors) / len(errors)) > 0.003:
            raise ValueError(f"Несогласованная MAE в summary: {model}")
        if abs(number(s, "mean_iou", 0, 1) - sum(ious) / len(ious)) > 0.00002:
            raise ValueError(f"Несогласованный IoU в summary: {model}")
    if len({frozenset(row["image"] for row in grouped[model]) for model in labels}) != 1:
        raise ValueError("Модели оценивались на разных снимках")
    return by_model, grouped


def validate_pilot(rows: list[dict]):
    seen = set()
    for row in rows:
        name = row["исходный_файл"]
        if name in seen:
            raise ValueError("Повторный пилотный снимок")
        seen.add(name)
        if row["статус_проверки"] != "ПРОВЕРЕНО ОПЕРАТОРОМ":
            raise ValueError("Есть пилотные маски без ручного подтверждения")
        reference = number(row, "ручная_облачность_%", 0, 100)
        prediction = number(row, "облачность_%", 0, 100)
        difference = number(row, "расхождение_пп", 0, 100)
        number(row, "IoU_с_ручной_маской", 0, 1)
        if abs(difference - abs(prediction - reference)) > 0.003:
            raise ValueError(f"Пилотный CSV противоречив для {name}")
        if row["решение_по_75%"] not in {"СОХРАНИТЬ", "ОТСЕЯТЬ"} or row["ручное_решение_по_75%"] not in {"СОХРАНИТЬ", "ОТСЕЯТЬ"}:
            raise ValueError(f"Некорректное решение пилота для {name}")
    return {
        "count": len(rows),
        "mae": sum(number(row, "расхождение_пп") for row in rows) / len(rows),
        "iou": sum(number(row, "IoU_с_ручной_маской") for row in rows) / len(rows),
        "correct": sum(row["решение_по_75%"] == row["ручное_решение_по_75%"] for row in rows),
        "reference_rejects": sum(row["ручное_решение_по_75%"] == "ОТСЕЯТЬ" for row in rows),
        "missed": sum(row["ручное_решение_по_75%"] == "ОТСЕЯТЬ" and row["решение_по_75%"] == "СОХРАНИТЬ" for row in rows),
        "changed": sum(number(row, "расхождение_пп") > 0.003 for row in rows),
    }


def sparkline(points: list[float], *, width: int = 880, height: int = 190) -> str:
    margin_x, margin_y = 32, 24
    lo, hi = 0.0, max(points) * 1.12 if points else 1
    scale_x = (width - margin_x * 2) / max(1, len(points) - 1)
    svg_points = " ".join(
        f"{margin_x + i * scale_x:.1f},{height - margin_y - (value - lo) / (hi - lo) * (height - 2 * margin_y):.1f}"
        for i, value in enumerate(points)
    )
    best = min(points)
    best_i = points.index(best)
    dot_x = margin_x + best_i * scale_x
    dot_y = height - margin_y - (best - lo) / (hi - lo) * (height - 2 * margin_y)
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="MAE по эпохам обучения v5; лучший результат на эпохе {best_i + 1}">
      <line x1="{margin_x}" y1="{height-margin_y}" x2="{width-margin_x}" y2="{height-margin_y}" stroke="#cbd5e1" />
      <polyline fill="none" stroke="#0a9189" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{svg_points}"/>
      <circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="6" fill="#d59e37" stroke="#fff" stroke-width="3"/>
      <text x="{dot_x + 12:.1f}" y="{max(20, dot_y - 10):.1f}" font-size="15" fill="#0b4b5a">Эпоха {best_i + 1} · {f(best,2)} п.п.</text>
      <text x="{margin_x}" y="{height-3}" font-size="13" fill="#52647b">1</text>
      <text x="{width-margin_x}" y="{height-3}" font-size="13" text-anchor="end" fill="#52647b">{len(points)} эпох</text>
    </svg>'''


def preview_embed(name: str, preview_dir: Path | None) -> str:
    if preview_dir is None:
        return ""
    relative = Path(name)
    if relative.name != name or relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return ""
    file = preview_dir / f"{relative.stem}_review.jpg"
    if not file.is_file() or file.is_symlink() or file.stat().st_size > 2_000_000:
        return ""
    encoded = base64.b64encode(file.read_bytes()).decode("ascii")
    return f'<img class="example-image" src="data:image/jpeg;base64,{encoded}" alt="Оригинал и розовая маска AI для {esc(name)}">'


def build_report(v5_summary: Path, v5_images: Path, pilot_csv: Path,
                 validation_csv: Path, history_csv: Path, old_summary: Path,
                 old_images: Path, output: Path, preview_dir: Path | None = None) -> dict:
    summary24 = load_rows(v5_summary, {"model", "test_count", "mae_pp", "mean_iou", "correct_actions", "reference_rejects", "wrong_rejects", "missed_rejects"})
    images24 = load_rows(v5_images, {"model", "image", "reference_percent", "predicted_percent", "absolute_error_pp", "iou", "predicted_action", "reference_action", "correct_action"})
    s24, imgs24 = validate_holdout(summary24, images24, {"v4-policy", "v5-pilot"})
    summary5 = load_rows(old_summary, {"model", "test_count", "mae_pp", "mean_iou", "correct_actions", "reference_rejects", "wrong_rejects", "missed_rejects"})
    images5 = load_rows(old_images, {"model", "image", "reference_percent", "predicted_percent", "absolute_error_pp", "iou", "predicted_action", "reference_action", "correct_action"})
    s5, _ = validate_holdout(summary5, images5, {"v3-policy", "v4-policy-frozen"})
    pilot = load_rows(pilot_csv, {"исходный_файл", "облачность_%", "решение_по_75%", "статус_проверки", "ручная_облачность_%", "ручное_решение_по_75%", "расхождение_пп", "IoU_с_ручной_маской"})
    p = validate_pilot(pilot)
    vals = load_rows(validation_csv, {"model", "best_epoch", "val_mae_pp", "val_iou", "val_decision_accuracy"})
    val_by_model = {x["model"]: x for x in vals}
    if set(val_by_model) != {"v4-policy", "v5-pilot"}:
        raise ValueError("В VAL должны присутствовать v4-policy и v5-pilot")
    history = load_rows(history_csv, {"epoch", "val_mae_pp", "val_iou", "train_loss"})
    epochs = [integer(x, "epoch") for x in history]
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError("Непоследовательные эпохи в history CSV")
    mae_history = [number(x, "val_mae_pp", 0, 100) for x in history]
    best_epoch = integer(val_by_model["v5-pilot"], "best_epoch")
    if best_epoch not in epochs or abs(mae_history[best_epoch-1] - number(val_by_model["v5-pilot"], "val_mae_pp")) > 0.01:
        raise ValueError("Лучшая эпоха VAL не совпадает с историей")
    if integer(s24["v4-policy"], "test_count") != integer(s24["v5-pilot"], "test_count"):
        raise ValueError("Ожидалась одна общая независимая выборка")
    n24 = integer(s24["v5-pilot"], "test_count")
    if integer(s24["v4-policy"], "reference_rejects") != integer(s24["v5-pilot"], "reference_rejects"):
        raise ValueError("У моделей v4 и v5 различается TEST-эталон")
    n_ref_reject = integer(s24["v5-pilot"], "reference_rejects")
    top = sorted(imgs24["v5-pilot"], key=lambda x: -number(x, "absolute_error_pp"))[:4]
    false_rejects = [x for x in imgs24["v5-pilot"] if x["predicted_action"] == "REJECT" and x["reference_action"] == "KEEP"]
    missed = [x for x in pilot if x["ручное_решение_по_75%"] == "ОТСЕЯТЬ" and x["решение_по_75%"] == "СОХРАНИТЬ"]
    top_pilot = max(pilot, key=lambda x: number(x, "расхождение_пп"))
    min_base = number(s24["v4-policy"], "mae_pp", 0, 100)
    min_new = number(s24["v5-pilot"], "mae_pp", 0, 100)
    v4_iou = number(s24["v4-policy"], "mean_iou", 0, 1)
    v5_iou = number(s24["v5-pilot"], "mean_iou", 0, 1)
    bar_max = max(min_base, min_new, 1)
    def bar(label, value, class_name, precision=2, max_value=None, suffix=""):
        bound = max_value or bar_max
        width = 100 * value / bound
        return f'<div class="bar-row"><div class="bar-label">{esc(label)}</div><div class="bar-track"><div class="bar-fill {class_name}" style="width:{width:.2f}%"></div></div><strong>{f(value, precision)}{suffix}</strong></div>'
    def metric_row(name, label, unit="", digits=2):
        return f'<tr><td>{esc(label)}</td><td>{f(number(s24["v4-policy"],name),digits)}{unit}</td><td>{f(number(s24["v5-pilot"],name),digits)}{unit}</td></tr>'
    correct4 = integer(s24["v4-policy"], "correct_actions")
    correct5 = integer(s24["v5-pilot"], "correct_actions")
    pilot_rows = "".join(f'<tr><td class="file">{esc(x["исходный_файл"])}</td><td>{f(number(x,"облачность_%"))}%</td><td>{f(number(x,"ручная_облачность_%"))}%</td></tr>' for x in missed)
    errors_rows = "".join(f'<tr><td class="file">{esc(x["image"])}</td><td>{f(number(x,"reference_percent"))}%</td><td>{f(number(x,"predicted_percent"))}%</td><td>{f(number(x,"absolute_error_pp"))} п.п.</td></tr>' for x in top)
    false_rows = "".join(f'<li><strong>{esc(x["image"])}</strong>: эталон {f(number(x,"reference_percent"),3)}%, v5 {f(number(x,"predicted_percent"),3)}%. Следовало сохранить, модель предложила отсеять.</li>' for x in false_rejects)
    preview_html = preview_embed(top_pilot["исходный_файл"], preview_dir)
    html_page = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CloudVision AI · технический отчёт · СПУТНИКС</title>
<style>
:root{{--ink:#142a39;--muted:#526577;--bg:#f3f6fa;--teal:#098d86;--navy:#173b69;--border:#dce5ec;--amber:#d6a04e;--red:#c54d57}}
*{{box-sizing:border-box}}body{{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}}
main{{max-width:1120px;margin:auto;padding:34px 24px 70px}}header{{background:linear-gradient(115deg,#12283c,#135f6b);color:white;padding:50px 6vw}}header .inner{{max-width:1120px;margin:auto}}h1{{font-size:clamp(2.1rem,5vw,3.7rem);line-height:1.08;letter-spacing:-.04em;margin:.7rem 0}}h2{{font-size:1.45rem;letter-spacing:-.025em;margin:0 0 15px}}h3{{font-size:1.04rem;margin:0 0 10px}}p{{line-height:1.6;margin:.7rem 0}}header p{{max-width:790px;color:#d7e6ec}}.kicker{{font-size:.79rem;letter-spacing:.13em;font-weight:800;text-transform:uppercase;color:#9ee4dc}}.eyebrow{{font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;color:#6a8194;font-weight:750}}.section{{margin-top:38px}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}}.card,.panel{{background:white;border:1px solid var(--border);border-radius:14px;padding:22px;box-shadow:0 3px 12px #16344c08}}.card strong{{font-size:2rem;display:block;letter-spacing:-.06em;line-height:1.2;margin:10px 0 5px}}.card small{{color:var(--muted);line-height:1.35;display:block}}.grid2{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}}.big{{font-size:1.2rem}}.bar-row{{display:grid;grid-template-columns:85px 1fr 88px;align-items:center;gap:14px;margin:15px 0}}.bar-track{{height:19px;background:#edf2f7;border-radius:6px;overflow:hidden}}.bar-fill{{height:100%;border-radius:6px}}.navy{{background:#2d587b}}.teal{{background:#189f96}}table{{width:100%;border-collapse:collapse;font-size:.95rem}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #e6edf3}}th{{color:#536779;font-size:.78rem;font-weight:750;letter-spacing:.035em}}tr:last-child td{{border-bottom:0}}.file{{font-family:ui-monospace,Consolas,monospace;font-size:.76rem;overflow-wrap:anywhere}}.notice{{border-left:4px solid var(--amber);padding:14px 18px;border-radius:5px;background:#fff9ee}}.important{{border-left-color:var(--red);background:#fff4f4}}.tag{{display:inline-block;border:1px solid #8fd5c7;border-radius:100px;padding:5px 12px;color:#b8f5ec;font-size:.78rem;font-weight:750;letter-spacing:.03em}}.subtitle{{color:#6a8496}}.chart svg{{width:100%;height:auto}}.example-image{{display:block;max-width:100%;border-radius:12px;border:1px solid var(--border);margin-top:14px}}.steps{{counter-reset:item;padding:0;margin:0;list-style:none}}.steps li{{counter-increment:item;padding:10px 0 12px 46px;position:relative;line-height:1.5}}.steps li::before{{content:counter(item);position:absolute;left:0;top:8px;width:30px;height:30px;display:grid;place-items:center;background:#d6f2ee;color:#075b57;border-radius:50%;font-weight:800}}.footer{{margin-top:42px;font-size:.82rem;color:#596e80;border-top:1px solid var(--border);padding-top:20px}}.muted{{color:var(--muted)}}.good{{color:#067c74}}.warn{{color:#ad393e}}@media(max-width:850px){{.cards{{grid-template-columns:repeat(2,1fr)}}.grid2{{grid-template-columns:1fr}}}}@media(max-width:520px){{main{{padding:22px 13px}}.cards{{grid-template-columns:1fr}}.card,.panel{{padding:16px}}.bar-row{{grid-template-columns:55px 1fr 80px}}th,td{{padding:9px 5px}}}}@page{{size:A4;margin:10mm 11mm}}@media print{{body{{background:white;font-size:12px}}header{{padding:24px 18px;-webkit-print-color-adjust:exact;print-color-adjust:exact}}h1{{font-size:36px}}h2{{font-size:19px}}h3{{font-size:14px}}p{{line-height:1.35;margin:6px 0}}main{{padding:12px 0}}.section{{margin-top:17px;break-before:auto}}.panel,.card{{break-inside:avoid;box-shadow:none;padding:12px}}.cards{{gap:8px}}.card strong{{font-size:24px}}.card small{{font-size:11px}}.grid2{{gap:10px}}th,td{{padding:7px 6px}}.chart svg{{height:100px}}.steps li{{padding:5px 0 6px 40px}}.steps li::before{{top:2px}}.notice{{padding:10px 13px}}}}
</style></head><body>
<header><div class="inner"><div class="kicker">SK Kids Challenge · СПУТНИКС · исследовательский прототип</div>
<h1>CloudVision AI</h1><p class="big">Выделение облаков на спутниковых RGB-снимках и поддержка решения по порогу <strong>строго больше 75%</strong>.</p>
<span class="tag">Проверенные данные · без автоматического удаления</span>
<p>Отчёт составлен из сохранённых CSV испытаний. Нейросеть и исходные фотографии при создании отчёта не обрабатывались.</p></div></header>
<main><section class="section"><div class="eyebrow">01 / Важнейшие результаты</div><h2>Что уже измерено</h2>
<div class="cards"><div class="card"><span class="eyebrow">Независимый v5 TEST</span><strong>{n24}</strong><small>новых снимка из отдельных групп съёмки</small></div>
<div class="card"><span class="eyebrow">MAE v5 · TEST</span><strong>{f(min_new)} <span style="font-size:.9rem">п.п.</span></strong><small>v4: {f(min_base)} п.п.; разница {f(min_base-min_new)} п.п.</small></div>
<div class="card"><span class="eyebrow">Верные действия</span><strong>{correct5}/{n24}</strong><small>у v5 и {correct4}/{n24} у v4 по порогу 75%</small></div>
<div class="card"><span class="eyebrow">Эталонные REJECT</span><strong>{n_ref_reject}</strong><small>на новом независимом TEST; чувствительность не измерена</small></div></div></section>
<section class="section"><div class="eyebrow">02 / Сравнение v4 и v5</div><h2>Новый независимый TEST · {n24} снимка</h2>
<div class="grid2"><div class="panel"><h3>Средняя абсолютная ошибка облачности ↓</h3><p class="muted">Процентные пункты. Меньше — лучше.</p>
{bar("U-Net v4",min_base,"navy",suffix=" п.п.")}{bar("U-Net v5",min_new,"teal",suffix=" п.п.")}
<p class="muted">Разница небольшая; на этой выборке существенное преимущество v5 не установлено.</p></div>
<div class="panel"><h3>Контрольные метрики</h3><table><thead><tr><th>Показатель</th><th>v4</th><th>v5</th></tr></thead><tbody>
{metric_row("mean_iou","Средний IoU",digits=5)}
<tr><td>Решения по 75%</td><td>{correct4}/{n24}</td><td>{correct5}/{n24}</td></tr>
<tr><td>Ложные отсевы</td><td>{integer(s24['v4-policy'],'wrong_rejects')}</td><td>{integer(s24['v5-pilot'],'wrong_rejects')}</td></tr>
<tr><td>Пропущенные отсевы</td><td>{integer(s24['v4-policy'],'missed_rejects')}</td><td>{integer(s24['v5-pilot'],'missed_rejects')}</td></tr>
</tbody></table></div></div>
<div class="notice important" style="margin-top:16px"><strong>Ограничение независимого TEST:</strong> среди {n24} ручных эталонов нет кадров с облачностью строго выше 75%. Значение «0 пропущенных отсевов» не доказывает способность обнаруживать такие кадры. Простое правило «сохранять всё» дало бы {n24}/{n24} верных решений на этом наборе.</div></section>
<section class="section"><div class="eyebrow">03 / Где ошибается v5</div><h2>Самые трудные независимые снимки</h2>
<div class="panel"><table><thead><tr><th>Снимок</th><th>Ручная</th><th>v5</th><th>Ошибка</th></tr></thead><tbody>{errors_rows}</tbody></table>
<p class="muted">Сортировка по абсолютной ошибке доли облаков. Эти кадры относятся к независимому TEST и не должны использоваться для подбора модели.</p></div>
<div class="notice" style="margin-top:16px"><strong>Ложный отсев в новом TEST.</strong><ul>{false_rows}</ul>До промышленного применения требуется ручное подтверждение каждого потенциального отсева.</div></section>
<section class="section"><div class="eyebrow">04 / История модели</div><h2>Проверка v3 → v4 и дообучение v5</h2>
<div class="grid2"><div class="panel"><h3>Ранее независимый TEST · 5 снимков</h3><table><thead><tr><th>Метрика</th><th>v3</th><th>v4</th></tr></thead><tbody>
<tr><td>MAE, п.п.</td><td>{f(number(s5['v3-policy'],'mae_pp'))}</td><td>{f(number(s5['v4-policy-frozen'],'mae_pp'))}</td></tr>
<tr><td>IoU</td><td>{f(number(s5['v3-policy'],'mean_iou'),5)}</td><td>{f(number(s5['v4-policy-frozen'],'mean_iou'),5)}</td></tr>
<tr><td>Решения</td><td>{integer(s5['v3-policy'],'correct_actions')}/5</td><td>{integer(s5['v4-policy-frozen'],'correct_actions')}/5</td></tr>
</tbody></table><p class="muted">И этот старый независимый TEST также не содержал эталонных REJECT. Результаты пяти и 24 снимков показаны отдельно.</p></div>
<div class="panel"><h3>Последовательная VAL · 10 снимков</h3><table><thead><tr><th>Метрика</th><th>v4</th><th>v5</th></tr></thead><tbody>
<tr><td>MAE, п.п.</td><td>{f(number(val_by_model['v4-policy'],'val_mae_pp'),3)}</td><td>{f(number(val_by_model['v5-pilot'],'val_mae_pp'),3)}</td></tr>
<tr><td>IoU</td><td>{f(number(val_by_model['v4-policy'],'val_iou'),4)}</td><td>{f(number(val_by_model['v5-pilot'],'val_iou'),4)}</td></tr>
<tr><td>Выбранная эпоха</td><td>{integer(val_by_model['v4-policy'],'best_epoch')}</td><td>{best_epoch}</td></tr></tbody></table><p class="muted">VAL применялась для выбора checkpoint: эти данные не являются новой независимой оценкой v5.</p></div></div>
<div class="panel chart" style="margin-top:18px"><h3>Кривая обучения v5: MAE на VAL</h3>{sparkline(mae_history)}<p class="muted">В обучении пройдено {len(mae_history)} эпох. Хранится checkpoint эпохи {best_epoch}; дальнейшие эпохи могли давать хуже результаты. TEST не использовался для выбора эпохи.</p></div></section>
<section class="section"><div class="eyebrow">05 / Пилотный аудит</div><h2>30 кадров с ручной проверкой по подсказке AI</h2>
<div class="grid2"><div class="panel"><p><strong>{p['count']} проверенных масок</strong> · MAE {f(p['mae'])} п.п. · средний IoU {f(p['iou'],3)}.</p><p><strong>{p['correct']}/{p['count']}</strong> верных решений модели. По ручной разметке {p['reference_rejects']} кадра следовало отсеять, и все {p['missed']} были пропущены AI.</p><div class="notice">Эти ручные маски создавались с AI-подсказками, а часть пилотных сцен вошла в последующее обучение v5. Поэтому пилот <strong>не является независимым TEST v5</strong>.</div></div>
<div class="panel"><h3>Пропущенные отсеваемые кадры в пилоте v4</h3><table><thead><tr><th>Кадр</th><th>AI</th><th>Ручная</th></tr></thead><tbody>{pilot_rows}</tbody></table></div></div>
<div class="panel" style="margin-top:18px"><h3>Типичная ошибка: поверхность Земли как облака</h3><p class="muted">{esc(top_pilot['исходный_файл'])}: AI {f(number(top_pilot,'облачность_%'))}%, ручная {f(number(top_pilot,'ручная_облачность_%'))}%, ошибка {f(number(top_pilot,'расхождение_пп'))} п.п.</p>{preview_html if preview_html else '<p>Парное изображение для этой сцены не было предоставлено генератору; открой локальную операторскую галерею.</p>'}</div></section>
<section class="section"><div class="eyebrow">06 / Как работает система</div><h2>Сценарий демонстрации жюри</h2>
<div class="grid2"><div class="panel"><ol class="steps"><li>Загрузить спутниковый RGB-снимок; показать оригинал.</li><li>Выделить облака с помощью U-Net, показать бинарную и розовую маску.</li><li>Посчитать долю облачных пикселей по всему снимку.</li><li>Предложить <strong>«ОТСЕЯТЬ» только если облачность строго выше 75%</strong>; ровно 75% — сохранить.</li><li>Отобразить результат и статус ручной проверки, не удаляя исходник.</li></ol></div><div class="panel"><h3>Что можно заявить</h3><p>✓ Работает GPU-сегментация спутниковых изображений.</p><p>✓ Есть пакетная обработка с возобновлением и SQLite.</p><p>✓ Есть операторская HTML-галерея, исправление масок и сохранение истории решений.</p><p>✓ Веса моделей и независимые наборы проверок фиксируются.</p><div class="notice important"><strong>Что пока нельзя заявить:</strong> что автоматический отсев снимков с облачностью выше 75% надёжен без контроля человека.</div></div></div></section>
<section class="section"><div class="eyebrow">07 / Следующая проверка</div><h2>Отдельный набор сильно облачных кадров</h2><div class="panel"><p>Для измерения чувствительности правила >75% требуется новый набор снимков с подтверждённой высокой облачностью. Его следует собирать и размечать <strong>без подсказок AI</strong> и явно назвать «сложной целевой выборкой», а не представительной случайной проверкой.</p><p>Сохранённые v4 и v5 нельзя подстраивать под уже опубликованные независимые TEST. Снимки из нового сложного набора, если они применялись для отбора/корректировки, должны быть отделены от финальной оценки.</p></div></section>
<div class="footer">CloudVision AI · СПУТНИКС / SK Kids Challenge · Исследовательский отчёт для конкурсной демонстрации. Основа: сохранённые CSV независимых TEST (5 и {n24}), VAL (10), пилот (30), история обучения v5. Цифры не заменяют оценку на более широкой репрезентативной выборке.</div></main></body></html>'''
    destination = output.resolve()
    inputs = [v5_summary, v5_images, pilot_csv, validation_csv, history_csv, old_summary, old_images]
    if any(destination == src.resolve() for src in inputs):
        raise ValueError("HTML нельзя сохранять поверх исходных CSV")
    if destination.suffix.lower() != ".html":
        raise ValueError("--out должен завершаться на .html")
    if destination.is_file():
        raise FileExistsError(f"Отчёт уже существует: {destination}. Выбери другое --out, чтобы сохранить прежнюю версию.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        file.write(html_page)
    return {"test": n24, "pilot": p['count'], "pilot_missed": p['missed'], "mean_mae_v4": min_base, "mean_mae_v5": min_new, "html": destination}


def main() -> None:
    parser = argparse.ArgumentParser(description="Офлайн отчёт CloudVision AI для защиты: только готовые CSV")
    parser.add_argument("--test-summary", type=Path, default=ROOT / "reports/v5/final_holdout/summary.csv")
    parser.add_argument("--test-images", type=Path, default=ROOT / "reports/v5/final_holdout/per_image.csv")
    parser.add_argument("--old-test-summary", type=Path, default=ROOT / "reports/final_test_v4/summary.csv")
    parser.add_argument("--old-test-images", type=Path, default=ROOT / "reports/final_test_v4/per_image.csv")
    parser.add_argument("--pilot", type=Path, default=ROOT / "reports/competition_pilot_v4/operator_review/operator_review.csv")
    parser.add_argument("--validation", type=Path, default=ROOT / "reports/training_v5/model_comparison.csv")
    parser.add_argument("--history", type=Path, default=ROOT / "reports/training_v5/training_history.csv")
    parser.add_argument("--previews", type=Path, default=ROOT / "reports/competition_pilot_v4/operator_review/previews")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/competition_final/cloudvision_report.html")
    args = parser.parse_args()
    try:
        summary = build_report(args.test_summary, args.test_images, args.pilot,
                               args.validation, args.history, args.old_test_summary,
                               args.old_test_images, args.out, args.previews)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(f"Отчёт создан: {summary['html']}")
    print(f"Независимый TEST {summary['test']} кадров | MAE v4={summary['mean_mae_v4']:.2f}, v5={summary['mean_mae_v5']:.2f} п.п.")
    print(f"Пилот {summary['pilot']} кадров | пропущенных отсевов {summary['pilot_missed']}")
    print("Ни один исходный снимок, маска, checkpoint или CSV не изменён.")


if __name__ == "__main__":
    main()
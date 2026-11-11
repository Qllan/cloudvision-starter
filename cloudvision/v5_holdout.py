"""Freeze two v4/v5 models BEFORE manual annotation of a NEW v5 holdout.

The manifest is read as TEXT only. TEST photographs are not opened by freeze
or hash verification. Reports cannot be silently overwritten.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cloudvision.v4_holdout import file_sha256, test_manifest_names


def freeze_v5(manifest: Path, v4: Path, v5: Path, path: Path) -> dict:
    names = test_manifest_names(manifest)
    if len(names) < 5:
        raise ValueError("Нужно хотя бы 5 новых независимых TEST")
    if not v4.is_file() or not v5.is_file():
        raise FileNotFoundError("Не найдены обе модели v4 и v5")
    if v4.resolve() == v5.resolve():
        raise ValueError("Контрольные точки v4 и v5 должны различаться")
    if path.exists() or (path.parent / "summary.csv").exists():
        raise FileExistsError("Эта независимая проверка уже зафиксирована")
    result = {
        "schema": 1,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "first comparison v4 vs v5 on fresh blind holdout",
        "manifest_sha256": file_sha256(manifest),
        "test_count": len(names),
        "v4": {"name": v4.name, "sha256": file_sha256(v4), "bytes": v4.stat().st_size},
        "v5": {"name": v5.name, "sha256": file_sha256(v5), "bytes": v5.stat().st_size},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return result


def verify_v5_freeze(manifest: Path, v4: Path, v5: Path, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("Нет frozen_models.json: сначала --freeze после выбора v5")
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if result.get("schema") != 1:
        raise ValueError("Неподдерживаемая схема зафиксированных checkpoint")
    if result["manifest_sha256"] != file_sha256(manifest):
        raise ValueError("Новый TEST CSV изменился после фиксации")
    if result["test_count"] != len(test_manifest_names(manifest)):
        raise ValueError("Число TEST изменилось")
    for key, model in (("v4", v4), ("v5", v5)):
        old = result[key]
        if old["name"] != model.name or old["sha256"] != file_sha256(model) or old["bytes"] != model.stat().st_size:
            raise ValueError(f"Checkpoint {key} изменился после фиксации: нельзя оценивать")
    return result

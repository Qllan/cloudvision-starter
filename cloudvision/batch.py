"""Crash-resumable, preview-only inference over a large image folder.

The module does not load a network or modify source images. The prediction
callable is injected so it is independently testable without a GPU.
"""
from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from cloudvision.rules import should_delete

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    source_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    model_id TEXT NOT NULL,
    source_size INTEGER NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    mask_path TEXT NOT NULL,
    cloud_percent REAL NOT NULL,
    decision TEXT NOT NULL,
    processed_utc TEXT NOT NULL,
    PRIMARY KEY (source_id, relative_path)
);
CREATE TABLE IF NOT EXISTS failures (
    source_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    error TEXT NOT NULL,
    PRIMARY KEY (source_id, relative_path)
);
"""


def discover_images(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Нет папки изображений: {root}")
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in EXTENSIONS:
            continue
        # Do not walk through manually linked subdirectories.
        if any(part.is_symlink() for part in path.parents if part != root):
            continue
        files.append(path)
    return sorted(files, key=lambda path: str(path.relative_to(root)).casefold())


def model_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    version = "v4" if "_v4_" in path.stem.casefold() else "v3"
    return f"unet-policy-{version}:" + digest.hexdigest()


def _save_binary_mask(mask: Image.Image, destination: Path, expected_size: tuple[int, int]) -> float:
    if mask.size != expected_size or mask.mode != "L":
        raise ValueError("Маска должна быть L, в исходном разрешении")
    a = np.asarray(mask, dtype=np.uint8)
    if not a.size or not np.isin(a, (0, 255)).all():
        raise ValueError("Маска должна содержать только 0 и 255")
    percent = float(np.count_nonzero(a)) * 100.0 / a.size
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".png", prefix=".pending-", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
        mask.save(tmp_path, format="PNG")
        os.replace(tmp_path, destination)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return percent


def run_batch(
    input_dir: Path,
    output_dir: Path,
    model_id: str,
    predict: Callable[[Image.Image], Image.Image],
    limit: int = 0,
    log: Callable[[str], None] = print,
    manifest: Path | None = None,
    splits: set[str] | None = None,
) -> dict[str, int]:
    """Process one image at a time; save SQLite progress after each success.

    Images are never copied, moved or deleted. Reprocess when the source or
    checkpoint changes, or if the saved mask has disappeared.
    """
    if limit < 0:
        raise ValueError("limit должен быть >= 0")
    source = input_dir.resolve()
    output = output_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Нет папки изображений: {source}")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Папки входных снимков и результатов не должны пересекаться")
    if splits is not None and manifest is None:
        raise ValueError("Укажи --manifest для выбора train/val/test")
    if manifest is not None:
        from cloudvision.selection import read_manifest
        files = read_manifest(manifest, source, splits)
    else:
        files = discover_images(source)
    if limit:
        files = files[:limit]
    output.mkdir(parents=True, exist_ok=True)
    source_id = hashlib.sha256(os.fsencode(str(source).casefold())).hexdigest()
    stats = {"found": len(files), "processed": 0, "skipped": 0, "failed": 0}
    # sqlite3.Connection.__exit__ commits or rolls back but does NOT close the file.
    # Windows locks progress.sqlite3 until the connection is explicitly closed.
    with closing(sqlite3.connect(output / "progress.sqlite3")) as db:
        db.executescript(SCHEMA)
        registered = db.execute("SELECT value FROM metadata WHERE key='source_id'").fetchone()
        if registered and registered[0] != source_id:
            raise ValueError("Эта папка отчётов уже используется для другого датасета. Укажи другой --out")
        if not registered:
            db.execute("INSERT INTO metadata VALUES ('source_id', ?)", (source_id,))
            db.commit()
        for index, path in enumerate(files, 1):
            relative = path.relative_to(source)
            rel = relative.as_posix()
            stat = path.stat()
            mask_rel = Path("masks") / relative.parent / (path.stem + "_policy_mask.png")
            mask_dest = output / mask_rel
            cached = db.execute(
                """SELECT model_id, source_size, source_mtime_ns, mask_path
                   FROM images WHERE source_id=? AND relative_path=?""",
                (source_id, rel),
            ).fetchone()
            if (
                cached is not None
                and cached == (model_id, stat.st_size, stat.st_mtime_ns, mask_rel.as_posix())
                and mask_dest.is_file()
            ):
                stats["skipped"] += 1
                continue
            try:
                with Image.open(path) as file:
                    original = ImageOps.exif_transpose(file).convert("RGB")
                mask = predict(original)
                percent = _save_binary_mask(mask, mask_dest, original.size)
                action = "ОТСЕЯТЬ" if should_delete(percent) else "СОХРАНИТЬ"
                db.execute(
                    """INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (source_id, rel, model_id, stat.st_size, stat.st_mtime_ns,
                     mask_rel.as_posix(), percent, action,
                     datetime.now(timezone.utc).isoformat()),
                )
                db.execute("DELETE FROM failures WHERE source_id=? AND relative_path=?", (source_id, rel))
                db.commit()
                stats["processed"] += 1
                log(f"[{index}/{len(files)}] {rel}: {percent:.2f}% — {action}")
            except (OSError, ValueError, RuntimeError) as error:
                stats["failed"] += 1
                db.execute(
                    "INSERT OR REPLACE INTO failures VALUES (?, ?, ?)",
                    (source_id, rel, f"{type(error).__name__}: {error}"),
                )
                db.commit()
                log(f"[{index}/{len(files)}] ОШИБКА {rel}: {error}")
        export_report(db, output / "operator_preview.csv", source_id, model_id)
    return stats


def export_report(db: sqlite3.Connection, path: Path, source_id: str, model_id: str) -> None:
    rows = db.execute(
        """SELECT relative_path, cloud_percent, decision, mask_path
           FROM images WHERE source_id=? AND model_id=? ORDER BY relative_path""",
        (source_id, model_id),
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["исходный_файл", "облачность_%", "решение_по_75%", "маска", "фактическое_действие"])
        for name, percent, decision, mask in rows:
            writer.writerow([name, f"{percent:.3f}", decision, mask, "ТОЛЬКО ПРЕДПРОСМОТР"])

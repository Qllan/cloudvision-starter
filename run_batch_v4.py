"""CloudVision AI v4: CUDA-capable, resumable, preview-only inference.

Examples:
  .\\.venv\\Scripts\\python.exe run_batch_v4.py --input data/new_raw --limit 5
  .\\.venv\\Scripts\\python.exe run_batch_v4.py --input data/new_raw --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cloudvision.batch import model_fingerprint, run_batch
from run_competition_demo import load_checkpoint, predict_mask

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudVision AI — массовый предпросмотр на GPU")
    parser.add_argument("--input", type=Path, default=ROOT / "data/input")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/batch_v4")
    parser.add_argument("--model", type=Path, default=ROOT / "models/cloudvision_unet_v3_policy.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=0, help="0 — все найденные снимки")
    parser.add_argument("--manifest", type=Path, default=None, help="CSV выборки от select_v4_samples.py")
    parser.add_argument("--splits", choices=("train,val", "train", "val", "test", "all"), default="all",
                        help="Для разметки обычно train,val — TEST оставляем независимым")
    args = parser.parse_args()
    if args.splits != "all" and args.manifest is None:
        parser.error("Для --splits нужно указать --manifest")
    requested_splits = None if args.splits == "all" else set(args.splits.split(","))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна в текущем PyTorch")
    device_name = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    try:
        model, size, checkpoint = load_checkpoint(args.model, "policy", device)
        fingerprint = model_fingerprint(args.model)
        print(f"CloudVision AI v4 | {device.type.upper()} | эпоха {checkpoint.get('best_epoch', '?')}")
        print("Режим только предпросмотра: исходники не копируются и не удаляются.")
        print("Снимки обрабатываются по одному; готовые маски повторно не вычисляются.")
        stats = run_batch(
            input_dir=args.input,
            output_dir=args.out,
            model_id=fingerprint,
            predict=lambda image: predict_mask(image, model, size, device),
            limit=args.limit,
            manifest=args.manifest,
            splits=requested_splits,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Найдено: {stats['found']}; новых: {stats['processed']}; "
          f"пропущено готовых: {stats['skipped']}; ошибок: {stats['failed']}")
    print("Отчёт:", args.out / "operator_preview.csv")
    print("Маски:", args.out / "masks")
    print("Прогресс:", args.out / "progress.sqlite3")
    print("Оригиналы не изменены.")


if __name__ == "__main__":
    main()

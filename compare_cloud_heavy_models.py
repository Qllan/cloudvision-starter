"""Diagnose one already-evaluated cloud-heavy TEST frame using frozen v4/v5.

Creates exactly one read-only, four-panel RGB/manual/v4/v5 overlay JPG.
Both checkpoints and the target challenge manifest are SHA-256 verified
against the EXISTING frozen_models.json before model inference. This does
NOT modify original photos, ground truth, checkpoints, or TEST CSVs.

Diagnostic visualization only, NOT a new evaluation/model-selection run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

from cloudvision.diagnostic_panels import render_four_panels
from cloudvision.selection import read_manifest
from cloudvision.v5_holdout import verify_v5_freeze

ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = "2025-11-30_22-18-23_SXC3-2334_1"


def select_case(manifest: Path, root: Path, labels: Path, query: str) -> tuple[Path, Path]:
    """Select only one image already listed in the frozen challenge manifest."""
    if not query.strip():
        raise ValueError("--query must not be empty")
    chosen = read_manifest(manifest, root, {"test"})
    matching = [path for path in chosen if query.strip().casefold() in path.name.casefold()]
    if not matching:
        raise FileNotFoundError("No image matched --query in the frozen challenge manifest")
    if len(matching) != 1:
        names = "\n".join(f"  {path.name}" for path in matching[:15])
        raise ValueError(f"Ambiguous --query; provide more of the filename:\n{names}")
    source_root, labels_root = root.resolve(), labels.resolve()
    if labels_root == source_root or labels_root.is_relative_to(source_root):
        raise ValueError("Human masks must be stored outside the source photos")
    image = matching[0]
    mask = labels_root / image.relative_to(source_root).with_suffix(".png")
    if not mask.is_file() or mask.is_symlink() or not mask.resolve().is_relative_to(labels_root):
        raise FileNotFoundError(f"Human verified PNG mask missing or unsafe: {mask}")
    return image, mask


def run(args: argparse.Namespace) -> Path:
    source, human_mask = select_case(args.manifest, args.input, args.labels, args.query)
    out_folder = args.out.resolve()
    frozen_dir = args.frozen.resolve().parent
    protected = [args.input.resolve(), args.labels.resolve(), args.v4.resolve().parent,
                 args.v5.resolve().parent, frozen_dir]
    if any(out_folder == path or out_folder.is_relative_to(path) or path.is_relative_to(out_folder)
           for path in protected):
        raise ValueError("Diagnostic output must be separate from originals, labels, weights and TEST evaluation")
    dest = out_folder / (source.stem + "_v4_v5_comparison.jpg")
    if dest.exists():
        raise FileExistsError(f"Diagnostic JPG already exists; use a different --out: {dest}")

    # IMPORTANT: verify the models and the exact challenge manifest before predictions.
    frozen = verify_v5_freeze(args.manifest, args.v4, args.v5, args.frozen)
    if frozen["test_count"] < 1:
        raise ValueError("Frozen challenge manifest is empty")

    # Torch is imported only when actually diagnosing; light unit tests need no CUDA.
    import torch
    from run_competition_demo import load_checkpoint, predict_mask

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    with Image.open(source) as file:
        image = ImageOps.exif_transpose(file).convert("RGB")
    with Image.open(human_mask) as file:
        manual = file.copy()

    net_v4, size_v4, info_v4 = load_checkpoint(args.v4, "policy", device)
    if info_v4.get("version") != 4:
        raise ValueError("Unexpected v4 checkpoint version")
    net_v5, size_v5, info_v5 = load_checkpoint(args.v5, "policy", device)
    if info_v5.get("version") != 5:
        raise ValueError("Unexpected v5 checkpoint version")
    pred_v4 = predict_mask(image, net_v4, size_v4, device)
    pred_v5 = predict_mask(image, net_v5, size_v5, device)
    picture, percentages = render_four_panels(
        image, manual, pred_v4, pred_v5, panel_width=args.panel_width)

    out_folder.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as file:
        file.write(picture)
    print(f"Case: {source.name}")
    print(f"Human cloud coverage: {percentages['human']:.3f}%")
    print(f"Frozen U-Net v4:      {percentages['v4']:.3f}%")
    print(f"Frozen U-Net v5:      {percentages['v5']:.3f}%")
    print("Exact >75% means REJECT; 75.000% means KEEP.")
    print("Diagnostic image:", dest)
    print("No labels, TEST results, checkpoints or source photos were changed.")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-panel v4/v5 frozen cloud-heavy case diagnostic")
    parser.add_argument("--query", default=DEFAULT_TARGET)
    parser.add_argument("--input", type=Path, default=ROOT / "data/new_raw")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/cloud_heavy_verified")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "reports/v5/cloud_heavy_challenge/challenge_manifest.csv")
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "reports/v5/cloud_heavy_challenge/evaluation/frozen_models.json")
    parser.add_argument("--v4", type=Path, default=ROOT / "models/cloudvision_unet_v4_policy.pt")
    parser.add_argument("--v5", type=Path, default=ROOT / "models/cloudvision_unet_v5_pilot.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/cloud_heavy_case_review")
    parser.add_argument("--panel-width", type=int, default=430)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

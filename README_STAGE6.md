# CloudVision AI — Stage 6: visible color masks

The `_tiled_mask.png` files are **class indices**, not a photograph. They store values 0, 1, 2, 3, so ordinary viewers show them almost black. This is expected.

## Windows 11 setup

1. Copy `review_tiling.py` next to `run_ai.py` in your existing project.
2. In VS Code's terminal, run:
   ```powershell
   .\.venv\Scripts\python.exe review_tiling.py
   ```
3. Open `reports/tiling_experiment/color_previews/`.

Each output contains: original, colored class map, and overlay. Colors are *only visual aids*: dark grey = clear; red = thick cloud; yellow = thin cloud; blue = shadow. Only thick and thin clouds count toward percentage. This script does **not** change the model, its predictions, the reports, or the source files.

## Research status

The tiled approach improves mean absolute cloud cover error from 23.53 to 13.07 percentage points across the three labeled samples. Mean IoU increases from 0.1841 to 0.2210. Three samples are too few for a quality claim; collect more accurate whole-area labels, including images exceeding 75% cloud cover, and evaluate on a held-out set.

Keep the 75% deletion action disabled until the model is validated. Test it first on duplicates or in a quarantine folder.

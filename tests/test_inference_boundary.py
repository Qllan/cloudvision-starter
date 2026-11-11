"""Keep the frozen U-Net inference implementation out of CLI scripts.

These dependency-boundary tests run without PyTorch, CUDA or source photos.
They guard a design rule: evaluation and batch processing must not import the
demo entry point, which contains a separate, explicitly guarded copy-delete
mode. Runtime evaluation is still covered by historical frozen experiments.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class InferenceBoundaryTests(unittest.TestCase):
    def test_shared_inference_is_the_single_definition(self):
        module = ast.parse((ROOT / "cloudvision/inference.py").read_text(encoding="utf-8"))
        names = {n.name for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue({"load_checkpoint", "predict_mask"}.issubset(names))
        # The shared inference layer must have no source-deleting or copying code.
        source = (ROOT / "cloudvision/inference.py").read_text(encoding="utf-8")
        for unsafe in (".unlink(", ".rmtree(", "shutil.", ".copy2("):
            with self.subTest(unsafe=unsafe):
                self.assertNotIn(unsafe, source)

    def test_cli_and_evaluators_import_from_the_side_effect_free_module(self):
        users = (
            "run_competition_demo.py",
            "run_batch_v4.py",
            "evaluate_v4_holdout.py",
            "evaluate_v5_holdout.py",
            "compare_cloud_heavy_models.py",
        )
        for rel in users:
            with self.subTest(rel=rel):
                tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
                inference_imports = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "cloudvision.inference"
                ]
                imported_names = {n.name for stmt in inference_imports for n in stmt.names}
                self.assertTrue({"load_checkpoint", "predict_mask"}.issubset(imported_names))
                self.assertFalse(any(
                    isinstance(node, ast.ImportFrom)
                    and node.module == "run_competition_demo"
                    for node in ast.walk(tree)
                ))
        # Historical import contracts stay available through the demo module.
        demo = ast.parse((ROOT / "run_competition_demo.py").read_text(encoding="utf-8"))
        self.assertFalse(any(
            isinstance(node, ast.FunctionDef)
            and node.name in {"load_checkpoint", "predict_mask"}
            for node in demo.body
        ))


if __name__ == "__main__":
    unittest.main()

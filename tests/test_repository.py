"""Repository integrity: layout, imports, and that quoted numbers trace to report files."""
import importlib
import json
import pkgutil
import py_compile
import re
import unittest

from tests._helpers import REPO_ROOT

REQUIRED_DIRS = [
    "src/models", "src/training", "src/inference", "src/preprocessing", "src/explainability",
    "src/utils", "configs", "data/raw", "data/processed_v4", "data/metadata", "checkpoints/v4",
    "checkpoints/calibration", "reports/figures", "reports/tables", "reports/ablations",
    "reports/confusion_matrices", "reports/datasets", "docs/thesis", "docs/viva",
    "docs/architecture", "demo/streamlit", "demo/cli", "demo/samples", "demo/outputs",
    "scripts", "tests", "archive",
]
ACTIVE_CODE = ["src", "scripts", "demo", "tests"]


class LayoutTests(unittest.TestCase):
    def test_required_directories_exist(self):
        missing = [d for d in REQUIRED_DIRS if not (REPO_ROOT / d).is_dir()]
        self.assertEqual(missing, [])

    def test_default_config_resolves(self):
        from src.utils.config import DEFAULT_PATH
        self.assertTrue(DEFAULT_PATH.exists(), DEFAULT_PATH)


class CodeTests(unittest.TestCase):
    def test_every_src_module_imports(self):
        import src
        for m in pkgutil.walk_packages(src.__path__, "src."):
            importlib.import_module(m.name)

    def test_active_code_compiles(self):
        for folder in ACTIVE_CODE:
            for py in (REPO_ROOT / folder).rglob("*.py"):
                py_compile.compile(str(py), doraise=True)

    def test_no_imports_of_the_pre_refactor_package(self):
        pattern = re.compile(r"^\s*(from|import)\s+fraggen\b|-m\s+fraggen\.", re.M)
        offenders = [str(p.relative_to(REPO_ROOT)) for folder in ACTIVE_CODE
                     for p in (REPO_ROOT / folder).rglob("*.py")
                     if pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])


class ResultsTraceabilityTests(unittest.TestCase):
    def test_v4_headline_matches_the_driver_summary(self):
        from src.utils import results
        m = results.v4_metric("A_plain", "test", "accuracy")
        self.assertIsNotNone(m)
        self.assertEqual(m.n, 3)
        rows = json.loads((results.V4_DIR / "driver_summary_A_plain.json").read_text())["rows"]
        expected = sum(r["cross_acc"] for r in rows) / len(rows)
        self.assertAlmostEqual(m.mean, expected, places=12)

    def test_cross_corpus_result_is_preserved(self):
        from src.utils import results
        cc = results.cross_corpus_4096("A_plain")
        self.assertIsNotNone(cc)
        self.assertEqual(round(cc.mean, 4), 0.6812)


if __name__ == "__main__":
    unittest.main()

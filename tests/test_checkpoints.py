"""Integration tests against the real trained checkpoints (skipped if absent).

These run on CPU so they never compete with a training job for GPU memory.
"""
import json
import os
import unittest

from tests._helpers import REPO_ROOT

CKPT_DIR = REPO_ROOT / "checkpoints" / "v4"
SAMPLES = REPO_ROOT / "demo" / "samples"
HAVE_MODEL = (CKPT_DIR / "A_plain_s1337.pt").exists()


@unittest.skipUnless(HAVE_MODEL, "checkpoints/v4/A_plain_s1337.pt not present")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("FRAGGEN_DEVICE", "cpu")
        from src.inference import FragGenClassifier
        cls.clf = FragGenClassifier(device="cpu")

    def test_every_v4_checkpoint_loads_with_the_v4_label_space(self):
        from src.models.checkpoint import load_model
        import torch
        ckpts = sorted(CKPT_DIR.glob("*.pt"))
        self.assertGreaterEqual(len(ckpts), 3)
        for p in ckpts:
            _, cfg = load_model(str(p), torch.device("cpu"))
            self.assertEqual([str(c) for c in cfg.classes],
                             ["txt", "csv", "html", "xml", "pdf", "jpg", "png", "gif", "zip"], p.name)

    def test_calibration_is_loaded(self):
        self.assertTrue(self.clf.calibrated)
        cal = json.loads((REPO_ROOT / "checkpoints/calibration/A_plain_s1337.calib.json").read_text())
        self.assertAlmostEqual(self.clf.temperature, cal["temperature"])
        self.assertEqual(self.clf.block_size, 4096)

    def test_bundled_samples_behave_as_documented(self):
        """The demo relies on these outcomes; this guards the bundled set, it is
        not an accuracy measurement."""
        for ext in ["pdf", "jpg", "png", "gif", "txt", "html", "xml", "csv", "zip"]:
            v = self.clf.classify_file(SAMPLES / f"sample.{ext}")
            self.assertEqual(v.final_prediction, ext, f"sample.{ext}")
        tricky = self.clf.classify_file(SAMPLES / "tricky_pdf_container_example.pdf")
        self.assertEqual(tricky.final_prediction, "zip")
        self.assertGreater(tricky.vote_counts["pdf"], 0)


if __name__ == "__main__":
    unittest.main()

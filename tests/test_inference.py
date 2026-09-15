"""Unit tests for src.inference.FragGenClassifier (no checkpoint required)."""
import unittest

import numpy as np

from tests._helpers import CLASSES, blocks, stub_classifier


class FragmentationTests(unittest.TestCase):
    def test_header_block_is_skipped_and_offsets_are_absolute(self):
        clf = stub_classifier()
        frags, start = clf.fragment_bytes(blocks([0, 1, 2, 3]))
        self.assertEqual(start, 16)
        self.assertEqual(frags.shape, (4, 16))
        v = clf.classify_bytes(blocks([0, 1, 2, 3]))
        self.assertEqual([f.byte_offset for f in v.fragments], [16, 32, 48, 64])
        self.assertEqual([f.index for f in v.fragments], [1, 2, 3, 4])

    def test_keep_header_scores_block_zero(self):
        v = stub_classifier().classify_bytes(blocks([0, 1], header=False), skip_header=False)
        self.assertEqual(v.n_fragments, 2)
        self.assertEqual(v.start_offset, 0)

    def test_input_without_a_post_header_block_is_too_small(self):
        clf = stub_classifier()
        v = clf.classify_bytes(b"x" * 31)          # header + 15 bytes: nothing to score
        self.assertTrue(v.too_small)
        self.assertEqual(v.final_prediction, "unknown")
        # regression: an input of exactly two blocks scores ONE fragment (offset 16),
        # it must never fall back to classifying the header block
        v = clf.classify_bytes(b"x" * 32)
        self.assertEqual(v.n_fragments, 1)
        self.assertEqual(v.fragments[0].byte_offset, 16)

    def test_trailing_partial_block_is_ignored(self):
        v = stub_classifier().classify_bytes(blocks([4, 4]) + b"\x08" * 10)
        self.assertEqual(v.n_fragments, 2)

    def test_base_offset_and_max_fragments(self):
        v = stub_classifier().classify_bytes(blocks([0, 1, 2]), base_offset=1000, max_fragments=2)
        self.assertEqual(v.n_fragments, 2)
        self.assertEqual(v.fragments[0].byte_offset, 1016)


class VotingTests(unittest.TestCase):
    def test_majority_vote(self):
        v = stub_classifier().classify_bytes(blocks([4, 4, 8]))     # pdf, pdf, zip
        self.assertEqual(v.final_prediction, "pdf")
        self.assertAlmostEqual(v.final_confidence, 2 / 3)
        self.assertEqual(v.vote_counts["zip"], 1)
        self.assertEqual(sum(v.vote_counts.values()), 3)

    def test_ties_go_to_the_earliest_class_in_model_order(self):
        v = stub_classifier().classify_bytes(blocks([8, 4]))        # zip, pdf
        self.assertEqual(v.final_prediction, "pdf")                 # pdf precedes zip

    def test_temperature_changes_confidence_but_never_the_class(self):
        data = blocks([0, 3, 5, 8, 8])
        raw = stub_classifier(temperature=1.0).classify_bytes(data)
        hot = stub_classifier(temperature=3.7).classify_bytes(data)
        self.assertEqual([f.predicted_class for f in raw.fragments],
                         [f.predicted_class for f in hot.fragments])
        self.assertTrue(all(h.confidence < r.confidence
                            for r, h in zip(raw.fragments, hot.fragments)))
        self.assertTrue(hot.calibrated)

    def test_probability_vectors_are_distributions_in_class_order(self):
        v = stub_classifier().classify_bytes(blocks([6]))
        p = np.array(v.fragments[0].probabilities)
        self.assertEqual(len(p), len(CLASSES))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertEqual(int(p.argmax()), CLASSES.index("png"))


if __name__ == "__main__":
    unittest.main()

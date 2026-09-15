"""Tests for the JSON/CSV report format and the PDF stream mapper."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.explainability import (FRAGMENT_FIELDS, dominant_region, explain_verdict,
                                fragment_region_shares, is_pdf, pdf_stream_regions, save_csv,
                                save_json, verdict_to_csv, verdict_to_dict)
from tests._helpers import CLASSES, blocks, stub_classifier


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.v = stub_classifier().classify_bytes(blocks([4, 4, 8, 5]), name="x.pdf")

    def test_json_has_every_documented_fragment_field(self):
        d = verdict_to_dict(self.v)
        self.assertEqual(len(d["fragments"]), 4)
        for rec in d["fragments"]:
            self.assertEqual(list(rec), FRAGMENT_FIELDS)
            self.assertEqual(list(rec["probability_vector"]), CLASSES)
        self.assertEqual(d["verdict"]["final_prediction"], "pdf")
        self.assertEqual(d["fragmentation"]["first_fragment_offset"], 16)

    def test_csv_header_and_rows(self):
        rows = list(csv.reader(io.StringIO(verdict_to_csv(self.v))))
        self.assertEqual(rows[0][:len(FRAGMENT_FIELDS)], FRAGMENT_FIELDS)
        self.assertEqual(rows[0][len(FRAGMENT_FIELDS):], [f"p_{c}" for c in CLASSES])
        self.assertEqual(len(rows), 1 + 4)
        self.assertEqual(json.loads(rows[1][4]).keys(), set(CLASSES))

    def test_files_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = save_json(self.v, Path(tmp) / "a" / "r.json")
            c = save_csv(self.v, Path(tmp) / "a" / "r.csv")
            self.assertEqual(json.loads(j.read_text())["file"]["name"], "x.pdf")
            self.assertTrue(c.read_text().startswith("fragment_index,"))

    def test_notes_flag_pdf_container_lookalikes(self):
        notes = explain_verdict(self.v)        # 2 pdf, 1 zip, 1 jpg -> 50% lookalikes
        self.assertTrue(any("FlateDecode" in n for n in notes))


class PdfStreamMapTests(unittest.TestCase):
    def _pdf(self):
        head = b"%PDF-1.4\n"
        obj1 = b"1 0 obj\n<< /Length 64 /Filter /FlateDecode >>\nstream\n"
        pay1 = b"\x11" * 64
        obj2 = b"\nendstream\nendobj\n2 0 obj\n<< /Filter [/FlateDecode /DCTDecode] >>\nstream\r\n"
        pay2 = b"\xff" * 32
        tail = b"\r\nendstream\nendobj\n%%EOF\n"
        data = head + obj1 + pay1 + obj2 + pay2 + tail
        s1 = len(head + obj1)
        s2 = len(head + obj1 + pay1 + obj2)
        return data, (s1, s1 + 64), (s2, s2 + 32)

    def test_regions_and_filters(self):
        data, r1, r2 = self._pdf()
        self.assertTrue(is_pdf(data))
        regions = pdf_stream_regions(data)
        self.assertEqual([r.label for r in regions], ["FlateDecode", "DCTDecode"])
        # payload boundaries exclude the EOL before "endstream" (\n and \r\n cases)
        self.assertEqual((regions[0].start, regions[0].end), r1)
        self.assertEqual((regions[1].start, regions[1].end), r2)

    def test_fragment_shares_sum_to_one(self):
        data, r1, _ = self._pdf()
        regions = pdf_stream_regions(data)
        shares = fragment_region_shares(regions, start_offset=0, block_size=32,
                                        n_fragments=len(data) // 32)
        for s in shares:
            self.assertAlmostEqual(sum(s.values()), 1.0)
        inside = r1[0] // 32 + 1               # a block fully inside the Flate payload
        self.assertEqual(dominant_region(shares[inside]), "FlateDecode")

    def test_non_pdf(self):
        self.assertFalse(is_pdf(b"PK\x03\x04..."))
        self.assertEqual(pdf_stream_regions(b"no streams here"), [])


if __name__ == "__main__":
    unittest.main()

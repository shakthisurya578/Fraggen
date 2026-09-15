"""Explainable outputs: JSON/CSV reports, verdict notes, and PDF stream mapping."""
from .report import (FRAGMENT_FIELDS, REPORT_SCHEMA, explain_verdict, fragment_records,
                     save_csv, save_json, verdict_to_csv, verdict_to_dict, verdict_to_json)
from .container_analysis import (REGION_LABELS, StreamRegion, dominant_region,
                                 fragment_region_shares, is_pdf, pdf_stream_regions)

__all__ = [
    "FRAGMENT_FIELDS", "REPORT_SCHEMA", "explain_verdict", "fragment_records", "save_csv",
    "save_json", "verdict_to_csv", "verdict_to_dict", "verdict_to_json",
    "REGION_LABELS", "StreamRegion", "dominant_region", "fragment_region_shares", "is_pdf",
    "pdf_stream_regions",
]

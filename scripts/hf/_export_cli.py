#!/usr/bin/env python3
"""Shared export CLI pieces for the Hub export scripts.

``export_calamari_onnx.py`` and ``export_ppocr_rec_onnx.py`` take the same
``--destination`` / ``--opset`` / ``--report-json`` flags and finish the same
way (print the provenance report, optionally also write it as JSON). This
module owns those flags and that finish so the two scripts cannot drift
apart. It is a library, not a script: no ``main``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Matches the exporter's own default. It is surfaced as a flag because the
# opset is part of what an artifact is, and a re-export at a different opset
# should be a deliberate, recorded choice rather than a silent consequence of
# upgrading Torch.
DEFAULT_OPSET_VERSION = 17


def add_destination_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --destination/--opset/--report-json, identical across export scripts."""
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Output .onnx path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help=f"ONNX opset version (default: {DEFAULT_OPSET_VERSION})",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Also write the printed report to this path as JSON",
    )


def write_report_json(report_json: Path | None, payload: dict[str, object]) -> None:
    """Print the provenance report, and also write it to --report-json."""
    import json

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

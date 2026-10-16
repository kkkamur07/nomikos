#!/usr/bin/env python3
"""Export a kraken PP-OCRv6 recognition ``.safetensors`` checkpoint to ``model.onnx``.

This is the step between training (which writes the kraken safetensors file)
and publishing. The artifact this prints the provenance of is what the runtime
opens; the checkpoint travels beside it as the export input.

The report is the operator's half of the registry entry:

* ``source_sha256`` identifies the checkpoint the artifact was traced from,
* ``artifact_sha256`` is what goes into ``registry.yaml`` as ``artifact_sha256``
  and is what the runtime verifies before opening the graph,
* ``classes`` / ``variant`` / ``line_height`` describe the model the artifact
  actually contains, read out of the checkpoint rather than restated elsewhere.

Example::

    PYTHONPATH=. python scripts/hf/export_ppocr_rec_onnx.py \\
        --checkpoint ppocr-syriac.safetensors \\
        --destination var/export/syriac-ppocr-v1/model.onnx \\
        --report-json var/export/syriac-ppocr-v1/export-report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.hf._export_cli import add_destination_arguments, write_report_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local kraken PP-OCRv6 recognition checkpoint (.safetensors)",
    )
    add_destination_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from nomikos_inference.export.ppocr_rec import export_ppocr_rec_onnx

    report = export_ppocr_rec_onnx(
        args.checkpoint,
        args.destination,
        opset_version=args.opset,
    )
    payload = report.to_dict()
    write_report_json(args.report_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

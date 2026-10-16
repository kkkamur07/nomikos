#!/usr/bin/env python3
"""Prove a PP-OCRv6 recognition ONNX artifact reproduces its kraken Torch model.

Both graphs run at batch size 1 on the same preprocessed tensors: real Syriac
line images cut from PAGE-XML pages with kraken's own parser, line extractor
and input transforms, plus a synthetic width sweep of seeded uniform noise.
The gate is exact agreement where it matters and a tight numeric bound where
it does not; anything else exits 2.

Inputs come from ``kraken.models.ctc`` territory on purpose. Each PAGE-XML is
parsed with kraken's ``XMLPage`` and each line is cut with kraken's
``extract_polygons`` on the ``baselines`` path, then run through
``ImageInputTransforms(1, height, 0, 3, (pad, 0))`` exactly as
``CTCRecognitionInferenceMixin._recognition_pred`` builds it. Two data quirks
are handled openly rather than worked around silently (both are counted in the
report):

* kraken 7.1.1 ``parse_page_custom`` rejects the platform's ``custom``
  attribute format (``source:kraken; kind:polygon`` has no ``tag{...}`` chunk),
  so ``custom="..."`` attributes are stripped from a copy held in memory.
  Geometry and text are untouched.
* Some pages store the baseline polyline in the boundary slot, which is not a
  polygon. Lines with a degenerate boundary fall back to kraken's own
  ``BaselineLine.to_bbox`` box extraction (the ``_recognize_box_lines`` family).
  Lines are never invented: a line neither path can cut is skipped and counted.

Example::

    PYTHONPATH=. python scripts/hf/verify_ppocr_rec_parity.py \\
        --checkpoint ppocr-syriac.safetensors \\
        --onnx var/export/syriac-ppocr-v1/model.onnx \\
        --pages-dir data/dataset/chapter4/pages \\
        --xml-dir data/dataset/chapter4/xml \\
        --lines tests/fixtures/manuscripts/syriac/transcribe_line.jpg \\
        --report-json var/export/syriac-ppocr-v1/parity-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.hf._ppocr_rec_inputs import collect_file_crops, collect_page_crops
from scripts.hf._ppocr_rec_parity_lib import (
    _cer_figure,
    _Evaluation,
    _ParityRow,
    _quirk_tables,
    _run_rows,
    _summarise,
)

# Uniform-noise widths. They straddle the backbone's stride boundaries (odd
# widths exercise the SAME-padding branches) and run from width 8 to a
# panorama wider than any manuscript line.
DEFAULT_SYNTHETIC_WIDTHS = (8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 320, 777, 1000, 2000, 3000, 4000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Reference .safetensors")
    parser.add_argument("--onnx", type=Path, required=True, help="Exported model.onnx")
    parser.add_argument("--pages-dir", type=Path, help="Directory of page images")
    parser.add_argument("--xml-dir", type=Path, help="Directory of PAGE-XML segmentations")
    parser.add_argument(
        "--lines",
        type=Path,
        nargs="*",
        default=[],
        help="Extra line image files, used as plain images",
    )
    parser.add_argument(
        "--synthetic-widths",
        type=int,
        nargs="*",
        default=None,
        help=f"Noise sweep widths (default: {list(DEFAULT_SYNTHETIC_WIDTHS)})",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Cap on real lines (XML plus --lines), for quick iterations",
    )
    parser.add_argument("--report-json", type=Path, help="Write the full report here as JSON")
    return parser.parse_args()


def _build_real_inputs(pages_dir, xml_dir, line_files, *, height, pad, max_lines):
    """Cut every line with kraken's own extractor; returns inputs and counters."""
    from kraken.lib.dataset import ImageInputTransforms

    records, counters = collect_page_crops(pages_dir, xml_dir, max_lines=max_lines)
    records.extend(collect_file_crops(line_files, already=len(records), max_lines=max_lines))
    inputs = []
    for record in records:
        transforms = ImageInputTransforms(1, height, 0, 3, (pad, 0), record.valid_norm)
        tensor = transforms(record.crop).numpy()[None]
        inputs.append({"name": record.name, "tensor": tensor, "truth": record.truth})
    return inputs, counters


def _load_reference(checkpoint: Path):
    """Read inference defaults and the one kraken model out of the checkpoint."""
    from kraken.configs.base import RecognitionInferenceConfig
    from kraken.models import load_models

    # Inference defaults, read out of kraken rather than restated: pad 16 is
    # the blank margin on each side of a line (configs/base.py), temperature
    # 1.0 leaves the softmax unchanged.
    inference_config = RecognitionInferenceConfig()
    pad = int(inference_config.padding)
    temperature = float(inference_config.temperature)

    models = load_models(str(checkpoint))
    if len(models) != 1:
        print(f"expected one model in {checkpoint}", file=sys.stderr)
        return None
    model = models[0]
    model.eval()
    height = int(model.input[2])
    num_classes = int(model.num_classes)
    labels: dict[int, str] = {}
    for grapheme, seq in model.codec.c2l.items():
        if len(seq) == 1:
            labels[seq[0]] = grapheme
    return model, height, pad, temperature, num_classes, labels


def _build_evaluation_inputs(args, height: int, pad: int):
    """Real kraken-cut lines plus the uniform-noise width sweep."""
    real_inputs, counters = _build_real_inputs(
        args.pages_dir,
        args.xml_dir,
        args.lines,
        height=height,
        pad=pad,
        max_lines=args.max_lines,
    )
    widths = (
        tuple(args.synthetic_widths)
        if args.synthetic_widths is not None
        else DEFAULT_SYNTHETIC_WIDTHS
    )
    rng = np.random.default_rng(0)
    synthetic = [
        {
            "name": f"synthetic-w{width}",
            "tensor": rng.random((1, 3, height, width), dtype=np.float32),
            "truth": None,
        }
        for width in widths
    ]
    return real_inputs, counters, synthetic


def _print_report(
    summary: dict[str, object],
    real_rows: list[_ParityRow],
    quirk_changed: list[dict[str, object]],
    args,
    display,
    rows: list[_ParityRow],
) -> None:
    """Print the summary, per-line samples and quirk rows; write --report-json."""
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    shown = 0
    for row in real_rows:
        if shown >= 5 or not row["truth"]:
            continue
        print(f"--- {row['name']}")
        print(f"  torch: {row['torch_text']}")
        print(f"  onnx : {row['onnx_text']}")
        print(f"  truth: {display(row['truth'])}")
        shown += 1

    for entry in quirk_changed:
        print(f"--- mask quirk {entry['name']}")
        print(f"  unmasked: {entry['unmasked_text']}")
        print(f"  masked  : {entry['masked_text']}")
        print(f"  truth   : {entry['truth_display']}")
        if "closer_to_truth" in entry:
            print(
                f"  levenshtein unmasked={entry['levenshtein_unmasked']} "
                f"masked={entry['levenshtein_masked']} "
                f"closer={entry['closer_to_truth']}"
            )

    if args.report_json is not None:
        payload = dict(summary)
        payload["per_line"] = rows
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main() -> int:
    args = _parse_args()

    import onnxruntime as ort
    import torch
    from kraken.lib.bidi import get_display

    from scripts.hf._ppocr_rec_parity_lib import _open_sessions

    reference = _load_reference(args.checkpoint)
    if reference is None:
        return 1
    model, height, pad, temperature, num_classes, labels = reference
    real_inputs, counters, synthetic = _build_evaluation_inputs(args, height, pad)
    if not real_inputs:
        print("no real line inputs were built", file=sys.stderr)
        return 1
    sessions = _open_sessions(args.onnx)
    rows, mask = _run_rows(
        real_inputs,
        synthetic,
        model=model,
        sessions=sessions,
        labels=labels,
        temperature=temperature,
        num_classes=num_classes,
    )
    real_rows = rows[: len(real_inputs)]
    synthetic_rows = rows[len(real_inputs) :]
    evaluation = _Evaluation(
        rows=rows,
        real_rows=real_rows,
        synthetic_rows=synthetic_rows,
        counters=counters,
        num_classes=num_classes,
        mask=mask,
    )
    cer = _cer_figure(real_rows, get_display)
    quirk_lines, quirk_violations, quirk_changed = _quirk_tables(real_rows, mask, get_display)
    summary, passed = _summarise(
        evaluation,
        checkpoint=str(args.checkpoint),
        onnx=str(args.onnx),
        ort_version=ort.__version__,
        torch_version=torch.__version__,
        cer=cer,
        quirk_lines=quirk_lines,
        quirk_violations=quirk_violations,
        quirk_changed=quirk_changed,
    )
    _print_report(summary, real_rows, quirk_changed, args, get_display, rows)
    # A decoded-string mismatch is the failure ADR 0006 was written about, so
    # it must be an exit status, not a line in a report nobody reads.
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

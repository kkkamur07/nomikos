#!/usr/bin/env python3
"""Prove the PP-OCR recognition serving adapter reproduces kraken's inference.

For every Chapter4 line: cut the line with kraken's own XML parser and line
extractor (bounding-box fallback where the XML has no polygons), encode the
crop as PNG bytes, and compare the adapter against kraken on the same pixels:

(a) the adapter's preprocessed tensor against kraken's
    ``ImageInputTransforms`` output: must be bitwise identical for every line;
(b) the adapter's final text against kraken's own result for the same line
    tensor at batch 1: the kraken Torch model run without ``seq_lens``,
    greedy-decoded with kraken's codec, converted to logical order with
    kraken's own ``BaselineOCRRecord.logical_order``: identical for all lines;
(c) for information, kraken's masked result (``seq_lens=[W]``) and how often
    it differs: a known kraken quirk the export doc explains, not a gate.

Exits 2 unless (a) and (b) hold on every processed line. Run with the kraken
venv (it owns torch and kraken) and ``PYTHONPATH`` pointing at the worktree so
the adapter under test is this checkout::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. $KPY scripts/hf/verify_ppocr_rec_adapter.py \\
        --checkpoint $MAIN/ppocr-syriac.safetensors \\
        --onnx $ONNX --pages-dir $MAIN/data/dataset/chapter4/pages \\
        --xml-dir $MAIN/data/dataset/chapter4/xml \\
        --lines $MAIN/tests/fixtures/manuscripts/syriac/transcribe_line.jpg \\
        --report-json $OUT/adapter-parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.hf._ppocr_rec_inputs import collect_file_crops, collect_page_crops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Reference .safetensors")
    parser.add_argument("--onnx", type=Path, required=True, help="Adapter model.onnx")
    parser.add_argument("--pages-dir", type=Path, help="Directory of page images")
    parser.add_argument("--xml-dir", type=Path, help="Directory of PAGE-XML segmentations")
    parser.add_argument("--lines", type=Path, nargs="*", default=[], help="Extra line images")
    parser.add_argument("--max-lines", type=int, default=None, help="Cap on real lines")
    parser.add_argument("--report-json", type=Path, help="Write the full report here as JSON")
    return parser.parse_args()


def _build_real_inputs(pages_dir, xml_dir, line_files, *, max_lines):
    """Cut every line with kraken's own extractor; returns crops and counters."""
    records, counters = collect_page_crops(pages_dir, xml_dir, max_lines=max_lines)
    records.extend(
        collect_file_crops(line_files, already=len(records), max_lines=max_lines)
    )
    inputs = []
    for record in records:
        output = BytesIO()
        record.crop.save(output, format="PNG")
        inputs.append(
            {
                "name": record.name,
                "crop": record.crop.copy(),
                "png": output.getvalue(),
                "truth": record.truth,
            }
        )
    return inputs, counters


def _kraken_reference(model, tensor, *, temperature, decoder):
    """Kraken's own decode of one line tensor, without ``seq_lens``.

    Mirrors ``TorchSeqRecognizer.forward`` plus the baseline recognition path:
    temperature softmax, greedy decode, codec to codepoints, then kraken's own
    ``BaselineOCRRecord.logical_order``. The record carries no geometry (only
    the frame cuts kraken itself emits): geometry shapes polygons, never text,
    and the conversion under test is the bidi reorder.
    """
    import torch
    from kraken.containers import BaselineOCRRecord

    with torch.inference_mode():
        logits, olens = model(torch.from_numpy(tensor))
    probs = (logits / temperature).softmax(1)
    outputs = probs.detach().squeeze(2)
    decoded = model.codec.decode(decoder(outputs, olens)[0])
    display = "".join(char for char, _, _, _ in decoded)
    cuts = [[start, end] for _, start, end, _ in decoded]
    confidences = [conf for _, _, _, conf in decoded]
    record = BaselineOCRRecord(
        prediction=display,
        cuts=cuts,
        confidences=confidences,
        line={"type": "baselines", "id": "verify"},
        display_order=True,
    )
    return record.logical_order().prediction


def _kraken_masked_text(model, tensor, width):
    """Kraken's decode with ``seq_lens=[W]``: the known quirk, info only."""
    import torch

    from scripts.hf._ppocr_rec_inputs import greedy_decode

    with torch.inference_mode():
        full, _ = model(torch.from_numpy(tensor), torch.tensor([width]))
    logits = full.numpy()[0, :, 0, :].T
    labels = {}
    for grapheme, seq in model.codec.c2l.items():
        if len(seq) == 1:
            labels[seq[0]] = grapheme
    parts = greedy_decode(logits, labels)
    from kraken.lib.bidi import get_display

    return get_display(parts, None)


def main() -> int:
    args = _parse_args()

    from kraken.configs.base import RecognitionInferenceConfig
    from kraken.lib.ctc_decoder import greedy_decoder
    from kraken.lib.dataset import ImageInputTransforms
    from kraken.models import load_models

    from nomikos_inference.architectures.ppocr_rec.adapter import (
        run_ppocr_rec_transcribe,
    )
    from nomikos_inference.architectures.ppocr_rec.preprocessing import (
        preprocess_line_image_bytes_to_ppocr_rec_tensor,
    )

    inference_config = RecognitionInferenceConfig()
    pad = int(inference_config.padding)
    temperature = float(inference_config.temperature)

    models = load_models(str(args.checkpoint))
    if len(models) != 1:
        print(f"expected one model in {args.checkpoint}", file=sys.stderr)
        return 1
    model = models[0]
    model.eval()
    height = int(model.input[2])

    inputs, counters = _build_real_inputs(
        args.pages_dir,
        args.xml_dir,
        args.lines,
        max_lines=args.max_lines,
    )
    if not inputs:
        print("no real line inputs were built", file=sys.stderr)
        return 1

    rows = []
    tensor_identical = 0
    text_identical = 0
    masked_differing = []
    for entry in inputs:
        transforms = ImageInputTransforms(1, height, 0, 3, (pad, 0), False)
        reference = transforms(entry["crop"]).numpy()[None]
        candidate = preprocess_line_image_bytes_to_ppocr_rec_tensor(
            entry["png"], line_height=height, pad=pad, pad_fill=255
        )
        same_tensor = bool(np.array_equal(candidate, reference))
        tensor_identical += int(same_tensor)

        width = int(reference.shape[3])
        kraken_text = _kraken_reference(
            model, reference, temperature=temperature, decoder=greedy_decoder
        )
        adapter_text = run_ppocr_rec_transcribe(entry["png"], checkpoint_path=args.onnx).text
        same_text = kraken_text == adapter_text
        text_identical += int(same_text)

        masked_text = _kraken_masked_text(model, reference, width)
        if masked_text != kraken_text:
            masked_differing.append(entry["name"])

        rows.append(
            {
                "name": entry["name"],
                "width": width,
                "tensor_identical": same_tensor,
                "adapter_text": adapter_text,
                "kraken_text": kraken_text,
                "text_identical": same_text,
                "masked_text": masked_text,
                "truth": entry.get("truth"),
            }
        )

    total = len(rows)
    gate = (
        total > 0
        and counters["skipped_lines"] == 0
        and tensor_identical == total
        and text_identical == total
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.onnx),
        "counters": counters,
        "lines": total,
        "tensor_identical": tensor_identical,
        "text_identical": text_identical,
        "masked_differing": masked_differing,
        "gate": gate,
        "rows": rows,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"lines {total}, preprocessing bitwise identical {tensor_identical}, "
        f"text identical {text_identical}, "
        f"kraken masked-quirk differing {len(masked_differing)}"
    )
    for row in rows[:5]:
        print(f"--- {row['name']}")
        print(f"  adapter: {row['adapter_text']}")
        print(f"  kraken:  {row['kraken_text']}")
        print(f"  truth:   {row['truth']}")
    mismatches = [
        row["name"] for row in rows if not (row["tensor_identical"] and row["text_identical"])
    ]
    if mismatches:
        print(f"mismatched lines ({len(mismatches)}): {mismatches[:20]}", file=sys.stderr)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())

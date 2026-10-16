#!/usr/bin/env python3
"""Stages of the PP-OCR recognition parity gate (library for verify_ppocr_rec_parity.py).

``verify_ppocr_rec_parity.py`` stays a thin entry script (flags, input
building, orchestration); everything that turns model outputs into the JSON
report lives here so the entry stays under the file-length gate and each
stage is small enough to test. Pure stages (``_levenshtein``,
``_greedy_decode``, ``_softmax``, ``_cer_figure``, ``_quirk_tables``,
``_summarise``) take plain data and need no weights; the run stage
(``_open_sessions``, ``_score_entry``, ``_run_rows``) needs torch,
onnxruntime and kraken, imported function-locally as in the entry.
"""

from __future__ import annotations

import dataclasses
from typing import TypedDict

import numpy as np

# Numeric gates. Typical agreement is ~1e-6 relative on logits of magnitude
# tens; the bounds sit an order of magnitude above the worst observed values
# (logits 1.56e-3, softmax 2.67e-4) so a real regression trips them while
# float kernel noise does not. The decisions have their own 100 percent gates.
MAX_ABS_LOGIT_DIFF = 5e-3
MAX_ABS_SOFTMAX_DIFF = 1e-3


def _levenshtein(a: str, b: str) -> int:
    """Plain codepoint edit distance; the CER sanity figure needs nothing more."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _greedy_decode(logits, labels):
    """Collapse repeats, drop blank 0, map through the codec (display order)."""
    best = logits.argmax(1)
    parts = []
    previous = -1
    for label in best:
        label = int(label)
        if label == 0:
            previous = 0
            continue
        if label != previous:
            parts.append(labels.get(label, ""))
        previous = label
    return "".join(parts)


def _softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class _ParityRow(TypedDict):
    """One scored line: torch vs ONNX logits, decodes and mask-quirk facts."""

    name: str
    width: int
    torch_shape: list[int]
    onnx_shape: list[int]
    bitwise_equal: bool
    max_abs_diff: float
    mean_abs_diff: float
    softmax_max_abs_diff: float
    argmax_agreement: float
    torch_text: str
    onnx_text: str
    masked_text: str | None
    masked_max_abs_diff: float | None
    masked_out_lens: int | None
    text_identical: bool
    truth: str | None
    noopt_bitwise_equal: bool
    single_thread_bitwise_equal: bool


@dataclasses.dataclass
class _MaskProbe:
    """The kraken masking quirk, measured but kept out of the gate."""

    worst: float = 0.0
    nonzero: list[str] = dataclasses.field(default_factory=list)
    changed: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Evaluation:
    """Everything _summarise needs beyond flags, versions and quirk tables."""

    rows: list[_ParityRow]
    real_rows: list[_ParityRow]
    synthetic_rows: list[_ParityRow]
    counters: dict[str, int]
    num_classes: int
    mask: _MaskProbe


def _open_sessions(onnx):
    """The model under the three session configs the gate compares."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    narrow = ort.SessionOptions()
    narrow.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_noopt = ort.InferenceSession(
        str(onnx), sess_options=narrow, providers=["CPUExecutionProvider"]
    )
    single = ort.SessionOptions()
    single.intra_op_num_threads = 1
    session_single = ort.InferenceSession(
        str(onnx), sess_options=single, providers=["CPUExecutionProvider"]
    )
    return session, session_noopt, session_single


def _score_entry(
    entry,
    *,
    position: int,
    real_count: int,
    model,
    sessions,
    labels,
    temperature: float,
    num_classes: int,
    mask: _MaskProbe,
) -> _ParityRow:
    """Run torch and ONNX on one input; returns its report row."""
    import torch

    session, session_noopt, session_single = sessions
    tensor = entry["tensor"]
    width = int(tensor.shape[3])
    masked_logits = None
    masked_out_lens = None
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(tensor))[0].numpy()[0, :, 0, :].T
        if position < real_count:
            masked_full, masked_lens = model(torch.from_numpy(tensor), torch.tensor([width]))
            masked_logits = masked_full.numpy()[0, :, 0, :].T
            if masked_lens is not None:
                masked_out_lens = int(masked_lens.item())
            mask_diff = float(np.abs(torch_logits - masked_logits).max())
            mask.worst = max(mask.worst, mask_diff)
            if mask_diff != 0.0:
                mask.nonzero.append(entry["name"])
    onnx_raw = np.asarray(session.run(["logits"], {"image": tensor})[0])
    onnx_logits = onnx_raw[0]
    torch_f64 = torch_logits.astype(np.float64)
    onnx_f64 = onnx_logits.astype(np.float64)
    difference = np.abs(torch_f64 - onnx_f64)
    torch_labels = torch_f64.argmax(axis=-1)
    onnx_labels = onnx_f64.argmax(axis=-1)
    torch_text = _greedy_decode(torch_f64, labels)
    onnx_text = _greedy_decode(onnx_f64, labels)
    masked_text = None
    if masked_logits is not None:
        masked_text = _greedy_decode(masked_logits.astype(np.float64), labels)
        if masked_text != torch_text:
            mask.changed.append(entry["name"])
    torch_softmax = _softmax(torch_f64 / temperature)
    onnx_softmax = _softmax(onnx_f64 / temperature)
    return _ParityRow(
        name=entry["name"],
        width=width,
        torch_shape=[1, int(torch_logits.shape[0]), num_classes],
        onnx_shape=[int(v) for v in onnx_raw.shape],
        bitwise_equal=bool(np.array_equal(torch_logits, onnx_logits)),
        max_abs_diff=float(difference.max()),
        mean_abs_diff=float(difference.mean()),
        softmax_max_abs_diff=float(np.abs(torch_softmax - onnx_softmax).max()),
        argmax_agreement=float(np.mean(torch_labels == onnx_labels)),
        torch_text=torch_text,
        onnx_text=onnx_text,
        masked_text=masked_text,
        masked_max_abs_diff=(
            float(np.abs(torch_logits - masked_logits).max())
            if masked_logits is not None
            else None
        ),
        masked_out_lens=masked_out_lens,
        text_identical=torch_text == onnx_text,
        truth=entry["truth"],
        noopt_bitwise_equal=bool(
            np.array_equal(
                torch_logits,
                np.asarray(session_noopt.run(["logits"], {"image": tensor})[0])[0],
            )
        ),
        single_thread_bitwise_equal=bool(
            np.array_equal(
                torch_logits,
                np.asarray(session_single.run(["logits"], {"image": tensor})[0])[0],
            )
        ),
    )


def _run_rows(
    real_inputs,
    synthetic,
    *,
    model,
    sessions,
    labels,
    temperature: float,
    num_classes: int,
) -> tuple[list[_ParityRow], _MaskProbe]:
    """Score every real and synthetic input; returns rows and the mask probe."""
    mask = _MaskProbe()
    rows = [
        _score_entry(
            entry,
            position=position,
            real_count=len(real_inputs),
            model=model,
            sessions=sessions,
            labels=labels,
            temperature=temperature,
            num_classes=num_classes,
            mask=mask,
        )
        for position, entry in enumerate(real_inputs + synthetic)
    ]
    return rows, mask


def _cer_figure(real_rows: list[_ParityRow], display) -> float | None:
    """Torch display-order decode vs display-order ground truth; info only."""
    cer_dist = cer_len = 0
    for row in real_rows:
        if row["truth"]:
            hypothesis = row["torch_text"]
            reference = display(row["truth"])
            cer_dist += _levenshtein(hypothesis, reference)
            cer_len += len(reference)
    return cer_dist / cer_len if cer_len else None


def _quirk_changed_entries(
    real_rows: list[_ParityRow], changed: list[str], display
) -> list[dict[str, object]]:
    """Per-line mask-quirk rows for lines whose decode changed; info only."""
    quirk_changed: list[dict[str, object]] = []
    for name in changed:
        row = next(row for row in real_rows if row["name"] == name)
        reference = display(row["truth"]) if row["truth"] else None
        masked_text = row["masked_text"]
        entry: dict[str, object] = {
            "name": name,
            "unmasked_text": row["torch_text"],
            "masked_text": masked_text,
            "truth_display": reference,
        }
        if reference is not None and masked_text is not None:
            lev_unmasked = _levenshtein(row["torch_text"], reference)
            lev_masked = _levenshtein(masked_text, reference)
            entry["levenshtein_unmasked"] = lev_unmasked
            entry["levenshtein_masked"] = lev_masked
            entry["closer_to_truth"] = "unmasked" if lev_unmasked <= lev_masked else "masked"
        quirk_changed.append(entry)
    return quirk_changed


def _quirk_tables(
    real_rows: list[_ParityRow], mask: _MaskProbe, display
) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    """Explain the kraken masking quirk line by line; info only, out of the gate."""
    # The masking quirk is kraken's, not the export's: _lengths_and_mask
    # (kraken/lib/ppocr/network.py) scales widths in float32, so floor can land
    # one frame short and the last frame is masked off. The exported graph
    # serves the unmasked result by design, so this is measured, explained and
    # kept out of the gate. Every nonzero line must satisfy out_lens == T - 1;
    # a line that breaks the explanation is an open question, not a silent row.
    quirk_lines: list[dict[str, object]] = []
    quirk_violations: list[str] = []
    for row in real_rows:
        if row["name"] not in mask.nonzero:
            continue
        time_steps = row["torch_shape"][1]
        out_lens = row["masked_out_lens"]
        quirk_lines.append(
            {
                "name": row["name"],
                "W": row["width"],
                "T_torch": time_steps,
                "out_lens": out_lens,
            }
        )
        if out_lens != time_steps - 1:
            quirk_violations.append(row["name"])
    quirk_changed = _quirk_changed_entries(real_rows, mask.changed, display)
    return quirk_lines, quirk_violations, quirk_changed


@dataclasses.dataclass
class _MismatchFigures:
    """Names and worst-case figures the gate reads; no thresholds here."""

    real_mismatched: list[str]
    real_argmax_bad: list[str]
    shape_bad: list[str]
    worst_all: float
    worst_real: float
    worst_softmax_all: float
    worst_softmax_real: float


def _mismatch_figures(
    rows: list[_ParityRow], real_rows: list[_ParityRow]
) -> _MismatchFigures:
    """Collect the per-line mismatches and worst-case diffs for the gate."""
    return _MismatchFigures(
        real_mismatched=[row["name"] for row in real_rows if not row["text_identical"]],
        real_argmax_bad=[row["name"] for row in real_rows if row["argmax_agreement"] != 1.0],
        shape_bad=[row["name"] for row in rows if row["onnx_shape"] != row["torch_shape"]],
        worst_all=max(row["max_abs_diff"] for row in rows),
        worst_real=max(row["max_abs_diff"] for row in real_rows),
        worst_softmax_all=max(row["softmax_max_abs_diff"] for row in rows),
        worst_softmax_real=max(row["softmax_max_abs_diff"] for row in real_rows),
    )


def _summarise(
    evaluation: _Evaluation,
    *,
    checkpoint: str,
    onnx: str,
    ort_version: str,
    torch_version: str,
    cer: float | None,
    quirk_lines: list[dict[str, object]],
    quirk_violations: list[str],
    quirk_changed: list[dict[str, object]],
) -> tuple[dict[str, object], bool]:
    """Apply the pass gate and assemble the JSON summary, in gate order."""
    real_rows = evaluation.real_rows
    figures = _mismatch_figures(evaluation.rows, real_rows)
    real_mismatched = figures.real_mismatched
    real_argmax_bad = figures.real_argmax_bad
    shape_bad = figures.shape_bad
    worst_all = figures.worst_all
    worst_real = figures.worst_real
    worst_softmax_all = figures.worst_softmax_all
    worst_softmax_real = figures.worst_softmax_real
    gate: dict[str, bool] = {
        "all_real_decodes_identical": not real_mismatched,
        "all_real_argmax_agree": not real_argmax_bad,
        "max_abs_diff_within_bound": worst_all <= MAX_ABS_LOGIT_DIFF,
        "softmax_within_bound": worst_softmax_all <= MAX_ABS_SOFTMAX_DIFF,
        "all_shapes_match_torch": not shape_bad,
    }
    passed = all(gate.values())
    summary: dict[str, object] = {
        "checkpoint": checkpoint,
        "onnx": onnx,
        "classes": evaluation.num_classes,
        "ort_version": ort_version,
        "torch_version": torch_version,
        "counters": evaluation.counters,
        "real_lines": len(real_rows),
        "synthetic_lines": len(evaluation.synthetic_rows),
        "bitwise_equal_real": sum(row["bitwise_equal"] for row in real_rows),
        "bitwise_equal_noopt_real": sum(row["noopt_bitwise_equal"] for row in real_rows),
        "bitwise_equal_single_thread_real": sum(
            row["single_thread_bitwise_equal"] for row in real_rows
        ),
        "worst_max_abs_diff_real": worst_real,
        "worst_max_abs_diff_all": worst_all,
        "worst_softmax_max_abs_diff_real": worst_softmax_real,
        "worst_softmax_max_abs_diff_all": worst_softmax_all,
        "mean_of_mean_abs_diff_real": float(np.mean([row["mean_abs_diff"] for row in real_rows])),
        "min_argmax_agreement_real": min(row["argmax_agreement"] for row in real_rows),
        "decoded_identical_real": len(real_rows) - len(real_mismatched),
        "decoded_mismatched_real": real_mismatched,
        "argmax_mismatched_real": real_argmax_bad,
        "shape_mismatched": shape_bad,
        "kraken_mask_quirk": {
            "max_abs_diff": evaluation.mask.worst,
            "lines": quirk_lines,
            "out_lens_always_T_minus_1": not quirk_violations,
            "explanation_violations": quirk_violations,
            "decode_changed": quirk_changed,
        },
        "torch_cer_vs_xml": cer,
        "gate": gate,
        "gate_passed": passed,
    }
    return summary, passed

"""Behaviour tests for the operator scripts in scripts/ (refactor pins).

Every test here exercises a pure script helper on small synthetic inputs, so
the complexity and file-length refactors in scripts/hf,
scripts/segmentation/ppocr and scripts/platform/seed cannot silently change
results. Nothing here needs weights, GPUs, paddle, kraken or a database.
"""

import hashlib
import json
import uuid

import numpy as np
import pytest
from PIL import Image as PILImage
from shapely.geometry import Polygon

from scripts.hf import _ppocr_rec_inputs as rec_inputs
from scripts.hf import _ppocr_rec_parity_lib as parity_lib
from scripts.platform.seed import _fixtures as seed_fixtures
from scripts.segmentation.ppocr import _fixtures as seg_fixtures
from scripts.segmentation.ppocr import evaluate_refinement, verify_adapter
from scripts.segmentation.ppocr.export_onnx import (
    _check_model_files,
    _require_dynamic_inputs,
    _tensor_dims,
)


class _FakeLine:
    def __init__(self, boundary, baseline):
        self.boundary = boundary
        self.baseline = baseline


def _parity_row(**overrides):
    row = {
        "name": "p#0",
        "width": 100,
        "torch_shape": [1, 25, 4],
        "onnx_shape": [1, 25, 4],
        "bitwise_equal": True,
        "max_abs_diff": 1e-6,
        "mean_abs_diff": 1e-7,
        "softmax_max_abs_diff": 1e-7,
        "argmax_agreement": 1.0,
        "torch_text": "abc",
        "onnx_text": "abc",
        "masked_text": None,
        "masked_max_abs_diff": None,
        "masked_out_lens": None,
        "text_identical": True,
        "truth": "abc",
        "noopt_bitwise_equal": True,
        "single_thread_bitwise_equal": True,
    }
    row.update(overrides)
    return row


def _evaluation(rows, real_rows):
    return parity_lib._Evaluation(
        rows=rows,
        real_rows=real_rows,
        synthetic_rows=rows[len(real_rows) :],
        counters={"pages": 1},
        num_classes=4,
        mask=parity_lib._MaskProbe(),
    )


# ---------------------------------------------------------------------------
# rec parity lib: pure decodes and figures
# ---------------------------------------------------------------------------


def test_levenshtein_known_distances():
    assert parity_lib._levenshtein("kitten", "sitting") == 3
    assert parity_lib._levenshtein("", "abc") == 3
    assert parity_lib._levenshtein("abc", "abc") == 0


def test_greedy_decode_collapses_repeats_and_blank():
    logits = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.9]])
    assert rec_inputs.greedy_decode(logits, {1: "a", 2: "b"}) == "ab"


def test_softmax_rows_sum_to_one():
    out = parity_lib._softmax(np.array([[1.0, 2.0, 3.0]]))
    assert abs(float(out.sum()) - 1.0) < 1e-12
    assert out[0, 2] > out[0, 1] > out[0, 0]


def test_cer_figure_perfect_and_empty():
    rows = [_parity_row(torch_text="abc", truth="abc")]
    assert parity_lib._cer_figure(rows, lambda text: text) == 0.0
    assert parity_lib._cer_figure([_parity_row(truth=None)], lambda text: text) is None


def test_quirk_tables_flag_t_minus_1_violation():
    mask = parity_lib._MaskProbe(worst=0.5, nonzero=["p#0"], changed=["p#0"])
    rows = [
        _parity_row(
            name="p#0",
            torch_shape=[1, 25, 4],
            masked_out_lens=20,
            masked_text="abd",
            torch_text="abc",
            truth="abc",
        )
    ]
    lines, violations, changed = parity_lib._quirk_tables(rows, mask, lambda text: text)
    assert lines == [{"name": "p#0", "W": 100, "T_torch": 25, "out_lens": 20}]
    assert violations == ["p#0"]
    assert changed[0]["closer_to_truth"] == "unmasked"
    assert changed[0]["levenshtein_unmasked"] == 0
    assert changed[0]["levenshtein_masked"] == 1


def test_summarise_gate_pass_and_decode_mismatch():
    good = _parity_row()
    summary, passed = parity_lib._summarise(
        _evaluation([good], [good]),
        checkpoint="c",
        onnx="o",
        ort_version="1",
        torch_version="2",
        cer=0.0,
        quirk_lines=[],
        quirk_violations=[],
        quirk_changed=[],
    )
    assert passed
    assert summary["gate"]["all_real_decodes_identical"] is True

    bad = _parity_row(name="p#1", onnx_text="abd", text_identical=False)
    summary, passed = parity_lib._summarise(
        _evaluation([good, bad], [good, bad]),
        checkpoint="c",
        onnx="o",
        ort_version="1",
        torch_version="2",
        cer=0.0,
        quirk_lines=[],
        quirk_violations=[],
        quirk_changed=[],
    )
    assert not passed
    assert summary["decoded_mismatched_real"] == ["p#1"]


def test_mismatch_figures_worst_case():
    rows = [
        _parity_row(name="p#0", max_abs_diff=1e-3, softmax_max_abs_diff=2e-4),
        _parity_row(name="p#1", max_abs_diff=4e-3, softmax_max_abs_diff=1e-4),
    ]
    figures = parity_lib._mismatch_figures(rows, rows)
    assert figures.worst_all == 4e-3
    assert figures.worst_softmax_all == 2e-4
    assert figures.real_mismatched == []


# ---------------------------------------------------------------------------
# rec shared inputs: cleaning, boundary, lookup
# ---------------------------------------------------------------------------


def test_strip_custom_attributes():
    assert (
        rec_inputs.strip_custom_attributes('<TextLine custom="a b">x</TextLine>')
        == "<TextLine>x</TextLine>"
    )


def test_has_real_boundary_cases():
    assert rec_inputs.has_real_boundary(_FakeLine(None, [(0, 0)])) is False
    assert rec_inputs.has_real_boundary(_FakeLine([(0, 0)], [(0, 0)])) is False
    assert rec_inputs.has_real_boundary(_FakeLine([(0, 1)], [(0, 0)])) is True


def test_find_page_image_found_and_missing(tmp_path):
    (tmp_path / "p1.png").write_bytes(b"x")
    assert rec_inputs.find_page_image(tmp_path, "p1") == tmp_path / "p1.png"
    assert rec_inputs.find_page_image(tmp_path, "nope") is None


# ---------------------------------------------------------------------------
# segmentation fixtures: hashes, lookup, config validation
# ---------------------------------------------------------------------------


def test_sha256_helpers_match_hashlib(tmp_path):
    data = b"manuscript-bytes"
    assert seg_fixtures.sha256_bytes(data) == hashlib.sha256(data).hexdigest()
    path = tmp_path / "a.bin"
    path.write_bytes(data)
    assert seg_fixtures.sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_resolve_image_file_searches_both_roots(tmp_path):
    grec = tmp_path / "grec"
    grec.mkdir()
    (grec / "grec-p1.png").write_bytes(b"x")
    assert seg_fixtures.resolve_image_file("grec-p1", tmp_path, grec) == grec / "grec-p1.png"
    with pytest.raises(FileNotFoundError):
        seg_fixtures.resolve_image_file("missing", tmp_path, grec)


def test_resolve_verified_image_ok_mismatch_missing(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "c13.jpg").write_bytes(b"bytes")
    fixture = {"image": "c13.jpg", "image_sha256": hashlib.sha256(b"bytes").hexdigest()}
    assert seg_fixtures.resolve_verified_image("c13", fixture, images, tmp_path) == b"bytes"
    with pytest.raises(ValueError, match="does not match fixture"):
        seg_fixtures.resolve_verified_image(
            "c13", {"image": "c13.jpg", "image_sha256": "0" * 64}, images, tmp_path
        )
    with pytest.raises(FileNotFoundError):
        seg_fixtures.resolve_verified_image(
            "nope", {"image": "nope.jpg", "image_sha256": "0" * 64}, images, tmp_path
        )


def test_load_targets_care_split(tmp_path):
    doc = {
        "pairing": {"text_lines": [{"paired_line_id": "l1"}, {}]},
        "lines": [
            {"id": "l1", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            {"id": "l2", "points": [[2, 2], [3, 2], [3, 3], [2, 3]]},
        ],
    }
    (tmp_path / "p-latest.json").write_text(json.dumps(doc))
    targets = seg_fixtures.load_targets(tmp_path, "p")
    assert [t["care"] for t in targets] == [True, False]


def test_decode_overlay_canvas_rejects_garbage():
    with pytest.raises(ValueError, match="could not decode"):
        seg_fixtures.decode_overlay_canvas(b"not-an-image", "c13")


def test_config_float_int_validation():
    cfg = {"thresh": 0.2, "max_candidates": 1000}
    assert seg_fixtures.config_float(cfg, "thresh", 0.5) == 0.2
    assert seg_fixtures.config_float({}, "thresh", 0.5) == 0.5
    assert seg_fixtures.config_int(cfg, "max_candidates", 1) == 1000
    with pytest.raises(ValueError, match="unexpected postprocess config"):
        seg_fixtures.config_float({"thresh": "high"}, "thresh", 0.5)
    with pytest.raises(ValueError, match="unexpected postprocess config"):
        seg_fixtures.config_int({"max_candidates": True}, "max_candidates", 1)


# ---------------------------------------------------------------------------
# evaluate_refinement: matching, pairs, sweeps
# ---------------------------------------------------------------------------


def _square_targets():
    poly = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
    return [{"id": "l1", "polygon": poly, "care": True}]


def test_score_page_identical_empty_and_ignore():
    scored = evaluate_refinement._score_page(
        _square_targets(), [Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])]
    )
    assert (scored["tp"], scored["fp"], scored["fn"], scored["alone"]) == (1, 0, 0, 1)

    scored = evaluate_refinement._score_page(_square_targets(), [])
    assert (scored["tp"], scored["fp"], scored["fn"]) == (0, 0, 1)

    targets = _square_targets() + [
        {"id": "l2", "polygon": Polygon([(0, 40), (100, 40), (100, 60), (0, 60)]), "care": False}
    ]
    preds = [
        Polygon([(0, 0), (100, 0), (100, 20), (0, 20)]),
        Polygon([(0, 40), (100, 40), (100, 60), (0, 60)]),
    ]
    scored = evaluate_refinement._score_page(targets, preds)
    assert (scored["tp"], scored["fp"], scored["ignored"]) == (1, 0, 1)


def test_pairs_above_separate_and_overlapping():
    separate = [
        Polygon([(0, 0), (100, 0), (100, 20), (0, 20)]),
        Polygon([(0, 40), (100, 40), (100, 60), (0, 60)]),
    ]
    assert evaluate_refinement._pairs_above(separate, 0.20) == 0
    overlapping = [
        Polygon([(0, 0), (100, 0), (100, 20), (0, 20)]),
        Polygon([(50, 0), (150, 0), (150, 20), (50, 20)]),
    ]
    assert evaluate_refinement._pairs_above(overlapping, 0.20) == 1


def test_sweep_ratios_split_and_empty():
    assert evaluate_refinement._sweep_ratios(["0.25, 0.5", "0.75"], "--merge-gap-ratio") == [
        0.25,
        0.5,
        0.75,
    ]
    with pytest.raises(ValueError, match="needs at least one number"):
        evaluate_refinement._sweep_ratios([], "--merge-gap-ratio")


# ---------------------------------------------------------------------------
# verify_adapter: matching, corners, gate
# ---------------------------------------------------------------------------


def test_match_by_centre_pairs_closest():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    b = np.array([[0.3, 0.1], [10.3, 0.1], [10.3, 5.1], [0.3, 5.1]])
    assert verify_adapter._match_by_centre([a], [b]) == [(0, 0)]
    assert verify_adapter._centre(a).tolist() == [5.0, 2.5]


def test_corner_set_distances_values():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    b = np.array([[0.3, 0.1], [10.3, 0.1], [10.3, 5.1], [0.3, 5.1]])
    distances = verify_adapter._corner_set_distances(a, b)
    assert len(distances) == 8
    assert all(abs(d - 0.31622776601683794) < 1e-12 for d in distances)


def test_gate_page_failures_branches():
    gate = verify_adapter._gate_page_failures
    assert gate("p", 2, 2, set(), set(), 0.1) == []
    assert len(gate("p", 2, 3, set(), {2}, 0.1)) == 2
    assert gate("p", 2, 2, set(), set(), 0.6) == ["p: max corner distance 0.600px exceeds 0.5px"]


# ---------------------------------------------------------------------------
# verify_parity: boxes and corner distance
# ---------------------------------------------------------------------------


def test_as_boxes_empty_and_flat():
    assert seg_fixtures.as_boxes([]).tolist() == []
    assert seg_fixtures.as_boxes([0, 0, 1, 0, 1, 1, 0, 1]).tolist() == [
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ]


def test_max_corner_distance_same_and_empty():
    boxes = np.array([[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]])
    assert seg_fixtures.max_corner_distance(boxes, boxes) == 0.0
    assert seg_fixtures.max_corner_distance(np.zeros((0, 4, 2)), np.zeros((0, 4, 2))) == 0.0


# ---------------------------------------------------------------------------
# export_onnx: file checks and axis gate
# ---------------------------------------------------------------------------


def test_check_model_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_model_files(tmp_path)


def test_require_dynamic_inputs_frozen_and_symbolic():
    with pytest.raises(ValueError, match="unexpected detection input shape"):
        _require_dynamic_inputs([{"dims": [1, 3, 0]}])
    with pytest.raises(ValueError, match="input axes look frozen"):
        _require_dynamic_inputs([{"dims": [2, 3, 100, 200]}])
    _require_dynamic_inputs([{"dims": ["N", 3, "H", "W"]}])


def test_tensor_dims_symbolic_and_valued():
    class _Dim:
        def __init__(self, param, value):
            self.dim_param = param
            self.dim_value = value

    class _Shape:
        dim = [_Dim("N", 0), _Dim("", 3), _Dim("H", 0)]

    class _Tensor:
        pass

    entry = _Tensor()
    entry.type = _Tensor()
    entry.type.tensor_type = _Tensor()
    entry.type.tensor_type.shape = _Shape()
    assert _tensor_dims(entry) == ["N", 3, "H"]


# ---------------------------------------------------------------------------
# seed fixtures: grouping, shaping, passwords
# ---------------------------------------------------------------------------


def test_group_pages_by_document():
    groups = seed_fixtures._group_pages_by_document(
        ["Grec_1360_6", "Grec_1360_7", "solo", " Vat-1r".strip()]
    )
    assert groups["Grec_1360"] == ["Grec_1360_6", "Grec_1360_7"]
    assert groups["solo"] == ["solo"]


def test_document_display_name_branches():
    assert seed_fixtures._document_display_name("solo", ["solo"]) == "solo"
    assert (
        seed_fixtures._document_display_name("Grec_1360_x", ["a", "b"])
        == "Grec 1360 - Harmenopulus (btv1b10721710m)"
    )
    assert seed_fixtures._document_display_name("some_doc", ["a", "b"]) == "some doc"


def test_sort_key_and_image_key_stem():
    assert seed_fixtures._page_sort_key("Grec_1360_10") == ("Grec_1360", 10)
    assert seed_fixtures._page_sort_key("solo") == ("solo", 0)
    assert seed_fixtures._stem_from_image_key("parts/abc.tif") == "abc"
    assert seed_fixtures._stem_from_image_key("nope") is None


def test_approved_text_branches():
    approved = seed_fixtures._approved_text
    assert approved({"text_override": "  hi "}) == "hi"
    assert approved({"text_override": "   "}) is None
    assert approved({}) is None


def test_history_line_and_snapshot_state_order():
    segments = [
        {"number": 2, "points": [[0, 0]], "text_override": "b"},
        {"number": 1, "points": [[1, 1]]},
    ]
    state = seed_fixtures._legacy_snapshot_state({"segments": segments})
    assert [line["order"] for line in state["lines"]] == [0, 1]
    assert state["lines"][1]["approved_text"] == "b"
    assert len({line["id"] for line in state["lines"]}) == 2
    assert isinstance(uuid.UUID(state["lines"][0]["id"]), uuid.UUID)


def test_parse_legacy_timestamp_branches():
    assert seed_fixtures._parse_legacy_timestamp(None) is None
    assert seed_fixtures._parse_legacy_timestamp("not-a-date") is None
    parsed = seed_fixtures._parse_legacy_timestamp("2026-01-02T03:04:05Z")
    assert (parsed.year, parsed.month, parsed.day) == (2026, 1, 2)


def test_load_json_and_transcription(tmp_path):
    payload = tmp_path / "a.json"
    payload.write_text(json.dumps({"segments": []}))
    assert seed_fixtures._load_json(payload) == {"segments": []}
    assert seed_fixtures._load_annotation(tmp_path, "missing") is None
    text = tmp_path / "p.txt"
    text.write_text("  \n")
    assert seed_fixtures._load_page_transcription(tmp_path, "p") is None
    text.write_text("line one\nline two\n")
    assert seed_fixtures._load_page_transcription(tmp_path, "p") == "line one\nline two\n"


def test_discover_and_resolve_page_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        seed_fixtures._discover_page_stems(tmp_path)
    pages = tmp_path / "manuscripts" / "pages"
    annotations = tmp_path / "annotations" / "pages"
    pages.mkdir(parents=True)
    annotations.mkdir(parents=True)
    (pages / "p1.jpg").write_bytes(b"img")
    (annotations / "p1.json").write_text("{}")
    (annotations / "p2.json").write_text("{}")
    assert seed_fixtures._discover_page_stems(tmp_path) == ["p1", "p2"]
    assert seed_fixtures._resolve_page_image(pages, "p1") == pages / "p1.jpg"
    assert seed_fixtures._resolve_page_image(pages, "p2") is None


def test_image_dimensions_valid_and_garbage(tmp_path):
    path = tmp_path / "page.png"
    PILImage.new("RGB", (30, 12)).save(path)
    assert seed_fixtures._image_dimensions(path.read_bytes()) == (30, 12)
    assert seed_fixtures._image_dimensions(b"not-an-image") == (None, None)


def test_resolve_dev_password_supplied_and_generated(monkeypatch):
    monkeypatch.setenv("DEV_USER_PASSWORD", "supplied-by-the-operator")
    assert seed_fixtures._resolve_dev_password() == ("supplied-by-the-operator", False)
    monkeypatch.delenv("DEV_USER_PASSWORD")
    first, first_generated = seed_fixtures._resolve_dev_password()
    second, second_generated = seed_fixtures._resolve_dev_password()
    assert first_generated and second_generated
    assert first != second
    assert len(first) >= 24


def test_list_legacy_history_files_skips_state(tmp_path):
    history = tmp_path / "p1"
    history.mkdir()
    (history / "_state.json").write_text("{}")
    (history / "b.json").write_text("{}")
    (history / "a.json").write_text("{}")
    files = seed_fixtures._list_legacy_history_files(tmp_path, "p1")
    assert {p.name for p in files} == {"a.json", "b.json"}
    assert seed_fixtures._list_legacy_history_files(tmp_path, "missing") == []

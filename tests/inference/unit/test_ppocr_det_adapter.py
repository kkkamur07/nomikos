"""Adapter tests for the PP-OCRv6 detection segmenter, with a fake session.

No network, no real weights: the ONNX session is a stub returning synthetic
probability maps, so these tests pin the contract mapping, the param
validation and the runner dispatch rather than the detector quality.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from nomikos_inference.architectures.ppocr_det import ppocr_det
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.ppocr_det import (
    PPOCRDetUnavailableError,
    run_ppocr_det_segment,
)
from nomikos_inference.architectures.ppocr_det.response import build_ppocr_det_response
from nomikos_inference.contracts.common import (
    MAX_SEGMENT_LINES,
    InferenceTask,
    RegistryArchitecture,
)
from nomikos_inference.contracts.segment import SegmentRunResponse
from nomikos_inference.jobs.runner import run_model


class FakeSession:
    """Stand-in for an onnxruntime session over a fixed probability map."""

    def __init__(self, prob_map: np.ndarray):
        self.prob_map = np.asarray(prob_map, dtype=np.float32)
        self.calls = 0

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="x")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="sigmoid_0.tmp_0")]

    def run(self, _names: list[str], _feed: dict) -> list[np.ndarray]:
        self.calls += 1
        return [self.prob_map]


def _page_bytes(size: tuple[int, int] = (128, 128)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def _two_line_map() -> np.ndarray:
    prob = np.zeros((1, 1, 128, 128), dtype=np.float32)
    prob[0, 0, 20:36, 20:110] = 1.0
    prob[0, 0, 70:86, 20:110] = 1.0
    return prob


def _run_with_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prob_map: np.ndarray,
    params: dict | None = None,
    size: tuple[int, int] = (128, 128),
) -> SegmentRunResponse:
    artifact = tmp_path / "ppocrv6.onnx"
    artifact.write_bytes(b"fake onnx graph")
    session = FakeSession(prob_map)
    monkeypatch.setattr(
        ppocr_det, "_load_ppocr_det_session", lambda _path, _fingerprint: (session, "x", "y")
    )
    return run_ppocr_det_segment(
        _page_bytes(size), model_path=artifact, artifact_sha256=None, params=params
    )


def test_response_validates_and_lines_read_top_to_bottom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response = _run_with_session(
        monkeypatch, tmp_path, _two_line_map(), params={"box_type": "quad"}
    )

    assert isinstance(response, SegmentRunResponse)
    assert len(response.blocks) == 1
    assert response.blocks[0].external_id == "ppocr-det-region-1"
    assert response.blocks[0].order == 0
    xs = [point[0] for line in response.lines for point in line.points]
    ys = [point[1] for line in response.lines for point in line.points]
    assert response.blocks[0].box == {
        "points": [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
    }
    assert len(response.lines) == 2
    assert [line.external_id for line in response.lines] == [
        "ppocr-det-line-1",
        "ppocr-det-line-2",
    ]
    assert [line.order for line in response.lines] == [0, 1]
    top = np.mean([point[1] for point in response.lines[0].points])
    bottom = np.mean([point[1] for point in response.lines[1].points])
    assert top < bottom
    for line in response.lines:
        assert line.block_external_id == "ppocr-det-region-1"
        assert len(line.points) == 4
        assert line.kraken_ceiling is None
        assert line.mask is None
        assert line.source_metadata["detector"] == "pp-ocrv6-det"
        assert line.source_metadata["baseline_source"] == "quad_axis"
        assert line.source_metadata["baseline_fraction"] == 0.75
        assert line.source_metadata["score"] > 0.9


def test_baselines_lie_inside_their_quads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    response = _run_with_session(
        monkeypatch, tmp_path, _two_line_map(), params={"box_type": "quad"}
    )

    assert len(response.lines) == 2
    for line in response.lines:
        contour = np.asarray(line.points, dtype=np.float32)
        assert len(line.baseline["points"]) == 2
        for x, y in line.baseline["points"]:
            assert cv2.pointPolygonTest(contour, (float(x), float(y)), True) >= -1e-6


def test_bad_params_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for params in (
        {"limit_side_len": 100},
        {"limit_side_len": 5000},
        {"reading_direction": "diagonal"},
        {"baseline_fraction": 1.5},
        {"baseline_fraction": -0.1},
        {"box_type": "circle"},
    ):
        with pytest.raises(ValueError):
            _run_with_session(monkeypatch, tmp_path, _two_line_map(), params=params)


def test_wrong_output_shape_is_an_unavailable_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = np.zeros((1, 1, 64, 64), dtype=np.float32)

    with pytest.raises(PPOCRDetUnavailableError):
        _run_with_session(monkeypatch, tmp_path, bad)


def test_empty_map_returns_an_empty_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response = _run_with_session(
        monkeypatch, tmp_path, np.zeros((1, 1, 128, 128), dtype=np.float32)
    )

    assert response.blocks == []
    assert response.lines == []


def test_past_the_line_cap_the_highest_scores_survive() -> None:
    quads = [
        DetectedQuad(
            points=[[0.0, float(i)], [100.0, float(i)], [100.0, float(i + 1)], [0.0, float(i + 1)]],
            score=float(i % 7),
        )
        for i in range(MAX_SEGMENT_LINES + 5)
    ]

    response = build_ppocr_det_response(200, MAX_SEGMENT_LINES + 5, quads)

    assert len(response.lines) == MAX_SEGMENT_LINES
    # The five dropped quads are the lowest ranking: score 0 with the largest
    # input positions, so y=0.0 survives and y=10003.0 does not.
    tops = {line.points[0][1] for line in response.lines}
    assert 0.0 in tops
    assert 10003.0 not in tops
    # Survivors are numbered in reading (top to bottom) order.
    assert response.lines[0].points[0][1] == 0.0
    assert response.lines[0].external_id == "ppocr-det-line-1"


def _fake_ort_session_class(seen: dict) -> type:
    """An InferenceSession stand-in that records its construction arguments."""

    class FakeORTSession:
        def __init__(
            self, path: str, sess_options: object = None, providers: object = None
        ) -> None:
            seen["path"] = path
            seen["options"] = sess_options
            seen["providers"] = providers

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="x", shape=[1, 3, None, None])]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="y", shape=[1, 1, None, None])]

    return FakeORTSession


def test_session_uses_extended_sequential_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ppocr_det._load_ppocr_det_session.cache_clear()
    monkeypatch.delenv("NOMIKOS_PPOCR_DET_THREADS", raising=False)
    seen: dict = {}
    monkeypatch.setattr(ort, "InferenceSession", _fake_ort_session_class(seen))
    artifact = tmp_path / "ppocrv6-threads-default.onnx"
    artifact.write_bytes(b"fake onnx graph")

    _, input_name, output_name = ppocr_det._load_ppocr_det_session(str(artifact), None)

    assert (input_name, output_name) == ("x", "y")
    assert seen["providers"] == ["CPUExecutionProvider"]
    assert (
        seen["options"].graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    assert seen["options"].execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert seen["options"].inter_op_num_threads == 1
    assert seen["options"].intra_op_num_threads == 4


def test_session_threads_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ppocr_det._load_ppocr_det_session.cache_clear()
    monkeypatch.setenv("NOMIKOS_PPOCR_DET_THREADS", "12")
    seen: dict = {}
    monkeypatch.setattr(ort, "InferenceSession", _fake_ort_session_class(seen))
    artifact = tmp_path / "ppocrv6-threads-env.onnx"
    artifact.write_bytes(b"fake onnx graph")

    ppocr_det._load_ppocr_det_session(str(artifact), None)

    assert seen["options"].intra_op_num_threads == 12


@pytest.mark.parametrize("raw", ["0", "-3", "65", "banana", "4.5"])
def test_bad_session_threads_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
) -> None:
    ppocr_det._load_ppocr_det_session.cache_clear()
    monkeypatch.setenv("NOMIKOS_PPOCR_DET_THREADS", raw)
    seen: dict = {}
    monkeypatch.setattr(ort, "InferenceSession", _fake_ort_session_class(seen))
    artifact = tmp_path / "ppocrv6-threads-bad.onnx"
    artifact.write_bytes(b"fake onnx graph")

    with pytest.raises(PPOCRDetUnavailableError, match="NOMIKOS_PPOCR_DET_THREADS"):
        ppocr_det._load_ppocr_det_session(str(artifact), None)


def _run_with_spied_builders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    params: dict | None,
) -> dict:
    from nomikos_inference.contracts.segment import SegmentRunResponse

    calls: dict = {}
    artifact = tmp_path / "ppocrv6-branch.onnx"
    artifact.write_bytes(b"fake onnx graph")
    session = FakeSession(_two_line_map())
    monkeypatch.setattr(
        ppocr_det, "_load_ppocr_det_session", lambda _path, _fingerprint: (session, "x", "y")
    )

    def fake_plain(*_args: object, **_kwargs: object) -> SegmentRunResponse:
        calls["plain"] = True
        return SegmentRunResponse(blocks=[], lines=[])

    def fake_refined(*_args: object, **_kwargs: object) -> SegmentRunResponse:
        calls["refined"] = True
        return SegmentRunResponse(blocks=[], lines=[])

    monkeypatch.setattr(ppocr_det, "build_ppocr_det_response", fake_plain)
    monkeypatch.setattr(ppocr_det, "build_refined_ppocr_det_response", fake_refined)
    run_ppocr_det_segment(_page_bytes(), model_path=artifact, artifact_sha256=None, params=params)
    return calls


def test_all_refinement_off_uses_the_plain_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _run_with_spied_builders(
        monkeypatch,
        tmp_path,
        {"merge_fragments": False, "resolve_overlaps": False, "noise_policy": "off"},
    )

    assert calls == {"plain": True}


def test_refinement_on_uses_the_refined_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _run_with_spied_builders(monkeypatch, tmp_path, None)

    assert calls == {"refined": True}


def test_bad_refinement_params_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for params in (
        {"merge_fragments": "yes"},
        {"resolve_overlaps": 1},
        {"noise_policy": "delete"},
    ):
        with pytest.raises(ValueError):
            _run_with_session(monkeypatch, tmp_path, _two_line_map(), params=params)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("thresh", 1.5),
        ("box_thresh", -0.1),
        ("unclip_ratio", 6.0),
        ("max_candidates", 0),
        ("baseline_fraction", 1.5),
        ("limit_side_len", 100),
        ("merge_gap_ratio", 11.0),
        ("merge_max_overlap_ratio", 2.5),
        ("merge_max_height_ratio", 0.5),
        ("overlap_cut_threshold", 0.01),
    ],
)
def test_out_of_range_params_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str, bad: object
) -> None:
    with pytest.raises(ValueError, match=key):
        _run_with_session(monkeypatch, tmp_path, _two_line_map(), params={key: bad})


@pytest.mark.parametrize("raw", [float("inf"), float("nan"), "wide", True])
@pytest.mark.parametrize(
    "key",
    [
        "thresh",
        "box_thresh",
        "unclip_ratio",
        "baseline_fraction",
        "merge_gap_ratio",
        "merge_max_overlap_ratio",
        "merge_max_height_ratio",
        "overlap_cut_threshold",
    ],
)
def test_non_finite_unparseable_or_bool_floats_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str, raw: object
) -> None:
    with pytest.raises(ValueError, match=key):
        _run_with_session(monkeypatch, tmp_path, _two_line_map(), params={key: raw})


@pytest.mark.parametrize("raw", [float("inf"), float("nan"), "huge", True])
def test_non_finite_unparseable_or_bool_side_len_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: object
) -> None:
    with pytest.raises(ValueError, match="limit_side_len"):
        _run_with_session(monkeypatch, tmp_path, _two_line_map(), params={"limit_side_len": raw})


@pytest.mark.parametrize("raw", [float("inf"), float("nan"), "many", True, 0, 10001])
def test_bad_max_candidates_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: object
) -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        _run_with_session(monkeypatch, tmp_path, _two_line_map(), params={"max_candidates": raw})


def test_run_model_dispatches_ppocr_det(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The runner hands a ``ppocr-det`` entry to the new adapter verbatim."""
    monkeypatch.setattr("nomikos_inference.jobs.runner.validate_image_bytes", lambda *_args: None)
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.validate_request_params", lambda *_args: None
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.get_inference_settings",
        lambda: SimpleNamespace(inference_registry_path=Path("registry.yaml")),
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.resolve_registry_entry",
        lambda **_kwargs: SimpleNamespace(
            architecture=RegistryArchitecture.ppocr_det,
            versions={
                "stable": SimpleNamespace(
                    weights_source="file://unused",
                    hub_revision=None,
                    artifact_sha256=None,
                )
            },
        ),
    )
    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"graph")
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.resolve_weights_source",
        lambda *_args, **_kwargs: weights,
    )
    seen: dict = {}
    sentinel = SegmentRunResponse(blocks=[], lines=[])

    def fake_run(image_bytes: bytes, **kwargs: object) -> SegmentRunResponse:
        seen["image_bytes"] = image_bytes
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr("nomikos_inference.jobs.runner.run_ppocr_det_segment", fake_run)
    page = _page_bytes()

    response = run_model(
        task=InferenceTask.segment,
        registry_model_id="ppocr-det-model",
        registry_tag="stable",
        image_bytes=page,
        params={"limit_side_len": 960},
    )

    assert response is sentinel
    assert seen["image_bytes"] == page
    assert seen["model_path"] == weights
    assert seen["artifact_sha256"] is None
    assert seen["params"] == {"limit_side_len": 960}

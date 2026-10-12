"""Unit tests for the local vs platform parity harness.

No network and no model weights: the API transport is faked and the
runner is stubbed where the full flow is exercised. Platform fixtures
use the real backend shapes: stored part lines for segment (with the
``external_id``/``job_id`` keys the merge adds to ``source_metadata``)
and the flat ``transcription_id``/``lines`` summary for transcribe.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "inference"
    / "parity"
    / "compare_local_vs_platform.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("nmk_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nmk = load_module()

FAKE_CATALOG = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "name": "ppocr",
        "task": "segment",
        "artifact_ref": "registry://ppocr-segment?tag=stable",
        "default_params": {},
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "name": "syriac",
        "task": "transcribe",
        "artifact_ref": "registry://syriac-ppocr-v1?tag=stable",
        "default_params": {"beam": 3},
    },
]

BOX_A = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
BOX_B = [[0.0, 20.0], [10.0, 20.0], [10.0, 30.0], [0.0, 30.0]]
BASE_A = [[0.0, 7.0], [10.0, 7.0]]


def seg_local(order, points, baseline=None, kind="polygon"):
    return {
        "external_id": f"ppocr-det-line-{order + 1}",
        "order": order,
        "kind": kind,
        "points": points,
        "baseline": {"points": baseline} if baseline is not None else {"points": []},
        "mask": None,
        "source_metadata": {"role": "line", "suspect": False},
    }


def seg_stored(
    line_id,
    order,
    points,
    baseline=None,
    created="2026-01-01T00:00:00",
    job_id="job-1",
    source="kraken",
):
    return {
        "id": line_id,
        "order": order,
        "points": points,
        "baseline": {"points": baseline} if baseline is not None else {"points": []},
        "mask": None,
        "kind": "polygon",
        "source": source,
        "manual_geometry": False,
        "source_metadata": {
            "role": "line",
            "suspect": False,
            "external_id": f"ppocr-det-line-{order + 1}",
            "job_id": job_id,
        },
        "created_at": created,
    }


def shift(points, delta):
    return [[x + delta, y + delta] for x, y in points]


def tr_local(line_id, index, text, confidence=0.9, error=None):
    return {
        "line_id": line_id,
        "line_index": index,
        "output": {"text": text, "confidence": confidence, "character_confidences": []},
        "error": error,
    }


def tr_flat(line_id, text, confidence=0.9):
    return {"line_id": line_id, "text": text, "confidence": confidence}


def test_resolve_model_by_display_name_registry_id_and_uuid():
    by_name = nmk.resolve_model(FAKE_CATALOG, "ppocr")
    assert by_name["artifact_ref"] == "registry://ppocr-segment?tag=stable"
    by_registry = nmk.resolve_model(FAKE_CATALOG, "ppocr-segment")
    assert by_registry["name"] == "ppocr"
    by_uuid = nmk.resolve_model(FAKE_CATALOG, "11111111-1111-1111-1111-111111111111")
    assert by_uuid["name"] == "ppocr"
    registry_id, tag = nmk.parse_artifact_ref(by_name["artifact_ref"])
    assert (registry_id, tag) == ("ppocr-segment", "stable")
    with pytest.raises(ValueError):
        nmk.resolve_model(FAKE_CATALOG, "no-such-model")


def test_segment_identical_against_stored_shape():
    local = [seg_local(0, BOX_A, BASE_A), seg_local(1, BOX_B)]
    platform = [seg_stored("a", 0, BOX_A, BASE_A), seg_stored("b", 1, BOX_B)]
    result = nmk.compare_segment_lines(local, platform)
    assert result["verdict"] == "IDENTICAL"
    assert result["local_line_count"] == result["platform_line_count"] == 2


def test_segment_numeric_shift():
    local = [seg_local(0, BOX_A, BASE_A)]
    platform = [seg_stored("a", 0, shift(BOX_A, 0.5), shift(BASE_A, 0.5))]
    result = nmk.compare_segment_lines(local, platform)
    assert result["verdict"] == "NUMERIC"
    assert result["max_coordinate_diff"] == pytest.approx(0.5)


def test_segment_reordered_is_mismatch_with_order_differences():
    local = [seg_local(0, BOX_A, BASE_A), seg_local(1, BOX_B)]
    platform = [seg_stored("a", 0, BOX_B), seg_stored("b", 1, BOX_A, BASE_A)]
    result = nmk.compare_segment_lines(local, platform)
    assert result["verdict"] == "MISMATCH"
    assert result["order_differences"]


def test_segment_missing_line():
    local = [seg_local(0, BOX_A, BASE_A), seg_local(1, BOX_B)]
    platform = [seg_stored("a", 0, BOX_A, BASE_A)]
    result = nmk.compare_segment_lines(local, platform)
    assert result["verdict"] == "MISMATCH"
    assert result["only_local"] == [1]
    assert result["platform_line_count"] == 1


def test_segment_empty_is_not_identical():
    result = nmk.compare_segment_lines([], [])
    assert result["verdict"] == "EMPTY"


def test_transcribe_identical_against_flat_shape():
    local = [tr_local("a", 0, "hello"), tr_local("b", 1, "world")]
    platform = [tr_flat("a", "hello", 0.9), tr_flat("b", "world", 0.9)]
    result = nmk.compare_transcribe_lines(local, platform)
    assert result["verdict"] == "IDENTICAL"


def test_transcribe_confidence_only():
    local = [tr_local("a", 0, "hello", 0.9)]
    platform = [tr_flat("a", "hello", 0.5)]
    result = nmk.compare_transcribe_lines(local, platform)
    assert result["verdict"] == "CONFIDENCE_ONLY"
    assert result["max_confidence_diff"] == pytest.approx(0.4)


def test_transcribe_text_diff_lists_both_texts():
    local = [tr_local("a", 0, "hello")]
    platform = [tr_flat("a", "hallo")]
    result = nmk.compare_transcribe_lines(local, platform)
    assert result["verdict"] == "MISMATCH"
    assert len(result["line_differences"]) == 1
    diff = result["line_differences"][0]
    assert diff["local_text"] == "hello"
    assert diff["platform_text"] == "hallo"
    assert result["max_cer"] > 0.0


def test_transcribe_error_mismatch_is_mismatch():
    local = [tr_local("a", 0, "hello", error="Line could not be transcribed")]
    platform = [tr_flat("a", "hello", 0.9)]
    result = nmk.compare_transcribe_lines(local, platform)
    assert result["verdict"] == "MISMATCH"


def test_transcribe_empty_is_not_identical():
    result = nmk.compare_transcribe_lines([], [])
    assert result["verdict"] == "EMPTY"


def test_transcribe_surfaces_failed_line_indexes():
    local = [tr_local("a", 0, "hello")]
    platform = [tr_flat("a", "hello", 0.9)]
    result = nmk.compare_transcribe_lines(local, platform, failed_line_indexes=[1])
    assert result["verdict"] == "IDENTICAL"
    assert result["failed_line_indexes"] == [1]


def rotated(rect, degrees):
    cx = sum(point[0] for point in rect) / len(rect)
    cy = sum(point[1] for point in rect) / len(rect)
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return [
        [
            cx + (x - cx) * cosine - (y - cy) * sine,
            cy + (x - cx) * sine + (y - cy) * cosine,
        ]
        for x, y in rect
    ]


def check_iou_properties():
    box = [[0.0, 0.0], [10.0, 0.0], [10.0, 2.0], [0.0, 2.0]]
    turned = rotated(box, 30)
    assert nmk.polygon_iou(turned, turned) == pytest.approx(1.0)
    far = [[x + 50.0, y + 50.0] for x, y in box]
    assert nmk.polygon_iou(box, far) == 0.0
    half = [[5.0, 0.0], [15.0, 0.0], [15.0, 2.0], [5.0, 2.0]]
    assert nmk.polygon_iou(box, half) == pytest.approx(1.0 / 3.0)
    for angle in (0, 15, 45, 90):
        for dx, dy in ((0, 0), (3, 1), (20, 0)):
            first = rotated(box, angle)
            second = [[x + dx, y + dy] for x, y in rotated(box, angle + 10)]
            score = nmk.polygon_iou(first, second)
            assert 0.0 <= score <= 1.0


def test_polygon_iou_with_shapely():
    assert nmk._HAS_SHAPELY
    check_iou_properties()


def test_polygon_iou_fallback(monkeypatch):
    monkeypatch.setattr(nmk, "_HAS_SHAPELY", False)
    check_iou_properties()


def test_build_transcribe_params_order_filter_index():
    part_lines = [
        {"id": "b", "order": 1, "points": BOX_B, "created_at": "2026-01-02"},
        {"id": "a", "order": 0, "points": BOX_A, "created_at": "2026-01-01"},
        {"id": "c", "order": 0, "points": BOX_A, "created_at": "2026-01-03"},
    ]
    params, ordered = nmk.build_transcribe_params(part_lines, ["c", "a"], {}, None)
    assert [line["id"] for line in ordered] == ["a", "c"]
    assert [region["line_index"] for region in params["lines"]] == [0, 1]
    assert [region["line_id"] for region in params["lines"]] == ["a", "c"]


def test_build_transcribe_params_empty_raises():
    with pytest.raises(nmk.ApiError):
        nmk.build_transcribe_params([], None, {}, None)
    with pytest.raises(nmk.ApiError):
        nmk.build_transcribe_params(
            [{"id": "a", "order": 0, "points": BOX_A}], ["missing"], {}, None
        )
    with pytest.raises(nmk.ApiError):
        nmk.build_transcribe_params([{"id": "a", "order": 0, "points": BOX_A}], None, {}, 0)


class FakeTransport:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        bare = url.split("?", 1)[0]
        for (route_method, route_path), handler in self.routes.items():
            if method == route_method and bare.endswith(route_path):
                return handler(method, url, body)
        raise AssertionError(f"unexpected request {method} {url}")


def json_route(payload, status=200):
    def handler(method, url, body):
        return status, json.dumps(payload).encode("utf-8")

    return handler


BASE_ARGS = [
    "--api",
    "http://127.0.0.1:9",
    "--project-id",
    "proj-1",
    "--document-id",
    "doc-1",
    "--part-id",
    "part-1",
]


def test_segment_enqueue_without_flag_makes_no_network_call():
    transport = FakeTransport({})
    code = nmk.main(
        [
            "--task",
            "segment",
            "--enqueue",
            "--model",
            "ppocr",
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code != 0
    assert transport.calls == []


def test_failed_job_path(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    job = {"id": "job-1", "status": "failed", "type": "segment", "error": "boom"}
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
        }
    )
    code = nmk.main(
        [
            "--task",
            "segment",
            "--model",
            "ppocr",
            "--job-id",
            "job-1",
            "--out",
            str(tmp_path / "out"),
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 2


def test_job_type_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    job = {"id": "job-1", "status": "done", "type": "segment", "payload": {}, "result": {}}
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
        }
    )
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--out",
            str(tmp_path / "out"),
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 2


def merge_summary(**overrides):
    summary = {
        "blocks_count": 1,
        "lines_count": 1,
        "added_lines": 1,
        "pruned_lines": 0,
        "preserved_manual_lines": 0,
        "preserved_transcribed_lines": 0,
        "skipped_covered_lines": 0,
    }
    summary.update(overrides)
    return summary


def stub_run_model(monkeypatch, payload):
    import nomikos_inference.jobs.runner as runner_module

    class FakeResponse:
        def model_dump(self, mode="json"):
            return payload

    monkeypatch.setattr(runner_module, "run_model", lambda **kwargs: FakeResponse())


def test_segment_not_comparable_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    job = {
        "id": "job-1",
        "status": "done",
        "type": "segment",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {}},
        "result": merge_summary(preserved_manual_lines=1),
    }
    stored = [seg_stored("a", 0, BOX_A, BASE_A)]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(stored),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "segment",
            "--model",
            "ppocr",
            "--job-id",
            "job-1",
            "--out",
            out_dir,
            "--trust-job-model",
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 3
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "NOT_COMPARABLE"
    assert report["header"]["merge_summary"]["preserved_manual_lines"] == 1
    markdown = Path(out_dir, "report.md").read_text(encoding="utf-8")
    assert "NOT_COMPARABLE" in markdown


def test_segment_allow_merged_compares(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    stub_run_model(monkeypatch, {"blocks": [], "lines": [seg_local(0, BOX_A, BASE_A)]})
    job = {
        "id": "job-1",
        "status": "done",
        "type": "segment",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "completed_at": "2026-01-02T00:00:00+00:00",
        "payload": {"ml_params": {}},
        "result": merge_summary(preserved_manual_lines=1),
    }
    stored = [seg_stored("a", 0, BOX_A, BASE_A)]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(stored),
            ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "segment",
            "--model",
            "ppocr",
            "--job-id",
            "job-1",
            "--allow-merged",
            "--trust-job-model",
            "--out",
            out_dir,
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 0


def test_secrets_never_appear_in_report(tmp_path, monkeypatch):
    secret = "super-secret-token-value-xyz"
    monkeypatch.setenv("NOMIKOS_TOKEN", secret)
    stub_run_model(
        monkeypatch,
        {
            "lines": [
                {
                    "line_id": "L1",
                    "line_index": 0,
                    "output": {"text": "hi", "confidence": 0.9, "character_confidences": []},
                }
            ]
        },
    )
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {"note": secret}, "line_ids": ["L1"]},
        "result": {
            "transcription_id": "t-1",
            "lines": [{"line_id": "L1", "text": "hi", "confidence": 0.9}],
        },
    }
    part_lines = [{"id": "L1", "order": 0, "points": BOX_A, "created_at": "2026-01-01"}]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(part_lines),
            ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--out",
            out_dir,
            "--trust-job-model",
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 0
    report_json = Path(out_dir, "report.json").read_text(encoding="utf-8")
    report_md = Path(out_dir, "report.md").read_text(encoding="utf-8")
    assert secret not in report_json
    assert secret not in report_md
    assert "[REDACTED]" in report_json
    assert os.environ["NOMIKOS_TOKEN"] == secret


def test_job_for_another_part_is_rejected_before_media(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    job = {
        "id": "job-9",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "other-part",
        "payload": {"ml_params": {}, "line_ids": ["L1"]},
        "result": {"transcription_id": "t-9", "lines": []},
    }
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-9"): json_route(job),
            ("GET", "/media/parts/other-part"): lambda m, u, b: (200, b"nope"),
        }
    )
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-9",
            "--out",
            str(tmp_path / "out"),
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 2
    assert not [call for call in transport.calls if "/media/" in call[1]]


def test_segment_staleness_matching_job_compares(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    stub_run_model(monkeypatch, {"blocks": [], "lines": [seg_local(0, BOX_A, BASE_A)]})
    job = {
        "id": "job-1",
        "status": "done",
        "type": "segment",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "completed_at": "2026-01-02T00:00:00+00:00",
        "payload": {"ml_params": {}},
        "result": merge_summary(),
    }
    stored = [seg_stored("a", 0, BOX_A, BASE_A, created="2026-01-01T12:00:00+00:00")]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(stored),
            ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
        }
    )
    code = nmk.main(
        [
            "--task",
            "segment",
            "--model",
            "ppocr",
            "--job-id",
            "job-1",
            "--trust-job-model",
            "--out",
            str(tmp_path / "out"),
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 0


def test_segment_staleness_newer_lines_are_not_comparable(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    job = {
        "id": "job-1",
        "status": "done",
        "type": "segment",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "completed_at": "2026-01-02T00:00:00+00:00",
        "payload": {"ml_params": {}},
        "result": merge_summary(),
    }
    stored = [
        seg_stored("a", 0, BOX_A, BASE_A, created="2026-01-03T00:00:00+00:00", job_id="job-9")
    ]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(stored),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "segment",
            "--model",
            "ppocr",
            "--job-id",
            "job-1",
            "--trust-job-model",
            "--out",
            out_dir,
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 3
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "NOT_COMPARABLE"
    assert report["comparison"]["reason"] == "lines no longer come from job job-1"


def test_transcribe_line_limit_narrows_both_sides(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    ids = [f"L{i}" for i in range(5)]
    stub_run_model(
        monkeypatch,
        {
            "lines": [
                {
                    "line_id": line_id,
                    "line_index": index,
                    "output": {
                        "text": f"text {line_id}",
                        "confidence": 0.9,
                        "character_confidences": [],
                    },
                }
                for index, line_id in enumerate(ids[:2])
            ]
        },
    )
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {}, "line_ids": ids},
        "result": {
            "transcription_id": "t-1",
            "lines": [
                {"line_id": line_id, "text": f"text {line_id}", "confidence": 0.9}
                for line_id in ids
            ],
        },
    }
    part_lines = [
        {"id": line_id, "order": index, "points": BOX_A, "created_at": "2026-01-01"}
        for index, line_id in enumerate(ids)
    ]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(part_lines),
            ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
        }
    )
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--line-limit",
            "2",
            "--trust-job-model",
            "--out",
            str(tmp_path / "out"),
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 0


def test_model_verified_by_payload_identifier(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    stub_run_model(
        monkeypatch,
        {
            "lines": [
                {
                    "line_id": "L1",
                    "line_index": 0,
                    "output": {"text": "hi", "confidence": 0.9, "character_confidences": []},
                }
            ]
        },
    )
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {"model": "syriac-ppocr-v1"}, "line_ids": ["L1"]},
        "result": {
            "transcription_id": "t-1",
            "lines": [{"line_id": "L1", "text": "hi", "confidence": 0.9}],
        },
    }
    part_lines = [{"id": "L1", "order": 0, "points": BOX_A, "created_at": "2026-01-01"}]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(part_lines),
            ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--out",
            out_dir,
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 0
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["header"]["model_verification"] == "verified"


def test_model_unverifiable_without_flag_is_not_comparable(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {}, "line_ids": ["L1"]},
        "result": {"transcription_id": "t-1", "lines": []},
    }
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--out",
            out_dir,
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 3
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "NOT_COMPARABLE"


def test_model_unverifiable_with_flag_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    stub_run_model(
        monkeypatch,
        {
            "lines": [
                {
                    "line_id": "L1",
                    "line_index": 0,
                    "output": {"text": "hi", "confidence": 0.9, "character_confidences": []},
                }
            ]
        },
    )
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {}, "line_ids": ["L1"]},
        "result": {
            "transcription_id": "t-1",
            "lines": [{"line_id": "L1", "text": "hi", "confidence": 0.9}],
        },
    }
    part_lines = [{"id": "L1", "order": 0, "points": BOX_A, "created_at": "2026-01-01"}]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(part_lines),
            ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--trust-job-model",
            "--out",
            out_dir,
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 0
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["header"]["model_verification"] == "model of the job not verified"


def test_model_identifier_disagreement_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {"model": "other-model"}, "line_ids": ["L1"]},
        "result": {"transcription_id": "t-1", "lines": []},
    }
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
        }
    )
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--trust-job-model",
            "--out",
            str(tmp_path / "out"),
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 2


def test_report_files_are_owner_only(tmp_path, monkeypatch):
    import stat as stat_module

    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    stub_run_model(
        monkeypatch,
        {
            "lines": [
                {
                    "line_id": "L1",
                    "line_index": 0,
                    "output": {"text": "hi", "confidence": 0.9, "character_confidences": []},
                }
            ]
        },
    )
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {"model": "syriac-ppocr-v1"}, "line_ids": ["L1"]},
        "result": {
            "transcription_id": "t-1",
            "lines": [{"line_id": "L1", "text": "hi", "confidence": 0.9}],
        },
    }
    part_lines = [{"id": "L1", "order": 0, "points": BOX_A, "created_at": "2026-01-01"}]
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(part_lines),
            ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--out",
            out_dir,
            *BASE_ARGS,
        ],
        transport=transport,
    )
    assert code == 0
    assert stat_module.S_IMODE(os.stat(out_dir).st_mode) == 0o700
    for name in ("report.json", "report.md"):
        mode = stat_module.S_IMODE(os.stat(Path(out_dir, name)).st_mode)
        assert mode == 0o600, (name, oct(mode))


def five_line_job(line_ids, result_ids=None, missing_page_ids=()):
    ids = list(line_ids)
    result = result_ids if result_ids is not None else ids
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {"model": "syriac-ppocr-v1"}, "line_ids": ids},
        "result": {
            "transcription_id": "t-1",
            "lines": [
                {"line_id": line_id, "text": f"text {line_id}", "confidence": 0.9}
                for line_id in result
            ],
        },
    }
    part_lines = [
        {"id": line_id, "order": index, "points": BOX_A, "created_at": "2026-01-01"}
        for index, line_id in enumerate(ids)
        if line_id not in missing_page_ids
    ]
    return job, part_lines


def stub_transcribe_run(monkeypatch, line_ids):
    stub_run_model(
        monkeypatch,
        {
            "lines": [
                {
                    "line_id": line_id,
                    "line_index": index,
                    "output": {
                        "text": f"text {line_id}",
                        "confidence": 0.9,
                        "character_confidences": [],
                    },
                }
                for index, line_id in enumerate(line_ids)
            ]
        },
    )


def run_transcribe_main(job, part_lines, tmp_path, extra_args):
    transport = FakeTransport(
        {
            ("GET", "/inference/models"): json_route(FAKE_CATALOG),
            ("GET", "/jobs/job-1"): json_route(job),
            ("GET", "/parts/part-1/lines"): json_route(part_lines),
            ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
        }
    )
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "transcribe",
            "--model",
            "syriac",
            "--job-id",
            "job-1",
            "--out",
            out_dir,
            *extra_args,
            *BASE_ARGS,
        ],
        transport=transport,
    )
    return code, out_dir


def test_limit_with_all_job_lines_present_is_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    ids = [f"L{i}" for i in range(5)]
    stub_transcribe_run(monkeypatch, ids[:2])
    job, part_lines = five_line_job(ids)
    code, out_dir = run_transcribe_main(job, part_lines, tmp_path, ["--line-limit", "2"])
    assert code == 0
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "IDENTICAL"
    assert report["header"]["lines_requested"] == 2
    assert report["header"]["lines_compared"] == 2


def test_limit_with_missing_job_line_is_not_comparable(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    ids = [f"L{i}" for i in range(5)]
    stub_transcribe_run(monkeypatch, ["L0"])
    job, part_lines = five_line_job(ids, missing_page_ids=("L1",))
    code, out_dir = run_transcribe_main(job, part_lines, tmp_path, ["--line-limit", "2"])
    assert code == 3
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "NOT_COMPARABLE"
    assert "job line L1 no longer exists on the page" in report["comparison"]["reason"]


def test_no_limit_with_missing_job_line_is_not_comparable(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    ids = [f"L{i}" for i in range(5)]
    stub_transcribe_run(monkeypatch, [i for i in ids if i != "L3"])
    job, part_lines = five_line_job(ids, missing_page_ids=("L3",))
    code, out_dir = run_transcribe_main(job, part_lines, tmp_path, [])
    assert code == 3
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "NOT_COMPARABLE"
    assert "job line L3 no longer exists on the page" in report["comparison"]["reason"]


def whole_page_job(result_ids, failed_indexes, extra_page_ids=()):
    ids = [f"L{i}" for i in range(5)]
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {"ml_params": {"model": "syriac-ppocr-v1"}},
        "result": {
            "transcription_id": "t-1",
            "lines": [
                {"line_id": line_id, "text": f"text {line_id}", "confidence": 0.9}
                for line_id in result_ids
            ],
            "failed_line_indexes": failed_indexes,
        },
    }
    page_ids = list(ids) + list(extra_page_ids)
    part_lines = [
        {"id": line_id, "order": index, "points": BOX_A, "created_at": "2026-01-01"}
        for index, line_id in enumerate(page_ids)
    ]
    return job, part_lines


def test_whole_page_failed_line_is_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    ids = [f"L{i}" for i in range(5)]
    stub_transcribe_run(monkeypatch, ids)
    job, part_lines = whole_page_job([i for i in ids if i != "L2"], [2])
    code, out_dir = run_transcribe_main(job, part_lines, tmp_path, [])
    assert code == 1
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "MISMATCH"
    failed_entries = [
        item for item in report["comparison"]["line_differences"] if item["line_id"] == "L2"
    ]
    assert len(failed_entries) == 1
    assert failed_entries[0]["platform_error"] is not None
    assert report["header"]["lines_requested"] == 5
    assert report["header"]["lines_compared"] == 5


def test_whole_page_failed_line_unmappable_is_not_comparable(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    ids = [f"L{i}" for i in range(5)]
    stub_transcribe_run(monkeypatch, ids)
    job, part_lines = whole_page_job([i for i in ids if i != "L2"], [2], extra_page_ids=("LX",))
    code, out_dir = run_transcribe_main(job, part_lines, tmp_path, [])
    assert code == 3
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "NOT_COMPARABLE"
    assert report["comparison"]["reason"] == "failed line indexes cannot be mapped to line ids"


def selective_job(payload_ids, page_ids, failed_id, failed_index):
    job = {
        "id": "job-1",
        "status": "done",
        "type": "transcribe",
        "document_id": "doc-1",
        "document_part_id": "part-1",
        "payload": {
            "ml_params": {"model": "syriac-ppocr-v1"},
            "line_ids": payload_ids,
        },
        "result": {
            "transcription_id": "t-1",
            "lines": [
                {"line_id": line_id, "text": f"text {line_id}", "confidence": 0.9}
                for line_id in page_ids
                if line_id != failed_id
            ],
            "failed_line_indexes": [failed_index],
        },
    }
    part_lines = [
        {"id": line_id, "order": index, "points": BOX_A, "created_at": "2026-01-01"}
        for index, line_id in enumerate(page_ids)
    ]
    return job, part_lines


def test_payload_ids_failed_index_is_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    page_ids = ["A", "B", "C", "D", "E"]
    stub_transcribe_run(monkeypatch, page_ids)
    job, part_lines = selective_job(["C", "A", "E", "B", "D"], page_ids, "D", 3)
    code, out_dir = run_transcribe_main(job, part_lines, tmp_path, [])
    assert code == 1
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "MISMATCH"
    failed_entries = [
        item for item in report["comparison"]["line_differences"] if item["line_id"] == "D"
    ]
    assert len(failed_entries) == 1
    assert failed_entries[0]["platform_error"] is not None


def test_reversed_payload_ids_fail_first_in_page_order(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    page_ids = ["A", "B", "C", "D", "E"]
    stub_transcribe_run(monkeypatch, page_ids)
    job, part_lines = selective_job(list(reversed(page_ids)), page_ids, "A", 0)
    code, out_dir = run_transcribe_main(job, part_lines, tmp_path, [])
    assert code == 1
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "MISMATCH"
    failed_entries = [
        item
        for item in report["comparison"]["line_differences"]
        if item["platform_error"] is not None
    ]
    assert [item["line_id"] for item in failed_entries] == ["A"]


def test_limit_ignores_failed_line_outside_subset(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMIKOS_TOKEN", "dummy")
    ids = [f"L{i}" for i in range(5)]
    stub_transcribe_run(monkeypatch, ids[:2])
    job, part_lines = whole_page_job([i for i in ids if i != "L4"], [4])
    code, out_dir = run_transcribe_main(job, part_lines, tmp_path, ["--line-limit", "2"])
    assert code == 0
    report = json.loads(Path(out_dir, "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"] == "IDENTICAL"
    assert report["header"]["lines_requested"] == 2
    assert report["header"]["lines_compared"] == 2

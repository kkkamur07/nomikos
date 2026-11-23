"""A malformed stored baseline falls back to the line's box instead of failing the PDF."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from pypdf import PdfReader

from backend.annotation.application.transcription_pdf_service import (
    TranscriptionPdfService,
    _baseline_points,
    _PdfLine,
)
from backend.document.infrastructure.orm_models import TranscriptionKind

_BOX = [[20, 20], [580, 20], [580, 70], [20, 70]]
_MALFORMED = [
    {"points": [[], []]},
    {"points": [["a", 1], [2]]},
    None,
    {"coords": [[20, 60], [580, 60]]},
    {"points": [[20, float("nan")], [580, 60]]},
    {"points": [[float("inf"), 60], [580, 60]]},
    {"points": "20,60 580,60"},
    [[20, 60], [580, 60]],
]


@pytest.mark.parametrize("baseline", _MALFORMED)
def test_a_malformed_baseline_reads_as_missing(baseline) -> None:
    assert _baseline_points(baseline) is None


def test_bad_points_are_dropped_and_good_ones_kept() -> None:
    baseline = {"points": [[20, 60], [], ["x", 1], [300, float("nan")], ["580", 60.5]]}
    assert _baseline_points(baseline) == [[20.0, 60.0], [580.0, 60.5]]


def _stored_line(baseline) -> SimpleNamespace:
    """What ``_pdf_lines`` reads off an ORM line, with the stored baseline as given."""
    layer = SimpleNamespace(kind=TranscriptionKind.ground_truth)
    return SimpleNamespace(
        baseline=baseline,
        points=_BOX,
        transcriptions=[SimpleNamespace(text="alpha beta", transcription=layer)],
    )


def _page_stream(content: bytes) -> bytes:
    return PdfReader(io.BytesIO(content)).pages[0].get_contents().get_data()


@pytest.mark.parametrize("baseline", _MALFORMED)
def test_pdf_renders_on_the_box_when_the_baseline_is_malformed(baseline) -> None:
    service = TranscriptionPdfService.__new__(TranscriptionPdfService)
    (prepared,) = service._pdf_lines([_stored_line(baseline)])
    assert prepared.baseline is None

    content = service._render_pdf(width=600, height=200, lines=[prepared])
    boxed = service._render_pdf(
        width=600, height=200, lines=[_PdfLine(text="alpha beta", points=_BOX)]
    )
    assert PdfReader(io.BytesIO(content)).pages[0].extract_text() == "alpha beta"
    # Drawn exactly where a line with no baseline at all is drawn: on its box.
    assert _page_stream(content) == _page_stream(boxed)

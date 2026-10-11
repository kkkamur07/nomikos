"""Which text a transcription PDF draws for a line, in which colour, and in which font."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

from pypdf import PdfReader

from backend.annotation.application.transcription_pdf_service import (
    _MODEL_TEXT_FILL,
    _TEXT_FILL,
    TranscriptionPdfService,
    _PdfLine,
)
from backend.document.infrastructure.orm_models import (
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)

_POINTS = [[0, 0], [10, 0], [10, 5], [0, 5]]


def _layer(kind: TranscriptionKind, *, age_days: int = 0) -> Transcription:
    return Transcription(
        name=kind.value,
        kind=kind,
        created_at=datetime(2026, 8, 29, tzinfo=UTC) - timedelta(days=age_days),
    )


def _line(*layers: tuple[Transcription, str]) -> Line:
    line = Line(baseline={"points": []}, points=_POINTS)
    for layer, text in layers:
        line.transcriptions.append(LineTranscription(transcription=layer, text=text))
    return line


def test_approved_text_wins_and_is_drawn_in_ink():
    lines = TranscriptionPdfService()._pdf_lines(
        [
            _line(
                (_layer(TranscriptionKind.model), "guess"),
                (_layer(TranscriptionKind.ground_truth), "word"),
            )
        ]
    )
    assert [(line.text, line.approved) for line in lines] == [("word", True)]
    assert lines[0].fill == _TEXT_FILL


def test_unapproved_model_text_is_drawn_in_grey_not_dropped():
    lines = TranscriptionPdfService()._pdf_lines(
        [_line((_layer(TranscriptionKind.model), "guess"))]
    )
    assert [(line.text, line.approved) for line in lines] == [("guess", False)]
    assert lines[0].fill == _MODEL_TEXT_FILL
    assert _MODEL_TEXT_FILL != _TEXT_FILL


def test_newest_model_layer_is_the_one_shown():
    lines = TranscriptionPdfService()._pdf_lines(
        [
            _line(
                (_layer(TranscriptionKind.model, age_days=3), "old guess"),
                (_layer(TranscriptionKind.model), "new guess"),
            )
        ]
    )
    assert [line.text for line in lines] == ["new guess"]


def test_blank_text_of_either_kind_is_not_a_line():
    lines = TranscriptionPdfService()._pdf_lines(
        [
            _line((_layer(TranscriptionKind.model), "   ")),
            _line((_layer(TranscriptionKind.ground_truth), "")),
            _line(),
            _line(
                (_layer(TranscriptionKind.ground_truth), ""),
                (_layer(TranscriptionKind.model), "guess"),
            ),
        ]
    )
    # An empty ground-truth row does not hide the model's text behind it.
    assert [(line.text, line.approved) for line in lines] == [("guess", False)]


def _box(top: int) -> list[list[float]]:
    return [[20, top], [580, top], [580, top + 50], [20, top + 50]]


def _render(*texts: str):
    lines = [_PdfLine(text=text, points=_box(20 + 70 * index)) for index, text in enumerate(texts)]
    content = TranscriptionPdfService()._render_pdf(width=600, height=400, lines=lines)
    return PdfReader(io.BytesIO(content)).pages[0]


def _font_names(page) -> set[str]:
    # Subset fonts are named ``ABCDEF+Family``; the family is what identifies the file.
    return {str(font["/BaseFont"]).split("+")[-1] for font in page["/Resources"]["/Font"].values()}


def test_greek_text_is_extractable_from_the_pdf():
    page = _render("Ἐν ἀρχῇ ἦν ὁ λόγος")
    assert "Ἐν ἀρχῇ ἦν ὁ λόγος" in page.extract_text()
    assert page.mediabox.width == 600
    assert page.mediabox.height == 400


def test_each_script_is_drawn_in_its_own_bundled_font():
    page = _render("Ἐν ἀρχῇ", "ܫܠܡܐ ܥܠܝܟܘܢ", "Բարեւ աշխարհ", "ⲡⲛⲟⲩⲧⲉ ⲡⲉ")
    assert {
        "NotoSans",
        "NotoSansSyriacEastern",
        "NotoSansArmenian",
        "NotoSansCoptic",
    } <= _font_names(page)
    # Shaped glyphs still map back to the characters they draw.
    text = page.extract_text()
    for word in ("ܫܠܡܐ", "Բարեւ", "ⲡⲛⲟⲩⲧⲉ"):
        assert word in text


def test_a_page_without_syriac_does_not_embed_the_syriac_font():
    assert "NotoSansSyriacEastern" not in _font_names(_render("alpha", "Ἐν ἀρχῇ"))

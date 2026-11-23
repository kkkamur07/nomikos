"""A transcription PDF sets each line on its baseline, in one size per page."""

from __future__ import annotations

import io

from pypdf import PdfReader

from backend.annotation.application.transcription_pdf_service import (
    _BASE_FAMILY,
    _MAX_FONT_SIZE,
    TranscriptionPdfService,
    _PdfLine,
    _TranscriptionPdf,
    draw_page,
)

_PAGE_W = 1000
_PAGE_H = 1000
_GAP = 40
_LEFT = 100
_RIGHT = 900


def _line(text: str, y: float, *, baseline: list[list[float]] | None = None) -> _PdfLine:
    # Polygons deliberately overlap their neighbours, as segmenter output does: the
    # layout must not be driven by them.
    polygon = [[_LEFT, y - 60], [_RIGHT, y - 60], [_RIGHT, y + 30], [_LEFT, y + 30]]
    if baseline is None:
        baseline = [[_LEFT, y], [_RIGHT, y]]
    return _PdfLine(text=text, points=polygon, baseline=baseline)


def _drawn(lines: list[_PdfLine]) -> list[tuple[str, float, float, float]]:
    """(text, size, x, baseline y from the top) for each run of text on the page."""
    content = TranscriptionPdfService()._render_pdf(width=_PAGE_W, height=_PAGE_H, lines=lines)
    page = PdfReader(io.BytesIO(content)).pages[0]
    runs: list[tuple[str, float, float, float]] = []

    def visit(text, _cm, tm, _font, size):
        if text.strip():
            runs.append((text, round(size * tm[0], 2), tm[4], _PAGE_H - tm[5]))

    page.extract_text(visitor_text=visit)
    return runs


def _page(*texts: str) -> list[_PdfLine]:
    return [_line(text, 100 + _GAP * index) for index, text in enumerate(texts)]


def test_every_line_of_a_page_gets_the_same_size_from_the_line_gap():
    runs = _drawn(_page("alpha beta", "gamma", "delta epsilon zeta", "eta"))
    sizes = {size for _, size, _, _ in runs}
    assert sizes == {28}  # 0.7 of the 40 point gap


def test_text_sits_on_its_baseline_at_the_baseline_start():
    runs = _drawn(_page("alpha", "beta", "gamma"))
    assert [round(y, 1) for _, _, _, y in runs] == [100.0, 140.0, 180.0]
    # The two points of cell padding sit between the baseline start and the first glyph.
    assert {round(x, 1) for _, _, x, _ in runs} == {_LEFT + 2}


def test_only_a_line_too_long_for_its_baseline_shrinks():
    lines = _page("alpha", "beta", "gamma")
    lines[1] = _line("beta " * 60, 140)
    runs = _drawn(lines)
    sizes = [size for _, size, _, _ in runs]
    assert sizes[0] == sizes[2] == 28
    assert sizes[1] < 28


def test_a_malformed_baseline_falls_back_to_the_box_bottom():
    lines = _page("alpha", "beta", "gamma", "delta")
    # A trace round the polygon: it climbs far more than one and a half line gaps.
    lines[2] = _line(
        "gamma", 180, baseline=[[_LEFT, 180], [_RIGHT, 180], [_RIGHT, 60], [_LEFT, 60]]
    )
    runs = _drawn(lines)
    y = runs[2][3]
    # The box bottom is 210; the baseline is lifted off it by the font's descent, which
    # is well under an em.
    assert 210 - 28 * 0.6 < y < 210
    assert [round(run[3], 1) for run in (runs[0], runs[1], runs[3])] == [100.0, 140.0, 220.0]


def test_a_right_to_left_line_ends_at_the_right_end_of_its_baseline():
    lines = _page("alpha", "ܡܠܟܐ", "gamma")
    runs = _drawn(lines)
    syriac = next(run for run in runs if "ܐ" in run[0])
    latin = runs[0]
    assert syriac[3] == latin[3] + _GAP
    # Right aligned: the run starts well right of the baseline start and ends at its end.
    assert syriac[2] > _RIGHT - 200


def _stub_line(text: str, top: float, *, left: float = _LEFT, right: float = _RIGHT) -> _PdfLine:
    """A line whose baseline is a stub: two equal points, as some segmenters store."""
    polygon = [[left, top], [right, top], [right, top + 30], [left, top + 30]]
    return _PdfLine(text=text, points=polygon, baseline=[[right - 50, top + 25]] * 2)


def test_a_page_of_stub_baselines_still_gets_one_size_from_the_box_pitch():
    lines = [_stub_line(text, 100 + 25 * index) for index, text in enumerate("abcdefgh")]
    # One short box among them must not pull its line to a size of its own.
    lines[3] = _PdfLine(text="d", points=[[_LEFT, 180], [_RIGHT, 180], [_RIGHT, 186], [_LEFT, 186]])
    sizes = {size for _, size, _, _ in _drawn(lines)}
    assert sizes == {17}  # 0.7 of the 25 point pitch between box centres


def test_a_baseline_stub_under_one_word_is_not_taken_for_the_whole_line():
    lines = _page("alpha", "beta", "gamma")
    # A real baseline that spans 30 of the box's 800 points.
    lines[1] = _line("beta", 140, baseline=[[_RIGHT - 30, 140], [_RIGHT, 140]])
    runs = _drawn(lines)
    assert runs[1][1] == 28
    assert round(runs[1][2], 1) == _LEFT + 2


def test_a_line_near_the_right_edge_never_ends_past_the_page():
    text = "οὕτω καὶ οἷα περ ἂν τὰ παρ' ἡμῶν ᾖ, ἐκ τῶν ὁμοίων παρέχεται"
    # The box itself runs past the page, as a crooked polygon on a page's margin can.
    lines = [_stub_line(text, 100 + 25 * index, left=700, right=1100) for index in range(4)]
    pdf = _TranscriptionPdf()
    draw_page(pdf, width=_PAGE_W, height=_PAGE_H, lines=lines)
    runs = _drawn(lines)
    pdf.set_font(_BASE_FAMILY, size=_MAX_FONT_SIZE)
    pdf.set_text_shaping(use_shaping_engine=True, direction="ltr")
    for _, size, x, _ in runs:
        pdf.set_font_size(size)
        assert x + pdf.get_string_width(text) <= _PAGE_W


def test_a_long_line_never_ends_past_its_own_box():
    text = "alpha beta gamma delta " * 8
    lines = [_stub_line(text, 100 + 25 * index, left=100, right=400) for index in range(3)]
    pdf = _TranscriptionPdf()
    pdf.add_page(format=(_PAGE_W, _PAGE_H))
    pdf.set_font(_BASE_FAMILY, size=_MAX_FONT_SIZE)
    pdf.set_text_shaping(use_shaping_engine=True, direction="ltr")
    for _, size, x, _ in _drawn(lines):
        pdf.set_font_size(size)
        assert x + pdf.get_string_width(text) <= 400

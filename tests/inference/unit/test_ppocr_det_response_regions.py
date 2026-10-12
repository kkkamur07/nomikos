"""PP-OCR regions: a line spanning the columns reads in a region of its own."""

from __future__ import annotations

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.response import build_ppocr_det_response


def _quad(x0: float, y0: float, x1: float, y1: float) -> DetectedQuad:
    return DetectedQuad(points=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], score=0.9)


def _two_columns(top: float, rows: int) -> list[DetectedQuad]:
    left = [_quad(100, top + 60 * i, 650, top + 40 + 60 * i) for i in range(rows)]
    right = [_quad(750, top + 60 * i, 1300, top + 40 + 60 * i) for i in range(rows)]
    return left + right


def _block_rank(response, line) -> int:
    return next(b.order for b in response.blocks if b.external_id == line.block_external_id)


def test_a_title_across_both_columns_is_its_own_first_region():
    response = build_ppocr_det_response(
        1400, 800, [_quad(100, 50, 1300, 90), *_two_columns(150, 4)]
    )

    title = response.lines[0]
    assert title.order == 0
    assert title.block_external_id is not None
    assert _block_rank(response, title) == 0
    assert [b.order for b in response.blocks] == [0, 1, 2]
    # Every line is in a region, and region-first numbering is the segmenter's own order.
    ranks = [_block_rank(response, line) for line in response.lines]
    assert ranks == sorted(ranks)


def test_a_footer_across_both_columns_is_its_own_last_region():
    response = build_ppocr_det_response(
        1400, 800, [*_two_columns(150, 4), _quad(100, 700, 1300, 740)]
    )

    footer = response.lines[-1]
    assert footer.block_external_id is not None
    assert _block_rank(response, footer) == len(response.blocks) - 1

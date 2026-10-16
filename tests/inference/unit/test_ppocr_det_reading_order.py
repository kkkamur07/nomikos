"""Column-aware reading order for PP-OCRv6 detection quads."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines, order_lines

DETECTION_DIR = Path(
    "/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_orli-env/server/"
    "ppocrv6-results-paddle322-20260919"
)


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    return DetectedQuad(
        points=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        score=score,
    )


def _ids_in_order(quads: list[DetectedQuad], direction: str = "ltr") -> list[int]:
    return order_lines(quads, direction=direction)


def test_single_column_orders_top_to_bottom() -> None:
    quads = [
        _quad(100, 300, 500, 330),
        _quad(100, 100, 500, 130),
        _quad(100, 200, 500, 230),
    ]

    assert _ids_in_order(quads) == [1, 2, 0]


def test_two_columns_read_left_page_then_right_page() -> None:
    quads = [
        _quad(600, 100, 900, 130),
        _quad(100, 200, 400, 230),
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
    ]

    assert _ids_in_order(quads, direction="ltr") == [3, 1, 0, 2]
    assert _ids_in_order(quads, direction="rtl") == [0, 2, 3, 1]


def test_narrow_marginal_quad_keeps_its_place_in_the_column() -> None:
    quads = [
        _quad(100, 100, 500, 130),
        _quad(520, 150, 560, 180),
        _quad(100, 200, 500, 230),
    ]

    assert _ids_in_order(quads) == [0, 1, 2]


def test_narrow_quad_beside_two_columns_joins_the_nearest() -> None:
    quads = [
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
        _quad(920, 150, 950, 180),
        _quad(100, 200, 400, 230),
        _quad(600, 100, 900, 130),
    ]

    assert _ids_in_order(quads, direction="ltr") == [1, 3, 4, 2, 0]


def test_shuffled_input_gives_the_same_order() -> None:
    quads = [
        _quad(600, 100, 900, 130),
        _quad(100, 200, 400, 230),
        _quad(920, 150, 950, 180),
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
        _quad(100, 300, 400, 330),
        _quad(600, 300, 900, 330),
    ]
    reference = [quads[i] for i in _ids_in_order(quads)]

    # A fixed non-identity permutation: reversing is enough to prove the
    # order does not follow the input sequence.
    shuffled = quads[::-1]
    rerun = [shuffled[i] for i in _ids_in_order(shuffled)]

    assert [id(quad) for quad in rerun] == [id(quad) for quad in reference]


def test_bad_direction_is_rejected() -> None:
    with pytest.raises(ValueError):
        order_lines([_quad(0, 0, 10, 10)], direction="diagonal")


def _load_quads(slug: str) -> list[DetectedQuad] | None:
    path = DETECTION_DIR / f"medium-1920-{slug}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        DetectedQuad(points=[[float(x), float(y)] for x, y in quad], score=float(score))
        for quad, score in zip(payload["polygons"], payload["scores"], strict=True)
    ]


def _assert_left_column_precedes_right(quads: list[DetectedQuad]) -> None:
    centres = [sum(point[0] for point in quad.points) / 4 for quad in quads]
    middle = max(point[0] for quad in quads for point in quad.points) / 2
    left = {i for i, cx in enumerate(centres) if cx < middle}
    right = {i for i, cx in enumerate(centres) if cx >= middle}
    positions = {quad_index: rank for rank, quad_index in enumerate(order_lines(quads))}
    if left and right:
        assert max(positions[i] for i in left) < min(positions[i] for i in right)


def test_real_spread_reads_left_column_before_right() -> None:
    """Left-before-right on the measured medium-1920 detector output.

    ``grec-p1`` is the file the brief names. In the measured JSON all 29 of
    its quads sit on the right folio (the left folio is blank, as the overlay
    beside the JSON shows), so the claim holds vacuously there; ``grec-p4``
    carries text on both folios and exercises the claim for real.
    """
    quads = _load_quads("grec-p1")
    if quads is None:
        pytest.skip("measured detector output is not present")
    _assert_left_column_precedes_right(quads)

    both = _load_quads("grec-p4")
    if both is None:
        pytest.skip("measured detector output is not present")
    middle = max(point[0] for quad in both for point in quad.points) / 2
    centres = [sum(point[0] for point in quad.points) / 4 for quad in both]
    assert any(cx < middle for cx in centres)
    assert any(cx >= middle for cx in centres)
    _assert_left_column_precedes_right(both)


def test_fragments_of_one_visual_line_share_a_row() -> None:
    left = _quad(0, 100, 300, 140)
    right = _quad(320, 97, 900, 137)
    below = _quad(0, 150, 900, 190)
    quads = [left, right, below]

    assert _ids_in_order(quads, direction="ltr") == [0, 1, 2]
    assert _ids_in_order(quads, direction="rtl") == [1, 0, 2]


def test_tall_initial_joins_the_first_row_without_chaining() -> None:
    initial = _quad(0, 100, 60, 260)
    first = _quad(70, 100, 900, 140)
    second = _quad(70, 160, 900, 200)
    third = _quad(70, 220, 900, 260)
    quads = [initial, first, second, third]

    assert _ids_in_order(quads) == [0, 1, 2, 3]

    flipped = quads[::-1]
    rerun = [flipped[i] for i in _ids_in_order(flipped)]
    assert [id(quad) for quad in rerun] == [id(quad) for quad in quads]


def _spread_with_header(header_ymin: float) -> list[DetectedQuad]:
    header = _quad(0, header_ymin, 1900, header_ymin + 40)
    left = [_quad(0, 100 + 60 * i, 900, 140 + 60 * i) for i in range(5)]
    right = [_quad(1000, 100 + 60 * i, 1900, 140 + 60 * i) for i in range(5)]
    return [header, *left, *right]


def test_spanning_header_reads_as_its_own_band() -> None:
    quads = _spread_with_header(50)

    assert _ids_in_order(quads, direction="ltr") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _ids_in_order(quads, direction="rtl") == [0, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5]


def test_spanning_header_between_rows_splits_the_bands() -> None:
    quads = _spread_with_header(250)

    assert _ids_in_order(quads, direction="ltr") == [1, 2, 3, 6, 7, 8, 0, 4, 5, 9, 10]


def test_stacked_overlapping_lines_read_top_to_bottom_in_every_permutation() -> None:
    """Adjacent lines whose boxes overlap are separate rows, not fragments.

    Quads 113, 109 and 96 of the measured c13 page share one x range and
    overlap vertically by more than half the shorter height; they are three
    tightly spaced Coptic lines and must read top first whatever order they
    arrive in.
    """
    top = _quad(873, 218, 1066, 261)
    middle = _quad(870, 236, 1069, 284)
    bottom = _quad(869, 299, 1073, 346)

    for permutation in itertools.permutations([top, middle, bottom]):
        arranged = list(permutation)
        assert [arranged[i] for i in _ids_in_order(arranged)] == [top, middle, bottom]


def _shuffled(count: int, seed: int) -> list[int]:
    """Deterministic Fisher-Yates permutation without the random module."""
    order = list(range(count))
    state = seed * 2 + 1
    for i in range(count - 1, 0, -1):
        state = (1103515245 * state + 12345) % (2**31)
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def test_layout_lines_exposes_the_rows_ordering_reads() -> None:
    quads = [
        _quad(600, 100, 900, 130),
        _quad(100, 200, 400, 230),
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
    ]

    layout = layout_lines(quads, direction="ltr")

    assert len(layout.columns) == 2
    assert layout.spanning == ()
    assert all(len(column.rows) == 2 for column in layout.columns)
    assert {tuple(sorted(column.members)) for column in layout.columns} == {(1, 3), (0, 2)}
    flat = [
        index for position in layout.column_order for index in layout.columns[position].sequence
    ]
    assert flat == order_lines(quads, direction="ltr")


def test_layout_lines_empty_input_gives_empty_layout() -> None:
    layout = layout_lines([])

    assert layout.columns == ()
    assert layout.spanning == ()
    assert order_lines([]) == []


def test_measured_pages_read_identically_under_shuffling() -> None:
    """Ten seeded shuffles of every measured page map back to the base order."""
    paths = sorted(DETECTION_DIR.glob("medium-1920-*.json"))
    if not paths:
        pytest.skip("measured detector output is not present")
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        quads = [
            DetectedQuad(points=[[float(x), float(y)] for x, y in quad], score=0.9)
            for quad in payload["polygons"]
        ]
        base = order_lines(quads)
        for seed in range(10):
            permutation = _shuffled(len(quads), seed)
            rerun = [permutation[i] for i in order_lines([quads[i] for i in permutation])]
            assert rerun == base, f"{path.name} seed={seed}"


def test_ten_thousand_stacked_lines_lay_out_in_seconds() -> None:
    """The admitted ``max_candidates`` ceiling: the all-pairs union took 28 s."""
    import time

    quads = [_quad(0, float(i), 100, float(i + 1)) for i in range(10000)]

    started = time.perf_counter()
    layout = layout_lines(quads)
    elapsed = time.perf_counter() - started

    assert len(layout.columns) == 1
    assert layout.columns[0].sequence == tuple(range(10000))
    assert elapsed < 5.0

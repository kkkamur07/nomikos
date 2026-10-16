"""``resolve_overlaps`` with a spatial index against the all-pairs original.

The production stage queries an STRtree for candidate pairs instead of
intersecting every live pair on every pass. The original quadratic loop is
kept here, and only here, as the oracle: on 200 random pages built to contain
stacked pairs, duplicates, containment and tall initials, both must leave
every work with the same geometry, baseline and flags.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import PageLayout, layout_lines
from nomikos_inference.architectures.ppocr_det.refinement import (
    _MAX_OVERLAP_PASSES,
    DUPLICATE_SHARED_RATIO,
    _cut_pair,
    _median_line_spacing,
    _shared_over_larger,
    _shared_ratio,
    merge_row_fragments,
    resolve_overlaps,
)
from nomikos_inference.architectures.ppocr_det.refinement import (
    _baseline_distance as baseline_distance,
)
from nomikos_inference.architectures.ppocr_det.refinement import _Work as Work


def _all_pairs_resolve_overlaps(
    works: list[Work],
    layout: PageLayout,
    *,
    cut_threshold: float,
    median_height: float,
) -> list[Work]:
    """The pre-index implementation, verbatim apart from the helper imports."""
    spacing = _median_line_spacing(layout, median_height)
    for _ in range(_MAX_OVERLAP_PASSES):
        pairs: list[tuple[float, int, int]] = []
        live = [i for i, work in enumerate(works) if not work.dropped]
        for pos_a, a in enumerate(live):
            for b in live[pos_a + 1 :]:
                ratio = _shared_ratio(works[a], works[b])
                if ratio >= cut_threshold:
                    pairs.append((-ratio, a, b))
        pairs.sort()
        changed = False
        for _, a, b in pairs:
            left, right = works[a], works[b]
            if left.dropped or right.dropped:
                continue
            ratio = _shared_ratio(left, right)
            if ratio < cut_threshold:
                continue
            distance = baseline_distance(left, right)
            if distance < spacing / 2:
                if left.role == "initial" or right.role == "initial":
                    left.overlap_unresolved = True
                    right.overlap_unresolved = True
                elif _shared_over_larger(left, right) > DUPLICATE_SHARED_RATIO:
                    if (right.score, right.members) > (left.score, left.members):
                        left.dropped = True
                    else:
                        right.dropped = True
                    changed = True
                else:
                    left.overlap_unresolved = True
                    right.overlap_unresolved = True
            elif _cut_pair(left, right):
                changed = True
            else:
                left.overlap_unresolved = True
                right.overlap_unresolved = True
        if not changed:
            break
    live_works = [work for work in works if not work.dropped]
    for pos_a, left in enumerate(live_works):
        for right in live_works[pos_a + 1 :]:
            if _shared_ratio(left, right) >= cut_threshold:
                left.overlap_unresolved = True
                right.overlap_unresolved = True
    return [work for work in works if not work.dropped]


def _box(x: float, y: float, width: float, height: float, score: float) -> DetectedQuad:
    return DetectedQuad(
        points=[[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        score=score,
    )


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _random_page(rng: np.random.Generator) -> list[DetectedQuad]:
    """A column or two of lines plus deliberate defects of every overlap kind."""
    quads: list[DetectedQuad] = []
    for column in range(int(rng.choice((1, 2)))):
        x0 = 20.0 + column * 420.0
        for row in range(int(rng.integers(3, 15))):
            y = 30.0 + row * 34.0 + _uniform(rng, -3, 3)
            height = _uniform(rng, 18, 26)
            width = _uniform(rng, 250, 380)
            quads.append(_box(x0 + _uniform(rng, -4, 4), y, width, height, _uniform(rng, 0, 1)))
            kind = _uniform(rng, 0, 1)
            if kind < 0.15:
                # Stacked neighbour pushed into this line.
                quads.append(_box(x0, y + height * 0.6, width, height, _uniform(rng, 0, 1)))
            elif kind < 0.25:
                # Near duplicate a pixel or two away.
                quads.append(
                    _box(
                        x0 + _uniform(rng, 0, 2),
                        y + _uniform(rng, 0, 2),
                        width,
                        height,
                        _uniform(rng, 0, 1),
                    )
                )
            elif kind < 0.32:
                # A small box inside the line.
                quads.append(_box(x0 + 40, y + 3, 30, height - 6, _uniform(rng, 0, 1)))
            elif kind < 0.37:
                # A tall initial over two lines.
                quads.append(_box(x0 - 30, y - 5, 28, height * 2.6, _uniform(rng, 0, 1)))
            elif kind < 0.45:
                # A row-mate fragment to merge.
                quads.append(_box(x0 + width + 3, y + 1, 60, height, _uniform(rng, 0, 1)))
    for _ in range(int(rng.integers(0, 4))):
        # Margin noise, sometimes overlapping text.
        quads.append(
            _box(
                _uniform(rng, 0, 800),
                _uniform(rng, 0, 500),
                _uniform(rng, 5, 60),
                15,
                _uniform(rng, 0, 1),
            )
        )
    return quads


def _snapshot(works: list[Work]) -> list[tuple[object, ...]]:
    return [
        (
            work.members,
            work.poly.wkb,
            work.baseline,
            work.area,
            work.dropped,
            work.overlap_unresolved,
            work.role,
            work.merged_from,
        )
        for work in works
    ]


def test_index_matches_the_all_pairs_original_on_200_random_pages() -> None:
    rng = np.random.default_rng(20261016)
    pages_with_defects = 0
    for _ in range(200):
        quads = _random_page(rng)
        layout = layout_lines(quads, direction="ltr")
        works = merge_row_fragments(quads, layout)
        threshold = float(rng.choice((0.05, 0.2, 0.5)))
        median_height = _uniform(rng, 15, 25)
        expected_works = copy.deepcopy(works)
        actual_works = copy.deepcopy(works)

        expected = _all_pairs_resolve_overlaps(
            expected_works, layout, cut_threshold=threshold, median_height=median_height
        )
        actual = resolve_overlaps(
            actual_works, layout, cut_threshold=threshold, median_height=median_height
        )

        assert _snapshot(actual) == _snapshot(expected)
        assert _snapshot(actual_works) == _snapshot(expected_works)
        if any(work.dropped or work.overlap_unresolved for work in expected_works) or any(
            a.poly.wkb != b.poly.wkb for a, b in zip(works, expected_works, strict=True)
        ):
            pages_with_defects += 1
    # The comparison is only worth something if the pages exercised the stage.
    assert pages_with_defects >= 150


def test_three_thousand_stacked_lines_resolve_in_seconds() -> None:
    """The default ``max_candidates``: all-pairs took about 105 s here."""
    quads = [
        DetectedQuad(
            points=[[0.0, float(i)], [100.0, float(i)], [100.0, float(i + 1)], [0.0, float(i + 1)]],
            score=float(i % 7),
        )
        for i in range(3000)
    ]
    layout = layout_lines(quads)
    works = merge_row_fragments(quads, layout)

    started = time.perf_counter()
    resolved = resolve_overlaps(works, layout)
    elapsed = time.perf_counter() - started

    assert len(resolved) == 3000
    assert elapsed < 10.0

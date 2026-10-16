"""Shared geometry for PP-OCRv6 refinement and response building.

Both the refinement stages (``merge``, ``overlaps``, ``refinement``) and the
response builder need the synthetic baseline and the refined line shape;
keeping them here lets ``response`` type its inputs without importing the
refinement stage that itself depends on the baseline helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad

Baseline = tuple[tuple[float, float], tuple[float, float]]


@dataclass
class RefinedLine:
    """One manuscript line after refinement, before ordering."""

    points: list[list[float]]
    baseline: list[list[float]]
    score: float
    members: tuple[int, ...] = ()
    role: str = "line"
    suspect: bool = False
    suspect_reason: str = ""
    merged_from: int = 1
    overlap_unresolved: bool = False


@dataclass
class Work:
    """Mutable geometry while merging and cutting."""

    poly: Polygon
    baseline: Baseline
    score: float
    members: tuple[int, ...]
    role: str = "line"
    merged_from: int = 1
    overlap_unresolved: bool = False
    dropped: bool = False
    suspect: bool = False
    suspect_reason: str = ""
    area: float = field(init=False)

    def __post_init__(self) -> None:
        self.area = self.poly.area


def synthetic_baseline_points(quad: list[list[float]], fraction: float) -> list[list[float]]:
    """Two-point baseline across a clockwise quad at ``fraction`` down."""
    top_left, top_right, bottom_right, bottom_left = quad
    left = [
        top_left[0] + fraction * (bottom_left[0] - top_left[0]),
        top_left[1] + fraction * (bottom_left[1] - top_left[1]),
    ]
    right = [
        top_right[0] + fraction * (bottom_right[0] - top_right[0]),
        top_right[1] + fraction * (bottom_right[1] - top_right[1]),
    ]
    return [left, right]


def bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def quad_height(quad: DetectedQuad) -> float:
    _, ymin, _, ymax = bbox(quad.points)
    return ymax - ymin


def median(values: list[float], fallback: float) -> float:
    """Upper median, or ``fallback`` for no values."""
    if not values:
        return fallback
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def is_initial(
    index: int,
    row: list[int] | tuple[int, ...],
    heights: list[float],
    fallback: float,
    ratio: float,
) -> bool:
    """A quad taller than ``ratio`` times its row mates' median height.

    With no row mates the reference is ``fallback`` (the page's builder
    median), so a lone tall quad is still recognised.
    """
    reference = median([heights[j] for j in row if j != index], fallback)
    return reference > 0 and heights[index] > ratio * reference


def work_of_quad(index: int, quad: DetectedQuad, baseline_fraction: float) -> Work:
    points = [[float(x), float(y)] for x, y in quad.points]
    baseline = synthetic_baseline_points(points, baseline_fraction)
    return Work(
        poly=Polygon(points),
        baseline=((baseline[0][0], baseline[0][1]), (baseline[1][0], baseline[1][1])),
        score=float(quad.score),
        members=(index,),
    )


def baseline_distance(left: Work, right: Work) -> float:
    """Perpendicular distance between the two baseline midpoints.

    When both baselines have zero length there is no direction to measure
    across, so the plain distance between the midpoints stands in.
    """
    first = np.asarray(left.baseline, dtype=float)
    second = np.asarray(right.baseline, dtype=float)
    first_dir = first[1] - first[0]
    second_dir = second[1] - second[0]
    if float(np.dot(first_dir, second_dir)) < 0:
        second_dir = -second_dir
    mean = first_dir + second_dir
    norm = float(np.linalg.norm(mean))
    offset = second.mean(axis=0) - first.mean(axis=0)
    if norm <= 0:
        return float(np.linalg.norm(offset))
    normal = np.array([-mean[1], mean[0]]) / norm
    return float(abs(np.dot(offset, normal)))


__all__ = [
    "Baseline",
    "RefinedLine",
    "Work",
    "baseline_distance",
    "bbox",
    "is_initial",
    "median",
    "quad_height",
    "synthetic_baseline_points",
    "work_of_quad",
]

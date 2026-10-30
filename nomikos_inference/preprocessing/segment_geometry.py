"""Polygon geometry helpers for segment preprocessing."""

from __future__ import annotations

import math
from typing import NotRequired, TypedDict

import numpy as np

MIN_VERTEX_SPACING_PX = 3.0


def distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def line_height(points: list[list[float]]) -> float:
    _, y0, _, y1 = bbox(points)
    return max(y1 - y0, 1.0)


def clean_polygon(
    points: list[list[float]],
    *,
    min_distance: float,
    min_vertices: int = 3,
) -> list[list[float]]:
    if len(points) < 2:
        return points

    working = [[float(point[0]), float(point[1])] for point in points]
    if len(working) >= 2 and distance(working[0], working[-1]) <= min_distance:
        working.pop()

    if len(working) < 2:
        return points

    cleaned: list[list[float]] = [working[0]]
    for point in working[1:]:
        if distance(point, cleaned[-1]) > min_distance:
            cleaned.append(point)

    if len(cleaned) >= 2 and distance(cleaned[0], cleaned[-1]) <= min_distance:
        cleaned.pop()

    return cleaned if len(cleaned) >= min_vertices else points


def approx_polygon(points: list[list[float]], *, epsilon: float) -> list[list[float]]:
    import cv2

    if len(points) < 3:
        return points
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
    if simplified is None or len(simplified) < 3:
        return points
    return [[float(x), float(y)] for x, y in simplified.reshape(-1, 2)]


def clamp_polygon_vertices(
    points: list[list[float]],
    *,
    max_points: int,
    min_points: int = 4,
) -> list[list[float]]:
    """Reduce a polygon to at most ``max_points`` vertices, never below four.

    The stored-geometry cap (``MAX_LINE_GEOMETRY_POINTS`` on the platform,
    mirrored by ``nomikos_inference.admission``) refuses polygons denser than
    ``max_points``. ``simplify_blla_boundary`` thins a raw boundary but does not
    target that cap, so every polygon is clamped here before it reaches the
    segment contract. The floor is four because the contract requires a polygon,
    not a triangle.
    """
    if len(points) <= max_points:
        return points

    result = points
    epsilon = 1.0
    for _ in range(40):
        candidate = approx_polygon(points, epsilon=epsilon)
        candidate = clean_polygon(candidate, min_distance=MIN_VERTEX_SPACING_PX)
        if len(candidate) < min_points:
            break
        result = candidate
        if len(result) <= max_points:
            return result
        epsilon *= 1.5

    # ``approxPolyDP`` flattened as far as it can without collapsing below a
    # triangle; keep every k-th vertex so the ring still follows the shape.
    if len(result) > max_points:
        step = len(result) / max_points
        result = [result[int(round(i * step))] for i in range(max_points)]
    return result


class BllaSimplificationMetrics(TypedDict):
    """Metrics ``simplify_blla_boundary`` reports alongside the cleaned ring.

    The spacing/epsilon entries are only present when the boundary had at
    least four points and was actually run through the simplifier.
    """

    simplification_status: str
    raw_point_count: int
    simplified_point_count: int
    min_vertex_spacing: NotRequired[float]
    epsilon: NotRequired[float]


def simplify_blla_boundary(
    points: list[list[float]],
    *,
    min_distance: float = MIN_VERTEX_SPACING_PX,
) -> tuple[list[list[float]], BllaSimplificationMetrics]:
    """Lightweight Douglas-Peucker + spacing cleanup for raw BLLA boundaries."""
    if len(points) < 4:
        return points, {
            "simplification_status": "unchanged",
            "raw_point_count": len(points),
            "simplified_point_count": len(points),
        }

    height_px = line_height(points)
    spacing = max(min_distance, 0.02 * height_px)
    reference = clean_polygon(points, min_distance=spacing)
    epsilon = max(2.0, 0.015 * height_px)
    simplified = approx_polygon(reference, epsilon=epsilon)
    simplified = clean_polygon(simplified, min_distance=spacing)
    if len(simplified) < 4:
        simplified = reference if len(reference) >= 4 else points
    status = "blla_boundary_simplified" if simplified != points else "blla_boundary_unchanged"

    return simplified, {
        "simplification_status": status,
        "raw_point_count": len(points),
        "simplified_point_count": len(simplified),
        "min_vertex_spacing": spacing,
        "epsilon": epsilon,
    }

"""BLLA baseline vectorization and reading order."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
from shapely import geometry as geom
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
from skimage import filters
from skimage.graph import MCP_Connect
from skimage.measure import approximate_polygon, label, regionprops
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

Baseline = list[list[int]]
#: Read-only points, so the reading order takes integer and float lines alike.
Points = Sequence[Sequence[float]]
#: A box as ``(rows, columns)`` slices, the shape the reading order compares.
Slices = tuple[slice, slice]

# A skeleton pixel with exactly one skeleton neighbour scores 10 + 1.
_ENDPOINT_KERNEL = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])


class _Region(Protocol):
    """The part of ``skimage.measure`` region properties tracing reads."""

    @property
    def coords(self) -> np.ndarray: ...


class _LineMCP(MCP_Connect):
    """Connect skeleton endpoints using the reference baseline scoring rule."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.connections: dict[tuple[int, int], tuple[object, object, float]] = {}
        self.scores: defaultdict[tuple[int, int], float] = defaultdict(lambda: np.inf)

    def create_connection(
        self,
        id1: int,
        id2: int,
        pos1: object,
        pos2: object,
        cost1: float,
        cost2: float,
    ) -> None:
        key = (min(id1, id2), max(id1, id2))
        score = cost1 + cost2
        if self.scores[key] > score:
            self.connections[key] = (pos1, pos2, score)
            self.scores[key] = score

    def get_connections(self) -> list[np.ndarray]:
        return [
            np.concatenate([self.traceback(pos1), self.traceback(pos2)[::-1]])
            for pos1, pos2, _score in self.connections.values()
        ]

    def goal_reached(self, _int_index: int, float_cumcost: float) -> int:
        return 2 if float_cumcost else 0


def _moore_neighborhood(current: np.ndarray, backtrack: np.ndarray) -> np.ndarray:
    operations = np.array(
        [
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
        ]
    )
    neighbors = (current + operations).astype(int)
    for index, point in enumerate(neighbors):
        if np.all(point == backtrack):
            return np.concatenate((neighbors[index:], neighbors[:index]))
    return np.empty((0, 2), dtype=int)


def _boundary_tracing(region: _Region) -> np.ndarray:
    coords = region.coords
    mins = np.amin(coords, axis=0)
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] - mins[0] + 3, maxs[1] - mins[1] + 3))
    y = coords[:, 0] - mins[0] + 1
    x = coords[:, 1] - mins[1] + 1
    binary[y, x] = 1

    index = 0
    while True:
        start = [y[index], x[index]]
        focus_start = binary[start[0] - 1 : start[0] + 2, start[1] - 1 : start[1] + 2]
        if np.sum(focus_start) > 1:
            break
        index += 1

    if binary[start[0] + 1, start[1]] == 0 and binary[start[0] + 1, start[1] - 1] == 0:
        backtrack_start = [start[0] + 1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = np.asarray(start)
    backtrack = np.asarray(backtrack_start)
    boundary: list[np.ndarray] = []
    while True:
        neighbors = _moore_neighborhood(current, backtrack)
        values = binary[neighbors[:, 0], neighbors[:, 1]]
        index = int(np.argmax(values))
        boundary.append(current)
        backtrack = neighbors[index - 1]
        current = neighbors[index]
        if np.all(current == start) and np.all(backtrack == backtrack_start):
            break
    return np.asarray(boundary) + [mins[0] - 1, mins[1] - 1]


def _region_boundaries(baseline_map: np.ndarray) -> list[BaseGeometry]:
    """Closed outlines of the baseline map's components, for extending ends."""
    boundaries: list[BaseGeometry] = []
    for region in regionprops(label(baseline_map)):
        if region.area < 6:
            continue
        try:
            boundary = _boundary_tracing(region)
            if len(boundary) > 3:
                boundaries.append(geom.Polygon(boundary).simplify(0.01).buffer(0))
        except Exception as error:  # noqa: BLE001 - one bad region is not a bad page
            # Skipped, not raised: a region shapely cannot close costs only its
            # own boundary, not the page's segmentation. Logged so a systematic
            # tracing failure doesn't look like a page with no extendable
            # baselines.
            logger.debug("boundary tracing skipped a region: %s", error)
            continue
    return boundaries


def _extend_end(baseline: Baseline, end: int, inner: int, outline: BaseGeometry) -> None:
    """Move ``baseline[end]`` out to ``outline`` along the ``inner`` to ``end`` direction.

    The point lands where a ray ten segments long meets the outline, or on the
    outline's nearest point when the ray does not meet it in one point.
    """
    point = baseline[end]
    if not outline.contains(geom.Point(point)):
        return
    ray = geom.LineString(
        [
            (
                point[0] - 10 * (baseline[inner][0] - point[0]),
                point[1] - 10 * (baseline[inner][1] - point[1]),
            ),
            point,
        ]
    )
    intersection = outline.boundary.intersection(ray)
    if intersection.geom_type != "Point":
        intersection = nearest_points(geom.Point(point), outline)[1]
    baseline[end] = np.asarray(intersection.coords[0], dtype=int).tolist()


def _extend_boundaries(baselines: list[Baseline], baseline_map: np.ndarray) -> list[Baseline]:
    """Extend each baseline's ends to the outline of the component holding it."""
    boundaries = _region_boundaries(baseline_map)
    for baseline in baselines:
        line = geom.LineString(baseline)
        outline = next((candidate for candidate in boundaries if candidate.contains(line)), None)
        if outline is None:
            continue
        _extend_end(baseline, 0, 1, outline)
        _extend_end(baseline, -1, -2, outline)
    return baselines


def _endpoint_budget(
    components: set[int],
    endpoint_counts: dict[int, int],
    skeleton_labels: np.ndarray,
    max_endpoints: int,
) -> set[int]:
    """The largest components whose endpoints fit in ``max_endpoints`` together."""
    component_sizes = np.bincount(skeleton_labels.ravel())
    by_size = sorted(
        components,
        key=lambda component_id: component_sizes[component_id],
        reverse=True,
    )
    kept: set[int] = set()
    budget = 0
    for component_id in by_size:
        endpoints = endpoint_counts[component_id]
        if budget + endpoints > max_endpoints:
            break
        kept.add(component_id)
        budget += endpoints
    return kept


def _prune_endpoints(
    extrema: np.ndarray, line_skeleton: np.ndarray, max_endpoints: int
) -> np.ndarray:
    """Drop the endpoints of noisy skeleton components until the budget holds.

    A component with more than ten endpoints is noise, not a line; if the
    rest still exceed ``max_endpoints``, the largest components are kept.
    """
    skeleton_labels = label(line_skeleton)
    extrema_components = skeleton_labels[extrema[:, 0], extrema[:, 1]]
    component_ids, component_counts = np.unique(extrema_components, return_counts=True)
    # `np.unique(..., return_counts=True)` returns the two arrays paired
    # element-for-element, so a length mismatch here would mean numpy broke.
    endpoint_counts = dict(zip(component_ids.tolist(), component_counts.tolist(), strict=True))
    valid_components = {
        component_id for component_id, count in endpoint_counts.items() if count <= 10
    }
    remaining_endpoints = sum(endpoint_counts[component_id] for component_id in valid_components)
    if remaining_endpoints > max_endpoints:
        valid_components = _endpoint_budget(
            valid_components, endpoint_counts, skeleton_labels, max_endpoints
        )
    return extrema[np.isin(extrema_components, list(valid_components))]


def _oriented(baseline: Baseline, filtered_start: np.ndarray, filtered_end: np.ndarray) -> Baseline:
    """Point the baseline from its start-heavy end to its end-heavy end.

    When the start and end maps do not agree, the line runs top to bottom in
    image rows, the reference decoder's horizontal fallback.
    """
    first = tuple(baseline[0])
    last = tuple(baseline[-1])
    first_lean = filtered_start[first] - filtered_end[first]
    last_lean = filtered_start[last] - filtered_end[last]
    if first_lean > 0.2 and last_lean < -0.2:
        return baseline
    if first_lean < -0.2 and last_lean > 0.2:
        return baseline[::-1]
    if baseline[0][1] > baseline[-1][1]:
        return baseline[::-1]
    return baseline


def vectorize_lines(
    channels: np.ndarray,
    *,
    threshold: float,
    min_length: float,
    max_endpoints: int = 400,
) -> list[Baseline]:
    """Apply baseline skeletonization and endpoint connection."""

    start_map, end_map, baseline_map = channels[:3]
    filtered_baseline = filters.sato(baseline_map, black_ridges=False, mode="constant")
    binary_baseline = filtered_baseline > threshold
    line_skeleton = skeletonize(binary_baseline)
    extrema = np.transpose(
        np.where((convolve2d(line_skeleton, _ENDPOINT_KERNEL, mode="same") == 11) * line_skeleton)
    )
    if len(extrema) > max_endpoints:
        extrema = _prune_endpoints(extrema, line_skeleton, max_endpoints)
    if len(extrema) < 2:
        return []

    mcp = _LineMCP(~line_skeleton)
    try:
        mcp.find_costs(extrema)
    except ValueError:
        return []

    lines = [approximate_polygon(line, 3).tolist() for line in mcp.get_connections()]
    lines = _extend_boundaries(lines, binary_baseline)
    filtered_start = maximum_filter(start_map, size=20)
    filtered_end = maximum_filter(end_map, size=20)

    oriented: list[Baseline] = []
    for baseline in lines:
        directed = _oriented(baseline, filtered_start, filtered_end)
        if geom.LineString(directed).length >= min_length:
            oriented.append([point[::-1] for point in directed])
    return oriented


def vectorize_regions(region_probability: np.ndarray) -> list[list[list[int]]]:
    """Vectorize text regions for reading order and ROIs."""

    boundaries = []
    for region in regionprops(label(region_probability > 0.5)):
        # Same region filter ``_extend_boundaries`` applies: a single isolated
        # pixel produces a component with no traversable boundary, and
        # ``_boundary_tracing`` would walk past the end of its coordinate array
        # and raise, failing the whole page for one stray pixel.
        if region.area < 6:
            continue
        try:
            boundary = _boundary_tracing(region)
        except Exception as error:  # noqa: BLE001 - one bad region is not a bad page
            logger.debug("boundary tracing skipped a region: %s", error)
            continue
        if len(boundary) > 2:
            boundaries.append(geom.Polygon(boundary))
    if not boundaries:
        return []
    merged = unary_union(boundaries)
    if merged.geom_type == "Polygon":
        polygons = [merged.boundary.simplify(10)]
    elif merged.geom_type == "MultiPolygon":
        polygons = [item.boundary.simplify(10) for item in merged.geoms]
    else:
        # ``unary_union`` can collapse degenerate regions into a LineString,
        # whose boundary carries no ``coords``. Nothing usable to vectorize.
        return []
    return [np.asarray(polygon.coords, dtype=np.uint)[:, [1, 0]].tolist() for polygon in polygons]


def is_in_region(line: geom.LineString, region: geom.Polygon) -> bool:
    midpoint = line.interpolate(0.5, normalized=True)
    return region.contains(midpoint)


def _x_overlaps(first: Slices, second: Slices) -> bool:
    return first[1].start < second[1].stop and first[1].stop > second[1].start


def _separates(middle: Slices, first: Slices, second: Slices) -> bool:
    """``middle`` sits between the pair in rows and spans the gap in columns."""
    if middle in (first, second):
        return False
    if middle[0].stop < min(first[0].start, second[0].start):
        return False
    if middle[0].start > max(first[0].stop, second[0].stop):
        return False
    return middle[1].start < first[1].stop and middle[1].stop > second[1].start


def _precedes(first: Slices, second: Slices, lines: list[Slices]) -> bool:
    """Column mates read top to bottom; others left to right unless separated."""
    if _x_overlaps(first, second):
        return first[0].start < second[0].start
    return first[1].stop < second[1].start and not any(
        _separates(middle, first, second) for middle in lines
    )


def _reading_order(lines: list[Slices]) -> np.ndarray:
    order = np.zeros((len(lines), len(lines)), dtype="uint8")
    for first_index, first in enumerate(lines):
        for second_index, second in enumerate(lines):
            if _precedes(first, second, lines):
                order[first_index, second_index] = 1
    return order


def _topological_sort(order: np.ndarray) -> list[int]:
    visited = np.zeros(len(order))
    result: list[int] = []

    def visit(index: int) -> None:
        if visited[index]:
            return
        visited[index] = 1
        predecessors = np.nonzero(np.ravel(order[:, index]))[0]
        for predecessor in predecessors:
            visit(int(predecessor))
        result.append(index)

    for index in range(len(order)):
        visit(index)
    return result


def _line_slices(baseline: Points) -> Slices:
    bounds = geom.LineString(baseline).bounds
    return slice(bounds[1], bounds[3]), slice(bounds[0], bounds[2])


def _ordered_within(items: list[tuple[int, Slices]]) -> list[int]:
    """Line indices of one region, in their reading order."""
    order = _topological_sort(_reading_order([item[1] for item in items]))
    return [items[position][0] for position in order]


def _assign_to_regions(
    baselines: Sequence[Points], region_polygons: list[geom.Polygon]
) -> tuple[list[list[tuple[int, Slices]]], list[Slices], dict[int, int | None]]:
    """Each line's first containing region, or the line on its own.

    Returns the lines per region, the boxes of the lines outside every region,
    and those lines' indices, each mapped to ``None`` for "no region".
    """
    region_lines: list[list[tuple[int, Slices]]] = [[] for _ in region_polygons]
    bounds: list[Slices] = []
    members: dict[int, int | None] = {}
    for line_index, baseline in enumerate(baselines):
        line = geom.LineString(baseline)
        home = next(
            (index for index, region in enumerate(region_polygons) if is_in_region(line, region)),
            None,
        )
        if home is None:
            bounds.append(_line_slices(baseline))
            members[line_index] = None
        else:
            region_lines[home].append((line_index, _line_slices(baseline)))
    return region_lines, bounds, members


def reading_order_indices(
    baselines: Sequence[Points],
    regions: Sequence[Points] | None = None,
) -> list[int]:
    """Return polygon-aware reading-order indices.

    Lines outside every region and the regions themselves are ordered as boxes
    on the page; each region then expands into its own lines' order.
    """

    if not regions:
        return _topological_sort(_reading_order([_line_slices(line) for line in baselines]))

    region_polygons = [geom.Polygon(region) for region in regions]
    region_lines, bounds, members = _assign_to_regions(baselines, region_polygons)
    intra_region_order: list[list[int]] = [[] for _ in region_polygons]
    last_line_index = len(baselines) - 1
    for region_index, region in enumerate(region_polygons):
        if region_lines[region_index]:
            intra_region_order[region_index] = _ordered_within(region_lines[region_index])
            region_bounds = region.bounds
            bounds.append(
                (
                    slice(region_bounds[1], region_bounds[3]),
                    slice(region_bounds[0], region_bounds[2]),
                )
            )
            members[last_line_index + region_index + 1] = region_index

    keys = sorted(members)
    ordered: list[int] = []
    for position in _topological_sort(_reading_order(bounds)):
        key = keys[position]
        home = members[key]
        if home is None:
            ordered.append(key)
        else:
            ordered.extend(intra_region_order[home])
    return ordered

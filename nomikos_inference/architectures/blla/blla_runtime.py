"""BLLA decoding and segment-contract conversion."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import SupportsFloat

import numpy as np
from PIL import Image

from nomikos_inference.architectures.blla.blla_decoder import decode_blla_heatmaps
from nomikos_inference.architectures.blla.blla_preprocessing import BLLAInput
from nomikos_inference.architectures.isolation import reraise_if_none_survived
from nomikos_inference.contracts.common import MAX_GEOMETRY_POINTS
from nomikos_inference.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse
from nomikos_inference.preprocessing.segment_geometry import (
    clamp_polygon_vertices,
    simplify_blla_boundary,
)

logger = logging.getLogger(__name__)


def _positive_float_param(params: Mapping[str, object], key: str, default: float) -> float:
    """Parse a caller-supplied positive number, falling back to the default.

    Upper bounds are *not* checked here: they belong to
    ``admission.validate_segment_params``, which every entry point into the
    runner passes through (sync run, queued job). Duplicating them would mean
    two places to keep in step, and the one that runs first is the one that can
    still return a 422 instead of a half-built response.
    """
    value = params.get(key, default)
    if not isinstance(value, str | bytes | SupportsFloat):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def build_blla_segment_response(
    image: Image.Image,
    logits: np.ndarray,
    prepared: BLLAInput,
    *,
    params: Mapping[str, object] | None = None,
) -> SegmentRunResponse:
    """Decode logits and preserve the native BLLA response contract."""

    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("BLLA logits must have shape (4, height, width)")

    params = params or {}
    threshold = _positive_float_param(params, "heatmap_threshold", 0.17)
    threshold = min(threshold, 0.99)

    width, height = image.size
    decoded_lines = decode_blla_heatmaps(
        values,
        image_size=(width, height),
        threshold=threshold,
        raw_logits=True,
        scaled_gray=prepared.scaled_gray,
    )

    block = SegmentBlock(
        external_id="blla-block-1",
        order=0,
        box={
            "points": [
                [0.0, 0.0],
                [float(width), 0.0],
                [float(width), float(height)],
                [0.0, float(height)],
            ]
        },
    )

    lines: list[SegmentLine] = []
    # Holding the first failure is what distinguishes "every line failed" from
    # "the decoder found nothing worth emitting"; a separate counter would say
    # the same thing twice.
    first_failure: Exception | None = None
    for order, decoded in enumerate(decoded_lines):
        baseline = decoded.baseline
        ceiling = decoded.polygon
        if len(ceiling) < 4 or len(baseline) < 2:
            continue

        source_metadata: dict[str, object] = {
            "adapter": "blla",
            "decoder": "native",
            "raw_order": order,
        }
        try:
            simplified_points, simplify_metrics = simplify_blla_boundary(ceiling)
        except Exception as error:  # noqa: BLE001 - one bad polygon is not a bad page
            # Simplification is per-line geometry work: a degenerate contour
            # that trips OpenCV must cost its own line, not the other
            # thirty-nine on the page. The line is dropped exactly like the
            # short-ceiling case above; the failure is kept so an all-failed
            # page can still raise.
            first_failure = first_failure or error
            logger.warning(
                "BLLA line simplification failed (raw_order=%s, ceiling_points=%s)",
                order,
                len(ceiling),
                exc_info=error,
            )
            continue

        source_metadata.update(simplify_metrics)
        # Clamp to the stored-geometry cap before the segment contract (which
        # enforces it with ``max_length``) rejects the line, so a denser ring is
        # coarsened rather than failing the whole page.
        capped_points = clamp_polygon_vertices(
            simplified_points,
            max_points=MAX_GEOMETRY_POINTS,
        )
        if len(capped_points) < 4:
            first_failure = first_failure or ValueError(
                "BLLA line simplified to fewer than four points"
            )
            logger.warning(
                "BLLA line simplification produced no polygon (raw_order=%s)",
                order,
            )
            continue
        lines.append(
            SegmentLine(
                external_id=f"blla-line-{order + 1}",
                order=len(lines),
                block_external_id=block.external_id,
                baseline={"points": baseline},
                mask={"points": capped_points},
                points=capped_points,
                kraken_ceiling=ceiling,
                source_metadata=source_metadata,
            )
        )

    # Every candidate line blew up: an empty response would read as "this page
    # has no text", which is a far worse lie than a 5xx. The same verdict is
    # reached from the Calamari batch loop, so the rule itself lives in
    # ``architectures.isolation``; a page of pure *skips* (short ceilings, no
    # failures) still returns empty, which is why ``first_failure`` gates it.
    reraise_if_none_survived(survivors=len(lines), first_failure=first_failure)

    return SegmentRunResponse(blocks=[block] if lines else [], lines=lines)

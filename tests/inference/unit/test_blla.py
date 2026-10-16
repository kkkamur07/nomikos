"""Focused tests for the BLLA preprocessing, decoder and runner path.

The graph itself, and the Kraken-oracle comparisons that used it, moved to
``tests/export`` with the Torch modules under ADR 0006. What is left here is
what a researcher's install actually contains: NumPy preprocessing, the
Torch-free decoder, and the runner that ties them to the ONNX session.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from nomikos_inference.architectures.blla.blla_decoder.common import resize_heatmaps_nearest
from nomikos_inference.architectures.blla.blla_preprocessing import (
    MAX_WIDTH_TO_HEIGHT_RATIO,
    preprocess_blla_image,
)


def test_blla_preprocessing_matches_fixed_height_rgb_inversion() -> None:
    image = Image.new("RGB", (20, 10), (0, 64, 255))

    prepared = preprocess_blla_image(image)

    assert prepared.array.shape == (3, 1800, 3600)
    assert prepared.array.dtype == np.float32
    np.testing.assert_allclose(
        prepared.array[:, 0, 0], np.array([1.0, 0.7490196, 0.0], dtype=np.float32), atol=1e-6
    )
    assert prepared.scaled_gray.shape == (1800, 3600)
    assert prepared.scale_xy == pytest.approx((20 / 3600, 10 / 1800))


def test_blla_preprocessing_caps_extreme_aspect_ratio_width() -> None:
    """A panorama within the pixel admission cap must not produce an unbounded array."""
    input_height = 90
    capped_width = input_height * MAX_WIDTH_TO_HEIGHT_RATIO
    image = Image.new("RGB", (4000, 10), (255, 255, 255))

    prepared = preprocess_blla_image(image, input_height=input_height)

    assert prepared.array.shape == (3, input_height, capped_width)
    # Coordinates still map back to source space through scale_xy.
    assert prepared.scale_xy == pytest.approx((4000 / capped_width, 10 / input_height))


def test_blla_width_clamp_bounds_numerics_and_not_only_memory() -> None:
    """The clamp's value is a measurement, so pin the width it actually admits.

    It was 8 while it only had to keep an extreme panorama from allocating a
    multi-gigabyte array. Under ADR 0006 it also bounds how far the ONNX graph
    may drift from the Torch oracle, because the scaled width is the free axis
    and the disagreement grows with it - roughly linearly in the RMS, from
    1.7e-05 on a real 2471-wide page to 1.9e-04 at 14400. Logits crossing the
    0.5 boundary, which the decoder is discontinuous at, go from none below 5400
    to a handful at every width from 7200 up.

    Three keeps the drift within a factor of three of the page ADR 0006
    validated and still leaves every codex shape untouched: a single leaf is
    near 0.7:1 and a two-page spread reaches about 2.5:1. What it costs is
    horizontal resolution on a stitched scroll, which is squeezed by up to 2.67x
    where 8:1 used to pass through.

    ``tests/export/test_blla_onnx.py`` holds the numeric half of this claim
    against the real weights; this half is what a researcher's Torch-free
    install can check.
    """
    panorama = Image.new("RGB", (4000, 100), (255, 255, 255))

    prepared = preprocess_blla_image(panorama)

    assert prepared.array.shape == (3, 1800, 5400)
    assert MAX_WIDTH_TO_HEIGHT_RATIO == 3


def test_blla_preprocessing_is_proportional_below_the_width_cap() -> None:
    image = Image.new("RGB", (20, 10), (255, 255, 255))

    prepared = preprocess_blla_image(image, input_height=90)

    assert prepared.array.shape == (3, 90, 180)


def test_nearest_resize_repeats_source_pixels_rather_than_blending() -> None:
    """The decoder's upsample is index arithmetic, so no new values may appear.

    This replaced ``torch.nn.functional.interpolate`` when Torch left the
    runtime. Interpolation with any other mode would invent intermediate
    probabilities and move the 0.17 threshold's boundary.
    """
    heatmaps = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)

    resized = resize_heatmaps_nearest(heatmaps, height=4, width=6)

    assert resized.shape == (4, 4, 6)
    assert set(np.unique(resized)) <= set(np.unique(heatmaps))
    # Row 0 of channel 0 is its two source pixels, each repeated three times.
    np.testing.assert_array_equal(resized[0, 0], np.array([0, 0, 0, 1, 1, 1], dtype=np.float32))


# `test_run_model_returns_a_blla_response_for_a_real_image` stood here and ran the
# published artifact through `run_model`, asserting `blocks == 1` and `len(lines) > 10`.
# `test_onnx_runtime.py::test_segment_runs_the_graph_on_real_weights` makes the same two
# assertions on the same artifact and the same page; the `run_model` dispatch increment is
# covered by `test_transcribe_batch_isolation.py`'s runner fixture and end to end by
# `test_cli_run.py`. Cutting it removes a second full model execution from the unit lane.

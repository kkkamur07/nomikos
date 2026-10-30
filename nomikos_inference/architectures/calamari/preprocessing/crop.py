"""Cut one line out of a page the way that model's training crops were cut.

There is one crop function, ``crop_polygon`` in
``src/preprocessing_data/syriac/xml_to_data.py`` (Armenian reaches it through
``src/preprocessing_data/armenian.py``), and every registry model was trained on
its output: the polygon's bounding box, widened by a padding and clamped to the
page, with every pixel outside the polygon painted white on the crop, then
``convert("L")``. What differs between models is only that padding, which is why
the registry declares it per model rather than this file assuming one. The Greek
finetuning crops were exported with 0 px and the Armenian and Syriac ones with
12; feeding Greek the Armenian padding costs it 125 exact lines out of 204 down
to 0 (CER 0.050 to 0.304) on its own corpus.

Serving has to cut the same picture the model was shown, or it reads a line it
was never trained for. ``cv2.fillPoly`` is kept rather than swapped for
``ImageDraw.polygon`` because the two rasterise polygon edges differently, and a
one-pixel disagreement along every edge is exactly the kind of skew this file
exists to prevent.

The array this module returns is not yet the picture the model saw. The training
exporter wrote each crop through ``save_crop`` as a grayscale JPEG at
``quality=82, optimize=True`` and the trainer's loader read that file back, so
the model was fitted on pixels that had been through a lossy round trip. The
runner re-encodes the same way before handing the crop on; see
``nomikos_inference/jobs/runner.py::_crop_line_image``.

Parity with the training function is asserted by
``tests/inference/unit/test_calamari_training_parity.py``.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from nomikos_inference.contracts.common import LineCrop

#: ``PADDING`` in ``src/preprocessing_data/syriac/xml_to_data.py`` and
#: ``src/preprocessing_data/armenian.py``. It is the exporter's default and so
#: the default here, but it is not universal: ``greek-calamari-v1`` was exported
#: with 0. Every transcribe entry in the registry states its own.
TRAINING_CROP_PADDING = 12


def crop_bounds(
    points: np.ndarray, *, width: int, height: int, padding: int
) -> tuple[int, int, int, int]:
    """Inclusive ``(x_min, y_min, x_max, y_max)`` of the padded, clamped box."""
    x_min = max(0, int(points[:, 0].min()) - padding)
    y_min = max(0, int(points[:, 1].min()) - padding)
    x_max = min(width - 1, int(points[:, 0].max()) + padding)
    y_max = min(height - 1, int(points[:, 1].max()) + padding)
    if x_max < x_min or y_max < y_min:
        raise ValueError(f"Invalid crop bounds: {(x_min, y_min, x_max, y_max)}")
    return x_min, y_min, x_max, y_max


def _integer_points(points: list[list[float]]) -> np.ndarray:
    """Round to integers exactly as the training exporter's ``parse_points`` does.

    The PAGE-XML parser upstream of ``crop_polygon`` turns every coordinate
    into ``int(round(float(value)))`` before the crop ever sees it. Segments
    stored by the platform carry fractional coordinates on a large share of
    lines (92 of 819 Armenian lines, 116 of 204 Greek lines), and truncating
    them instead moves the box by a pixel, which at a 30 px line height changes
    the resized width by several frames and the decoded text with it.
    """
    polygon = np.array(
        [[int(round(float(x))), int(round(float(y)))] for x, y in points], dtype=np.int32
    )
    if polygon.ndim != 2 or polygon.shape[0] == 0:
        raise ValueError("line has no points")
    return polygon


#: A page as ``crop_line`` accepts it: an open image, or the RGB ``uint8``
#: array ``page_pixels`` made from one. A batch converts the page once and
#: passes the array, so an N-line page pays one full-page conversion, not N.
Page = Image.Image | np.ndarray


def page_pixels(page: Page) -> np.ndarray:
    """The page as an RGB ``uint8`` array; an array is taken to be one already."""
    if isinstance(page, np.ndarray):
        return page
    return np.asarray(page.convert("RGB"))


def _cut(rgb: np.ndarray, polygon: np.ndarray, padding: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the RGB crop of the padded, clamped box and the box itself."""
    height, width = rgb.shape[:2]
    bounds = crop_bounds(polygon, width=width, height=height, padding=padding)
    x_min, y_min, x_max, y_max = bounds
    crop = rgb[y_min : y_max + 1, x_min : x_max + 1].copy()
    if crop.size == 0:
        raise ValueError(f"Empty crop for bounds {bounds}")
    return crop, bounds


def _to_grayscale(crop: np.ndarray) -> np.ndarray:
    # Through PIL rather than ``cv2.cvtColor``: the luminance coefficients agree
    # but the rounding does not, and the training exporter goes through PIL.
    return np.asarray(Image.fromarray(crop, mode="RGB").convert("L"), dtype=np.uint8)


def crop_line_on_white_rgb(
    page: Page,
    points: list[list[float]],
    *,
    padding: int = TRAINING_CROP_PADDING,
) -> np.ndarray:
    """Return the RGB polygon-on-white crop of ``points`` from ``page``.

    Fewer than three points cannot be filled as a polygon; the training exporter
    skipped such lines outright, so here they fall back to the padded box without
    a mask rather than failing the whole page.
    """
    polygon = _integer_points(points)
    crop, (x_min, y_min, _x_max, _y_max) = _cut(page_pixels(page), polygon, padding)

    if polygon.shape[0] >= 3:
        shifted: npt.NDArray[np.int32] = polygon - np.array([[x_min, y_min]], dtype=np.int32)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shifted], (255,))
        masked = np.full_like(crop, 255)
        masked[mask == 255] = crop[mask == 255]
        crop = masked

    return crop


def crop_line_on_white(
    page: Page,
    points: list[list[float]],
    *,
    padding: int = TRAINING_CROP_PADDING,
) -> np.ndarray:
    """Return the grayscale polygon-on-white crop of ``points`` from ``page``."""
    return _to_grayscale(crop_line_on_white_rgb(page, points, padding=padding))


def _require_known_crop(line_crop: LineCrop) -> None:
    if line_crop != LineCrop.polygon_white:
        raise ValueError(f"unsupported line crop: {line_crop!r}")


def crop_line(
    page: Page,
    points: list[list[float]],
    line_crop: LineCrop,
    *,
    padding: int,
) -> np.ndarray:
    """Cut the line the way this model's registry entry says its training data was cut.

    ``padding`` has no default on purpose. It is the one number the models
    disagree about, and a default here would silently reinstate the guess the
    registry field exists to remove.
    """
    _require_known_crop(line_crop)
    return crop_line_on_white(page, points, padding=padding)


def crop_line_rgb(
    page: Page,
    points: list[list[float]],
    line_crop: LineCrop,
    *,
    padding: int,
) -> np.ndarray:
    """``crop_line`` without the grayscale step, for models trained on colour crops."""
    _require_known_crop(line_crop)
    return crop_line_on_white_rgb(page, points, padding=padding)

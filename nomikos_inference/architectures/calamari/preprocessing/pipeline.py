"""Calamari line preprocessing for the ONNX inference runtime.

**Serving must reproduce the training loader exactly.** The models in the
registry were trained by ``src/models/calamari/trainer.py`` through
``src/models/calamari/data.py::_load_line_image``, which does precisely three
things to a line crop:

1. ``Image.convert("L")``,
2. an aspect-preserving resize to the model's line height with
   ``Image.Resampling.BILINEAR`` (``width = max(1, round(w * h_target / h))``),
3. a transpose to ``time x height`` with the raw ``uint8`` values kept as they
   are (the graph divides by 255 itself).

Dark ink on a light background, never inverted, never dewarped, never padded.
This module is that recipe and nothing more. Until 2026-09-07 it mirrored the
legacy TensorFlow Calamari processors instead (centre-line dewarping, inversion
to white-on-black, a 16 px zero pad), which no registry model was ever trained
on; on Armenian training pages that produced 1 exact line in 30 where the
training recipe produces 30 in 30 on the same ``best.onnx``. See ADR 0007.

If the trainer's loader changes, this file changes with it, and
``tests/inference/unit/test_calamari_training_parity.py`` is what says so.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from nomikos_inference.admission import open_image_bytes


def preprocess_line_image_bytes_to_calamari_tensor(
    image_bytes: bytes,
    *,
    line_height: int = 48,
) -> np.ndarray:
    """Return Calamari model input for encoded image bytes."""
    with open_image_bytes(image_bytes) as image:
        image_array = np.asarray(image.convert("L"), dtype=np.uint8)
    return preprocess_line_array_to_calamari_tensor(image_array, line_height=line_height)


def preprocess_line_array_to_calamari_tensor(
    image: np.ndarray,
    *,
    line_height: int = 48,
) -> np.ndarray:
    """Return Calamari model input for a grayscale uint8 line array.

    This is ``src/models/calamari/data.py::_load_line_image`` without the
    ``torch.from_numpy``: the same PIL resize, the same rounding of the target
    width, the same transpose, the same untouched ``uint8`` values.
    """
    if line_height <= 0:
        raise ValueError("line_height must be positive")
    if image.ndim != 2:
        raise ValueError(f"expected a grayscale height x width array, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixels, got {image.dtype}")
    if image.size == 0:
        raise ValueError("line image is empty")

    line = Image.fromarray(image, mode="L")
    width = max(1, round(line.width * line_height / line.height))
    line = line.resize((width, line_height), Image.Resampling.BILINEAR)
    # ``time x height``, one channel, batch of one.
    return np.asarray(line, dtype=np.uint8).T[None, :, :, None]

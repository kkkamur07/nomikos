"""TrOCR line augmentation using the Calamari easy / mild / hard pools.

Each training copy uses the shared 5-copy scheme (two easy, two mild, one
hard) plus one unaugmented original. TrOCR does not apply Calamari's Syriac
width stretch or squash, and it never uses Calamari's CTC tensor layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
from PIL import Image

from ..calamari.augmentation import augment_grayscale_line


@dataclass
class LineAugmentation:
    """Apply the Calamari easy, mild, or hard stack to one TrOCR training copy."""

    probability: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("TrOCR augmentation probability must be between zero and one.")

    def apply(self, image: Image.Image, strength: str, operation_count: int) -> Image.Image:
        """Return an RGB line image using the Calamari easy, mild, or hard pool."""
        if numpy.random.random() >= self.probability:
            return image
        pixels = numpy.asarray(image.convert("L"))
        augmented = augment_grayscale_line(pixels, strength, operation_count)
        if float(augmented.max(initial=0.0)) <= 1.0:
            augmented = augmented * 255.0
        gray = numpy.clip(augmented, 0.0, 255.0).astype(numpy.uint8)
        return Image.fromarray(gray, mode="L").convert("RGB")

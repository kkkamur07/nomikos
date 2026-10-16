"""Calamari line-image preprocessing: the training recipe, and only that."""

from nomikos_inference.architectures.calamari.preprocessing.conversion import (
    load_line_image_grayscale,
)
from nomikos_inference.architectures.calamari.preprocessing.crop import (
    TRAINING_CROP_PADDING,
    Page,
    crop_line,
    crop_line_on_white,
    crop_line_rgb,
    page_pixels,
)
from nomikos_inference.architectures.calamari.preprocessing.pipeline import (
    preprocess_line_array_to_calamari_tensor,
    preprocess_line_image_bytes_to_calamari_tensor,
    preprocess_line_image_to_calamari_tensor,
)

__all__ = [
    "TRAINING_CROP_PADDING",
    "Page",
    "crop_line",
    "crop_line_on_white",
    "crop_line_rgb",
    "page_pixels",
    "load_line_image_grayscale",
    "preprocess_line_array_to_calamari_tensor",
    "preprocess_line_image_bytes_to_calamari_tensor",
    "preprocess_line_image_to_calamari_tensor",
]

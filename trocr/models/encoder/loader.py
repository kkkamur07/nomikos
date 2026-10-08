"""Helpers for TrOCR's DeiT image encoder."""

from __future__ import annotations

from transformers import AutoImageProcessor, VisionEncoderDecoderModel


def load_image_processor(model_source: str):
    """Load the image resize and normalization settings stored with TrOCR."""
    return AutoImageProcessor.from_pretrained(model_source)


def freeze_encoder(model: VisionEncoderDecoderModel) -> int:
    """Freeze visual-encoder weights and return their parameter count."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    return sum(parameter.numel() for parameter in model.encoder.parameters())

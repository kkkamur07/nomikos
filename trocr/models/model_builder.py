"""Construct the Hugging Face TrOCR model."""

from __future__ import annotations

import logging

from transformers.utils import logging as transformers_logging

from .decoder import configure_decoder
from .encoder import freeze_encoder
from .encoder.lora import configure_encoder_lora
from .model_loader import TrOCRVisionEncoderDecoderModel


LOGGER = logging.getLogger(__name__)
_IGNORED_MISSING_KEYS = {
    # Tied to decoder token embeddings and omitted by SafeTensors serialization.
    "decoder.output_projection.weight",
    "encoder.pooler.dense.bias",
    "encoder.pooler.dense.weight",
}


def load_model(model_source: str) -> TrOCRVisionEncoderDecoderModel:
    """Load Microsoft TrOCR weights into the local DeiT and TrOCR classes."""
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        model, loading_info = TrOCRVisionEncoderDecoderModel.from_pretrained(
            model_source,
            output_loading_info=True,
        )
    finally:
        transformers_logging.set_verbosity(previous_verbosity)

    missing_keys = set(loading_info.get("missing_keys", []))
    unexpected_keys = set(loading_info.get("unexpected_keys", []))
    if missing_keys - _IGNORED_MISSING_KEYS or unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint loading mismatch: "
            f"missing={sorted(missing_keys - _IGNORED_MISSING_KEYS)}, "
            f"unexpected={sorted(unexpected_keys)}"
        )
    if "decoder.output_projection.weight" in missing_keys:
        # SafeTensors only drops this tensor when it shared storage with the token
        # embeddings at save time; restore that tie instead of keeping a random init.
        decoder = model.decoder
        decoder.get_output_embeddings().weight = decoder.get_input_embeddings().weight
    if missing_keys & _IGNORED_MISSING_KEYS and hasattr(model.encoder, "pooler"):
        model.encoder.pooler = None
    return model


def build_model(
    model_source: str,
    tokenizer,
    *,
    max_target_length: int,
    freeze_visual_encoder: bool = True,
    reinitialize_decoder: bool | str = True,
    tie_decoder_embeddings: bool = False,
    decoder_dropout: float = 0.1,
    lora_config: dict[str, object] | None = None,
) -> TrOCRVisionEncoderDecoderModel:
    """Load TrOCR and adapt its decoder to a replacement tokenizer."""
    model = load_model(model_source)
    decoder_parameters = configure_decoder(
        model,
        tokenizer,
        max_target_length=max_target_length,
        reinitialize=reinitialize_decoder,
        tie_embeddings=tie_decoder_embeddings,
        dropout=decoder_dropout,
    )
    if freeze_visual_encoder:
        encoder_parameters = freeze_encoder(model)
        LOGGER.info("Frozen visual encoder (%d parameters).", encoder_parameters)
    lora_adapters = configure_encoder_lora(model.encoder, lora_config)
    model.config.encoder_lora = dict(lora_config) if lora_config else None
    if lora_adapters:
        LOGGER.info("Added LoRA to %d encoder attention projections.", lora_adapters)
    LOGGER.info("Decoder parameter count: %d", decoder_parameters)
    return model

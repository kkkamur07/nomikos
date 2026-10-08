"""Configure TrOCR's decoder for a replacement tokenizer."""

from __future__ import annotations

from torch import nn
from transformers import PreTrainedTokenizerBase, VisionEncoderDecoderModel


def configure_decoder(
    model: VisionEncoderDecoderModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_target_length: int,
    reinitialize: bool | str = True,
    tie_embeddings: bool = False,
    dropout: float = 0.1,
) -> int:
    """Resize and synchronize the decoder with ``tokenizer``.

    The original TrOCR text vocabulary is incompatible with a replacement
    tokenizer, so the decoder's input embeddings and untied output head are
    resized. Reinitializing the decoder trains its text mapping from scratch
    while retaining the visual encoder.
    """
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define BOS and EOS token IDs.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a PAD token ID.")

    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    model.config.decoder.vocab_size = len(tokenizer)

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.max_length = max_target_length

    model.config.decoder.bos_token_id = tokenizer.bos_token_id
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id
    model.config.decoder.tie_word_embeddings = tie_embeddings
    model.config.decoder.dropout = dropout
    model.decoder.config.tie_word_embeddings = tie_embeddings
    model.decoder.config.dropout = dropout

    decoder = model.decoder.model.decoder
    decoder.dropout = dropout
    for layer in decoder.layers:
        layer.dropout = dropout

    if isinstance(reinitialize, bool):
        reinitialize_mode = "all" if reinitialize else "none"
    else:
        reinitialize_mode = str(reinitialize)
    if reinitialize_mode not in {"all", "token_layers", "none"}:
        raise ValueError(
            "Decoder reinitialize mode must be 'all', 'token_layers', or 'none'; "
            f"received {reinitialize_mode!r}."
        )

    if reinitialize_mode == "all":
        for module in model.decoder.modules():
            if hasattr(module, "_is_hf_initialized"):
                delattr(module, "_is_hf_initialized")
        model.decoder.init_weights()
    elif reinitialize_mode == "token_layers":
        initializer_std = float(model.decoder.config.init_std)
        input_embeddings = model.decoder.get_input_embeddings()
        output_embeddings = model.decoder.get_output_embeddings()
        nn.init.normal_(input_embeddings.weight, mean=0.0, std=initializer_std)
        if input_embeddings.padding_idx is not None:
            nn.init.zeros_(input_embeddings.weight[input_embeddings.padding_idx])
        if output_embeddings.weight is not input_embeddings.weight:
            nn.init.normal_(
                output_embeddings.weight,
                mean=0.0,
                std=initializer_std,
            )
        if output_embeddings.bias is not None:
            nn.init.zeros_(output_embeddings.bias)
    if tie_embeddings:
        model.decoder.tie_weights()

    return sum(parameter.numel() for parameter in model.decoder.parameters())

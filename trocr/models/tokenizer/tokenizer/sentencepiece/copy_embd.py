"""Transfer TrOCR vocabulary weights into a compact tokenizer's ID space."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from ....model_builder import load_model
from ... import bundled_tokenizer_path
from ...builder import load_tokenizer
from .new import old_to_new_token_ids


def _resize_output_projection(decoder, vocabulary_size: int) -> None:
    """Ensure an untied output projection has the requested vocabulary size."""
    output_projection = decoder.get_output_embeddings()
    if output_projection.out_features == vocabulary_size:
        return

    replacement = nn.Linear(
        output_projection.in_features,
        vocabulary_size,
        bias=output_projection.bias is not None,
    ).to(device=output_projection.weight.device, dtype=output_projection.weight.dtype)
    decoder.set_output_embeddings(replacement)


def _synchronize_special_token_config(model, tokenizer: PreTrainedTokenizerBase) -> None:
    """Update model and generation configuration after token IDs change."""
    model.config.vocab_size = len(tokenizer)
    model.config.decoder.vocab_size = len(tokenizer)
    model.decoder.config.vocab_size = len(tokenizer)

    for config in (model.config, model.config.decoder, model.generation_config):
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id


def copy_vocabulary_weights(
    model,
    original_tokenizer: PreTrainedTokenizerBase,
    compact_tokenizer: PreTrainedTokenizerBase,
) -> dict[int, int]:
    """Resize the decoder and copy vocabulary rows by exact token string.

    The TrOCR decoder has untied input embeddings and output projection weights,
    so both matrices must be copied.  It intentionally does not use
    ``resize_token_embeddings``' positional copy as the compact tokenizer's
    IDs no longer match the original tokenizer's IDs.
    """
    mapping = old_to_new_token_ids(original_tokenizer, compact_tokenizer)
    if len(mapping) != len(compact_tokenizer):
        raise ValueError(
            "The compact tokenizer contains pieces absent from the original tokenizer; "
            "this pruning transfer supports retained original pieces only."
        )

    original_input = model.decoder.get_input_embeddings().weight.detach().clone()
    original_output = model.decoder.get_output_embeddings().weight.detach().clone()
    original_output_bias = (
        model.decoder.get_output_embeddings().bias.detach().clone()
        if model.decoder.get_output_embeddings().bias is not None
        else None
    )
    if original_input.shape[0] < len(original_tokenizer) or original_output.shape[0] < len(original_tokenizer):
        raise ValueError("Model vocabulary matrices are smaller than the original tokenizer")

    model.decoder.resize_token_embeddings(len(compact_tokenizer))
    _resize_output_projection(model.decoder, len(compact_tokenizer))

    compact_input = model.decoder.get_input_embeddings()
    compact_output = model.decoder.get_output_embeddings()
    with torch.no_grad():
        for old_id, new_id in mapping.items():
            compact_input.weight[new_id].copy_(original_input[old_id])
            compact_output.weight[new_id].copy_(original_output[old_id])
            if compact_output.bias is not None and original_output_bias is not None:
                compact_output.bias[new_id].copy_(original_output_bias[old_id])

    compact_input.padding_idx = compact_tokenizer.pad_token_id
    model.decoder.model.decoder.padding_idx = compact_tokenizer.pad_token_id
    _synchronize_special_token_config(model, compact_tokenizer)
    return mapping


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for vocabulary-weight transfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Original TrOCR checkpoint directory or model name.")
    parser.add_argument(
        "--original-tokenizer",
        type=Path,
        default=bundled_tokenizer_path("trocr"),
        help="Directory containing the tokenizer used by the original checkpoint.",
    )
    parser.add_argument(
        "--compact-tokenizer",
        type=Path,
        required=True,
        help="Directory written by new.py.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination checkpoint directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --output.")
    return parser.parse_args()


def main() -> None:
    """Write a checkpoint whose decoder vocabulary matches a compact tokenizer."""
    args = parse_args()
    output_directory = args.output.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    original_tokenizer = load_tokenizer(str(args.original_tokenizer), use_fast=False)
    compact_tokenizer = load_tokenizer(str(args.compact_tokenizer), use_fast=False)
    model = load_model(args.model)
    mapping = copy_vocabulary_weights(model, original_tokenizer, compact_tokenizer)

    model.save_pretrained(output_directory)
    compact_tokenizer.save_pretrained(output_directory)
    print(
        f"Copied {len(mapping):,} token rows into a {len(compact_tokenizer):,}-token "
        f"checkpoint at {output_directory}"
    )


if __name__ == "__main__":
    main()

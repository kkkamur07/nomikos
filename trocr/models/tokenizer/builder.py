"""Tokenizer loading and TrOCR processor assembly."""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase, TrOCRProcessor

from ..encoder import load_image_processor


def load_tokenizer(
    tokenizer_path: str,
    *,
    use_fast: bool,
    pad_token: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load any Hugging Face tokenizer and add padding only when needed.

    Args:
        tokenizer_path: Local directory or locally cached Hugging Face tokenizer.
        use_fast: Whether to load the tokenizer's Rust-backed fast variant.
        pad_token: Token to add only if the loaded tokenizer has no pad token.
            Use ``None`` to reject tokenizers without an existing pad token.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=use_fast,
    )
    if tokenizer.pad_token_id is None:
        if pad_token is None:
            raise ValueError(
                "Tokenizer has no pad token. Supply a pad_token to add one."
            )
        tokenizer.add_special_tokens({"pad_token": pad_token})
    return tokenizer


def build_processor(
    model_source: str,
    tokenizer: PreTrainedTokenizerBase,
) -> TrOCRProcessor:
    """Pair TrOCR image preprocessing with the supplied text tokenizer."""
    return TrOCRProcessor(
        image_processor=load_image_processor(model_source),
        tokenizer=tokenizer,
    )

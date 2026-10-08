"""Canonical text normalization applied before OCR tokenization.

This module deliberately preserves transcription content.  It only selects a
single Unicode representation for canonically equivalent text and converts
non-breaking spaces to ordinary spaces.
"""

from __future__ import annotations

import unicodedata
from typing import Any


NON_BREAKING_SPACE = "\u00a0"


def normalize_transcription(text: str) -> str:
    """Return the NFC representation used as input to text tokenizers.

    Greek accents and Syriac combining marks are preserved; this function does
    not strip, lowercase, expand, or otherwise rewrite transcription content.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")
    return unicodedata.normalize("NFC", text.replace(NON_BREAKING_SPACE, " "))


def configure_bpe_normalizer(tokenizer: object) -> None:
    """Embed this normalization policy in a Hugging Face fast tokenizer.

    The tokenizer then normalizes every string before its byte-level
    pre-tokenizer and BPE merges run, including when it is reloaded later.
    """
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is None:
        raise TypeError("NFC BPE normalization requires a fast tokenizer backend")

    from tokenizers import normalizers

    backend_tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(NON_BREAKING_SPACE, " "),
            normalizers.NFC(),
        ]
    )


def decode_normalized(
    decoder: Any,
    token_ids: Any,
    *,
    skip_special_tokens: bool = True,
) -> list[str]:
    """Decode token IDs and return canonical NFC transcription strings."""
    decoded = decoder.batch_decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    return [normalize_transcription(text) for text in decoded]

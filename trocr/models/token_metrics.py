"""Tokenizer-aware OCR metric adapters for TrOCR."""
#! for decoding strategy

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from ...metrics.metrics import compute_sequence_length_metrics, compute_text_metrics


def decode_predictions_and_labels(
    predictions: Any,
    labels: Any,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[str], list[str]]:
    """Decode generated IDs and masked labels without altering OCR whitespace."""
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define PAD before metrics can be calculated.")

    labels_tensor = torch.as_tensor(labels).detach().cpu().clone()
    labels_tensor[labels_tensor == -100] = tokenizer.pad_token_id
    hypotheses = tokenizer.batch_decode(
        torch.as_tensor(predictions).detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    references = tokenizer.batch_decode(
        labels_tensor.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return references, hypotheses


def compute_token_metrics(
    predictions: Any,
    labels: Any,
    tokenizer: PreTrainedTokenizerBase,
    *,
    prefix: str = "",
    include_sequence_length_metrics: bool = False,
) -> dict[str, float]:
    """Decode token IDs and return every shared OCR metric."""
    references, hypotheses = decode_predictions_and_labels(predictions, labels, tokenizer)
    metrics = compute_text_metrics(references, hypotheses)
    if include_sequence_length_metrics:
        metrics.update(compute_sequence_length_metrics(references, hypotheses))
    return {f"{prefix}{name}": value for name, value in metrics.items()}


def character_error_rate(
    predictions: Any,
    labels: Any,
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    """Compatibility helper returning strict corpus CER."""
    return compute_token_metrics(predictions, labels, tokenizer)["cer"]

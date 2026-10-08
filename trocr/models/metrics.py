"""Framework-independent OCR metrics for Hugging Face TrOCR training."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizerBase


def edit_distance(reference: Sequence[Any], prediction: Sequence[Any]) -> int:
    """Return the Levenshtein distance between two character or token sequences."""
    previous_row = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current_row = [reference_index]
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[prediction_index] + 1,
                    previous_row[prediction_index - 1]
                    + (reference_item != prediction_item),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _words(text: str) -> list[str]:
    """Split Unicode OCR text into whitespace-delimited words."""
    return text.split()


def compute_text_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, float]:
    """Compute corpus OCR metrics; all reported values are fractions in [0, 1].

    ``wpa`` and ``exact_match_accuracy`` are aliases: the fraction of complete
    transcriptions that exactly match their reference. SROIE F1 follows the
    legacy implementation's bag-of-whitespace-token matching rule.
    """
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length.")
    if not references:
        return {
            "cer": 0.0,
            "wer": 0.0,
            "wpa": 0.0,
            "normalized_edit_distance": 0.0,
            "exact_match_accuracy": 0.0,
            "sroie_precision": 0.0,
            "sroie_recall": 0.0,
            "sroie_f1": 0.0,
        }

    character_edits = 0
    reference_characters = 0
    word_edits = 0
    reference_words = 0
    normalized_distances = []
    exact_matches = 0
    matched_sroie_words = 0
    predicted_sroie_words = 0
    reference_sroie_words = 0

    for reference, prediction in zip(references, predictions, strict=True):
        character_distance = edit_distance(reference, prediction)
        character_edits += character_distance
        reference_characters += len(reference)
        normalized_distances.append(
            character_distance / max(len(reference), len(prediction), 1)
        )
        exact_matches += int(reference == prediction)

        reference_tokens = _words(reference)
        prediction_tokens = _words(prediction)
        word_edits += edit_distance(reference_tokens, prediction_tokens)
        reference_words += len(reference_tokens)

        reference_counter = Counter(reference_tokens)
        prediction_counter = Counter(prediction_tokens)
        matched_sroie_words += sum(
            (reference_counter & prediction_counter).values()
        )
        reference_sroie_words += len(reference_tokens)
        predicted_sroie_words += len(prediction_tokens)

    cer = character_edits / reference_characters if reference_characters else 0.0
    wer = word_edits / reference_words if reference_words else 0.0
    exact_match_accuracy = exact_matches / len(references)
    sroie_precision = (
        matched_sroie_words / predicted_sroie_words
        if predicted_sroie_words
        else 0.0
    )
    sroie_recall = (
        matched_sroie_words / reference_sroie_words
        if reference_sroie_words
        else 0.0
    )
    sroie_f1 = (
        2 * sroie_precision * sroie_recall / (sroie_precision + sroie_recall)
        if sroie_precision + sroie_recall
        else 0.0
    )
    return {
        "cer": cer,
        "wer": wer,
        "wpa": exact_match_accuracy,
        "normalized_edit_distance": sum(normalized_distances)
        / len(normalized_distances),
        "exact_match_accuracy": exact_match_accuracy,
        "sroie_precision": sroie_precision,
        "sroie_recall": sroie_recall,
        "sroie_f1": sroie_f1,
    }


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
) -> dict[str, float]:
    """Decode token IDs and return every OCR metric with an optional prefix."""
    references, hypotheses = decode_predictions_and_labels(predictions, labels, tokenizer)
    return {
        f"{prefix}{name}": value
        for name, value in compute_text_metrics(references, hypotheses).items()
    }


def character_error_rate(
    predictions: Any,
    labels: Any,
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    """Compatibility helper returning strict corpus CER."""
    return compute_token_metrics(predictions, labels, tokenizer)["cer"]

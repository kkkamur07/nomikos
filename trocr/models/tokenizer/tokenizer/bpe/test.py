"""Round-trip validation for trained byte-level BPE tokenizers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TokenizationStatistics:
    """Summary statistics for validating a trained tokenizer."""

    records: int
    tokens: int
    unknown_tokens: int
    maximum_length: int


def validate_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    transcriptions: Iterable[str],
) -> TokenizationStatistics:
    """Verify normalized transcriptions round-trip and collect token statistics."""
    records = 0
    token_count = 0
    unknown_count = 0
    maximum_length = 0

    for transcription in transcriptions:
        token_ids = tokenizer.encode(transcription, add_special_tokens=False)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != transcription:
            raise ValueError(
                "Tokenizer does not round-trip a transcription exactly: "
                f"{transcription!r} became {decoded!r}"
            )
        records += 1
        token_count += len(token_ids)
        unknown_count += token_ids.count(tokenizer.unk_token_id)
        maximum_length = max(maximum_length, len(token_ids))

    return TokenizationStatistics(records, token_count, unknown_count, maximum_length)

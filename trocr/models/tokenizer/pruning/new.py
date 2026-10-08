"""Build a compact tokenizer by pruning the original TrOCR vocabulary.

The original tokenizer is a SentencePiece Unigram model.  Pruning it changes
which pieces can segment a line, so the selection always retains all special
and single-character fallback pieces before adding the most frequent pieces.
``coverage`` refers to the fraction of *observed token occurrences*, not the
fraction of vocabulary entries.
"""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2 as sentencepiece_pb2
from transformers import PreTrainedTokenizerBase

from .. import bundled_tokenizer_path
from ..builder import load_tokenizer
from .count import load_frequencies


SENTENCEPIECE_MODEL_NAME = "sentencepiece.bpe.model"


@dataclass(frozen=True)
class VocabularySelection:
    """The old token IDs retained for a compact tokenizer."""

    token_ids: tuple[int, ...]
    retained_token_occurrences: int
    total_token_occurrences: int

    @property
    def coverage(self) -> float:
        """Return the fraction of observed token occurrences represented."""
        if self.total_token_occurrences == 0:
            return 0.0
        return self.retained_token_occurrences / self.total_token_occurrences


def _is_protected_piece(piece: str) -> bool:
    """Return whether a piece is needed as a character-level fallback."""
    if piece == "▁":
        return True
    surface = piece.removeprefix("▁")
    if len(surface) == 1:
        return True
    return any(unicodedata.category(character)[0] in {"P", "S", "M"} for character in surface)


def required_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Return special, punctuation/symbol, and atomic fallback token IDs."""
    required = set(tokenizer.all_special_ids)
    for piece, token_id in tokenizer.get_vocab().items():
        if _is_protected_piece(piece):
            required.add(token_id)
    return required


def select_vocabulary(
    tokenizer: PreTrainedTokenizerBase,
    frequencies: Counter[int],
    *,
    coverage: float = 0.95,
) -> VocabularySelection:
    """Keep protected tokens plus frequent tokens covering the requested mass."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in the interval (0, 1]")

    vocabulary_size = len(tokenizer)
    invalid_ids = sorted(token_id for token_id in frequencies if token_id >= vocabulary_size)
    if invalid_ids:
        raise ValueError(f"Frequency table contains token IDs outside the tokenizer: {invalid_ids[:5]}")

    total = sum(frequencies.values())
    if total == 0:
        raise ValueError("Cannot select a vocabulary from an empty frequency table")

    kept = required_token_ids(tokenizer)
    retained = sum(frequencies[token_id] for token_id in kept)
    target = total * coverage
    for token_id, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0])):
        if retained >= target:
            break
        if token_id not in kept:
            kept.add(token_id)
            retained += count

    return VocabularySelection(
        token_ids=tuple(sorted(kept)),
        retained_token_occurrences=retained,
        total_token_occurrences=total,
    )


def _load_sentencepiece_model(tokenizer_directory: Path) -> sentencepiece_pb2.ModelProto:
    model_path = tokenizer_directory / SENTENCEPIECE_MODEL_NAME
    if not model_path.is_file():
        raise FileNotFoundError(f"SentencePiece model is missing: {model_path}")
    model = sentencepiece_pb2.ModelProto()
    model.ParseFromString(model_path.read_bytes())
    return model


def _pruned_sentencepiece_model(
    original_model: sentencepiece_pb2.ModelProto,
    tokenizer: PreTrainedTokenizerBase,
    selection: VocabularySelection,
) -> sentencepiece_pb2.ModelProto:
    """Return the original model with unselected normal pieces removed."""
    selected_pieces = {
        tokenizer.convert_ids_to_tokens(token_id)
        for token_id in selection.token_ids
        if token_id not in tokenizer.all_special_ids
    }
    # SentencePiece's built-in UNKNOWN and CONTROL pieces must stay intact;
    # XLM-R's PAD and MASK are added from the Hugging Face configuration.
    pieces = [
        piece
        for piece in original_model.pieces
        if piece.type != sentencepiece_pb2.ModelProto.SentencePiece.NORMAL
        or piece.piece in selected_pieces
    ]
    if not pieces:
        raise ValueError("Pruning removed every SentencePiece entry")

    compact_model = sentencepiece_pb2.ModelProto()
    compact_model.CopyFrom(original_model)
    compact_model.ClearField("pieces")
    for piece in pieces:
        compact_model.pieces.add().CopyFrom(piece)
    compact_model.trainer_spec.vocab_size = len(compact_model.pieces)
    return compact_model


def old_to_new_token_ids(
    original_tokenizer: PreTrainedTokenizerBase,
    compact_tokenizer: PreTrainedTokenizerBase,
) -> dict[int, int]:
    """Map each retained old token ID to its compact-tokenizer ID by piece."""
    original_vocabulary = original_tokenizer.get_vocab()
    compact_vocabulary = compact_tokenizer.get_vocab()
    return {
        old_id: compact_vocabulary[piece]
        for piece, old_id in original_vocabulary.items()
        if piece in compact_vocabulary
    }


def write_compact_tokenizer(
    tokenizer_directory: Path,
    frequencies: Counter[int],
    *,
    output_directory: Path,
    coverage: float = 0.95,
    overwrite: bool = False,
) -> VocabularySelection:
    """Prune the bundled tokenizer and write its files and ID mapping."""
    tokenizer_directory = tokenizer_directory.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    original_tokenizer = load_tokenizer(str(tokenizer_directory), use_fast=False)
    selection = select_vocabulary(original_tokenizer, frequencies, coverage=coverage)
    compact_model = _pruned_sentencepiece_model(
        _load_sentencepiece_model(tokenizer_directory),
        original_tokenizer,
        selection,
    )

    for source in tokenizer_directory.glob("*.json"):
        shutil.copy2(source, output_directory / source.name)
    (output_directory / SENTENCEPIECE_MODEL_NAME).write_bytes(compact_model.SerializeToString())

    compact_tokenizer = load_tokenizer(str(output_directory), use_fast=False)
    old_to_new = old_to_new_token_ids(original_tokenizer, compact_tokenizer)
    expected_tokens = {
        original_tokenizer.convert_ids_to_tokens(token_id) for token_id in selection.token_ids
    }
    missing_tokens = sorted(expected_tokens - set(compact_tokenizer.get_vocab()))
    if missing_tokens:
        raise RuntimeError(f"Compact tokenizer lost retained tokens: {missing_tokens[:5]}")

    mapping_path = output_directory / "old_to_new_token_ids.json"
    mapping_path.write_text(
        json.dumps(
            {
                "original_vocabulary_size": len(original_tokenizer),
                "compact_vocabulary_size": len(compact_tokenizer),
                "selected_piece_count": len(selection.token_ids),
                "retained_token_occurrences": selection.retained_token_occurrences,
                "total_token_occurrences": selection.total_token_occurrences,
                "observed_coverage": selection.coverage,
                "old_to_new": {str(old_id): new_id for old_id, new_id in sorted(old_to_new.items())},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return selection


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for vocabulary selection and pruning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequencies", type=Path, required=True, help="JSON output from count.py.")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=bundled_tokenizer_path("trocr"),
        help="Directory containing the original TrOCR tokenizer.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for the compact tokenizer.")
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Observed token-occurrence mass to retain after protected tokens (default: 0.95).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --output.")
    return parser.parse_args()


def main() -> None:
    """Write a compact tokenizer from a token-frequency table."""
    args = parse_args()
    selection = write_compact_tokenizer(
        args.tokenizer,
        load_frequencies(args.frequencies),
        output_directory=args.output,
        coverage=args.coverage,
        overwrite=args.overwrite,
    )
    print(
        f"Retained {len(selection.token_ids):,} selected tokens with "
        f"{selection.coverage:.2%} observed token coverage in {args.output}"
    )


if __name__ == "__main__":
    main()

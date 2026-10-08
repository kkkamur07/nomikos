"""Train language-specific compact GPT byte-level BPE tokenizers.

The existing GPT tokenizer has a 50k-token vocabulary.  BPE merges only add
tokens, so it cannot be extended and then reduced to 1,700 entries.  This
script instead reuses its GPT-2 byte-level pre-tokenization, byte fallback, and
special-token configuration, then learns a fresh set of merges from the 256 base
byte tokens on each language's processed pretraining and finetuning
transcriptions.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from ... import TOKENIZER_ROOT, bundled_tokenizer_path
from ...builder import load_tokenizer


DEFAULT_VOCABULARY_SIZE = 500
LANGUAGE_CORPUS_DIRECTORIES: dict[str, tuple[Path, Path]] = {
    "armenian": (
        Path("data/processed/armenian/pretraining"),
        Path("data/processed/armenian/finetuning"),
    ),
    "greek": (
        Path("data/processed/greek/pretraining"),
        Path("data/processed/greek/finetuning"),
    ),
    "syriac": (
        Path("data/processed/syriac/pretraining"),
        Path("data/processed/syriac/finetuning"),
    ),
}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_CORPUS_DIRECTORIES)


@dataclass(frozen=True)
class TokenizationStatistics:
    """Summary statistics for validating a trained tokenizer."""

    records: int
    tokens: int
    unknown_tokens: int
    maximum_length: int


@dataclass(frozen=True)
class LanguageTrainingResult:
    """Validation result and destination for one language tokenizer."""

    language: str
    output_directory: Path
    statistics: TokenizationStatistics


def corpus_directories_for_language(language: str) -> tuple[Path, Path]:
    """Return processed pretraining and finetuning directories for ``language``."""
    try:
        return LANGUAGE_CORPUS_DIRECTORIES[language]
    except KeyError as error:
        available = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(f"Unsupported language {language!r}; choose one of: {available}") from error


def iter_transcriptions(corpus_directories: Sequence[Path]) -> Iterator[str]:
    """Yield every transcription from ``gt_*.txt`` corpus manifests."""
    for corpus_directory in corpus_directories:
        corpus_directory = corpus_directory.expanduser().resolve()
        manifests = sorted(corpus_directory.glob("gt_*.txt"))
        if not manifests:
            raise FileNotFoundError(f"No gt_*.txt manifests found in {corpus_directory}")
        for manifest in manifests:
            for line_number, line in enumerate(
                manifest.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                try:
                    _, transcription = line.split("\t", maxsplit=1)
                except ValueError as error:
                    raise ValueError(
                        f"{manifest}:{line_number} must contain image<TAB>transcription"
                    ) from error
                yield transcription


def train_byte_bpe(
    corpus_directories: Sequence[Path],
    *,
    source_tokenizer_directory: Path,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
) -> PreTrainedTokenizerBase:
    """Train byte-level GPT BPE merges while retaining its base byte alphabet."""
    if vocabulary_size < 259:
        raise ValueError("vocabulary_size must allow 256 byte tokens and three special tokens")

    source_tokenizer = load_tokenizer(
        str(source_tokenizer_directory),
        use_fast=True,
        pad_token="<|pad|>",
    )
    trained_tokenizer = source_tokenizer.train_new_from_iterator(
        iter_transcriptions(corpus_directories),
        vocab_size=vocabulary_size,
    )
    if len(trained_tokenizer) != vocabulary_size:
        raise RuntimeError(
            f"Expected a {vocabulary_size}-token vocabulary, got {len(trained_tokenizer)}"
        )
    if trained_tokenizer.pad_token_id is None or trained_tokenizer.unk_token_id is None:
        raise RuntimeError("Trained tokenizer is missing required PAD or UNK token")
    return trained_tokenizer


def validate_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    corpus_directories: Sequence[Path],
) -> TokenizationStatistics:
    """Verify exact round trips and report sequence-length/unknown-token totals."""
    records = 0
    token_count = 0
    unknown_count = 0
    maximum_length = 0
    for transcription in iter_transcriptions(corpus_directories):
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


def train_and_save(
    corpus_directories: Sequence[Path],
    *,
    source_tokenizer_directory: Path,
    output_directory: Path,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    overwrite: bool = False,
) -> TokenizationStatistics:
    """Train, validate, and save a compact byte-level BPE tokenizer."""
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    tokenizer = train_byte_bpe(
        corpus_directories,
        source_tokenizer_directory=source_tokenizer_directory,
        vocabulary_size=vocabulary_size,
    )
    statistics = validate_tokenizer(tokenizer, corpus_directories)
    if statistics.unknown_tokens:
        raise ValueError(f"Byte-level tokenizer emitted {statistics.unknown_tokens} unknown tokens")
    tokenizer.save_pretrained(output_directory)
    return statistics


def train_language(
    language: str,
    *,
    source_tokenizer_directory: Path,
    output_root: Path = TOKENIZER_ROOT,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    overwrite: bool = False,
) -> LanguageTrainingResult:
    """Train and save a tokenizer for one language's complete processed corpus."""
    corpus_directories = corpus_directories_for_language(language)
    output_directory = output_root / f"gpt_{language}_{vocabulary_size}"
    statistics = train_and_save(
        corpus_directories,
        source_tokenizer_directory=source_tokenizer_directory,
        output_directory=output_directory,
        vocabulary_size=vocabulary_size,
        overwrite=overwrite,
    )
    return LanguageTrainingResult(language, output_directory, statistics)


def train_languages(
    languages: Sequence[str],
    *,
    source_tokenizer_directory: Path,
    output_root: Path = TOKENIZER_ROOT,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    overwrite: bool = False,
) -> list[LanguageTrainingResult]:
    """Train independent tokenizers for every requested supported language."""
    selected_languages = SUPPORTED_LANGUAGES if "all" in languages else tuple(languages)
    invalid_languages = set(selected_languages).difference(SUPPORTED_LANGUAGES)
    if invalid_languages:
        invalid = ", ".join(sorted(invalid_languages))
        raise ValueError(f"Unsupported language selection: {invalid}")
    if not selected_languages:
        raise ValueError("Choose at least one language or 'all'")
    return [
        train_language(
            language,
            source_tokenizer_directory=source_tokenizer_directory,
            output_root=output_root,
            vocabulary_size=vocabulary_size,
            overwrite=overwrite,
        )
        for language in selected_languages
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for BPE tokenizer training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=(*SUPPORTED_LANGUAGES, "all"),
        nargs="+",
        default=["all"],
        help="Languages to train independently (default: all).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        help="Custom corpus directories containing gt_*.txt manifests.",
    )
    parser.add_argument(
        "--source-tokenizer",
        type=Path,
        default=bundled_tokenizer_path("gpt_tokenizer"),
        help="Current GPT tokenizer used as the byte-level template.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory for one tokenizer trained from --corpus.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=TOKENIZER_ROOT,
        help="Parent directory for language-specific tokenizer outputs.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCABULARY_SIZE,
        help="Total vocabulary size including base bytes and special tokens (default: 500).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --output.")
    return parser.parse_args()


def main() -> None:
    """Train and save compact language-specific GPT BPE tokenizers."""
    args = parse_args()
    if args.corpus:
        if args.output is None:
            raise ValueError("--output is required when --corpus is supplied")
        statistics = train_and_save(
            args.corpus,
            source_tokenizer_directory=args.source_tokenizer,
            output_directory=args.output,
            vocabulary_size=args.vocab_size,
            overwrite=args.overwrite,
        )
        results = [
            LanguageTrainingResult(
                language="custom",
                output_directory=args.output,
                statistics=statistics,
            )
        ]
    elif args.output is not None:
        raise ValueError("--output can only be used together with --corpus")
    else:
        results = train_languages(
            args.language,
            source_tokenizer_directory=args.source_tokenizer,
            output_root=args.output_root,
            vocabulary_size=args.vocab_size,
            overwrite=args.overwrite,
        )

    for result in results:
        statistics = result.statistics
        print(
            f"Saved {args.vocab_size:,}-token {result.language} byte-level BPE tokenizer to "
            f"{result.output_directory}; {statistics.records:,} records, {statistics.tokens:,} "
            f"tokens, {statistics.unknown_tokens} unknowns, max length {statistics.maximum_length}"
        )


if __name__ == "__main__":
    main()

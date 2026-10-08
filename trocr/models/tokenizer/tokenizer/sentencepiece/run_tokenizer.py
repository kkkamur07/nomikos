"""Tokenize the Greek and Syriac TrOCR manifests with the bundled tokenizer.

The output is JSON Lines so later pruning stages can use the exact token IDs
that the current TrOCR decoder sees.  Tokenization deliberately goes through
``XLMRobertaTokenizer`` rather than calling SentencePiece directly: that
preserves the bundled tokenizer's normalization, whitespace handling, special
token IDs, and unknown-token behaviour.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from ... import bundled_tokenizer_path
from ...builder import load_tokenizer


DEFAULT_CORPUS_DIRECTORIES = (
    Path("data/trocr_processed/greek/pretraining"),
    Path("data/trocr_processed/syriac/pretraining"),
)


def iter_manifest_records(corpus_directory: Path) -> Iterator[dict[str, str]]:
    """Yield the transcription text in every ground-truth manifest."""
    corpus_directory = corpus_directory.expanduser().resolve()
    manifests = sorted(corpus_directory.glob("gt_*.txt"))
    if not manifests:
        raise FileNotFoundError(f"No gt_*.txt manifests found in {corpus_directory}")

    corpus = corpus_directory.parent.name
    for manifest in manifests:
        split = manifest.stem.removeprefix("gt_")
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                image_name, text = line.split("\t", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"{manifest}:{line_number} must contain image<TAB>transcription") from error
            yield {
                "corpus": corpus,
                "split": split,
                "image_name": image_name,
                "text": text,
            }


def tokenize_corpora(
    corpus_directories: Sequence[Path],
    *,
    tokenizer_directory: Path,
    output_path: Path,
) -> int:
    """Tokenize manifests and write records with the current decoder token IDs."""
    tokenizer = load_tokenizer(str(tokenizer_directory), use_fast=False)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record_count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for corpus_directory in corpus_directories:
            for record in iter_manifest_records(corpus_directory):
                # TrOCR labels do not add BOS/EOS automatically; the collator appends EOS.
                record["token_ids"] = tokenizer.encode(record["text"], add_special_tokens=False)
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_count += 1
    return record_count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for corpus tokenization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CORPUS_DIRECTORIES),
        help="Processed corpus directories containing gt_*.txt manifests.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=bundled_tokenizer_path("trocr"),
        help="Directory containing the original TrOCR tokenizer.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSONL destination for tokenized records.")
    return parser.parse_args()


def main() -> None:
    """Tokenize both processed corpora from the command line."""
    args = parse_args()
    record_count = tokenize_corpora(
        args.corpus,
        tokenizer_directory=args.tokenizer,
        output_path=args.output,
    )
    print(f"Wrote {record_count:,} tokenized records to {args.output}")


if __name__ == "__main__":
    main()

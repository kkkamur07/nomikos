"""Count current TrOCR token usage from tokenized corpus records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


def count_token_frequencies(tokenized_records_path: Path) -> Counter[int]:
    """Return token-ID frequencies from JSONL created by ``run_tokenizer``."""
    frequencies: Counter[int] = Counter()
    with tokenized_records_path.expanduser().open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            try:
                record = json.loads(line)
                token_ids = record["token_ids"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"{tokenized_records_path}:{line_number} is not a tokenized record") from error
            if not isinstance(token_ids, list) or any(
                not isinstance(token_id, int) or token_id < 0 for token_id in token_ids
            ):
                raise ValueError(f"{tokenized_records_path}:{line_number} has invalid token_ids")
            frequencies.update(token_ids)
    return frequencies


def write_frequencies(frequencies: Mapping[int, int], output_path: Path) -> None:
    """Write counts in a stable, human-readable JSON format."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {str(token_id): frequencies[token_id] for token_id in sorted(frequencies)}
    payload = {
        "total_tokens": sum(frequencies.values()),
        "unique_observed_tokens": len(frequencies),
        "counts": counts,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_frequencies(input_path: Path) -> Counter[int]:
    """Load token frequencies written by :func:`write_frequencies`."""
    try:
        payload = json.loads(input_path.expanduser().read_text(encoding="utf-8"))
        counts = payload["counts"]
    except (json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"{input_path} is not a token-frequency JSON file") from error

    if not isinstance(counts, dict):
        raise ValueError(f"{input_path} contains invalid counts")
    frequencies: Counter[int] = Counter()
    for token_id, count in counts.items():
        try:
            integer_token_id = int(token_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{input_path} contains an invalid token ID: {token_id!r}") from error
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"{input_path} contains an invalid count for token {token_id!r}")
        frequencies[integer_token_id] = count
    return frequencies


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for token counting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL output from run_tokenizer.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON frequency table.")
    return parser.parse_args()


def main() -> None:
    """Count and persist frequencies from a tokenized corpus."""
    args = parse_args()
    frequencies = count_token_frequencies(args.input)
    write_frequencies(frequencies, args.output)
    print(
        f"Counted {sum(frequencies.values()):,} tokens across "
        f"{len(frequencies):,} observed token IDs; wrote {args.output}"
    )


if __name__ == "__main__":
    main()

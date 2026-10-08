"""Normalize all TrOCR datasets to deduplicated, page-disjoint 80/10/10 splits.

Only manifest files are rewritten; images remain in place. For every
language/partition pool:

1. Transcriptions are NFC-normalized (the manifests are the source of truth;
   this deliberately overrides the non-NFC Greek GT of the Greek importer).
   Syriac rows then go through ``syriac.normalize_text`` and Greek rows
   through ``greek.normalize_text``.
2. Exact duplicate transcriptions are dropped, then near duplicates (same
   loose key of >= 20 chars, see ``loose_key``). One copy is kept, preferring
   a non-``esteban__`` crop, then the smallest image name.
3. Every line gets a page key derived from its image name (see
   ``PAGE_KEY_RULES``). Lines are grouped into connected components over
   shared page keys and shared transcriptions. A kept line with no page in
   its name inherits the page of any row dropped in its favour.
4. Balance groups with at most ``EXHAUSTIVE_MAX_GROUPS`` components use the
   assignment closest to 80/10/10; larger ones shuffle components and fill
   train to 80% of lines, val to 90%, the rest to test.

Combined manifests are rebuilt from the language-specific assignments, so a
crop always has the same split in its language dataset and the combined
dataset. Current manifests are backed up once to ``BACKUP_DIR_NAME`` and the
backup is the input pool on later runs, so re-running is deterministic.

Run from the repository root:

    uv run python -m src.preprocessing_data.resplit --root /path/to/data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

try:  # python -m src.preprocessing_data.resplit
    from .armenian import apply_ink_rules, normalize_manifest_text as normalize_armenian_text
    from .greek import apply_editorial_rules, normalize_text as normalize_greek_text
    from .syriac import normalize_symbol_encoding, normalize_text as normalize_syriac_text
except ImportError:  # python src/preprocessing_data/resplit.py
    from armenian import apply_ink_rules, normalize_manifest_text as normalize_armenian_text
    from greek import apply_editorial_rules, normalize_text as normalize_greek_text
    from syriac import normalize_symbol_encoding, normalize_text as normalize_syriac_text


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
LANGUAGES = ("greek", "syriac", "armenian", "coptic")
PARTITIONS = ("pretraining", "finetuning")
SPLITS = ("train", "val", "test")
SEED = 1111
CUMULATIVE_BOUNDS = (0.8, 0.9)
TARGET_FRACTIONS = (0.8, 0.1, 0.1)
EXHAUSTIVE_MAX_GROUPS = 12
NEAR_DUPLICATE_MIN_KEY_LENGTH = 20
# Ignored by the loose key: Esteban's copies of Stavronikita lines omit the
# kai sign that the Stavronikita GT writes as U+03D7.
LOOSE_KEY_IGNORED = frozenset({"\u03d7"})
BACKUP_DIR_NAME = "splits_backup_2026-10-07"
REPORT_NAME = "split_report.json"

Row = tuple[str, str]

# (source pattern, page pattern or None, how the page key is known).
# Order matters: the first matching rule wins. A name that matches no rule
# raises, so a new source cannot silently lose its page grouping.
PAGE_KEY_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "esteban",
        re.compile(r"^esteban__(?P<source>.+)_page_(?P<page>\d+)_line_\d+\.png$"),
        "name 'esteban__<ms>_page_<NNN>_line_<MMM>.png' (workbook preproc_file_name, "
        "greek_pretraining.py:253); icdar_nt_ct has exactly one line per 'page', so its "
        "page field is a line counter and is ignored",
    ),
    (
        "labelled",
        re.compile(r"^labelled__(?P<source>.+)_(?P<page>\d+)_\d+\.png$"),
        "name 'labelled__<book>_<page>_<line>.png' (Kraken export name, greek_pretraining.py:302)",
    ),
    (
        "grec",
        re.compile(r"^(?P<source>Grec_.+)_(?P<page>\d+)_\d+\.jpg$"),
        "name 'Grec_..._<page>_<line>.jpg'",
    ),
    (
        "dresden",
        re.compile(r"^(?P<source>dresden_a151)_facs_(?P<page>\d+)_\d+\.jpg$"),
        "name 'dresden_a151_facs_<facs>_<line>.jpg'; generator not in repo, but every facs "
        "number carries a contiguous 0-based line index (66 facs, 6-22 lines each)",
    ),
    (
        "no_page",
        re.compile(
            r"^(?P<source>syriac_2025_coordinates|syriac_2024|syriac_2025|greek_hpgtr"
            r"|greek_eparchos|greek_htr_cpgr23|greek_stavronikita_ms_\d+|greek_bessarion"
            r"|greek_ljs380_excerpts|armenian_datalab_dulaurier)_\d+\.(?:jpg|png)$"
        ),
        "no page in name; built outside this repo (data_nomicous); build reports only record "
        "page counts, not a line->page map, so page provenance is unrecoverable",
    ),
    (
        "page_xml",
        re.compile(r"^(?P<source>.+?)__(?P<page>.+)__\d{3}\.jpg$"),
        "name '<slug>__<PAGE-XML stem>__<line:03d>.jpg' (armenian.py:268, "
        "replace_armenian_ms.py:208, coptic.py:285, syriac.py:387 + chapter4_import.json "
        "page_splits)",
    ),
)
# Esteban sub-corpora whose "page" number is really a per-line counter.
ESTEBAN_LINE_COUNTER_SOURCES = frozenset({"icdar_nt_ct"})


def parse_manifest(path: Path) -> list[Row]:
    """Read one TrOCR filename/transcription manifest."""
    rows: list[Row] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            image_name, text = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid manifest row {path}:{line_number}") from exc
        rows.append((image_name, text))
    return rows


def read_splits(root: Path) -> dict[str, list[Row]]:
    """Read the three split manifests of one directory."""
    return {split: parse_manifest(root / f"gt_{split}.txt") for split in SPLITS}


def page_key(image_name: str) -> tuple[str, str | None, str]:
    """Return (source, page key or None, rule name) for one crop name."""
    for rule, pattern, _ in PAGE_KEY_RULES:
        match = pattern.match(image_name)
        if not match:
            continue
        source = match.group("source")
        page = match.groupdict().get("page")
        if rule == "esteban":
            if source in ESTEBAN_LINE_COUNTER_SOURCES:
                page = None
            source = f"esteban__{source}"
        elif rule == "labelled":
            source = f"labelled__{source}"
        if page is None:
            return source, None, rule
        return source, f"{source}/{page}", rule
    raise ValueError(f"No page-key rule matches image name: {image_name}")


def fuzzy_text(text: str) -> str:
    """Strip diacritics, punctuation, symbols and spacing for near-duplicate checks."""
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    return "".join(
        char for char in decomposed if unicodedata.category(char)[0] not in {"M", "P", "Z", "S", "C"}
    )


def source_group(language: str, partition: str, image_name: str) -> str:
    """Keep imported Greek corpora independently balanced."""
    if language == "greek" and partition == "pretraining":
        if image_name.startswith("esteban__"):
            return "esteban"
        if image_name.startswith("labelled__"):
            return "labelled"
        return "greek_base"
    return language


def keep_preference(image_name: str) -> tuple[bool, str]:
    """Order duplicate candidates: non-Esteban first, then smallest name."""
    return image_name.startswith("esteban__"), image_name


def drop_duplicate_texts(rows: list[Row]) -> tuple[list[Row], list[dict[str, object]]]:
    """Keep one row per identical transcription, preferring non-Esteban crops."""
    by_text: dict[str, list[str]] = defaultdict(list)
    for image_name, text in rows:
        by_text[text].append(image_name)
    kept: list[Row] = []
    dropped: list[dict[str, object]] = []
    for text, names in by_text.items():
        ordered = sorted(names, key=keep_preference)
        kept.append((ordered[0], text))
        for name in ordered[1:]:
            dropped.append({"dropped": name, "kept": ordered[0], "text_length": len(text)})
    kept.sort()
    dropped.sort(key=lambda item: str(item["dropped"]))
    return kept, dropped


def loose_key(text: str) -> str:
    """NFD, drop Mn, P*, S*, whitespace and ``LOOSE_KEY_IGNORED``, casefold."""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
        and unicodedata.category(char)[0] not in {"P", "S"}
        and not char.isspace()
        and char not in LOOSE_KEY_IGNORED
    ).casefold()


def drop_near_duplicates(rows: list[Row]) -> tuple[list[Row], list[dict[str, object]]]:
    """Keep one row per loose key of at least ``NEAR_DUPLICATE_MIN_KEY_LENGTH`` chars."""
    by_key: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        key = loose_key(row[1])
        if len(key) >= NEAR_DUPLICATE_MIN_KEY_LENGTH:
            by_key[key].append(row)
    dropped_names: set[str] = set()
    dropped: list[dict[str, object]] = []
    for key, candidates in by_key.items():
        ordered = sorted(candidates, key=lambda row: keep_preference(row[0]))
        kept_name, kept_text = ordered[0]
        for name, text in ordered[1:]:
            dropped_names.add(name)
            dropped.append(
                {
                    "dropped": name,
                    "kept": kept_name,
                    "dropped_text": text,
                    "kept_text": kept_text,
                    "loose_key_length": len(key),
                }
            )
    dropped.sort(key=lambda item: str(item["dropped"]))
    return sorted(row for row in rows if row[0] not in dropped_names), dropped


def dropped_twins(*dropped_lists: list[dict[str, object]]) -> dict[str, list[str]]:
    """Map each kept name to every row dropped in its favour, following chains."""
    parent = {
        str(item["dropped"]): str(item["kept"]) for dropped in dropped_lists for item in dropped
    }

    def final(name: str) -> str:
        while name in parent:
            name = parent[name]
        return name

    twins: dict[str, list[str]] = defaultdict(list)
    for name in sorted(parent):
        twins[final(name)].append(name)
    return twins


class UnionFind:
    """Minimal union-find over string ids."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def connected_components(kept: list[Row], twins: dict[str, list[str]]) -> list[list[Row]]:
    """Group kept rows into connected components over shared page keys and texts.

    A kept row whose name carries no page inherits the page keys of the rows
    dropped in its favour, exact or near duplicates (``greek_eparchos_*`` takes
    the page of its ``esteban__eparchos_ct`` twin). Dropped rows never join two
    pages that both still have kept rows: identical short lines (numerals,
    headings) on different pages are different physical lines.
    """
    union_find = UnionFind()
    for image_name, text in kept:
        union_find.union(f"row:{image_name}", f"text:{text}")
        key = page_key(image_name)[1]
        pages = {key} if key is not None else {page_key(twin)[1] for twin in twins[image_name]}
        for page in sorted(pages - {None}):
            union_find.union(f"row:{image_name}", f"page:{page}")
    components: dict[str, list[Row]] = defaultdict(list)
    for row in kept:
        components[union_find.find(f"row:{row[0]}")].append(row)
    return sorted((sorted(rows) for rows in components.values()), key=lambda rows: rows[0][0])


def split_groups_exhaustive(groups: list[list[Row]]) -> dict[str, list[Row]]:
    """Pick the assignment of few groups closest to 80/10/10 by line count.

    Every assignment with all splits non-empty is scored by the sum of absolute
    deviations of line fractions from the targets. Ties prefer the larger train
    split, then the lexicographically smallest assignment over groups sorted by
    first image name (train=0, val=1, test=2).
    """
    ordered = sorted(groups, key=lambda rows: rows[0][0])
    sizes = [len(rows) for rows in ordered]
    total = sum(sizes)
    best_key: tuple[int, int, tuple[int, ...]] | None = None
    best: tuple[int, ...] = ()
    for choice in itertools.product(range(len(SPLITS)), repeat=len(ordered)):
        counts = [0] * len(SPLITS)
        for split_index, size in zip(choice, sizes):
            counts[split_index] += size
        if not all(counts):
            continue
        # Deviation scaled by 10 * total so the comparison stays in integers.
        deviation = sum(
            abs(10 * count - round(10 * target) * total)
            for count, target in zip(counts, TARGET_FRACTIONS)
        )
        key = (deviation, -counts[0], choice)
        if best_key is None or key < best_key:
            best_key, best = key, choice
    assignments: dict[str, list[Row]] = {split: [] for split in SPLITS}
    for split_index, rows in zip(best, ordered):
        assignments[SPLITS[split_index]].extend(rows)
    return assignments


def split_groups(groups: list[list[Row]], seed: int) -> tuple[dict[str, list[Row]], str]:
    """Split whole groups 80/10/10 by lines; return the assignment and the policy used.

    With at most ``EXHAUSTIVE_MAX_GROUPS`` groups every assignment is tried.
    Otherwise groups are shuffled, train is filled to >=80% and val to >=90% of
    lines, the rest is test, and one group is held back for each later split.
    """
    if len(groups) < len(SPLITS):
        raise ValueError(f"Need at least {len(SPLITS)} groups to split, got {len(groups)}")
    if len(groups) <= EXHAUSTIVE_MAX_GROUPS:
        return split_groups_exhaustive(groups), "exhaustive"
    ordered = sorted(groups, key=lambda rows: rows[0][0])
    random.Random(seed).shuffle(ordered)
    total = sum(len(rows) for rows in ordered)
    assignments: dict[str, list[Row]] = {split: [] for split in SPLITS}
    phase = 0
    cumulative = 0
    for index, rows in enumerate(ordered):
        groups_left = len(ordered) - index
        while phase < len(CUMULATIVE_BOUNDS) and (
            (assignments[SPLITS[phase]] and cumulative >= CUMULATIVE_BOUNDS[phase] * total)
            or groups_left <= len(SPLITS) - 1 - phase
        ):
            phase += 1
        assignments[SPLITS[phase]].extend(rows)
        cumulative += len(rows)
    return assignments, "greedy"


def leakage_metrics(splits: dict[str, list[Row]]) -> dict[str, dict[str, int]]:
    """Count val/test lines whose text, near-text, or page key also occurs in train."""
    train_texts = {text for _, text in splits["train"]}
    train_fuzzy = {fuzzy_text(text) for _, text in splits["train"]}
    train_pages = {page_key(name)[1] for name, _ in splits["train"]} - {None}
    metrics: dict[str, dict[str, int]] = {}
    for split in ("val", "test"):
        rows = splits[split]
        metrics[split] = {
            "lines": len(rows),
            "text_in_train": sum(text in train_texts for _, text in rows),
            "page_in_train": sum(page_key(name)[1] in train_pages for name, _ in rows),
            "near_text_in_train": sum(fuzzy_text(text) in train_fuzzy for _, text in rows),
            "lines_without_page_key": sum(page_key(name)[1] is None for name, _ in rows),
        }
    return metrics


def split_counts(splits: dict[str, list[Row]]) -> dict[str, object]:
    """Summarize split sizes and fractions."""
    total = sum(len(rows) for rows in splits.values())
    return {
        **{split: len(splits[split]) for split in SPLITS},
        "total": total,
        "fractions": {
            split: round(len(splits[split]) / total, 4) if total else 0.0 for split in SPLITS
        },
    }


def split_by_key(splits: dict[str, list[Row]], key_fn) -> dict[str, dict[str, object]]:
    """Split counts broken down by a per-name key."""
    keyed: dict[str, dict[str, list[Row]]] = defaultdict(lambda: {split: [] for split in SPLITS})
    for split in SPLITS:
        for row in splits[split]:
            keyed[key_fn(row[0])][split].append(row)
    return {key: split_counts(keyed[key]) for key in sorted(keyed)}


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest_atomic(path: Path, rows: list[Row]) -> None:
    """Replace one manifest atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for image_name, text in rows:
            output.write(f"{image_name}\t{text}\n")
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    """Write one JSON report."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def backup_manifests(root: Path) -> dict[str, object]:
    """Copy current manifests to the backup directory once; never overwrite it."""
    backup = root / BACKUP_DIR_NAME
    if backup.exists():
        return {"path": str(backup), "created": False}
    backup.mkdir()
    for split in SPLITS:
        shutil.copy2(root / f"gt_{split}.txt", backup / f"gt_{split}.txt")
    return {"path": str(backup), "created": True}


def codepoints(text: str) -> str:
    """Render a string as space-separated U+XXXX codepoints."""
    return " ".join(f"U+{ord(char):04X}" for char in text) or "(none)"


def nfc_changes(text: str) -> list[str]:
    """List 'before->after' codepoint mappings that NFC applies to one string.

    The text is cut at every starter (combining class 0) so each change is
    reported on its own; if NFC ever composes across those cuts the whole row
    is reported as one mapping.
    """
    clusters: list[str] = []
    for char in text:
        if clusters and unicodedata.combining(char):
            clusters[-1] += char
        else:
            clusters.append(char)
    normalized = [unicodedata.normalize("NFC", cluster) for cluster in clusters]
    if "".join(normalized) != unicodedata.normalize("NFC", text):
        return [f"{codepoints(text)}->{codepoints(unicodedata.normalize('NFC', text))}"]
    return [
        f"{codepoints(cluster)}->{codepoints(after)}"
        for cluster, after in zip(clusters, normalized)
        if cluster != after
    ]


def syriac_normalizer_changes(text: str) -> list[str]:
    """List 'before->after' codepoint mappings the Syriac symbol normaliser applies."""
    changes = [
        f"{codepoints(char)}->{codepoints(normalize_symbol_encoding(char))}"
        for char in text
        if normalize_symbol_encoding(char) != char
    ]
    if normalize_syriac_text(text) != normalize_symbol_encoding(text):
        changes.append("other (whitespace/NFC in syriac.normalize_text)")
    return changes


def greek_normalizer_changes(text: str, image_name: str) -> list[str]:
    """List the Greek editorial-rule substitutions applied to one transcription."""
    changed, counts = apply_editorial_rules(text, image_name)
    changes = [name for name, count in counts.items() for _ in range(count)]
    if "<" in changed or ">" in changed:
        changes.append("unresolved angle-bracket token left as-is (rows)")
    return changes


def armenian_normalizer_changes(text: str, _image_name: str) -> list[str]:
    """List the Armenian ink-rule substitutions applied to one transcription."""
    _, counts = apply_ink_rules(text)
    return [name for name, count in counts.items() for _ in range(count)]


# language -> (normalizer(text, image_name) applied after NFC,
#              change describer(text, image_name), normalizer name)
LANGUAGE_NORMALIZERS = {
    "armenian": (
        lambda text, _image_name: normalize_armenian_text(text),
        armenian_normalizer_changes,
        "src/preprocessing_data/armenian.py normalize_manifest_text (after NFC)",
    ),
    "syriac": (
        lambda text, _image_name: normalize_syriac_text(text),
        lambda text, _image_name: syriac_normalizer_changes(text),
        "src/preprocessing_data/syriac.py normalize_text (after NFC)",
    ),
    "greek": (
        normalize_greek_text,
        greek_normalizer_changes,
        "src/preprocessing_data/greek.py normalize_text (after NFC)",
    ),
}


def original_splits(
    root: Path, language: str | None = None
) -> tuple[dict[str, list[Row]], dict[str, object], dict[str, object] | None]:
    """Read the pre-resplit manifests (the backup once it exists), normalized.

    Every transcription is NFC-normalized. Rows of a language listed in
    ``LANGUAGE_NORMALIZERS`` then go through that language's ``normalize_text``
    so the GT matches current conventions while the backups stay untouched.
    """
    backup = root / BACKUP_DIR_NAME
    raw = read_splits(backup if backup.is_dir() else root)
    mappings: Counter[str] = Counter()
    changed: list[str] = []
    language_mappings: Counter[str] = Counter()
    language_changed: list[str] = []
    splits: dict[str, list[Row]] = {}
    for split in SPLITS:
        splits[split] = []
        for image_name, text in raw[split]:
            normalized = unicodedata.normalize("NFC", text)
            if normalized != text:
                changed.append(image_name)
                mappings.update(nfc_changes(text))
            if language in LANGUAGE_NORMALIZERS:
                normalize, describe, _ = LANGUAGE_NORMALIZERS[language]
                language_normalized = normalize(normalized, image_name)
                if language_normalized != normalized:
                    language_changed.append(image_name)
                    language_mappings.update(describe(normalized, image_name))
                normalized = language_normalized
            splits[split].append((image_name, normalized))
    changed_by_source = Counter(page_key(name)[0] for name in changed)
    stats = {
        "rows_changed": len(changed),
        "rows_changed_by_source": dict(sorted(changed_by_source.items())),
        "codepoint_mappings": dict(mappings.most_common()),
    }
    language_stats = None
    if language in LANGUAGE_NORMALIZERS:
        language_stats = {
            "normalizer": LANGUAGE_NORMALIZERS[language][2],
            "rows_changed": len(language_changed),
            "rows_changed_by_source": dict(
                sorted(Counter(page_key(name)[0] for name in language_changed).items())
            ),
            "codepoint_mappings": dict(language_mappings.most_common()),
        }
    return splits, stats, language_stats


def pooled_rows(root: Path, splits: dict[str, list[Row]]) -> list[Row]:
    """Pool all original splits and reject duplicate or missing image names."""
    rows = [row for split in SPLITS for row in splits[split]]
    names = [image_name for image_name, _ in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate image names across splits under {root}")
    for image_name in names:
        if not (root / "image" / image_name).is_file():
            raise FileNotFoundError(f"Missing manifest image: {root / 'image' / image_name}")
    return rows


def page_coverage(pool: list[Row], kept: list[Row], twins: dict[str, list[str]]) -> dict:
    """Report, per source, how many lines carry a page key."""
    derivations = {rule: description for rule, _, description in PAGE_KEY_RULES}
    kept_names = {name for name, _ in kept}
    coverage: dict[str, dict[str, object]] = {}
    for image_name, _ in pool:
        source, key, rule = page_key(image_name)
        entry = coverage.setdefault(
            source,
            {
                "rule": rule,
                "derivation": derivations[rule],
                "rows_before_dedup": 0,
                "rows_after_dedup": 0,
                "rows_with_own_page_key": 0,
                "rows_grouped_via_dropped_twin_page": 0,
                "rows_without_any_page_info": 0,
            },
        )
        entry["rows_before_dedup"] += 1
        if image_name not in kept_names:
            continue
        entry["rows_after_dedup"] += 1
        if key is not None:
            entry["rows_with_own_page_key"] += 1
        elif any(page_key(twin)[1] is not None for twin in twins[image_name]):
            entry["rows_grouped_via_dropped_twin_page"] += 1
        else:
            entry["rows_without_any_page_info"] += 1
    for entry in coverage.values():
        after = int(entry["rows_after_dedup"])
        entry["fraction_without_own_page_key"] = (
            round(1 - int(entry["rows_with_own_page_key"]) / after, 4) if after else None
        )
        entry["fraction_without_any_page_info"] = (
            round(int(entry["rows_without_any_page_info"]) / after, 4) if after else None
        )
    return dict(sorted(coverage.items()))


def resplit_language_partition(root: Path, language: str, partition: str) -> dict:
    """Dedup and page-disjointly resplit one language partition."""
    backup = backup_manifests(root)
    before, nfc_stats, language_stats = original_splits(root, language)
    pool = pooled_rows(root, before)
    exact_kept, dropped = drop_duplicate_texts(pool)
    kept, near_dropped = drop_near_duplicates(exact_kept)
    twins = dropped_twins(dropped, near_dropped)
    components = connected_components(kept, twins)

    balance_groups: dict[str, list[list[Row]]] = defaultdict(list)
    cross_group_components = 0
    for rows in components:
        votes = Counter(source_group(language, partition, name) for name, _ in rows)
        if len(votes) > 1:
            cross_group_components += 1
        top = max(votes.values())
        balance_groups[min(group for group, count in votes.items() if count == top)].append(rows)

    # Seeds follow the sorted group names before any merge. A group with too few
    # components to fill three splits joins the largest group.
    seeds = {group: SEED + index for index, group in enumerate(sorted(balance_groups))}
    merged_balance_groups: dict[str, str] = {}
    largest = max(sorted(balance_groups), key=lambda group: len(balance_groups[group]))
    for group in sorted(balance_groups):
        if group != largest and len(balance_groups[group]) < len(SPLITS):
            balance_groups[largest].extend(balance_groups.pop(group))
            merged_balance_groups[group] = largest

    after: dict[str, list[Row]] = {split: [] for split in SPLITS}
    policies: dict[str, str] = {}
    for group in sorted(balance_groups):
        assignments, policies[group] = split_groups(balance_groups[group], seeds[group])
        for split in SPLITS:
            after[split].extend(assignments[split])
    for split in SPLITS:
        after[split].sort()

    sizes = sorted((len(rows) for rows in components), reverse=True)
    report = {
        "dataset": f"{language}/{partition}",
        "root": str(root),
        "generated_by": "src/preprocessing_data/resplit.py",
        "seed": SEED,
        "input_manifests": str(root / BACKUP_DIR_NAME),
        "backup": backup,
        "split_policy": (
            "transcriptions NFC-normalized; exact duplicates, then near duplicates (loose key "
            ">= 20 chars) dropped; connected components over shared page key "
            "and shared transcription, page-less kept rows inheriting the page of dropped "
            "twins (exact or near). Balance groups with <= 12 components: exhaustive search "
            "over all assignments with non-empty splits, minimising the summed absolute "
            "deviation of line fractions from 0.8/0.1/0.1 (ties: larger train, then "
            "lexicographic). Otherwise greedy: components shuffled with "
            "random.Random(seed + group_index), train filled to >=80% of lines, val to >=90%, "
            "rest test (one component held back per later split)"
        ),
        "counts": {"before": split_counts(before), "after": split_counts(after)},
        "nfc": nfc_stats,
        "language_normalization": language_stats,
        "duplicates": {
            "rows_before": len(pool),
            "rows_after": len(exact_kept),
            "dropped_count": len(dropped),
            "dropped": dropped,
        },
        "near_duplicates": {
            "rule": (
                f"after NFC and exact dedup: loose key = NFD, drop Mn, P*, S*, whitespace and U+03D7 "
                f"(kai sign, omitted by Esteban copies of Stavronikita lines), "
                f"casefold; keys of length >= {NEAR_DUPLICATE_MIN_KEY_LENGTH} shared by several "
                "rows keep one row (non-esteban, then smallest name)"
            ),
            "rows_before": len(exact_kept),
            "rows_after": len(kept),
            "dropped_count": len(near_dropped),
            "dropped": near_dropped,
        },
        "groups": {
            "components": len(components),
            "largest_component_lines": sizes[:5],
            "singleton_components": sum(size == 1 for size in sizes),
            "components_spanning_balance_groups": cross_group_components,
            "balance_groups": {group: len(balance_groups[group]) for group in sorted(balance_groups)},
            "split_policy_per_balance_group": policies,
            "merged_balance_groups": merged_balance_groups,
        },
        "page_key_coverage": page_coverage(pool, kept, twins),
        "achieved_fractions": {
            "overall": split_counts(after)["fractions"],
            "per_balance_group": split_by_key(
                after, lambda name: source_group(language, partition, name)
            ),
            "per_source": split_by_key(after, lambda name: page_key(name)[0]),
        },
        "leakage": {"before": leakage_metrics(before), "after": leakage_metrics(after)},
    }
    return {"report": report, "after": after}


def rebuild_combined_partition(partition: str, root: Path = PROCESSED_ROOT) -> dict:
    """Rebuild combined manifests from the normalized language manifests.

    A language is included only if its crops are present in the combined image
    directory; rows whose crop is absent there are left out and reported.
    """
    combined_root = root / "combined" / partition
    backup = backup_manifests(combined_root)
    before, nfc_stats, _ = original_splits(combined_root)
    combined_images = {path.name for path in (combined_root / "image").iterdir()}

    after: dict[str, list[Row]] = {split: [] for split in SPLITS}
    languages: dict[str, dict[str, object]] = {}
    seen_names: set[str] = set()
    for language in LANGUAGES:
        language_root = root / language / partition
        if not language_root.is_dir():
            continue
        language_splits = read_splits(language_root)
        all_rows = [row for split in SPLITS for row in language_splits[split]]
        present = sum(name in combined_images for name, _ in all_rows)
        missing_sources = Counter(
            page_key(name)[0] for name, _ in all_rows if name not in combined_images
        )
        included = present > 0
        languages[language] = {
            "included": included,
            "rows": len(all_rows),
            "rows_included": present if included else 0,
            "rows_missing_from_combined_image_dir_by_source": dict(sorted(missing_sources.items())),
        }
        if not included:
            continue
        for split in SPLITS:
            for image_name, text in language_splits[split]:
                if image_name not in combined_images:
                    continue
                if image_name in seen_names:
                    raise ValueError(
                        f"Duplicate image name in combined {partition}/{split}: {image_name}"
                    )
                seen_names.add(image_name)
                after[split].append((image_name, text))

    for split in SPLITS:
        after[split].sort()
        write_manifest_atomic(combined_root / f"gt_{split}.txt", after[split])
    report = {
        "dataset": f"combined/{partition}",
        "root": str(combined_root),
        "generated_by": "src/preprocessing_data/resplit.py",
        "input_manifests": "language manifests written by this run",
        "backup": backup,
        "nfc_of_previous_manifests": nfc_stats,
        "languages": languages,
        "counts": {"before": split_counts(before), "after": split_counts(after)},
        "achieved_fractions": {
            "overall": split_counts(after)["fractions"],
            "per_source": split_by_key(after, lambda name: page_key(name)[0]),
        },
        "leakage": {"before": leakage_metrics(before), "after": leakage_metrics(after)},
        "cross_language_identical_texts_across_splits": cross_language_shared_texts(
            after, language_of_names(root, partition)
        ),
        "sha256": {
            f"gt_{split}.txt": sha256(combined_root / f"gt_{split}.txt") for split in SPLITS
        },
    }
    write_json(combined_root / REPORT_NAME, report)
    return report["counts"]["after"]


def language_of_names(root: Path, partition: str) -> dict[str, str]:
    """Map each crop name in the language manifests of one partition to its language."""
    return {
        image_name: language
        for language in LANGUAGES
        if (root / language / partition).is_dir()
        for split in SPLITS
        for image_name, _ in parse_manifest(root / language / partition / f"gt_{split}.txt")
    }


def cross_language_shared_texts(rows: dict[str, list[Row]], language_of: dict[str, str]) -> list:
    """List identical transcriptions that different languages put in different splits."""
    owners: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for split in SPLITS:
        for image_name, text in rows[split]:
            owners[text].add((split, language_of.get(image_name, ""), image_name))
    return [
        {"text": text, "rows": [list(owner) for owner in sorted(found)]}
        for text, found in sorted(owners.items())
        if len({split for split, _, _ in found}) > 1
    ]


def validate_partition(root: Path, language_of: dict[str, str] | None = None) -> None:
    """Fail if a name, transcription or page key spans splits, or an image is missing.

    In combined datasets, pass ``language_of`` so transcriptions are compared
    within one language only: a page numeral such as "43" in Greek and in
    Syriac is two different physical lines.
    """
    language_of = language_of or {}
    owners: dict[str, dict[object, str]] = {"image name": {}, "transcription": {}, "page key": {}}
    for split in SPLITS:
        rows = parse_manifest(root / f"gt_{split}.txt")
        names = {image_name for image_name, _ in rows}
        if len(names) != len(rows):
            raise ValueError(f"Duplicate rows in {root}/gt_{split}.txt")
        for image_name, text in rows:
            if not (root / "image" / image_name).is_file():
                raise FileNotFoundError(f"Missing image: {root / 'image' / image_name}")
            values = {
                "image name": image_name,
                "transcription": (language_of.get(image_name, ""), text),
                "page key": page_key(image_name)[1],
            }
            for kind, value in values.items():
                if value is None:
                    continue
                owner = owners[kind].setdefault(value, split)
                if owner != split:
                    raise ValueError(
                        f"{kind} {value!r} is in both {owner} and {split} under {root}"
                    )


def main() -> None:
    """Normalize language and combined TrOCR manifests."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROCESSED_ROOT)
    root = parser.parse_args().root.resolve()

    summary: dict[str, dict] = {}
    for partition in PARTITIONS:
        for language in LANGUAGES:
            language_root = root / language / partition
            if not language_root.is_dir():
                continue
            result = resplit_language_partition(language_root, language, partition)
            for split in SPLITS:
                write_manifest_atomic(language_root / f"gt_{split}.txt", result["after"][split])
            report = result["report"]
            report["sha256"] = {
                f"gt_{split}.txt": sha256(language_root / f"gt_{split}.txt") for split in SPLITS
            }
            write_json(language_root / REPORT_NAME, report)
            summary[f"{language}/{partition}"] = report["counts"]["after"]
        if (root / "combined" / partition).is_dir():
            summary[f"combined/{partition}"] = rebuild_combined_partition(partition, root)

    for partition in PARTITIONS:
        for dataset in (*LANGUAGES, "combined"):
            dataset_root = root / dataset / partition
            if not dataset_root.is_dir():
                continue
            validate_partition(
                dataset_root,
                language_of_names(root, partition) if dataset == "combined" else None,
            )
            if partition == "finetuning":
                names = {
                    image_name
                    for split in SPLITS
                    for image_name, _ in parse_manifest(dataset_root / f"gt_{split}.txt")
                }
                if any(name.startswith(("esteban__", "labelled__")) for name in names):
                    raise ValueError(f"External pretraining data leaked into {dataset_root}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

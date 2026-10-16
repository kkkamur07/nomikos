#!/usr/bin/env python3
"""Pure import helpers for the annotated-corpus dev seed (library, no DB).

``seed_dev_annotated_data.py`` stays a thin entry script (flags, ``run_seed``,
the dev guard); the filesystem walk, the legacy-JSON shaping and the import
statistics live here so the entry stays under the file-length gate. Nothing
here touches the database or the backend: that keeps these helpers importable
(and testable) without settings or a session.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TypedDict, cast
from uuid import UUID

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
_PAGE_SUFFIX_RE = re.compile(r"^(.+)_(\d+)$")
_IMAGE_KEY_STEM_RE = re.compile(r"/([^/]+)\.[a-z0-9]{1,16}$")


class SeedSegment(TypedDict, total=False):
    """One legacy annotation segment, as shaped by the on-disk corpus."""

    number: int
    kind: str
    points: list[list[float]]
    source: str
    source_metadata: dict[str, object] | None
    kraken_ceiling: float | None
    text_override: str | None
    baseline: dict[str, object]
    mask: dict[str, object]
    paired_text_line_index: int


class SeedAnnotation(TypedDict, total=False):
    """One legacy page annotation: an optional list of segments."""

    segments: list[SeedSegment]


class HistoryLine(TypedDict):
    """One history-snapshot line, as stored on the platform snapshot."""

    id: str
    block_id: None
    order: int
    kind: str
    points: list[list[float]]
    source: str
    source_metadata: dict[str, object] | None
    kraken_ceiling: float | None
    approved_text: str | None


class SnapshotState(TypedDict):
    """The legacy snapshot state converted to platform history lines."""

    lines: list[HistoryLine]


def _resolve_dev_password() -> tuple[str, bool]:
    """Return (password, generated) for the dev account this seed may create.

    Never falls back to a committed literal: a seed script that ships its own
    password hands out a working login to anyone who reads the repository.
    Callers that want a predictable account export DEV_USER_PASSWORD; otherwise
    we mint a throwaway and print it once, since only the hash survives.
    """
    import os
    import secrets

    supplied = os.environ.get("DEV_USER_PASSWORD")
    if supplied:
        return supplied, False
    return secrets.token_urlsafe(24), True


@dataclass
class ImportStats:
    documents_created: int = 0
    documents_skipped: int = 0
    parts: int = 0
    segments: int = 0
    history_snapshots: int = 0
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: ImportStats) -> None:
        self.documents_created += other.documents_created
        self.documents_skipped += other.documents_skipped
        self.parts += other.parts
        self.segments += other.segments
        self.history_snapshots += other.history_snapshots
        self.warnings.extend(other.warnings)


def _page_sort_key(stem: str) -> tuple[str, int]:
    match = _PAGE_SUFFIX_RE.match(stem)
    if match is None:
        return (stem, 0)
    return (match.group(1), int(match.group(2)))


def _group_pages_by_document(page_stems: list[str]) -> dict[str, list[str]]:
    """Group page stems into documents; multi-page manuscripts share a prefix."""
    prefix_counts: dict[str, int] = {}
    for stem in page_stems:
        match = _PAGE_SUFFIX_RE.match(stem)
        if match is not None:
            prefix_counts[match.group(1)] = prefix_counts.get(match.group(1), 0) + 1

    groups: dict[str, list[str]] = {}
    for stem in sorted(page_stems):
        match = _PAGE_SUFFIX_RE.match(stem)
        if match is not None and prefix_counts[match.group(1)] > 1:
            groups.setdefault(match.group(1), []).append(stem)
        else:
            groups[stem] = [stem]

    for doc_key, stems in groups.items():
        if len(stems) > 1:
            groups[doc_key] = sorted(stems, key=_page_sort_key)
    return groups


def _document_display_name(doc_key: str, page_stems: list[str]) -> str:
    if len(page_stems) == 1:
        return page_stems[0]
    if doc_key.startswith("Grec_1360_"):
        return "Grec 1360 - Harmenopulus (btv1b10721710m)"
    return doc_key.replace("_", " ")


def _stem_from_image_key(image_key: str) -> str | None:
    match = _IMAGE_KEY_STEM_RE.search(image_key)
    return match.group(1) if match else None


def _resolve_page_image(pages_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = pages_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _load_annotation(annotations_dir: Path, stem: str) -> SeedAnnotation | None:
    path = annotations_dir / f"{stem}.json"
    if not path.is_file():
        return None
    return cast("SeedAnnotation", _load_json(path))


def _load_page_transcription(transcriptions_dir: Path, stem: str) -> str | None:
    path = transcriptions_dir / f"{stem}.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


def _approved_text(segment: SeedSegment) -> str | None:
    text = segment.get("text_override")
    if text is None:
        return None
    text = str(text).strip()
    return text or None


def _segment_to_history_line(segment: SeedSegment, line_id: UUID) -> HistoryLine:
    return {
        "id": str(line_id),
        "block_id": None,
        "order": int(segment.get("number", 1)) - 1,
        "kind": segment.get("kind", "polygon"),
        "points": segment["points"],
        "source": segment.get("source", "manual"),
        "source_metadata": segment.get("source_metadata"),
        "kraken_ceiling": segment.get("kraken_ceiling"),
        "approved_text": _approved_text(segment),
    }


def _legacy_snapshot_state(annotation: SeedAnnotation) -> SnapshotState:
    import uuid

    segments = annotation.get("segments") or []
    lines = [
        _segment_to_history_line(segment, uuid.uuid4())
        for segment in sorted(segments, key=lambda item: int(item.get("number", 1)))
    ]
    return {"lines": lines}


def _parse_legacy_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _discover_page_stems(data_root: Path) -> list[str]:
    pages_dir = data_root / "manuscripts" / "pages"
    annotations_dir = data_root / "annotations" / "pages"
    if not pages_dir.is_dir():
        raise FileNotFoundError(f"Missing page images directory: {pages_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")

    stems: set[str] = set()
    for path in pages_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            stems.add(path.stem)
    for path in annotations_dir.glob("*.json"):
        stems.add(path.stem)
    return sorted(stems)


def _list_legacy_history_files(history_dir: Path, page_stem: str) -> list[Path]:
    page_history_dir = history_dir / page_stem
    if not page_history_dir.is_dir():
        return []
    return sorted(
        (path for path in page_history_dir.glob("*.json") if path.name != "_state.json"),
        key=lambda path: path.stat().st_mtime,
    )


def _image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as image:
            return image.size
    except OSError:
        return None, None

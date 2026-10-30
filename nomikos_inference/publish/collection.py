"""Load and validate Hub collection metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from nomikos_inference.publish.paths import DEFAULT_COLLECTION_PATH


@dataclass(frozen=True, slots=True)
class CollectionItemRef:
    slug: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    namespace: str
    slug: str
    title: str
    description: str
    hub_slug: str | None
    models: tuple[CollectionItemRef, ...]
    datasets: tuple[CollectionItemRef, ...]

    def repo_id(self, item_slug: str) -> str:
        return f"{self.namespace}/{item_slug}"


def _parse_items(raw_items: object) -> tuple[CollectionItemRef, ...]:
    if raw_items is None:
        return ()
    if not isinstance(raw_items, list):
        raise ValueError("collection items must be a list")

    items: list[CollectionItemRef] = []
    for entry in raw_items:
        if isinstance(entry, str):
            items.append(CollectionItemRef(slug=entry))
            continue
        if isinstance(entry, dict) and "slug" in entry:
            items.append(
                CollectionItemRef(
                    slug=str(entry["slug"]),
                    note=entry.get("note"),
                )
            )
            continue
        raise ValueError(f"invalid collection item entry: {entry!r}")
    return tuple(items)


def load_collection_spec(path: Path | None = None) -> CollectionSpec:
    collection_path = path or DEFAULT_COLLECTION_PATH
    if not collection_path.is_file():
        raise ValueError(f"collection file not found: {collection_path}")

    data = yaml.safe_load(collection_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"collection file must be a mapping: {collection_path}")

    for key in ("namespace", "slug", "title", "description"):
        if key not in data or not str(data[key]).strip():
            raise ValueError(f"collection file missing required field: {key}")

    return CollectionSpec(
        namespace=str(data["namespace"]).strip(),
        slug=str(data["slug"]).strip(),
        title=str(data["title"]).strip(),
        description=str(data["description"]).strip(),
        hub_slug=str(data["hub_slug"]).strip() if data.get("hub_slug") else None,
        models=_parse_items(data.get("models")),
        datasets=_parse_items(data.get("datasets")),
    )

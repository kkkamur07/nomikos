"""Collection spec loading: exact values and every rejection branch."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from nomikos_inference.publish.collection import _parse_items, load_collection_spec


def _write_spec(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_parse_items_accepts_strings_and_notes() -> None:
    items = _parse_items(["a", {"slug": "b", "note": "hi"}])
    assert [(item.slug, item.note) for item in items] == [("a", None), ("b", "hi")]


def test_parse_items_empty_is_empty() -> None:
    assert _parse_items(None) == ()
    assert _parse_items([]) == ()


def test_parse_items_rejects_non_lists_and_bad_entries() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        _parse_items(cast("list[object]", {"slug": "a"}))
    with pytest.raises(ValueError, match="invalid collection item entry"):
        _parse_items([42])


def test_load_collection_spec_pins_exact_values(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path / "collection.yaml",
        "namespace: nomikos\n"
        "slug: nomos\n"
        "title: Nomos collection\n"
        "description: All models.\n"
        "hub_slug: nomos\n"
        "models:\n"
        "  - greek-htr-calamari\n"
        "  - slug: syriac-htr-calamari\n"
        "    note: v2\n"
        "datasets: []\n",
    )
    spec = load_collection_spec(path)
    assert spec.namespace == "nomikos"
    assert spec.hub_slug == "nomos"
    assert [item.slug for item in spec.models] == ["greek-htr-calamari", "syriac-htr-calamari"]
    assert spec.models[1].note == "v2"
    assert spec.datasets == ()
    assert spec.repo_id("greek-htr-calamari") == "nomikos/greek-htr-calamari"


def test_load_collection_spec_rejects_bad_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collection file not found"):
        load_collection_spec(tmp_path / "absent.yaml")
    bad_list = _write_spec(tmp_path / "list.yaml", "- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_collection_spec(bad_list)
    missing = _write_spec(tmp_path / "missing.yaml", "namespace: nomikos\nslug: nomos\n")
    with pytest.raises(ValueError, match="missing required field: title"):
        load_collection_spec(missing)

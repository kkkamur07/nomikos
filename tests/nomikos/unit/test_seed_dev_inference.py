"""Dev inference seed display names: pair parsing and defaults, no database."""

import sys
from pathlib import Path

import pytest

SCRIPTS_PLATFORM = Path(__file__).resolve().parents[3] / "scripts" / "platform"
sys.path.insert(0, str(SCRIPTS_PLATFORM))

import seed_dev_inference as seed  # noqa: E402

from nomikos_inference.contracts.common import RegistryArchitecture  # noqa: E402


def test_parse_entry_splits_registry_id_and_display_name() -> None:
    assert seed._parse_entry("syriac-ppocr-v1:syriac-v2") == ("syriac-ppocr-v1", "syriac-v2")


def test_parse_entry_without_display_name_keeps_registry_id() -> None:
    assert seed._parse_entry("blla-segment") == ("blla-segment", "blla-segment")


def test_pairs_override_uses_registry_id_display_name_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFAULT_SEGMENT_MODEL", "ppocr-segment:seg-v2, blla-segment")
    assert seed._pairs("DEFAULT_SEGMENT_MODEL", ()) == (
        ("ppocr-segment", "seg-v2"),
        ("blla-segment", "blla-segment"),
    )


def test_default_segment_pairs_cover_both_segmenters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEFAULT_SEGMENT_MODEL", raising=False)
    assert seed._pairs("DEFAULT_SEGMENT_MODEL", seed.SEGMENT_MODELS) == (
        ("blla-segment", "seg-v1"),
        ("ppocr-segment", "seg-v2"),
    )


def test_default_transcribe_pairs_retire_calamari_syriac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEFAULT_TRANSCRIBE_MODEL", raising=False)
    pairs = seed._pairs("DEFAULT_TRANSCRIBE_MODEL", seed.TRANSCRIBE_MODELS)
    assert pairs == (
        ("greek-calamari-v1", "greek-v1"),
        ("armenian-calamari-v1", "armenian-v1"),
        ("coptic-calamari-v1", "coptic-v1"),
        ("syriac-ppocr-v1", "syriac-v2"),
    )
    assert "syriac-calamari-v2" not in [registry_id for registry_id, _ in pairs]


def test_provider_map_covers_ppocr_detection() -> None:
    assert seed._PROVIDER_BY_ARCHITECTURE[RegistryArchitecture.ppocr_det] == "ppocr"
    assert seed._PROVIDER_BY_ARCHITECTURE[RegistryArchitecture.ppocr_rec] == "ppocr"

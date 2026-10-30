"""PP-OCR recognition bidi: the vendored reorder and the adapter's use of it.

The network emits characters in display order (left to right on the image);
kraken's ``BaselineOCRRecord.logical_order`` runs that string through its bidi
``get_display_map`` and permutes the confidences with the same map. These
tests pin the vendored copy's behavior on the cases serving depends on: a
Syriac display string comes back in logical order with a bijective
permutation, and a Latin string passes through unchanged.
"""

from __future__ import annotations

from typing import Literal, cast

import pytest

from nomikos_inference.architectures.ppocr_rec.adapter import _reorder_to_logical
from nomikos_inference.architectures.ppocr_rec.bidi import UCD_VERSION, get_display_map
from nomikos_inference.architectures.ppocr_rec.bidi._resolve import resolve_levels


def test_syriac_display_order_becomes_logical_order() -> None:
    text, order = get_display_map("ܝܪܡ", None)
    assert text == "ܡܪܝ"
    assert order == [2, 1, 0]


def test_permutation_is_a_bijection() -> None:
    text, order = get_display_map("ܝܪܡ", None)
    assert sorted(order) == list(range(len("ܝܪܡ")))
    assert len(text) == len(order)


def test_latin_string_is_unchanged() -> None:
    text, order = get_display_map("hello", None)
    assert text == "hello"
    assert order == [0, 1, 2, 3, 4]


def test_empty_string_reorders_to_empty() -> None:
    assert get_display_map("", None) == ("", [])


def test_adapter_reorder_permutes_confidences_with_the_text() -> None:
    text, confidences = _reorder_to_logical("ܝܪܡ", [0.1, 0.2, 0.3])
    assert text == "ܡܪܝ"
    assert confidences == [0.3, 0.2, 0.1]


def test_adapter_reorder_leaves_latin_confidences_in_place() -> None:
    text, confidences = _reorder_to_logical("ab", [0.5, 0.6])
    assert text == "ab"
    assert confidences == [0.5, 0.6]


def test_vendored_tables_match_the_pinned_unicode_version() -> None:
    assert UCD_VERSION == "17.0.0"


def test_rtl_run_reverses_with_a_bijective_permutation() -> None:
    text, order = get_display_map("שלום", None)
    assert text == "םולש"
    assert order == [3, 2, 1, 0]


def test_mixed_direction_text_reorders_each_run() -> None:
    text, order = get_display_map("אבג def", None)
    assert text == "def גבא"
    assert order == [4, 5, 6, 3, 2, 1, 0]


def test_brackets_around_rtl_resolve_and_mirror() -> None:
    text, order = get_display_map("(א)", None)
    assert text == "(א)"
    assert order == [2, 1, 0]


def test_explicit_embeddings_are_removed_from_the_output() -> None:
    text, order = get_display_map("a\u202bB\u202cc", None)
    assert text == "aBc"
    assert order == [0, 2, 4]


def test_isolates_are_removed_from_the_output() -> None:
    text, order = get_display_map("a\u2066B\u2069c", None)
    assert text == "aBc"
    assert order == [0, 2, 4]


def test_nsm_follows_its_base_character() -> None:
    text, order = get_display_map("א\u0301x", None)
    assert text == "x\u0301א"
    assert order == [2, 1, 0]


def test_european_number_after_arabic_letter_stays_rtl() -> None:
    text, _order = get_display_map("أ\u0661\u0662\u0663", None)
    assert text == "\u0661\u0662\u0663أ"


def test_explicit_base_direction_overrides_first_strong() -> None:
    assert get_display_map("אבג", "L") == ("גבא", [2, 1, 0])
    assert get_display_map("abc", "R") == ("abc", [0, 1, 2])


def test_invalid_base_direction_is_rejected() -> None:
    bad = cast("Literal['L', 'R'] | None", "X")
    with pytest.raises(ValueError, match="Invalid base direction"):
        get_display_map("abc", bad)


def test_explicit_levels_remove_x9_characters_and_bump_l() -> None:
    levels, visual = resolve_levels(["RLE", "L", "PDF", "R"], [0x202B, 65, 0x202C, 0x05D0], 0)
    assert levels == [None, 2, None, 1]
    assert visual == [3, 1]

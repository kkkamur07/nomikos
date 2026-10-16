"""Table-driven UAX #9 rule tests with exact expected outputs.

Each case pins the exact levels, visual order, display string or helper
return the algorithm produces, so a mutant that changes any threshold,
set membership, boundary comparison or rule branch fails here. Values
were verified against the pre-split vendored copy (165-case differential
check, zero mismatches).
"""

from __future__ import annotations

import pytest

from nomikos_inference.architectures.ppocr_rec.bidi._common import (
    MAX_DEPTH,
    MAX_PAIRING_DEPTH,
    bidi_class,
)
from nomikos_inference.architectures.ppocr_rec.bidi._explicit import (
    explicit_levels,
    first_strong_level,
    match_isolates,
)
from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _find_bracket_pairs
from nomikos_inference.architectures.ppocr_rec.bidi._resolve import (
    get_display_map,
    resolve_levels,
)


@pytest.mark.parametrize(
    ("codepoint", "expected"),
    [
        (0x0000, "BN"),
        (0x0001, "BN"),
        (0x0008, "BN"),
        (0x0009, "S"),
        (0x000A, "B"),
        (0x0020, "WS"),
        (0x0030, "EN"),
        (0x0041, "L"),
        (0x05D0, "R"),
        (0x0661, "AN"),
        (0x0627, "AL"),
        (0x0301, "NSM"),
        (0x0028, "ON"),
        (0x002C, "CS"),
        (0x002B, "ES"),
        (0x0023, "ET"),
        (0x202B, "RLE"),
        (0x202A, "LRE"),
        (0x202C, "PDF"),
        (0x202D, "LRO"),
        (0x202E, "RLO"),
        (0x2066, "LRI"),
        (0x2067, "RLI"),
        (0x2068, "FSI"),
        (0x2069, "PDI"),
        (0x200E, "L"),
        (0x200F, "R"),
        (0xFEFF, "BN"),
        (0x10FFFF, "BN"),
        (0x110000, "L"),
    ],
)
def test_bidi_class_table(codepoint: int, expected: str) -> None:
    assert bidi_class(codepoint) == expected


def test_max_depth_is_125() -> None:
    assert MAX_DEPTH == 125
    assert MAX_PAIRING_DEPTH == 63


@pytest.mark.parametrize(
    ("depth", "expected_level", "removed"),
    [(62, 123, 62), (63, 125, 63), (64, 125, 64), (70, 125, 70)],
)
def test_embedding_depth_boundary(depth: int, expected_level: int, removed: int) -> None:
    """The 126th open embedding overflows instead of pushing level 127."""
    types = ["RLE"] * depth + ["L"]
    levels = explicit_levels(types, list(types), {}, 0)
    assert levels[-1] == expected_level
    assert levels.count(None) == removed


@pytest.mark.parametrize(("opens", "expected_pairs"), [(1, 1), (62, 62), (63, 63), (64, 0)])
def test_bracket_pairing_depth_boundary(opens: int, expected_pairs: int) -> None:
    """The 64th open bracket aborts pairing: the BD16 stack holds 63."""
    seq = list(range(opens)) + list(range(opens, 2 * opens))
    wtypes = ["ON"] * (2 * opens)
    cps = [0x28] * opens + [0x29] * opens
    assert len(_find_bracket_pairs(seq, wtypes, cps)) == expected_pairs


@pytest.mark.parametrize(
    ("types", "expected_levels", "expected_wtypes"),
    [
        (["LRE", "L", "PDF"], [None, 2, None], ["LRE", "L", "PDF"]),
        (["RLE", "L", "PDF"], [None, 1, None], ["RLE", "L", "PDF"]),
        (["RLO", "L", "PDF"], [None, 1, None], ["RLO", "R", "PDF"]),
        (["LRO", "R", "PDF"], [None, 2, None], ["LRO", "L", "PDF"]),
        (["RLI", "L", "PDI"], [0, 1, 0], ["RLI", "L", "PDI"]),
        (["FSI", "L", "PDI"], [0, 2, 0], ["FSI", "L", "PDI"]),
        (["FSI", "R", "PDI"], [0, 1, 0], ["FSI", "R", "PDI"]),
        (["BN", "L", "BN"], [None, 0, None], ["BN", "L", "BN"]),
        (["L", "PDF", "L"], [0, None, 0], ["L", "PDF", "L"]),
    ],
)
def test_explicit_levels_table(
    types: list[str], expected_levels: list[int | None], expected_wtypes: list[str]
) -> None:
    wtypes = list(types)
    assert explicit_levels(types, wtypes, match_isolates(types)[0], 0) == expected_levels
    assert wtypes == expected_wtypes


def test_match_isolates_table() -> None:
    assert match_isolates(["LRI", "LRI", "PDI"]) == ({1: 2, 0: None}, {2})
    assert match_isolates(["LRI", "L"]) == ({0: None}, set())
    assert match_isolates(["PDI", "L"]) == ({}, set())
    assert match_isolates(["RLI", "L", "PDI"]) == ({0: 2}, {2})


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (["L"], 0),
        (["R"], 1),
        (["AL"], 1),
        (["EN"], 0),
        (["LRI", "R", "PDI", "L"], 0),
        (["LRI", "L", "PDI", "R"], 1),
        ([], 0),
    ],
)
def test_first_strong_level_table(types: list[str], expected: int) -> None:
    assert first_strong_level(types, 0, len(types)) == expected


@pytest.mark.parametrize(
    ("types", "expected_levels"),
    [
        # W7: EN after strong L becomes L.
        (["L", "EN", "L"], [0, 0, 0]),
        # W3: AL becomes R, then I2 bumps it at an even level.
        (["L", "AL", "L"], [0, 1, 0]),
        # W4: CS between two ENs joins them; W7 already made them L here.
        (["EN", "CS", "EN"], [0, 0, 0]),
        # W4 in an RTL run: CS joins, I1 lifts the numbers.
        (["R", "EN", "CS", "EN"], [1, 2, 2, 2]),
        # W5: ET next to EN joins it.
        (["R", "ET", "EN"], [1, 2, 2]),
        (["R", "EN", "ET"], [1, 2, 2]),
        # W6: a lone ES becomes ON and resolves neutral.
        (["R", "ES", "EN"], [1, 1, 2]),
        # W2: EN after AL becomes AN.
        (["AL", "EN"], [1, 2]),
        # W1: NSM takes the preceding type, including sos at run start.
        (["L", "NSM", "R"], [0, 0, 1]),
        (["R", "NSM", "L"], [1, 1, 0]),
        (["LRE", "NSM", "L"], [None, 2, 2]),
        # N1/N2: neutrals take the surrounding type, or the embedding.
        (["R", "ON", "R"], [1, 1, 1]),
        (["L", "ON", "L"], [0, 0, 0]),
        (["L", "ON", "R"], [0, 0, 1]),
        (["R", "ON", "L"], [1, 0, 0]),
        (["LRE", "ON", "R"], [None, 2, 3]),
        (["LRE", "R", "ON"], [None, 3, 2]),
        (["LRE", "EN", "L"], [None, 2, 2]),
        (["RLE", "ON", "L"], [None, 1, 2]),
        (["RLE", "L", "ON"], [None, 2, 1]),
        # I1/I2/L1: stray formatting, BN removal, trailing reset.
        (["L", "PDI", "L"], [0, 0, 0]),
        (["L", "PDF", "L"], [0, None, 0]),
        (["R", "L", "WS"], [1, 0, 0]),
        (["L", "R", "WS"], [0, 1, 0]),
        (["L", "WS", "R"], [0, 0, 1]),
    ],
)
def test_implicit_and_neutral_levels_table(
    types: list[str], expected_levels: list[int | None]
) -> None:
    levels, _visual = resolve_levels(types, None, 0)
    assert levels == expected_levels


@pytest.mark.parametrize(
    ("text", "expected_text", "expected_order"),
    [
        ("שלום", "םולש", [3, 2, 1, 0]),
        ("אבג def", "def גבא", [4, 5, 6, 3, 2, 1, 0]),
        ("(א)", "(א)", [2, 1, 0]),
        ("<א>", "<א>", [2, 1, 0]),
        ("\u3008\u05d0\u3009", "\u3008\u05d0\u3009", [2, 1, 0]),
        ("\u05d0(\u05d1a)", "(a\u05d1)\u05d0", [4, 3, 2, 1, 0]),
        ("1א2", "2א1", [2, 1, 0]),
        ("أ\u0661\u0662\u0663", "\u0661\u0662\u0663أ", [1, 2, 3, 0]),
        ("a,b\u05d0", "a,b\u05d0", [0, 1, 2, 3]),
        ("a b\u05d0", "a b\u05d0", [0, 1, 2, 3]),
        ("a!\u05d0", "a!\u05d0", [0, 1, 2]),
        ("a\u00a0\u05d0", "a\u00a0\u05d0", [0, 1, 2]),
        ("a\ufeffb", "ab", [0, 2]),
        ("a\u202aB\u202cc", "aBc", [0, 2, 4]),
        ("a\u202dB\u202cc", "aBc", [0, 2, 4]),
        ("a\u202eB\u202cc", "aBc", [0, 2, 4]),
        ("a\u2066B\u2069c", "aBc", [0, 2, 4]),
        ("a\u2068\u05d0\u2069c", "a\u05d0c", [0, 2, 4]),
        ("\u2067B\u2069", "B", [1]),
        ("12,5", "12,5", [0, 1, 2, 3]),
        ("1(2)", "1(2)", [0, 1, 2, 3]),
    ],
)
def test_display_table(text: str, expected_text: str, expected_order: list[int]) -> None:
    assert get_display_map(text, None) == (expected_text, expected_order)


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        # Isolates inside neutral runs take the resolved fill, which decides
        # their I1/I2 bump: a surviving LRI/RLI/FSI/PDI bumps differently.
        (["RLE", "LRI", "R", "PDI", "PDF"], ([None, 1, 3, 0, None], [2, 1, 3])),
        (["RLE", "RLI", "R", "PDI", "PDF"], ([None, 1, 3, 0, None], [2, 1, 3])),
        (["RLE", "FSI", "R", "PDI", "PDF"], ([None, 1, 3, 0, None], [2, 1, 3])),
        (["RLE", "LRI", "L", "PDI", "PDF"], ([None, 1, 2, 0, None], [2, 1, 3])),
        (["RLE", "R", "PDI", "L", "PDF"], ([None, 1, 1, 2, None], [3, 2, 1])),
    ],
)
def test_isolates_in_neutral_runs(
    types: list[str], expected: tuple[list[int | None], list[int]]
) -> None:
    assert resolve_levels(types, None, 0) == expected


def test_level_at_rejects_a_removed_character() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._common import level_at

    with pytest.raises(ValueError, match="removed character"):
        level_at([None], 0)


def _explicit(types: list[str]) -> tuple[list[int | None], list[str]]:
    wtypes = list(types)
    levels = explicit_levels(types, wtypes, match_isolates(types)[0], 0)
    return levels, wtypes


def test_embedding_overflow_sticks_at_max_depth() -> None:
    levels, _wtypes = _explicit(["RLE"] * 70 + ["L"] + ["PDF"] * 70)
    assert levels[-3:] == [125, 125, None]
    assert levels.count(None) == 140


def test_isolate_push_at_exactly_max_depth() -> None:
    levels, wtypes = _explicit(["RLE"] * 62 + ["RLI", "L", "PDI", "R"])
    assert levels[-4:] == [123, 125, 123, 123]
    assert wtypes[62] == "RLI"


def test_override_applies_through_an_isolate() -> None:
    levels, wtypes = _explicit(["RLO", "RLI", "L", "PDI", "PDF"])
    assert levels == [None, 1, 3, 1, None]
    assert wtypes == ["RLO", "R", "L", "R", "PDF"]


def test_lre_inside_an_odd_embedding_goes_even() -> None:
    levels, _wtypes = _explicit(["RLE", "LRE", "L", "PDF", "PDF"])
    assert levels == [None, None, 2, None, None]


def test_fsi_without_strong_content_pushes_even() -> None:
    levels, _wtypes = _explicit(["FSI", "WS", "PDI", "R"])
    assert levels == [0, 2, 0, 0]


def test_stray_pdi_does_not_pop_past_paragraph() -> None:
    levels, wtypes = _explicit(["PDI", "R"])
    assert levels == [0, 0]
    assert wtypes == ["PDI", "R"]


def test_rle_override_default_leaves_type() -> None:
    levels, wtypes = _explicit(["RLE", "L", "PDF"])
    assert levels == [None, 1, None]
    assert wtypes == ["RLE", "L", "PDF"]


def test_isolate_overflow_and_unwind() -> None:
    levels, _wtypes = _explicit(["RLE"] * 64 + ["FSI", "L", "PDI"] + ["PDF"])
    assert levels[-4:] == [125, 125, 125, None]

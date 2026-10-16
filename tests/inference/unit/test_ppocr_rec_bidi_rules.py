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
    isolating_run_sequences,
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

    with pytest.raises(ValueError) as excinfo:
        level_at([None], 0)
    assert str(excinfo.value) == "isolating run sequence contains a removed character"


def _explicit(types: list[str]) -> tuple[list[int | None], list[str]]:
    wtypes = list(types)
    levels = explicit_levels(types, wtypes, match_isolates(types)[0], 0)
    return levels, wtypes


def test_embedding_overflow_sticks_at_max_depth() -> None:
    levels, _wtypes = _explicit(["RLE"] * 70 + ["L"] + ["PDF"] * 70)
    assert levels[70] == 125
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


@pytest.mark.parametrize(
    ("types", "para", "expected"),
    [
        # W2: EN after AL becomes AN; the strong tracking decides it.
        (["L", "AL", "EN"], 1, ([2, 1, 2], [2, 1, 0])),
        (["AL", "L", "EN"], 1, ([1, 2, 2], [1, 2, 0])),
        (["AL", "R", "EN"], 1, ([1, 1, 2], [2, 1, 0])),
        # W4: only a separator at index 1 with equal number neighbors joins.
        (["R", "EN", "ES", "EN"], 0, ([1, 2, 2, 2], [1, 2, 3, 0])),
        (["R", "EN", "ES", "AN"], 0, ([1, 2, 1, 2], [3, 2, 1, 0])),
        (["EN", "ES", "EN", "L"], 0, ([0, 0, 0, 0], [0, 1, 2, 3])),
        (["R", "EN", "CS", "EN"], 0, ([1, 2, 2, 2], [1, 2, 3, 0])),
        (["R", "AN", "CS", "AN"], 0, ([1, 2, 2, 2], [1, 2, 3, 0])),
        # W5: terminator runs read the true neighbor, not sos, at index 1.
        (["ET", "L", "EN"], 1, ([1, 2, 2], [1, 2, 0])),
        (["EN", "ET", "L"], 1, ([2, 2, 2], [0, 1, 2])),
        (["ET", "EN", "L"], 1, ([2, 2, 2], [0, 1, 2])),
        # W7: the strong tracker decides EN after L, even at odd levels.
        (["L", "EN", "L"], 1, ([2, 2, 2], [0, 1, 2])),
        (["R", "L", "EN"], 0, ([1, 0, 0], [0, 1, 2])),
        # W1: a leading NSM takes sos, including inside isolates.
        (["NSM", "L"], 1, ([1, 2], [1, 0])),
        (["LRI", "NSM", "L"], 0, ([0, 2, 2], [0, 1, 2])),
    ],
)
def test_weak_rule_tracking_table(
    types: list[str], para: int, expected: tuple[list[int | None], list[int]]
) -> None:
    assert resolve_levels(types, None, para) == expected


def test_nested_isolate_depth_counts() -> None:
    assert first_strong_level(["LRI", "LRI", "PDI", "R"], 0, 4) == 0
    assert first_strong_level(["PDI", "R"], 0, 2) == 1


def test_isolate_overflow_at_exactly_max_depth() -> None:
    levels, _wtypes = _explicit(["RLI"] * 64 + ["L"])
    assert levels[-2:] == [125, 125]


def test_valid_isolate_count_gates_pdi_pops() -> None:
    two = ["RLI", "RLI", "L", "PDI", "PDI", "R"]
    assert _explicit(two)[0] == [0, 1, 3, 1, 0, 0]
    three = ["RLI", "RLI", "L", "PDI", "PDI", "PDI", "R"]
    assert _explicit(three)[0] == [0, 1, 3, 1, 0, 0, 0]


def test_overflow_counters_gate_later_pushes() -> None:
    deep = ["RLE"] * 64
    levels, _wtypes = _explicit(deep + ["FSI", "FSI", "PDI", "RLE", "PDI", "PDF", "PDF", "N"])
    assert levels[-1] == 123
    levels, _wtypes = _explicit(deep + ["FSI", "PDI", "PDF", "RLE", "PDF", "PDF", "N"])
    assert levels[-1] == 123


def test_valid_pop_resets_embedding_overflow() -> None:
    levels, _wtypes = _explicit(["RLI"] + ["RLE"] * 63 + ["PDI", "M"] + ["RLE"] * 63 + ["N"])
    assert levels[64:66] == [0, 0]
    assert levels[-1] == 125


def test_pdf_unwind_arithmetic() -> None:
    levels, _wtypes = _explicit(["RLE"] * 64 + ["PDF"] * 3 + ["M"])
    assert levels[-2:] == [None, 121]


def test_paragraph_separator_takes_paragraph_level() -> None:
    levels, _wtypes = _explicit(["RLE", "B", "R"])
    assert levels == [None, 0, 1]


@pytest.mark.parametrize(
    "typ",
    ["L", "R", "EN", "AN", "ON", "NSM", "B", "WS", "LRI", "PDI", "RLE", "PDF"],
)
def test_strong_inside_table(typ: str) -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _strong_inside

    assert _strong_inside(typ) == (
        "L" if typ == "L" else ("R" if typ in ("R", "EN", "AN") else None)
    )


@pytest.mark.parametrize(
    ("wtypes", "sos", "embedding", "expected"),
    [
        # N0b: embedding-direction strong inside decides.
        (["ON", "R", "ON"], "L", "R", "R"),
        # N0d: no strong inside resolves nothing.
        (["ON", "ON", "ON"], "R", "R", None),
        # N0c: opposite strong inside resolves from the preceding context.
        (["ON", "L", "ON"], "L", "R", "L"),
        (["ON", "L", "ON"], "R", "R", "R"),
        (["ON", "R", "ON"], "L", "L", "L"),
    ],
)
def test_pair_type_table(wtypes: list[str], sos: str, embedding: str, expected: str | None) -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _pair_type

    assert _pair_type([0, 1, 2], sos, wtypes, 0, 2, embedding) == expected


def test_pair_type_reads_preceding_context() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _pair_type

    seq = [0, 1, 2, 3, 4]
    assert _pair_type(seq, "R", ["R", "L", "ON", "L", "ON"], 2, 4, "R") == "L"


def test_pair_type_scans_only_inside_the_pair() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _pair_type

    seq = [0, 1, 2, 3, 4]
    assert _pair_type(seq, "R", ["L", "ON", "ON", "ON", "R"], 1, 3, "R") is None


def test_pair_type_falls_back_to_sos_without_context() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _pair_type

    seq = [0, 1, 2, 3, 4, 5, 6]
    assert _pair_type(seq, "L", ["ON", "ON", "ON", "L", "ON", "ON", "R"], 2, 4, "R") == "L"


def test_pair_type_context_skips_two_at_a_time_is_wrong() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _pair_type

    seq = [0, 1, 2, 3, 4, 5, 6]
    assert _pair_type(seq, "R", ["R", "L", "ON", "ON", "L", "ON", "ON"], 3, 5, "R") == "L"


def test_bracket_pairs_table() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _find_bracket_pairs

    parens = ["ON", "ON", "ON"]
    assert _find_bracket_pairs([0, 1, 2], parens, [0x28, 0x28, 0x29]) == [(1, 2)]
    assert _find_bracket_pairs([0, 1, 2], parens, [0x28, 0x3008, 0x29]) == [(0, 2)]
    assert _find_bracket_pairs([0, 1, 2], parens, [0x3008, 0x5D0, 0x3009]) == [(0, 2)]
    nested = ["ON", "ON", "ON", "ON"]
    assert _find_bracket_pairs([0, 1, 2, 3], nested, [0x28, 0x28, 0x29, 0x29]) == [
        (0, 3),
        (1, 2),
    ]
    assert _find_bracket_pairs([0, 1, 2], ["L", "ON", "ON"], [65, 0x28, 0x29]) == [(1, 2)]
    assert _find_bracket_pairs([0, 1], ["ON", "L"], [0x29, 65]) == []


def test_bracket_resolution_retypes_pair_and_following_nsms() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import resolve_brackets

    seq = [0, 1, 2]
    wtypes = ["ON", "R", "ON"]
    resolve_brackets(seq, "R", ["ON", "R", "ON"], wtypes, [0x28, 0x5D0, 0x29], "R")
    assert wtypes == ["R", "R", "R"]


def test_bracket_resolution_stops_nsms_at_non_nsm() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import resolve_brackets

    seq = [0, 1, 2, 3, 4]
    wtypes = ["ON", "NSM", "L", "ON", "L"]
    resolve_brackets(seq, "L", list(wtypes), wtypes, [0x28, 0x301, 0x61, 0x29, 0x62], "L")
    assert wtypes == ["L", "L", "L", "L", "L"]


def test_bracket_resolution_leaves_later_nsms_past_a_break() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import resolve_brackets

    seq = [0, 1, 2, 3, 4, 5]
    wtypes = ["ON", "NSM", "L", "ON", "L", "NSM"]
    resolve_brackets(seq, "L", list(wtypes), wtypes, [0x28, 0x301, 0x61, 0x29, 0x62, 0x303], "L")
    assert wtypes == ["L", "L", "L", "L", "L", "NSM"]


def test_neutral_run_treats_an_as_r() -> None:
    from nomikos_inference.architectures.ppocr_rec.bidi._neutral import _fill_neutral_run

    wtypes = ["AN", "WS", "WS", "WS", "AN"]
    _fill_neutral_run([0, 1, 2, 3, 4], wtypes, 1, 4, "L", "L", "L")
    assert wtypes == ["AN", "R", "R", "R", "AN"]


def test_run_sequences_pin_sos_eos_and_membership() -> None:
    sequences = isolating_run_sequences(["L", "L", "L"], [2, 0, 1], [0, 1, 2], {}, set(), 0)
    assert sequences == [([0], "L", "L"), ([1], "L", "R"), ([2], "R", "R")]


def test_run_sequence_at_zero_ignores_last_keep() -> None:
    sequences = isolating_run_sequences(["L", "L", "L"], [0, 0, 1], [0, 1, 2], {}, set(), 0)
    assert sequences[0] == ([0, 1], "L", "R")


def test_run_sequence_starting_mid_keep_reads_previous() -> None:
    sequences = isolating_run_sequences(["L", "L", "L"], [1, 0, 0], [0, 1, 2], {}, set(), 0)
    assert sequences[1] == ([1, 2], "R", "L")


def test_run_sequence_at_zero_reads_paragraph_level() -> None:
    sequences = isolating_run_sequences(["L", "L", "L"], [1, 0, 1], [0, 1, 2], {}, set(), 0)
    assert sequences[0] == ([0], "R", "R")


def test_run_sequences_extend_through_matched_isolates() -> None:
    sequences = isolating_run_sequences(
        ["L", "RLI", "R", "PDI", "L"],
        [0, 0, 1, 0, 0],
        [0, 1, 2, 3, 4],
        {1: 3},
        {3},
        0,
    )
    assert sequences == [
        ([0, 1, 3, 4], "L", "L"),
        ([2], "R", "R"),
    ]


def test_run_sequences_skip_matched_pdi_runs() -> None:
    sequences = isolating_run_sequences(
        ["RLI", "PDI", "L", "PDI", "R"],
        [1, 0, 0, 1, 1],
        [0, 1, 2, 3, 4],
        {0: 1},
        {1},
        0,
    )
    assert sequences == [
        ([0, 1, 2], "R", "R"),
        ([3, 4], "R", "R"),
    ]


def test_run_sequence_ending_in_unmatched_isolate_uses_paragraph() -> None:
    sequences = isolating_run_sequences(
        ["L", "RLI", "PDI", "RLI", "R"],
        [0, 0, 1, 1, 2],
        [0, 1, 2, 3, 4],
        {1: 2, 3: None},
        {2},
        0,
    )
    assert sequences[0] == ([0, 1, 2, 3], "L", "R")


def test_l1_resets_after_paragraph_separator() -> None:
    levels, visual = resolve_levels(["RLE", "RLI", "PDI", "B", "PDF"], None, 0)
    assert levels == [None, 0, 0, 0, None]
    assert visual == [1, 2, 3]


def test_l1_trailing_whitespace_resets() -> None:
    levels, visual = resolve_levels(["RLE", "L", "WS", "PDF"], None, 0)
    assert levels == [None, 2, 0, None]
    assert visual == [1, 2]


def test_l1_segment_separator_resets() -> None:
    levels, visual = resolve_levels(["RLE", "L", "S", "PDF"], None, 0)
    assert levels == [None, 2, 0, None]
    assert visual == [1, 2]


def test_l1_trailing_isolates_reset() -> None:
    for initiator in ("LRI", "RLI", "FSI"):
        levels, visual = resolve_levels(["RLE", "L", initiator, "PDF"], None, 0)
        assert levels == [None, 2, 0, None]
        assert visual == [1, 2]


def test_explicit_base_direction_mixed() -> None:
    assert get_display_map("aא", "L") == ("aא", [0, 1])
    assert get_display_map("אב", "R") == ("בא", [1, 0])


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        # W2 feeds W5: an AN terminator-neighbor stays, an EN joins.
        (["AL", "EN", "ET"], ([1, 2, 1], [2, 1, 0])),
        (["AL", "L", "EN", "ET"], ([1, 2, 2, 2], [1, 2, 3, 0])),
        (["AL", "R", "EN", "ET"], ([1, 1, 2, 2], [2, 3, 1, 0])),
        (["L", "AL", "EN", "ET"], ([2, 1, 2, 1], [3, 2, 1, 0])),
    ],
)
def test_weak_feeds_terminator_resolution(
    types: list[str], expected: tuple[list[int | None], list[int]]
) -> None:
    assert resolve_levels(types, None, 1) == expected


def test_nsm_after_isolate_joins_neutral_runs() -> None:
    levels, visual = resolve_levels(["RLE", "LRI", "R", "NSM", "R", "PDF"], None, 0)
    assert levels == [None, 1, 3, 3, 3, None]
    assert visual == [4, 3, 2, 1]


def test_no_mirroring_at_even_levels() -> None:
    assert get_display_map("a\u202a(\u05d0)\u202c", None) == ("a(א)", [0, 2, 3, 4])


def test_invalid_base_direction_message_is_exact() -> None:
    from typing import Literal, cast

    bad = cast("Literal['L', 'R'] | None", "X")
    with pytest.raises(ValueError) as excinfo:
        get_display_map("abc", bad)
    assert str(excinfo.value) == "Invalid base direction 'X'"

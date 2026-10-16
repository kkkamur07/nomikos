"""Isolating run sequences, implicit levels and visual reorder.

Split out of ``bidi/__init__.py``: rule X10/BD13 sequence building, rules
I1/I2/L1/L2, and the ``get_display`` entry points. Algorithm unchanged from
kraken 7.1.1 (literal port of UAX #9).
"""

from __future__ import annotations

from typing import Literal

from nomikos_inference.architectures.ppocr_rec.bidi._common import (
    ISOLATE_INITIATORS,
    ISOLATES,
    bidi_class,
    direction,
    level_at,
)
from nomikos_inference.architectures.ppocr_rec.bidi._data import MIRRORED
from nomikos_inference.architectures.ppocr_rec.bidi._explicit import (
    explicit_levels,
    first_strong_level,
    match_isolates,
)
from nomikos_inference.architectures.ppocr_rec.bidi._neutral import (
    resolve_brackets,
    resolve_neutrals,
)
from nomikos_inference.architectures.ppocr_rec.bidi._weak import resolve_weak

__all__ = ["resolve_levels", "get_display", "get_display_map"]


def _compact_runs(
    levels: list[int | None], keep: list[int]
) -> tuple[list[list[int]], dict[int, int]]:
    """Group X9-compacted characters into runs of equal level."""
    runs: list[list[int]] = []
    run_of: dict[int, int] = {}
    for pos in keep:
        if runs and levels[pos] == levels[runs[-1][-1]]:
            runs[-1].append(pos)
        else:
            runs.append([pos])
        run_of[pos] = len(runs) - 1
    return runs, run_of


def _extend_with_initiator(
    seq: list[int],
    runs: list[list[int]],
    run_of: dict[int, int],
    types: list[str],
    matching_pdi: dict[int, int | None],
) -> None:
    """Append the run holding the PDI that matches a trailing initiator."""
    while types[seq[-1]] in ISOLATE_INITIATORS and matching_pdi[seq[-1]] is not None:
        pdi = matching_pdi[seq[-1]]
        if pdi is None:
            break
        seq.extend(runs[run_of[pdi]])


def _sequence_sos(
    first: int,
    keep_index: dict[int, int],
    levels: list[int | None],
    keep: list[int],
    para_level: int,
) -> str:
    k = keep_index[first]
    prev_level = level_at(levels, keep[k - 1]) if k > 0 else para_level
    return direction(max(level_at(levels, first), prev_level))


def _sequence_eos(
    last: int,
    types: list[str],
    matching_pdi: dict[int, int | None],
    keep_index: dict[int, int],
    levels: list[int | None],
    keep: list[int],
    para_level: int,
) -> str:
    if types[last] in ISOLATE_INITIATORS and matching_pdi[last] is None:
        next_level = para_level
    else:
        k = keep_index[last]
        next_level = level_at(levels, keep[k + 1]) if k + 1 < len(keep) else para_level
    return direction(max(level_at(levels, last), next_level))


def isolating_run_sequences(
    types: list[str],
    levels: list[int | None],
    keep: list[int],
    matching_pdi: dict[int, int | None],
    matched_pdis: set[int],
    para_level: int,
) -> list[tuple[list[int], str, str]]:
    """Rules X10/BD13: isolating run sequences with their sos/eos types.

    Returns (character indices, sos, eos) triples.
    """
    if not keep:
        return []
    runs, run_of = _compact_runs(levels, keep)
    keep_index = {pos: k for k, pos in enumerate(keep)}
    sequences: list[tuple[list[int], str, str]] = []
    for run in runs:
        if types[run[0]] == "PDI" and run[0] in matched_pdis:
            # Continuation of the sequence containing the matching initiator.
            continue
        seq = list(run)
        _extend_with_initiator(seq, runs, run_of, types, matching_pdi)
        first, last = seq[0], seq[-1]
        sos = _sequence_sos(first, keep_index, levels, keep, para_level)
        eos = _sequence_eos(last, types, matching_pdi, keep_index, levels, keep, para_level)
        sequences.append((seq, sos, eos))
    return sequences


def _apply_implicit_levels(levels: list[int | None], keep: list[int], wtypes: list[str]) -> None:
    """Rules I1/I2: bump levels of rightward and numeric characters."""
    for pos in keep:
        level = level_at(levels, pos)
        if level % 2:
            if wtypes[pos] != "R":
                levels[pos] = level + 1
        elif wtypes[pos] == "R":
            levels[pos] = level + 1
        elif wtypes[pos] in ("EN", "AN"):
            levels[pos] = level + 2


def _apply_line_break_levels(
    levels: list[int | None], keep: list[int], types: list[str], para_level: int
) -> None:
    """Rule L1: reset levels of trailing whitespace and isolates."""
    reset = True
    for pos in reversed(keep):
        if types[pos] in ("B", "S"):
            levels[pos] = para_level
            reset = True
        elif types[pos] in ("WS", "LRI", "RLI", "FSI", "PDI"):
            if reset:
                levels[pos] = para_level
        else:
            reset = False


def _reverse_level_run(visual: list[int], levels: list[int | None], level: int) -> None:
    """L2: reverse maximal runs at or above ``level``, in place."""
    k = 0
    while k < len(visual):
        if level_at(levels, visual[k]) >= level:
            end = k
            while end < len(visual) and level_at(levels, visual[end]) >= level:
                end += 1
            visual[k:end] = visual[k:end][::-1]
            k = end
        else:
            k += 1


def _visual_order(levels: list[int | None], keep: list[int]) -> list[int]:
    """Rule L2: reorder kept indices from logical to visual order."""
    visual = list(keep)
    if not visual:
        return visual
    highest = max(level_at(levels, pos) for pos in visual)
    odd_levels = [level_at(levels, pos) for pos in visual if level_at(levels, pos) % 2]
    if not odd_levels:
        return visual
    for level in range(highest, min(odd_levels) - 1, -1):
        _reverse_level_run(visual, levels, level)
    return visual


def resolve_levels(
    types: list[str], cps: list[int] | None, para_level: int
) -> tuple[list[int | None], list[int]]:
    """Run rules X1-L2 over a paragraph of bidi character types.

    ``cps`` holds the corresponding code points (needed for bracket pair
    resolution in rule N0). Returns the resolved embedding levels (``None``
    for characters removed by rule X9) and the visual order of the
    surviving character indices from left to right.
    """
    wtypes = list(types)
    matching_pdi, matched_pdis = match_isolates(types)
    levels = explicit_levels(types, wtypes, matching_pdi, para_level)
    keep = [idx for idx, level in enumerate(levels) if level is not None]

    for seq, sos, eos in isolating_run_sequences(
        types, levels, keep, matching_pdi, matched_pdis, para_level
    ):
        embedding_dir = direction(level_at(levels, seq[0]))
        resolve_weak(seq, sos, eos, wtypes)
        if cps is not None:
            resolve_brackets(seq, sos, types, wtypes, cps, embedding_dir)
        resolve_neutrals(seq, sos, eos, wtypes, embedding_dir)

    _apply_implicit_levels(levels, keep, wtypes)
    _apply_line_break_levels(levels, keep, types, para_level)
    return levels, _visual_order(levels, keep)


def get_display_map(text: str, base_dir: Literal["L", "R"] | None = None) -> tuple[str, list[int]]:
    """Reorder a string from logical order into display order (UAX #9).

    The text is treated as a single paragraph. Returns the reordered string
    and the index of each of its characters in ``text``. Mirrored characters
    (e.g. parentheses in right-to-left runs) are replaced by their mirror
    image; explicit directional formatting characters (LRE, RLE, LRO, RLO,
    PDF, LRI, RLI, FSI, PDI) and BN-class characters are removed.
    """
    if base_dir not in (None, "L", "R"):
        raise ValueError(f"Invalid base direction {base_dir!r}")
    cps = [ord(ch) for ch in text]
    types = [bidi_class(cp) for cp in cps]
    if base_dir is None:
        para_level = first_strong_level(types, 0, len(types))
    else:
        para_level = 0 if base_dir == "L" else 1
    levels, visual = resolve_levels(types, cps, para_level)
    display = []
    order = []
    for k in visual:
        if types[k] in ISOLATES:
            continue
        cp = cps[k]
        # L4
        if level_at(levels, k) % 2 and cp in MIRRORED:
            cp = MIRRORED[cp]
        display.append(chr(cp))
        order.append(k)
    return "".join(display), order


def get_display(text: str, base_dir: Literal["L", "R"] | None = None) -> str:
    """Reorder a string from logical order into display order (UAX #9).

    Mirrored characters are replaced by their mirror image; directional
    formatting characters and BN-class characters are removed.
    """
    return get_display_map(text, base_dir)[0]

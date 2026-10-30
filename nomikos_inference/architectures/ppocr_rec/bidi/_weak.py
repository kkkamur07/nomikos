"""Rules W1-W7: resolve weak types over one isolating run sequence.

Split out of ``bidi/__init__.py``; one helper per rule so each function
stays under the complexity gates. Algorithm unchanged from kraken 7.1.1
(literal port of UAX #9).
"""

from __future__ import annotations

from nomikos_inference.architectures.ppocr_rec.bidi._common import ISOLATES

__all__ = ["resolve_weak"]


def _w1_nsm(seq: list[int], sos: str, wtypes: list[str]) -> None:
    """W1: an NSM takes the type of the preceding character."""
    prev = sos
    for pos in seq:
        if wtypes[pos] == "NSM":
            wtypes[pos] = "ON" if prev in ISOLATES else prev
        prev = wtypes[pos]


def _w2_en_after_al(seq: list[int], sos: str, wtypes: list[str]) -> None:
    """W2: a European number after Arabic letter becomes Arabic number."""
    strong = sos
    for pos in seq:
        if wtypes[pos] == "EN" and strong == "AL":
            wtypes[pos] = "AN"
        if wtypes[pos] in ("L", "R", "AL"):
            strong = wtypes[pos]


def _w3_al_to_r(seq: list[int], wtypes: list[str]) -> None:
    """W3: Arabic letters become right-to-left."""
    for pos in seq:
        if wtypes[pos] == "AL":
            wtypes[pos] = "R"


def _w4_single_separators(seq: list[int], wtypes: list[str]) -> None:
    """W4: a single separator between two numbers of the same type joins them."""
    for k in range(1, len(seq) - 1):
        typ = wtypes[seq[k]]
        prev_typ, next_typ = wtypes[seq[k - 1]], wtypes[seq[k + 1]]
        if typ == "ES" and prev_typ == next_typ == "EN":
            wtypes[seq[k]] = "EN"
        elif typ == "CS" and prev_typ == next_typ and prev_typ in ("EN", "AN"):
            wtypes[seq[k]] = prev_typ


def _w5_terminator_run(seq: list[int], wtypes: list[str], start: int, sos: str, eos: str) -> int:
    """W5: one run of European terminators; returns the index past the run."""
    end = start
    while end < len(seq) and wtypes[seq[end]] == "ET":
        end += 1
    before = wtypes[seq[start - 1]] if start > 0 else sos
    after = wtypes[seq[end]] if end < len(seq) else eos
    if before == "EN" or after == "EN":
        for m in range(start, end):
            wtypes[seq[m]] = "EN"
    return end


def _w5_terminators(seq: list[int], sos: str, eos: str, wtypes: list[str]) -> None:
    """W5: a terminator sequence adjacent to a European number joins it."""
    k = 0
    while k < len(seq):
        if wtypes[seq[k]] == "ET":
            k = _w5_terminator_run(seq, wtypes, k, sos, eos)
        else:
            k += 1


def _w6_to_on(seq: list[int], wtypes: list[str]) -> None:
    """W6: remaining separators and terminators become other neutrals."""
    for pos in seq:
        if wtypes[pos] in ("ET", "ES", "CS"):
            wtypes[pos] = "ON"


def _w7_en_after_l(seq: list[int], sos: str, wtypes: list[str]) -> None:
    """W7: a European number after strong left-to-right becomes L."""
    strong = sos
    for pos in seq:
        if wtypes[pos] == "EN" and strong == "L":
            wtypes[pos] = "L"
        if wtypes[pos] in ("L", "R"):
            strong = wtypes[pos]


def resolve_weak(seq: list[int], sos: str, eos: str, wtypes: list[str]) -> None:
    """Rules W1-W7 over one isolating run sequence, in rule order."""
    _w1_nsm(seq, sos, wtypes)
    _w2_en_after_al(seq, sos, wtypes)
    _w3_al_to_r(seq, wtypes)
    _w4_single_separators(seq, wtypes)
    _w5_terminators(seq, sos, eos, wtypes)
    _w6_to_on(seq, wtypes)
    _w7_en_after_l(seq, sos, wtypes)

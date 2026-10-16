"""Rules N0-N2: paired brackets and neutral resolution.

Split out of ``bidi/__init__.py`` so each function stays under the
complexity gates. Algorithm unchanged from kraken 7.1.1 (literal port of
UAX #9, rules N0 with BD14-BD16 and N1/N2).
"""

from __future__ import annotations

from nomikos_inference.architectures.ppocr_rec.bidi._common import (
    CANON_BRACKETS,
    MAX_PAIRING_DEPTH,
    NI,
)
from nomikos_inference.architectures.ppocr_rec.bidi._data import BRACKETS

__all__ = ["resolve_brackets", "resolve_neutrals"]


def _strong_inside(typ: str) -> str | None:
    """Within N0, EN and AN count as R."""
    if typ == "L":
        return "L"
    elif typ in ("R", "EN", "AN"):
        return "R"
    return None


def _match_closer(
    stack: list[tuple[int, int]], pairs: list[tuple[int, int]], opener: int, closer_k: int
) -> None:
    for si in range(len(stack) - 1, -1, -1):
        if stack[si][0] == opener:
            pairs.append((stack[si][1], closer_k))
            del stack[si:]
            break


def _find_bracket_pairs(seq: list[int], wtypes: list[str], cps: list[int]) -> list[tuple[int, int]]:
    """BD16: pair open brackets with their closers, in sequence order."""
    stack: list[tuple[int, int]] = []
    pairs: list[tuple[int, int]] = []
    for k, pos in enumerate(seq):
        if wtypes[pos] != "ON":
            continue
        bracket = BRACKETS.get(cps[pos])
        if bracket is None:
            continue
        if bracket[1] == "o":
            if len(stack) == MAX_PAIRING_DEPTH:
                break
            stack.append((CANON_BRACKETS.get(cps[pos], cps[pos]), k))
        else:
            _match_closer(stack, pairs, CANON_BRACKETS.get(bracket[0], bracket[0]), k)
    pairs.sort()
    return pairs


def _pair_type(
    seq: list[int],
    sos: str,
    wtypes: list[str],
    open_k: int,
    close_k: int,
    embedding_dir: str,
) -> str | None:
    """N0 b-d: the resolved type of one bracket pair, if any."""
    found_opposite = False
    for k in range(open_k + 1, close_k):
        strong = _strong_inside(wtypes[seq[k]])
        if strong == embedding_dir:
            return embedding_dir
        elif strong is not None:
            found_opposite = True
    if not found_opposite:
        return None
    opposite = "L" if embedding_dir == "R" else "R"
    context = sos
    for k in range(open_k - 1, -1, -1):
        strong = _strong_inside(wtypes[seq[k]])
        if strong is not None:
            context = strong
            break
    return opposite if context == opposite else embedding_dir


def _retarget_pair_nsms(
    seq: list[int], types: list[str], wtypes: list[str], pair: tuple[int, int], new_type: str
) -> None:
    """NSMs following a re-typed bracket take its type."""
    open_k, close_k = pair
    wtypes[seq[open_k]] = new_type
    wtypes[seq[close_k]] = new_type
    for bracket_k in (open_k, close_k):
        for k in range(bracket_k + 1, len(seq)):
            if types[seq[k]] != "NSM":
                break
            wtypes[seq[k]] = new_type


def resolve_brackets(
    seq: list[int],
    sos: str,
    types: list[str],
    wtypes: list[str],
    cps: list[int],
    embedding_dir: str,
) -> None:
    """Rule N0 with BD14-BD16: resolve paired brackets in one sequence."""
    for pair in _find_bracket_pairs(seq, wtypes, cps):
        new_type = _pair_type(seq, sos, wtypes, pair[0], pair[1], embedding_dir)
        if new_type is not None:
            _retarget_pair_nsms(seq, types, wtypes, pair, new_type)


def _fill_neutral_run(
    seq: list[int],
    wtypes: list[str],
    start: int,
    end: int,
    sos: str,
    eos: str,
    embedding_dir: str,
) -> None:
    """N1/N2: resolve one run of neutral and isolate formatting characters."""
    before = wtypes[seq[start - 1]] if start > 0 else sos
    after = wtypes[seq[end]] if end < len(seq) else eos
    if before in ("EN", "AN"):
        before = "R"
    if after in ("EN", "AN"):
        after = "R"
    fill = before if before == after else embedding_dir
    for m in range(start, end):
        wtypes[seq[m]] = fill


def resolve_neutrals(
    seq: list[int], sos: str, eos: str, wtypes: list[str], embedding_dir: str
) -> None:
    """Rules N1/N2: resolve neutral and isolate formatting runs."""
    k = 0
    seq_len = len(seq)
    while k < seq_len:
        if wtypes[seq[k]] in NI:
            end = k
            while end < seq_len and wtypes[seq[end]] in NI:
                end += 1
            _fill_neutral_run(seq, wtypes, k, end, sos, eos, embedding_dir)
            k = end
        else:
            k += 1

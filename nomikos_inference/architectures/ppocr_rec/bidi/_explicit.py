"""Rules X1-X9: explicit embeddings, overrides, isolates.

Split out of ``bidi/__init__.py``; one handler per rule group so each
function stays under the complexity gates. Algorithm unchanged from
kraken 7.1.1 (literal port of UAX #9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nomikos_inference.architectures.ppocr_rec.bidi._common import (
    EXPLICIT_EMBEDDINGS,
    ISOLATE_INITIATORS,
    MAX_DEPTH,
)

__all__ = ["first_strong_level", "match_isolates", "explicit_levels"]


@dataclass
class _ExplicitState:
    """The directional status stack plus the X-rule overflow counters."""

    stack: list[tuple[int, str, bool]] = field(default_factory=list)
    overflow_isolates: int = 0
    overflow_embeddings: int = 0
    valid_isolates: int = 0


def first_strong_level(types: list[str], start: int, end: int) -> int:
    """Rules P2/P3: embedding level of the first strong character in
    ``types[start:end]``, skipping characters between an isolate initiator
    and its matching PDI. Defaults to 0.
    """
    depth = 0
    for typ in types[start:end]:
        if typ in ISOLATE_INITIATORS:
            depth += 1
        elif typ == "PDI":
            if depth > 0:
                depth -= 1
        elif depth == 0:
            if typ == "L":
                return 0
            elif typ in ("R", "AL"):
                return 1
    return 0


def match_isolates(types: list[str]) -> tuple[dict[int, int | None], set[int]]:
    """Rule BD9: match isolate initiators with their PDIs.

    Returns a map from initiator index to PDI index (``None`` if unmatched)
    and the set of matched PDI indices.
    """
    matching_pdi: dict[int, int | None] = {}
    matched_pdis: set[int] = set()
    stack: list[int] = []
    for idx, typ in enumerate(types):
        if typ in ISOLATE_INITIATORS:
            stack.append(idx)
        elif typ == "PDI" and stack:
            initiator = stack.pop()
            matching_pdi[initiator] = idx
            matched_pdis.add(idx)
    for initiator in stack:
        matching_pdi[initiator] = None
    return matching_pdi, matched_pdis


def _apply_embedding(state: _ExplicitState, levels: list[int | None], idx: int, typ: str) -> None:
    """Rules X2-X5: push an explicit embedding or override."""
    levels[idx] = None
    top = state.stack[-1][0]
    new_level = (top + 1) | 1 if typ[0] == "R" else (top + 2) & ~1
    if new_level <= MAX_DEPTH and not state.overflow_isolates and not state.overflow_embeddings:
        state.stack.append((new_level, {"RLO": "R", "LRO": "L"}.get(typ, "N"), False))
    elif not state.overflow_isolates:
        state.overflow_embeddings += 1


def _apply_isolate_initiator(
    state: _ExplicitState,
    types: list[str],
    wtypes: list[str],
    levels: list[int | None],
    matching_pdi: dict[int, int | None],
    idx: int,
    typ: str,
) -> None:
    """Rules X5a-X5c: push an isolate embedding."""
    level, override, _ = state.stack[-1]
    levels[idx] = level
    if override != "N":
        wtypes[idx] = override
    if typ == "FSI":
        pdi = matching_pdi[idx]
        rtl = first_strong_level(types, idx + 1, pdi if pdi is not None else len(types)) == 1
    else:
        rtl = typ == "RLI"
    new_level = (level + 1) | 1 if rtl else (level + 2) & ~1
    if new_level <= MAX_DEPTH and not state.overflow_isolates and not state.overflow_embeddings:
        state.valid_isolates += 1
        state.stack.append((new_level, "N", True))
    else:
        state.overflow_isolates += 1


def _apply_pdi(
    state: _ExplicitState, wtypes: list[str], levels: list[int | None], idx: int
) -> None:
    """Rule X6a: pop back to the matching isolate initiator."""
    if state.overflow_isolates:
        state.overflow_isolates -= 1
    elif state.valid_isolates:
        state.overflow_embeddings = 0
        while not state.stack[-1][2]:
            state.stack.pop()
        state.stack.pop()
        state.valid_isolates -= 1
    level, override, _ = state.stack[-1]
    levels[idx] = level
    if override != "N":
        wtypes[idx] = override


def _apply_pdf(state: _ExplicitState, levels: list[int | None], idx: int) -> None:
    """Rule X7: close the most recent embedding."""
    levels[idx] = None
    if state.overflow_isolates:
        return
    if state.overflow_embeddings:
        state.overflow_embeddings -= 1
    elif not state.stack[-1][2] and len(state.stack) > 1:
        state.stack.pop()


def _apply_override(
    state: _ExplicitState, wtypes: list[str], levels: list[int | None], idx: int
) -> None:
    """Rule X6: inherit the current embedding level and override."""
    level, override, _ = state.stack[-1]
    levels[idx] = level
    if override != "N":
        wtypes[idx] = override


def explicit_levels(
    types: list[str], wtypes: list[str], matching_pdi: dict[int, int | None], para_level: int
) -> list[int | None]:
    """Rules X1-X9: resolve explicit embedding levels.

    Marks the characters removed by X9 (RLE, LRE, RLO, LRO, PDF, BN) with a
    level of ``None``. Directional overrides are applied to ``wtypes`` in
    place.
    """
    levels: list[int | None] = [para_level] * len(types)
    # (embedding level, directional override, directional isolate status)
    state = _ExplicitState(stack=[(para_level, "N", False)])

    for idx, typ in enumerate(types):
        if typ in EXPLICIT_EMBEDDINGS:
            _apply_embedding(state, levels, idx, typ)
        elif typ in ISOLATE_INITIATORS:
            _apply_isolate_initiator(state, types, wtypes, levels, matching_pdi, idx, typ)
        elif typ == "PDI":
            _apply_pdi(state, wtypes, levels, idx)
        elif typ == "PDF":
            _apply_pdf(state, levels, idx)
        elif typ == "B":
            levels[idx] = para_level
        elif typ == "BN":
            levels[idx] = None
        else:
            _apply_override(state, wtypes, levels, idx)
    return levels

"""Shared tables, constants and tiny helpers for the vendored UAX #9.

Split out of ``bidi/__init__.py`` so the rule modules below stay under the
complexity gates. Algorithm unchanged from kraken 7.1.1.
"""

from __future__ import annotations

from bisect import bisect_right
from functools import cache

from nomikos_inference.architectures.ppocr_rec.bidi._data import (
    BIDI_CLASS_RANGES,
    UCD_VERSION,
)

__all__ = ["UCD_VERSION", "bidi_class"]

# Maximum explicit embedding depth (BD2).
MAX_DEPTH = 125
# Open bracket stack limit (BD16).
MAX_PAIRING_DEPTH = 63

ISOLATE_INITIATORS = frozenset(("LRI", "RLI", "FSI"))
ISOLATES = frozenset(("LRI", "RLI", "FSI", "PDI"))
EXPLICIT_EMBEDDINGS = frozenset(("RLE", "LRE", "RLO", "LRO"))
# Neutral or isolate formatting characters (NI).
NI = frozenset(("B", "S", "WS", "ON", "LRI", "RLI", "FSI", "PDI"))

# Canonical equivalences for bracket matching (N0).
CANON_BRACKETS = {0x3008: 0x2329, 0x3009: 0x232A}

_CLASS_STARTS = tuple(r[0] for r in BIDI_CLASS_RANGES)


@cache
def bidi_class(codepoint: int) -> str:
    """Return the Bidi_Class of a code point."""
    idx = bisect_right(_CLASS_STARTS, codepoint) - 1
    if idx >= 0 and codepoint <= BIDI_CLASS_RANGES[idx][1]:
        return BIDI_CLASS_RANGES[idx][2]
    return "L"


def direction(level: int) -> str:
    """Embedding direction of a level: odd is right-to-left."""
    return "R" if level % 2 else "L"


def level_at(levels: list[int | None], pos: int) -> int:
    """The resolved level at ``pos``; ``pos`` is always a kept character.

    Levels are ``None`` only for characters removed by rule X9, and every
    caller here walks ``keep`` (or a sequence built from it), so ``None``
    is unreachable. Raise rather than assert so the invariant holds under
    ``python -O`` too.
    """
    level = levels[pos]
    if level is None:
        raise ValueError("isolating run sequence contains a removed character")
    return level

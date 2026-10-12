"""Which writing system a line of transcription is in, read off its characters.

A document carries no script or direction field, and the exports that need one (the
Word file picks a font and a paragraph direction, the PDF picks a font and a shaping
direction) cannot ask. The characters can answer: every script this platform
transcribes has its own Unicode block, so a count over the line's letters says which
one it is written in.

The answer is per line, not per document, because a page can mix them: a Syriac
manuscript with a Greek gloss in the margin is two scripts on one page, and the gloss
must not come out right to left.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum


class Script(StrEnum):
    latin = "latin"
    greek = "greek"
    coptic = "coptic"
    armenian = "armenian"
    syriac = "syriac"


#: The one right-to-left script here. Syriac is also a joining script, which is why the
#: PDF has to shape it and cannot draw it one codepoint at a time.
RTL_SCRIPTS = frozenset({Script.syriac})

_RANGES: tuple[tuple[int, int, Script], ...] = (
    (0x0700, 0x074F, Script.syriac),
    (0x0530, 0x058F, Script.armenian),
    (0x2C80, 0x2CFF, Script.coptic),
    # The Coptic letters Unicode left inside the Greek block (shei to dei). Checked
    # before the Greek range that contains them.
    (0x03E2, 0x03EF, Script.coptic),
    (0x0370, 0x03FF, Script.greek),
    (0x1F00, 0x1FFF, Script.greek),
)


def _char_script(char: str) -> Script | None:
    codepoint = ord(char)
    for low, high, script in _RANGES:
        if low <= codepoint <= high:
            return script
    return None


def detect_script(text: str) -> Script:
    """The script most of ``text``'s letters belong to; Latin when none is recognised.

    Coptic is written in Greek letters plus a handful of its own, and much Coptic
    transcription uses the Greek block for everything those few do not cover. So a line
    with any Coptic-only letter is Coptic whenever Greek and Coptic together outnumber
    the other scripts: one shei is enough to tell a Coptic line from a Greek one, and
    counting it against twenty Greek-block alphas would call every such line Greek.
    """
    counts = Counter(script for script in map(_char_script, text) if script is not None)
    if not counts:
        return Script.latin
    if counts[Script.coptic]:
        counts[Script.coptic] += counts.pop(Script.greek, 0)
    return counts.most_common(1)[0][0]


def is_rtl(script: Script) -> bool:
    return script in RTL_SCRIPTS

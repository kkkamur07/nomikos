"""Keep combining marks in the same shaping run as the letter they sit on (fpdf2 2.8.9).

fpdf2's ``FPDF._parse_chars`` starts a new text fragment whenever
``get_unicode_script`` changes from one character to the next, and every fragment is
shaped by HarfBuzz on its own. Combining marks (U+0300 to U+036F, U+FE20 to U+FE2F,
and the rest of Unicode's ``Inherited`` script) report ``UnicodeScript.INHERITED``, so
``"ⲙⲛ̇ⲛⲓ"`` is split into ``'ⲙⲛ'``, ``'̇'`` and ``'ⲛⲓ'``. The mark is then shaped with no
base letter: it loses its GPOS anchor and is drawn as a free-standing glyph after the
letter, and a Syriac seyame (U+0308) breaks the cursive join of the word it is in.

Unicode's own rule (UAX #24) is that an ``Inherited`` character takes the script of the
character before it, which for run splitting is the same as not splitting at it. Mapping
``INHERITED`` to ``COMMON`` gets exactly that, because fpdf2 already lets ``COMMON``
characters join the run around them.

fpdf2 calls ``get_unicode_script.cache_clear()`` on output, so the replacement must stay
an ``functools.lru_cache`` object. The patch is applied once at import and is idempotent.
``tests/nomikos/unit/test_transcription_pdf_marks.py`` carries a canary that fails once
upstream stops splitting, which is the signal to delete this module.
"""

from __future__ import annotations

import functools

import fpdf.fpdf as _fpdf_module
from fpdf.unicode_script import UnicodeScript

_PATCH_MARKER = "_nomikos_keeps_marks"

original_get_unicode_script = getattr(
    _fpdf_module.get_unicode_script, "__wrapped__", _fpdf_module.get_unicode_script
)


@functools.cache
def _script_keeping_marks(char: str) -> UnicodeScript:
    script = original_get_unicode_script(char)
    return UnicodeScript.COMMON if script == UnicodeScript.INHERITED else script


setattr(_script_keeping_marks, _PATCH_MARKER, True)


def apply() -> None:
    """Install the mark-keeping lookup in fpdf2, once."""
    if not getattr(_fpdf_module.get_unicode_script, _PATCH_MARKER, False):
        _fpdf_module.get_unicode_script = _script_keeping_marks


apply()

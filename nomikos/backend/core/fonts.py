"""Resolve Unicode-capable fonts for generated artifacts."""

from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_ASSETS_FONT = _ASSETS_DIR / "NotoSans-Regular.ttf"

#: Syriac ships in three Noto styles. The base NotoSansSyriac-Regular.ttf is v3.000
#: (Estrangela); NotoSansSyriacEastern-Regular.ttf and NotoSansSyriacWestern-Regular.ttf
#: are v3.001, unhinted TTFs from https://github.com/notofonts/syriac/releases
#: (NotoSansSyriacEastern-v3.001.zip, NotoSansSyriacWestern-v3.001.zip).
#: The bundled font for each script ``detect_script`` can name, keyed by its value.
#: Noto Sans itself covers Latin and Greek; the others are the Noto families for
#: their script, all under the OFL in ``assets/fonts/OFL.txt``.
SCRIPT_FONT_FILES = {
    "latin": "NotoSans-Regular.ttf",
    "greek": "NotoSans-Regular.ttf",
    "coptic": "NotoSansCoptic-Regular.ttf",
    "armenian": "NotoSansArmenian-Regular.ttf",
    "syriac": "NotoSansSyriacEastern-Regular.ttf",
}

#: The family name each of those fonts declares. A Word document names its fonts
#: rather than embedding them, so this is what the reader's machine is asked for.
SCRIPT_FONT_FAMILIES = {
    "latin": "Noto Sans",
    "greek": "Noto Sans",
    "coptic": "Noto Sans Coptic",
    "armenian": "Noto Sans Armenian",
    "syriac": "Noto Sans Syriac Eastern",
}

#: The bundled file for each Syriac style. The default is Eastern because the
#: platform's Syriac manuscripts are East Syriac, so a later per-document choice
#: has somewhere to plug in.
SYRIAC_STYLE_FONT_FILES = {
    "eastern": "NotoSansSyriacEastern-Regular.ttf",
    "western": "NotoSansSyriacWestern-Regular.ttf",
    "estrangela": "NotoSansSyriac-Regular.ttf",
}

_OS_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/opt/homebrew/share/fonts/DejaVuSans.ttf"),
)

_FONT_MISSING_MESSAGE = (
    "Transcription PDF could not be generated because no Unicode font is available. "
    "The app expects its bundled Noto Sans font; contact support if this keeps happening."
)


def resolve_unicode_font() -> Path:
    """Return a Greek-capable Unicode TTF, preferring the bundled font."""
    if _ASSETS_FONT.is_file():
        return _ASSETS_FONT
    for candidate in _OS_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise RuntimeError(_FONT_MISSING_MESSAGE)


def font_for_script(script: str) -> Path:
    """The bundled TTF for ``script``, or the general Unicode font when it is missing.

    Takes the script's string value rather than the enum so this module stays below the
    document layer that defines it.
    """
    bundled = _ASSETS_DIR / SCRIPT_FONT_FILES.get(script, SCRIPT_FONT_FILES["latin"])
    if bundled.is_file():
        return bundled
    return resolve_unicode_font()

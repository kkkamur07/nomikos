"""Resolve Unicode-capable fonts for generated artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_ASSETS_FONT = _ASSETS_DIR / "NotoSans-Regular.ttf"


@dataclass(frozen=True)
class FontChoice:
    """One bundled font a document may pick for a script.

    ``file`` is the TTF under ``assets/fonts``; ``family`` is the family name that file
    declares in its name table (a Word document names its fonts rather than embedding
    them, so this is what the reader's machine is asked for); ``label`` is what the
    editor shows in its font picker.
    """

    file: str
    family: str
    label: str


#: Every font a document can choose, keyed by the script value ``detect_script`` names
#: and then by choice id. Noto Sans covers Latin and Greek; the others are the Noto
#: families for their script, all under the OFL in ``assets/fonts/OFL.txt``.
#:
#: Syriac ships in three Noto styles. The base NotoSansSyriac-Regular.ttf is v3.000
#: (Estrangela); NotoSansSyriacEastern-Regular.ttf and NotoSansSyriacWestern-Regular.ttf
#: are v3.001, unhinted TTFs from https://github.com/notofonts/syriac/releases
#: (NotoSansSyriacEastern-v3.001.zip, NotoSansSyriacWestern-v3.001.zip).
#:
#: Adding a font is data only: an entry here, the @font-face and choice rule in
#: nomikos/frontend/src/styles/script-fonts.css, the entry in
#: nomikos/frontend/src/components/document/scriptFonts.ts, and the file copied to
#: nomikos/frontend/public/fonts/ as well as assets/fonts here. A test enforces that
#: the backend and frontend lists agree.
SCRIPT_FONTS: dict[str, dict[str, FontChoice]] = {
    "latin": {
        "noto-sans": FontChoice("NotoSans-Regular.ttf", "Noto Sans", "Noto Sans"),
    },
    "greek": {
        "noto-sans": FontChoice("NotoSans-Regular.ttf", "Noto Sans", "Noto Sans"),
    },
    "coptic": {
        "noto-sans-coptic": FontChoice(
            "NotoSansCoptic-Regular.ttf", "Noto Sans Coptic", "Noto Sans Coptic"
        ),
    },
    "armenian": {
        "noto-sans-armenian": FontChoice(
            "NotoSansArmenian-Regular.ttf", "Noto Sans Armenian", "Noto Sans Armenian"
        ),
    },
    "syriac": {
        "estrangela": FontChoice("NotoSansSyriac-Regular.ttf", "Noto Sans Syriac", "Estrangela"),
        "eastern": FontChoice(
            "NotoSansSyriacEastern-Regular.ttf", "Noto Sans Syriac Eastern", "East Syriac"
        ),
        "western": FontChoice(
            "NotoSansSyriacWestern-Regular.ttf", "Noto Sans Syriac Western", "West Syriac"
        ),
    },
}

#: The choice a document gets for each script until it picks another. Syriac defaults
#: to Estrangela, the classical style, rather than either regional one; a document of
#: East or West Syriac manuscripts picks its style per document.
DEFAULT_FONT_CHOICE: dict[str, str] = {
    "latin": "noto-sans",
    "greek": "noto-sans",
    "coptic": "noto-sans-coptic",
    "armenian": "noto-sans-armenian",
    "syriac": "estrangela",
}

_FALLBACK_SCRIPT = "latin"


def _choice_for(script: str, choice: str | None) -> FontChoice:
    """The registry entry for ``choice``, the script default, or Latin's default."""
    choices = SCRIPT_FONTS.get(script)
    if choices is None:
        script, choices = _FALLBACK_SCRIPT, SCRIPT_FONTS[_FALLBACK_SCRIPT]
    if choice is not None and choice in choices:
        return choices[choice]
    return choices[DEFAULT_FONT_CHOICE[script]]


def font_file_for(script: str, choice: str | None = None) -> Path:
    """The bundled TTF for ``script`` in ``choice``, or the general Unicode font.

    ``None`` or an unknown choice means the script's default; an unknown script means
    Latin. Takes the script's string value rather than the enum so this module stays
    below the document layer that defines it.
    """
    bundled = _ASSETS_DIR / _choice_for(script, choice).file
    if bundled.is_file():
        return bundled
    return resolve_unicode_font()


def font_family_for(script: str, choice: str | None = None) -> str:
    """The family name of the font ``font_file_for`` would pick, with the same fallbacks."""
    return _choice_for(script, choice).family


def validate_script_fonts(value: Mapping[str, str]) -> dict[str, str]:
    """Check a per-document ``{script: choice}`` map against the registry.

    Raises ``ValueError`` naming the offending key for an unknown script or a choice the
    script does not offer. Returns a new plain dict, so a caller can assign it to a JSONB
    column without sharing the input object.
    """
    result: dict[str, str] = {}
    for script, choice in value.items():
        choices = SCRIPT_FONTS.get(script)
        if choices is None:
            raise ValueError(f"unknown script {script!r}")
        if choice not in choices:
            raise ValueError(f"unknown font choice {choice!r} for script {script!r}")
        result[script] = choice
    return result


#: Back-compat views of the registry defaults, keyed by script value.
SCRIPT_FONT_FILES: dict[str, str] = {
    script: SCRIPT_FONTS[script][choice].file for script, choice in DEFAULT_FONT_CHOICE.items()
}
SCRIPT_FONT_FAMILIES: dict[str, str] = {
    script: SCRIPT_FONTS[script][choice].family for script, choice in DEFAULT_FONT_CHOICE.items()
}
#: The bundled file for each Syriac style.
SYRIAC_STYLE_FONT_FILES: dict[str, str] = {
    choice: font.file for choice, font in SCRIPT_FONTS["syriac"].items()
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
    """The default bundled TTF for ``script``; the same as ``font_file_for(script)``."""
    return font_file_for(script)

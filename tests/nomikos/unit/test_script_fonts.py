"""Every script sample is drawn from its own bundled font, with no fallback gaps."""

from __future__ import annotations

from fontTools.ttLib import TTFont

from backend.core.fonts import SCRIPT_FONT_FILES, font_for_script

_ASSETS = font_for_script("latin").parent

# Representative text per script: Greek polytonic with Greek Extended letters and
# combining marks; Coptic block letters plus the Coptic letters in the Greek block
# U+03E2 to U+03EF; Armenian upper and lower case with U+0589; Syriac consonants
# with Eastern vowel points U+0730 to U+074A, QUSHSHAYA/RUKKAKHA marks and digits.
SAMPLES = {
    "greek": "Ἐν ἀρχῇ ἦν ὁ λόγος ᾈ ᾖ ὦ έ̓ ᾲ",
    "coptic": "ⲡⲛⲟⲩⲧⲉ ⲡⲉ ϢϣϤϥϦϧϨϩϪϫϬϭϮϯ",
    "armenian": "Բարեւ աշխարհ ԱԲԳ աբգ։",
    "syriac": "ܫܠܡܐ ܥܠܝܟܘܢ ܘܥܠ ܟܠܗܘܢ ܐܚܝ̈ܢ ܫܠܳܡܳܐ ܒ݁ܩܼ ١٢٣ ܀",
}


def _cmap(script: str):
    return TTFont(_ASSETS / SCRIPT_FONT_FILES[script]).getBestCmap()


def test_every_sample_codepoint_is_in_its_script_font():
    for script, sample in SAMPLES.items():
        cmap = _cmap(script)
        missing = sorted(
            f"U+{ord(char):04X}" for char in set(sample) if char != " " and ord(char) not in cmap
        )
        assert not missing, f"{script} sample has no glyph for: {', '.join(missing)}"


def test_syriac_default_is_the_eastern_style():
    assert SCRIPT_FONT_FILES["syriac"] == "NotoSansSyriacEastern-Regular.ttf"
    family = TTFont(_ASSETS / SCRIPT_FONT_FILES["syriac"])["name"].getDebugName(1)
    assert family == "Noto Sans Syriac Eastern"

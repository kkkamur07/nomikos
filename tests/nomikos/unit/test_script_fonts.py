"""Every script sample is drawn from its own bundled font, with no fallback gaps."""

from __future__ import annotations

import pytest
from fontTools.ttLib import TTFont

from backend.core.fonts import (
    DEFAULT_FONT_CHOICE,
    SCRIPT_FONT_FAMILIES,
    SCRIPT_FONT_FILES,
    SCRIPT_FONTS,
    SYRIAC_STYLE_FONT_FILES,
    font_family_for,
    font_file_for,
    font_for_script,
    validate_script_fonts,
)

_ASSETS = font_for_script("latin").parent

# Representative text per script: Greek polytonic with Greek Extended letters and
# combining marks; Coptic block letters plus the Coptic letters in the Greek block
# U+03E2 to U+03EF; Armenian upper and lower case with U+0589; Syriac consonants
# with Eastern vowel points U+0730 to U+074A, QUSHSHAYA/RUKKAKHA marks and digits.
SAMPLES = {
    "greek": "Ἐν ἀρχῇ ἦν ὁ λόγος ᾈ ᾖ ὦ έ̓ ᾲ",
    "coptic": "ⲡⲛⲟⲩⲧⲉ ⲡⲉ ϢϣϤϥϦϧϨϩϪϫϬϭϮϯ",
    "armenian": "Բարեւ աշխարհ ԱԲԳ աբգ։",
    "syriac": "ܫܠܡܐ ܥܠܝܟܘܢ ܘܥܠ ܟܠܗܘܢ ܐܚܝ̈ܢ ܫܠܳܡܳܐ ܒ݁ܩܼ ١٢٣ ܀",
}

_ALL_CHOICES = [
    (script, choice_id) for script, choices in SCRIPT_FONTS.items() for choice_id in choices
]


def _family(file: str) -> str:
    return TTFont(_ASSETS / file)["name"].getDebugName(1)


@pytest.mark.parametrize(("script", "choice_id"), [(s, c) for s, c in _ALL_CHOICES if s in SAMPLES])
def test_every_sample_codepoint_is_in_every_font_for_its_script(script, choice_id):
    cmap = TTFont(_ASSETS / SCRIPT_FONTS[script][choice_id].file).getBestCmap()
    missing = sorted(
        f"U+{ord(char):04X}"
        for char in set(SAMPLES[script])
        if char != " " and ord(char) not in cmap
    )
    assert not missing, f"{script}/{choice_id} has no glyph for: {', '.join(missing)}"


@pytest.mark.parametrize(("script", "choice_id"), _ALL_CHOICES)
def test_every_choice_family_matches_its_font_name_table(script, choice_id):
    choice = SCRIPT_FONTS[script][choice_id]
    assert (_ASSETS / choice.file).is_file()
    assert _family(choice.file) == choice.family


def test_every_script_has_a_default_it_offers():
    assert set(DEFAULT_FONT_CHOICE) == set(SCRIPT_FONTS)
    for script, choice_id in DEFAULT_FONT_CHOICE.items():
        assert choice_id in SCRIPT_FONTS[script]


def test_syriac_default_is_estrangela():
    assert DEFAULT_FONT_CHOICE["syriac"] == "estrangela"
    assert SCRIPT_FONT_FILES["syriac"] == "NotoSansSyriac-Regular.ttf"
    assert SCRIPT_FONT_FAMILIES["syriac"] == "Noto Sans Syriac"
    assert font_file_for("syriac").name == "NotoSansSyriac-Regular.ttf"
    assert font_for_script("syriac") == font_file_for("syriac")
    assert _family(font_file_for("syriac").name) == "Noto Sans Syriac"
    assert font_family_for("syriac") == "Noto Sans Syriac"


def test_syriac_styles_view_lists_every_syriac_choice():
    assert SYRIAC_STYLE_FONT_FILES == {
        "estrangela": "NotoSansSyriac-Regular.ttf",
        "eastern": "NotoSansSyriacEastern-Regular.ttf",
        "western": "NotoSansSyriacWestern-Regular.ttf",
    }


def test_a_named_choice_picks_its_font():
    assert font_file_for("syriac", "eastern").name == "NotoSansSyriacEastern-Regular.ttf"
    assert font_family_for("syriac", "western") == "Noto Sans Syriac Western"


@pytest.mark.parametrize("choice", [None, "meltho", "noto-sans"])
def test_no_or_unknown_choice_falls_back_to_the_script_default(choice):
    assert font_file_for("syriac", choice).name == "NotoSansSyriac-Regular.ttf"
    assert font_family_for("syriac", choice) == "Noto Sans Syriac"


def test_unknown_script_falls_back_to_latin():
    assert font_file_for("klingon").name == "NotoSans-Regular.ttf"
    assert font_file_for("klingon", "estrangela").name == "NotoSans-Regular.ttf"
    assert font_family_for("klingon") == "Noto Sans"


def test_missing_bundled_file_falls_back_to_the_unicode_font(monkeypatch, tmp_path):
    import backend.core.fonts as fonts

    monkeypatch.setattr(fonts, "_ASSETS_DIR", tmp_path)
    monkeypatch.setattr(fonts, "resolve_unicode_font", lambda: tmp_path / "fallback.ttf")
    assert fonts.font_file_for("syriac", "eastern") == tmp_path / "fallback.ttf"


def test_validate_script_fonts_returns_a_plain_copy():
    source = {"syriac": "eastern", "greek": "noto-sans"}
    result = validate_script_fonts(source)
    assert result == source
    assert result is not source
    assert type(result) is dict
    assert validate_script_fonts({}) == {}


def test_validate_script_fonts_rejects_an_unknown_script():
    with pytest.raises(ValueError, match="klingon"):
        validate_script_fonts({"klingon": "x"})


def test_validate_script_fonts_rejects_an_unknown_choice():
    with pytest.raises(ValueError, match="meltho"):
        validate_script_fonts({"syriac": "meltho"})

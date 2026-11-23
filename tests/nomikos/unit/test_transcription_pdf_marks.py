"""Combining marks stay in the same shaping run as their letter in a transcription PDF."""

from __future__ import annotations

import functools

import fpdf.fpdf
import pytest

from backend.annotation.application.transcription_pdf_service import _TranscriptionPdf
from backend.core import fpdf_marks
from backend.document.application.script_detection import Script

# Letters with U+0307 (dot above) on the second nu.
_COPTIC_DOT = "ⲙⲛ̇ⲛⲓ"
# Letters with U+FE26 (combining conjoining macron), a supralinear stroke over a pair.
_COPTIC_MACRON = "ⲓ︦ⲥ︦"
# Mem with seyame (U+0308), nun, waw and nun with vowel points (U+0735).
_SYRIAC_SEYAME = "ܡ̈ܢܵܘܵܢ"


def _fragments(script: Script, direction: str, text: str) -> list:
    pdf = _TranscriptionPdf()
    pdf.add_page(format=(1000, 1000))
    pdf.set_font(pdf.family_for(script), size=48)
    pdf.set_text_shaping(use_shaping_engine=True, direction=direction)
    return list(pdf._parse_chars(text, False))


def test_a_coptic_letter_and_its_dot_above_are_one_fragment():
    assert len(_fragments(Script.coptic, "ltr", _COPTIC_DOT)) == 1


def test_coptic_letters_and_their_conjoining_macrons_are_one_fragment():
    assert len(_fragments(Script.coptic, "ltr", _COPTIC_MACRON)) == 1


def test_a_syriac_word_with_seyame_and_vowel_points_is_one_fragment():
    assert len(_fragments(Script.syriac, "rtl", _SYRIAC_SEYAME)) == 1


def test_canary_unpatched_fpdf2_still_splits_at_a_combining_mark(monkeypatch):
    """When this fails, fpdf2 fixed the split upstream and backend/core/fpdf_marks.py can go."""
    # fpdf calls cache_clear on its lookup, so the stand-in must be an lru_cache too.
    monkeypatch.setattr(
        fpdf.fpdf,
        "get_unicode_script",
        functools.lru_cache(maxsize=None)(fpdf_marks.original_get_unicode_script),
    )
    assert len(_fragments(Script.coptic, "ltr", _COPTIC_DOT)) > 1


def test_applying_the_patch_twice_keeps_the_one_cached_lookup():
    fpdf_marks.apply()
    fpdf_marks.apply()
    assert fpdf.fpdf.get_unicode_script is fpdf_marks._script_keeping_marks
    assert callable(fpdf.fpdf.get_unicode_script.cache_clear)


@pytest.fixture(autouse=True)
def _patch_is_installed():
    fpdf_marks.apply()

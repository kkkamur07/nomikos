"""Which script, and so which font and direction, a line's characters name."""

from __future__ import annotations

import pytest

from backend.document.application.script_detection import Script, detect_script, is_rtl


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("ܫܠܡܐ ܥܠܝܟܘܢ", Script.syriac),
        ("Բարեւ աշխարհ", Script.armenian),
        ("ⲡⲛⲟⲩⲧⲉ ⲡⲉ", Script.coptic),
        # Coptic written in Greek-block letters: one shei makes it Coptic.
        ("ϣαχε μπνουτε", Script.coptic),
        ("Ἐν ἀρχῇ ἦν ὁ λόγος", Script.greek),
        ("καὶ ὁ λόγος", Script.greek),
        ("In principio erat verbum", Script.latin),
        ("", Script.latin),
        ("12 - 34", Script.latin),
        # A Syriac line with a page number and a Greek siglum is still Syriac.
        ("ܫܠܡܐ ܥܠܝܟܘܢ 12 α", Script.syriac),
    ],
)
def test_detect_script(text: str, script: Script) -> None:
    assert detect_script(text) == script


def test_only_syriac_runs_right_to_left() -> None:
    assert [script for script in Script if is_rtl(script)] == [Script.syriac]

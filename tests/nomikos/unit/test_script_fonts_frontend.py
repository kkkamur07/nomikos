"""The frontend draws each script from the same font registry the exports use.

``backend/core/fonts.py`` is the source of truth. The browser mirrors it in two
places, the @font-face and choice rules in ``script-fonts.css`` and the choice list
in ``scriptFonts.ts``; these tests fail as soon as either drifts.
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.core.fonts import DEFAULT_FONT_CHOICE, SCRIPT_FONTS

_ROOT = Path(__file__).resolve().parents[3] / "nomikos"
_CSS = (_ROOT / "frontend/src/styles/script-fonts.css").read_text(encoding="utf-8")
_TS = (_ROOT / "frontend/src/components/document/scriptFonts.ts").read_text(encoding="utf-8")
_PUBLIC = _ROOT / "frontend/public/fonts"
_ASSETS = _ROOT / "backend/core/assets/fonts"


def _ts_registry() -> dict[str, tuple[str, list[str]]]:
    registry: dict[str, tuple[str, list[str]]] = {}
    for match in re.finditer(
        r"^  (\w+): \{\s*defaultChoice: \"([\w-]+)\",\s*choices: \[(.*?)\],\s*\},",
        _TS,
        re.MULTILINE | re.DOTALL,
    ):
        script, default, body = match.groups()
        registry[script] = (default, re.findall(r'id: "([\w-]+)"', body))
    return registry


def test_ts_registry_matches_the_backend_registry():
    expected = {
        script: (DEFAULT_FONT_CHOICE[script], list(choices))
        for script, choices in SCRIPT_FONTS.items()
    }
    assert _ts_registry() == expected


def test_css_has_a_face_for_every_registry_file_and_a_rule_for_every_choice():
    served = set(re.findall(r'url\("/fonts/([^"]+)"\)', _CSS))
    for script, choices in SCRIPT_FONTS.items():
        assert f"--script-font-{script}:" in _CSS
        assert f'.script-text[data-script="{script}"]' in _CSS
        for choice_id, choice in choices.items():
            assert choice.file in served, choice.file
            assert f'[data-font-{script}="{choice_id}"]' in _CSS, (script, choice_id)


def test_every_registry_file_is_served_byte_identical():
    for choices in SCRIPT_FONTS.values():
        for choice in choices.values():
            assert (_PUBLIC / choice.file).read_bytes() == (_ASSETS / choice.file).read_bytes()
    assert (_PUBLIC / "OFL.txt").read_bytes() == (_ASSETS / "OFL.txt").read_bytes()

"""The reading rule for exports: approved text first, else the newest model output."""

from __future__ import annotations

from datetime import UTC, datetime

from backend.document.application.line_text_selection import LineText, reading_text
from backend.document.infrastructure.orm_models import (
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)


def _line(*layers: tuple[TranscriptionKind, str, datetime | None]) -> Line:
    line = Line(baseline={"points": []}, points=[[0, 0], [10, 0], [10, 5], [0, 5]])
    for kind, text, created_at in layers:
        layer = Transcription(name=kind.value, kind=kind, created_at=created_at)
        line.transcriptions.append(LineTranscription(transcription=layer, text=text))
    return line


_GT = TranscriptionKind.ground_truth
_MODEL = TranscriptionKind.model
_OLD = datetime(2026, 1, 1, tzinfo=UTC)
_NEW = datetime(2026, 6, 1, tzinfo=UTC)


def test_ground_truth_wins_verbatim_over_model_output() -> None:
    line = _line((_MODEL, "guess", _NEW), (_GT, "  approved \n", None))
    assert reading_text(line) == LineText("  approved ", approved=True)


def test_without_ground_truth_the_newest_non_empty_model_output_is_unapproved() -> None:
    line = _line((_MODEL, "old", _OLD), (_MODEL, "new", _NEW), (_MODEL, "   ", None))
    assert reading_text(line) == LineText("new", approved=False)


def test_a_whitespace_only_ground_truth_falls_back_to_the_model() -> None:
    line = _line((_GT, "  ", None), (_MODEL, "guess", _OLD))
    assert reading_text(line) == LineText("guess", approved=False)


def test_a_line_with_no_text_of_either_kind_is_skipped() -> None:
    assert reading_text(_line()) is None
    assert reading_text(_line((_GT, " ", None), (_MODEL, "\n", None))) is None

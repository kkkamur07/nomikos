"""The member and public line DTO builders project transcriptions the same way."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

from backend.document.api.line_responses import line_response, public_line_response
from backend.document.infrastructure.orm_models import (
    Line,
    LineGeometryKind,
    LineSource,
    TranscriptionKind,
)


def _line() -> Line:
    transcription = SimpleNamespace(
        id=uuid4(),
        transcription_id=uuid4(),
        transcription=SimpleNamespace(kind=TranscriptionKind.ground_truth),
        text="καὶ",
        confidence=0.75,
    )
    fake = SimpleNamespace(
        id=uuid4(),
        part_id=uuid4(),
        block_id=None,
        order=3,
        baseline={"points": [[0, 10], [100, 10]]},
        mask={"points": [[0, 0], [100, 0], [100, 20], [0, 20]]},
        kind=LineGeometryKind.polygon,
        points=[[0, 0], [100, 0], [100, 20], [0, 20]],
        source=LineSource.manual,
        source_metadata={"engine": "kraken"},
        kraken_ceiling=None,
        manual_geometry=True,
        transcriptions=[transcription],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return cast(Line, fake)


def test_public_line_response_carries_geometry_and_text_only() -> None:
    line = _line()

    response = public_line_response(line)

    assert response.id == line.id
    assert response.part_id == line.part_id
    assert response.order == 3
    assert response.points == line.points
    [only] = response.line_transcriptions
    source = line.transcriptions[0]
    assert (only.id, only.transcription_id, only.text, only.confidence) == (
        source.id,
        source.transcription_id,
        "καὶ",
        0.75,
    )
    assert only.transcription_kind == TranscriptionKind.ground_truth
    assert "source_metadata" not in response.model_dump()


def test_member_and_public_responses_share_the_transcription_projection() -> None:
    line = _line()

    member = line_response(line)
    public = public_line_response(line)

    assert member.line_transcriptions == public.line_transcriptions
    assert member.source_metadata == {"engine": "kraken"}
    assert member.manual_geometry is True

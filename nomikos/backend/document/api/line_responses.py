"""Shared DTO builders for document line responses."""

from backend.document.api.schemas import LineResponse, LineTranscriptionResponse
from backend.document.infrastructure.orm_models import Line
from nomikos_inference.contracts.transcribe import CharacterConfidence


def line_transcription_response(row) -> LineTranscriptionResponse:
    """Build the API view of one stored line transcription.

    The stored floats are zipped with the code points of the stored text into
    the [{char, confidence}] shape the editor reads. A row whose scores are
    missing or no longer describe its text (a human rewrote it, or it predates
    the column) reports null, and the editor falls back to the line score.
    """
    scores = getattr(row, "character_confidences", None)
    chars = list(row.text)
    paired: list[CharacterConfidence] | None = None
    if scores is not None and len(scores) == len(chars):
        paired = [
            CharacterConfidence(char=char, confidence=score)
            for char, score in zip(chars, scores, strict=True)
        ]
    return LineTranscriptionResponse(
        id=row.id,
        transcription_id=row.transcription_id,
        transcription_kind=row.transcription.kind,
        text=row.text,
        confidence=row.confidence,
        character_confidences=paired,
    )


def line_response(line: Line) -> LineResponse:
    return LineResponse(
        id=line.id,
        part_id=line.part_id,
        block_id=line.block_id,
        order=line.order,
        baseline=line.baseline,
        mask=line.mask,
        kind=line.kind,
        points=line.points,
        source=line.source,
        source_metadata=line.source_metadata,
        kraken_ceiling=line.kraken_ceiling,
        manual_geometry=line.manual_geometry,
        line_transcriptions=[line_transcription_response(row) for row in line.transcriptions],
        created_at=line.created_at,
    )

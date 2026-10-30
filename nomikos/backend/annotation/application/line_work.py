"""Has a person worked on this line?

One definition, shared by everything that is about to destroy a row. It used to
live as two private static methods on :class:`SegmentHealthService`; the free
merge asks exactly the same question, and two copies of "what counts as somebody's
work" is how one of them quietly stops counting a pairing.
"""

from __future__ import annotations

from uuid import UUID

from backend.document.infrastructure.orm_models import Line, TranscriptionKind


def has_text(line: Line) -> bool:
    """Any layer at all, not just the approved one.

    A draft nobody has approved yet is still someone's work, and a model
    prediction is the thing a reviewer is about to correct. Restricting this
    to ground truth would make every un-approved line deletable.
    """
    return any(
        isinstance(entry.text, str) and entry.text.strip() for entry in (line.transcriptions or [])
    )


def is_paired(line: Line, paired_ids: set[UUID]) -> bool:
    """A human's decision about this line, in either of the two places it lands.

    A non-blank ground-truth transcription is the usual one. A row in
    ``page_transcription_lines`` pointing here is the other, and the first
    does not imply it: pairing writes the ground-truth row and un-pairing
    removes it, so the two agree most of the time, but text imported and
    paired before anyone approved it carries the pairing without the layer.
    Reading only the layers would call such a line untouched and offer it up
    for deletion, discarding the pairing with it.

    ``segment_merge_service._protected_line_ids`` takes the same union for
    the same reason, and re-segmentation trusts it to decide what to keep.
    """
    if line.id in paired_ids:
        return True
    return any(
        entry.transcription is not None
        and entry.transcription.kind == TranscriptionKind.ground_truth
        and isinstance(entry.text, str)
        and entry.text.strip()
        for entry in (line.transcriptions or [])
    )


__all__ = ["has_text", "is_paired"]

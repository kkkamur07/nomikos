"""A line text save can name the model layer it corrected, in the same write.

Saving an edited model reading used to copy the uncorrected model text into
Ground truth first and write the correction second. A failed second write left
the wrong text approved. The PATCH now takes ``seeded_from_transcription_id``
so the corrected text and its provenance land in one checked write.
"""

from __future__ import annotations

import uuid

import pytest

from backend.core.exceptions import ConflictError, ValidationError
from backend.document.infrastructure.orm_models import (
    LineTranscription,
    Transcription,
    TranscriptionKind,
)
from tests.nomikos.unit.test_ground_truth_line_text_conflict import (
    _patched_service,
    _session_holding,
    _stored,
)


def _with_layers(service, *layers: Transcription) -> None:
    by_id = {layer.id: layer for layer in layers}

    async def get_transcription_in_document(_session, document_id, transcription_id):
        layer = by_id.get(transcription_id)
        if layer is None or layer.document_id != document_id:
            return None
        return layer

    service._documents.get_transcription_in_document = get_transcription_in_document


def _model_layer(document_id: uuid.UUID) -> Transcription:
    return Transcription(
        id=uuid.uuid4(), document_id=document_id, name="model", kind=TranscriptionKind.model
    )


async def _patch(service, session, part, line, layer, **kwargs):
    return await service.patch_ground_truth_line_text(
        session, object(), uuid.uuid4(), part.document_id, layer.id, line.id, **kwargs
    )


@pytest.mark.asyncio
async def test_a_first_save_records_the_model_layer_it_corrected() -> None:
    service, part, line, layer = _patched_service()
    model = _model_layer(part.document_id)
    _with_layers(service, model)
    session = _session_holding(None)

    await _patch(
        service,
        session,
        part,
        line,
        layer,
        text="corrected",
        previous_text="",
        seeded_from_transcription_id=model.id,
    )

    (row,) = session.added
    assert isinstance(row, LineTranscription)
    assert row.text == "corrected"
    assert row.seeded_from_transcription_id == model.id
    assert session.commits == 1


@pytest.mark.asyncio
async def test_a_row_without_a_seed_gets_one() -> None:
    service, part, line, layer = _patched_service()
    model = _model_layer(part.document_id)
    _with_layers(service, model)
    stored = _stored(line, layer, "")
    session = _session_holding(stored)

    await _patch(
        service,
        session,
        part,
        line,
        layer,
        text="corrected",
        previous_text="",
        seeded_from_transcription_id=model.id,
    )

    assert stored.text == "corrected"
    assert stored.seeded_from_transcription_id == model.id


@pytest.mark.asyncio
async def test_an_existing_seed_is_kept() -> None:
    service, part, line, layer = _patched_service()
    first = _model_layer(part.document_id)
    second = _model_layer(part.document_id)
    _with_layers(service, first, second)
    stored = _stored(line, layer, "seeded")
    stored.seeded_from_transcription_id = first.id
    session = _session_holding(stored)

    await _patch(
        service,
        session,
        part,
        line,
        layer,
        text="corrected",
        previous_text="seeded",
        seeded_from_transcription_id=second.id,
    )

    assert stored.text == "corrected"
    assert stored.seeded_from_transcription_id == first.id


@pytest.mark.asyncio
async def test_a_layer_of_another_document_is_a_422_and_writes_nothing() -> None:
    service, part, line, layer = _patched_service()
    foreign = _model_layer(uuid.uuid4())
    _with_layers(service, foreign)
    session = _session_holding(None)

    with pytest.raises(ValidationError):
        await _patch(
            service,
            session,
            part,
            line,
            layer,
            text="corrected",
            seeded_from_transcription_id=foreign.id,
        )

    assert session.added == []
    assert session.commits == 0


@pytest.mark.asyncio
async def test_a_ground_truth_layer_as_seed_is_a_422() -> None:
    service, part, line, layer = _patched_service()
    _with_layers(service, layer)
    session = _session_holding(None)

    with pytest.raises(ValidationError):
        await _patch(
            service,
            session,
            part,
            line,
            layer,
            text="corrected",
            seeded_from_transcription_id=layer.id,
        )

    assert session.commits == 0


@pytest.mark.asyncio
async def test_a_seeded_save_over_text_written_elsewhere_is_still_a_409() -> None:
    service, part, line, layer = _patched_service()
    model = _model_layer(part.document_id)
    _with_layers(service, model)
    stored = _stored(line, layer, "OTHER_BROWSER")
    session = _session_holding(stored)

    with pytest.raises(ConflictError):
        await _patch(
            service,
            session,
            part,
            line,
            layer,
            text="corrected",
            previous_text="",
            seeded_from_transcription_id=model.id,
        )

    assert stored.text == "OTHER_BROWSER"
    assert stored.seeded_from_transcription_id is None
    assert session.commits == 0

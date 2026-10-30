"""A history snapshot carries each line's model origin, and a restore puts it back.

The origin columns are not client-writable, so the restore is the one path that
may write them, and it does so only from what the server itself saved.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from backend.annotation.application.history_service import AnnotationHistoryService
from backend.document.infrastructure.orm_models import LineGeometryKind, LineSource


class _DocumentService:
    def __init__(self) -> None:
        self.replaced: dict | None = None

    async def get_document(self, *_args, **_kwargs):
        return SimpleNamespace(id=_DOCUMENT_ID)

    async def replace_part_lines(self, *_args, **kwargs):
        self.replaced = kwargs
        return []


class _Documents:
    async def get_part(self, _session, part_id):
        return SimpleNamespace(id=part_id, document_id=_DOCUMENT_ID)


class _Session:
    def __init__(self, snapshot) -> None:
        self._snapshot = snapshot

    async def get(self, _model, _snapshot_id):
        return self._snapshot


_DOCUMENT_ID = uuid.uuid4()


def _line(run_id: uuid.UUID | None, external_id: str | None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        block_id=None,
        order=0,
        kind=LineGeometryKind.polygon,
        points=[[0, 0], [10, 0], [10, 5], [0, 5]],
        source=LineSource.kraken,
        source_metadata={},
        kraken_ceiling=None,
        transcriptions=[],
        segmentation_run_id=run_id,
        run_external_id=external_id,
    )


def test_a_snapshot_records_the_lines_origin() -> None:
    run_id = uuid.uuid4()
    state = AnnotationHistoryService(documents=_Documents())._line_state(_line(run_id, "l1"))

    assert state["segmentation_run_id"] == str(run_id)
    assert state["run_external_id"] == "l1"


@pytest.mark.asyncio
async def test_a_restore_hands_the_origin_back_on_the_trusted_path() -> None:
    run_id = uuid.uuid4()
    service = AnnotationHistoryService(documents=_Documents(), document_service=_DocumentService())
    with_origin = service._line_state(_line(run_id, "l1"))
    # A snapshot taken before runs were recorded has no origin keys at all.
    legacy = {
        key: value
        for key, value in service._line_state(_line(None, None)).items()
        if key not in {"segmentation_run_id", "run_external_id"}
    }
    part_id = uuid.uuid4()
    snapshot = SimpleNamespace(part_id=part_id, state={"lines": [with_origin, legacy]})

    await service.restore_snapshot(
        _Session(snapshot), object(), uuid.uuid4(), _DOCUMENT_ID, part_id, uuid.uuid4()
    )

    replaced = service._document_service.replaced
    assert replaced["restore_origin"] is True
    restored, old = replaced["lines"]
    assert restored["segmentation_run_id"] == run_id
    assert restored["run_external_id"] == "l1"
    assert old["segmentation_run_id"] is None

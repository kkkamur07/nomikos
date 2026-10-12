"""The projects list names each project's owner, not only an id."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.project.api import projects as projects_api


def _project(owner_id: uuid.UUID | None) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid.uuid4(),
        name="Grec 1360",
        slug="grec-1360",
        guidelines=None,
        owner_id=owner_id,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_list_items_carry_owner_username(monkeypatch: pytest.MonkeyPatch) -> None:
    owner_id = uuid.uuid4()
    owned = _project(owner_id)
    orphan = _project(None)
    monkeypatch.setattr(
        projects_api._document_repo,
        "count_documents_by_project_ids",
        AsyncMock(return_value={owned.id: 2}),
    )
    result = MagicMock()
    result.all.return_value = [(owner_id, "alice")]
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)

    items = await projects_api._projects_with_counts(db, [owned, orphan])

    assert items[0].owner_username == "alice"
    assert items[0].document_count == 2
    assert items[1].owner_username is None

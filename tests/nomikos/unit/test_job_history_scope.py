"""Unit tests for DELETE /jobs/history: whose finished jobs a member may delete.

The service and the project check are stubbed; the DELETE's SQL is checked in
test_job_lifecycle.py and the rows themselves in the integration lane.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from backend.jobs.api import jobs as jobs_routes


async def test_the_route_clears_the_signed_in_users_history(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    class _Service:
        async def clear_project_job_history(self, project_id, *, user_id):
            calls.append((project_id, user_id))
            return 1

    async def _get_project(_db, _user, _project_id):
        return None

    monkeypatch.setattr(jobs_routes._project_service, "get_project", _get_project)
    user = SimpleNamespace(id=uuid.uuid4())
    project_id = uuid.uuid4()

    response = await jobs_routes.clear_job_history(
        project_id=project_id, db=None, service=_Service(), current_user=user
    )

    assert response.deleted == 1
    assert calls == [(project_id, user.id)]

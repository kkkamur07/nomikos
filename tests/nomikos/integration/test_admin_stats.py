"""The admin dashboard's SQL on real Postgres, for the cases a mocked session cannot see."""

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from backend.core.application import admin_stats_service
from infrastructure.db import sync_engine

_NOW = datetime.now(UTC)


def _stats(function, **kwargs):
    async def run():
        engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=NullPool)
        try:
            async with AsyncSession(engine) as session:
                return await function(session, **kwargs)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def _seed_user(connection, email: str) -> uuid.UUID:
    user_id = uuid.uuid4()
    connection.execute(
        text(
            "INSERT INTO users (id, email, username, hashed_password) "
            "VALUES (:id, :email, :email, 'x')"
        ),
        {"id": user_id, "email": email},
    )
    return user_id


def _seed_page_with_snapshots(connection, owner_id: uuid.UUID, counts: list[int]) -> None:
    project_id, document_id, part_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    connection.execute(
        text("INSERT INTO projects (id, name, slug, owner_id) VALUES (:id, 'p', :slug, :owner)"),
        {"id": project_id, "slug": f"p-{project_id}", "owner": owner_id},
    )
    connection.execute(
        text(
            "INSERT INTO documents (id, project_id, name, workflow) "
            "VALUES (:id, :project, 'd', 'draft')"
        ),
        {"id": document_id, "project": project_id},
    )
    connection.execute(
        text(
            'INSERT INTO document_parts (id, document_id, "order", image_key) '
            "VALUES (:id, :document, 0, 'k')"
        ),
        {"id": part_id, "document": document_id},
    )
    for days_ago, count in zip(range(len(counts), 0, -1), counts, strict=True):
        connection.execute(
            text(
                "INSERT INTO annotation_history_snapshots "
                "(id, part_id, state, line_count, paired_line_count, created_at) "
                "VALUES (:id, :part, '{}'::jsonb, 120, :count, :at)"
            ),
            {
                "id": uuid.uuid4(),
                "part": part_id,
                "count": count,
                "at": _NOW - timedelta(days=days_ago),
            },
        )


def test_the_oldest_kept_snapshot_does_not_count_as_approvals() -> None:
    # 100 lines approved long ago, then five saves this week approving 2 more. History
    # kept only these five snapshots, so the first has no predecessor in the table.
    with sync_engine.begin() as connection:
        ada = _seed_user(connection, "ada@example.com")
        _seed_page_with_snapshots(connection, ada, [100, 101, 101, 102, 102])

    users = _stats(admin_stats_service.users_stats, days=30)
    activity = _stats(admin_stats_service.activity_stats, days=30)

    assert [(u["email"], u["lines_approved"]) for u in users["top_users"]] == [
        ("ada@example.com", 2)
    ]
    assert sum(day["approved"] for day in activity["annotations"]) == 2
    assert sum(day["corrections"] for day in activity["annotations"]) == 5


def test_a_session_refreshed_today_counts_as_active_this_week() -> None:
    with sync_engine.begin() as connection:
        ada = _seed_user(connection, "ada@example.com")
        connection.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, token_hash, csrf_token_hash, expires_at, created_at, updated_at) "
                "VALUES (:id, :user, 't', 'c', :expires, :created, :updated)"
            ),
            {
                "id": uuid.uuid4(),
                "user": ada,
                "expires": _NOW + timedelta(days=4),
                "created": _NOW - timedelta(days=10),
                "updated": _NOW - timedelta(hours=1),
            },
        )

    users = _stats(admin_stats_service.users_stats, days=30)

    assert (users["active_7d"], users["active_30d"], users["active_in_window"]) == (1, 1, 1)


def test_overview_answers_for_an_admin_and_reuses_its_postgres_cards(
    client, registered_user, auth_headers
) -> None:
    from backend.core.api import admin

    admin.clear_overview_cache()
    with sync_engine.begin() as connection:
        connection.execute(
            text("UPDATE users SET is_admin = true WHERE email = :email"),
            {"email": registered_user["email"]},
        )

    first = client.get("/admin/overview?days=7", headers=auth_headers)
    with sync_engine.begin() as connection:
        _seed_user(connection, "late@example.com")
    second = client.get("/admin/overview?days=7", headers=auth_headers)
    admin.clear_overview_cache()

    assert first.status_code == 200, first.text
    assert first.json()["users"]["total"] == 1
    # Within the cache minute the new account is not seen yet.
    assert second.json()["users"]["total"] == 1

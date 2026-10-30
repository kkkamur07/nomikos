"""The migration chain must reproduce ``Base.metadata`` exactly.

This is the ``alembic check`` the project was missing. 001_initial_schema used to
build the baseline with ``Base.metadata.create_all``, which made a missing
migration structurally undetectable: a fresh database migrated to head always
matched the ORM because head *was* the ORM. The revisions that had to be written
afterwards to add ``jobs.claimed_by`` and ``document_parts.width`` are what that
cost us - schema changes that reached production before anyone noticed they had
no migration. They are folded back into the baseline now, but only because this
test exists to catch the next one.

So this test migrates a *scratch* database from empty to head and asserts that
alembic's autogenerate comparison against ``Base.metadata`` finds nothing. Add a
column to an ORM model without a migration and this goes red.

The scratch database is created and dropped per run rather than reusing the test
database: the point is to exercise the chain from nothing, and running it against
the shared database would collide with every other integration test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text

import infrastructure.models  # noqa: F401 - register all ORM tables
from backend.core.settings import get_infrastructure_settings
from infrastructure.db import Base

pytestmark = pytest.mark.integration

_SCRATCH_DATABASE = "nomikos_migration_check"

_ALEMBIC_INI = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../nomikos/infrastructure/alembic.ini",
)

# Tables the chain creates that are deliberately absent from ``Base.metadata``.
# ``alembic_version`` is alembic's own bookkeeping. ``inference_jobs`` used to be
# listed here because the inference service owned its own mapper; ADR 0003
# collapsed that queue into ``jobs`` and the squashed baseline never creates the
# table, so the exemption is gone.
_UNMANAGED_TABLES = frozenset({"alembic_version"})


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{database}", query=""))


def _maintenance_url(url: str) -> str:
    """A connection that is not to the database being created or dropped."""
    return _with_database(url, "postgres")


@contextmanager
def _migrator_url(url: str) -> Iterator[None]:
    """Point alembic's env.py at ``url``, restoring the previous value after."""
    # alembic's env.py resolves its URL from the cached settings object, so the
    # override has to go through the environment and the cache has to be cleared
    # on both sides of the run or the rest of the session keeps the scratch URL.
    previous = os.environ.get("MIGRATOR_DATABASE_URL")
    os.environ["MIGRATOR_DATABASE_URL"] = url
    get_infrastructure_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MIGRATOR_DATABASE_URL", None)
        else:
            os.environ["MIGRATOR_DATABASE_URL"] = previous
        get_infrastructure_settings.cache_clear()


def _run_ddl(url: str, statement: str) -> None:
    # CREATE/DROP DATABASE cannot run inside a transaction block.
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            connection.execute(text(statement))
    finally:
        engine.dispose()


@pytest.fixture
def migrated_scratch_database() -> Iterator[str]:
    """Create an empty database, migrate it to head, yield its URL, drop it."""
    base_url = get_infrastructure_settings().migrator_database_url
    maintenance_url = _maintenance_url(base_url)
    scratch_url = _with_database(base_url, _SCRATCH_DATABASE)

    _run_ddl(maintenance_url, f'DROP DATABASE IF EXISTS "{_SCRATCH_DATABASE}" WITH (FORCE)')
    _run_ddl(maintenance_url, f'CREATE DATABASE "{_SCRATCH_DATABASE}"')

    with _migrator_url(scratch_url):
        command.upgrade(Config(_ALEMBIC_INI), "head")
        yield scratch_url
    _run_ddl(maintenance_url, f'DROP DATABASE IF EXISTS "{_SCRATCH_DATABASE}" WITH (FORCE)')


def test_migration_chain_matches_orm_metadata(migrated_scratch_database: str) -> None:
    """An empty database migrated to head has no autogenerate diff vs the ORM.

    A non-empty diff means a model changed without a migration. Read the printed
    operations: ``add_column``/``remove_column``/``add_table`` name exactly what
    the missing revision has to do.
    """

    def include_name(name, type_, parent_names) -> bool:
        return type_ != "table" or name not in _UNMANAGED_TABLES

    engine = create_engine(migrated_scratch_database)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={
                    "include_name": include_name,
                    "compare_type": True,
                    "compare_server_default": True,
                },
            )
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    assert diff == [], (
        "The migration chain no longer reproduces Base.metadata. "
        "Add a migration for these changes instead of editing 001_initial_schema:\n"
        + "\n".join(repr(entry) for entry in diff)
    )


@pytest.fixture
def scratch_database_before_public_sharing() -> Iterator[str]:
    """An empty database migrated to the revision *before* 006, and left there."""
    base_url = get_infrastructure_settings().migrator_database_url
    maintenance_url = _maintenance_url(base_url)
    scratch_url = _with_database(base_url, _SCRATCH_DATABASE)

    _run_ddl(maintenance_url, f'DROP DATABASE IF EXISTS "{_SCRATCH_DATABASE}" WITH (FORCE)')
    _run_ddl(maintenance_url, f'CREATE DATABASE "{_SCRATCH_DATABASE}"')

    with _migrator_url(scratch_url):
        command.upgrade(Config(_ALEMBIC_INI), "005_case_insensitive_email")
        yield scratch_url
    _run_ddl(maintenance_url, f'DROP DATABASE IF EXISTS "{_SCRATCH_DATABASE}" WITH (FORCE)')


def test_006_gives_already_published_documents_a_share_token(
    scratch_database_before_public_sharing: str,
) -> None:
    """Documents that were public before 006 must come out of it with a token.

    Every ``/public/*`` route reads a null token as "this document is not shareable",
    so without the backfill an owner whose chapter was already live would open the
    sharing panel and be told that only the owner can get the link, while being the
    owner. The links themselves cannot be rescued either way - a URL sent last week
    carries no ``t`` at all - but the owner must have a working one to re-send without
    unpublishing and republishing first.
    """
    engine = create_engine(scratch_database_before_public_sharing)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users (id, email, username, hashed_password) VALUES "
                    "('11111111-1111-1111-1111-111111111111', 'a@example.com', 'a', 'x')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO projects (id, name, slug, owner_id) VALUES "
                    "('22222222-2222-2222-2222-222222222222', 'P', 'p', "
                    "'11111111-1111-1111-1111-111111111111')"
                )
            )
            for document_id, workflow in (
                ("33333333-3333-3333-3333-333333333333", "published"),
                ("44444444-4444-4444-4444-444444444444", "published"),
                ("55555555-5555-5555-5555-555555555555", "draft"),
            ):
                connection.execute(
                    text(
                        "INSERT INTO documents (id, project_id, name, workflow) VALUES "
                        "(:id, '22222222-2222-2222-2222-222222222222', 'D', :workflow)"
                    ),
                    {"id": document_id, "workflow": workflow},
                )
    finally:
        engine.dispose()

    with _migrator_url(scratch_database_before_public_sharing):
        command.upgrade(Config(_ALEMBIC_INI), "head")

    engine = create_engine(scratch_database_before_public_sharing)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT workflow::text, public_share_token FROM documents ORDER BY id")
            ).fetchall()
    finally:
        engine.dispose()

    published = [token for workflow, token in rows if workflow == "published"]
    drafts = [token for workflow, token in rows if workflow == "draft"]

    assert all(token for token in published), published
    # Distinct, not merely present: the column is uniquely indexed, and one token
    # shared between two documents would make one link open the other.
    assert len(set(published)) == len(published) == 2
    # A draft has nothing to link to yet, so it must come out with no token to leak.
    assert drafts == [None]

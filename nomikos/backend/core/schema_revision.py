"""Compare the database's alembic revision with the head the running code ships.

Production never runs alembic: the API image starts uvicorn only and schema SQL is
applied by hand. A deploy whose code needs a migration that was not applied therefore
starts cleanly and then fails every query that touches the new columns. This module
gives ``/health`` the facts to notice that before traffic does.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# backend/core/schema_revision.py -> <root>/infrastructure/alembic, where <root> is
# ``nomikos/`` in the repository and ``/app`` in the API image. Resolved from this
# file rather than the CWD so the probe does not depend on where uvicorn started.
DEFAULT_SCRIPT_LOCATION = Path(__file__).resolve().parents[2] / "infrastructure" / "alembic"

SchemaState = Literal["ok", "behind", "ahead", "unknown"]


@dataclass(frozen=True)
class CodeRevisions:
    """The alembic head shipped with the code and every revision below it."""

    head: str
    ancestors: frozenset[str]


def load_code_revisions(script_location: Path = DEFAULT_SCRIPT_LOCATION) -> CodeRevisions | None:
    """Read the revision chain from ``script_location``, or ``None`` if it cannot be read.

    ``None`` covers a script directory that was not shipped, a revision file that
    fails to import, and a chain with more than one head: in each case the code's
    expected revision is not known, so nothing can be compared.
    """
    try:
        script = ScriptDirectory(str(script_location))
        head = script.get_current_head()
        if head is None:
            raise ValueError("the revision chain is empty")
        ancestors = frozenset(
            rev.revision for rev in script.walk_revisions("base", head) if rev.revision != head
        )
    except Exception:
        logger.warning(
            "could not read the alembic revision chain at %s: schema checks are disabled",
            script_location,
            exc_info=True,
        )
        return None
    return CodeRevisions(head=head, ancestors=ancestors)


@lru_cache(maxsize=1)
def code_revisions() -> CodeRevisions | None:
    """The shipped revision chain, read once per process."""
    return load_code_revisions()


def compare_schema_revision(db_revision: str | None, code: CodeRevisions | None) -> SchemaState:
    """Classify the database revision against the code's head.

    ``behind`` only for a revision the code knows as an ancestor of its head. A
    revision the code does not know is ``ahead``: the release order applies SQL
    before merge, so the old code legitimately runs against a newer schema for a
    while.
    """
    if db_revision is None or code is None:
        return "unknown"
    if db_revision == code.head:
        return "ok"
    if db_revision in code.ancestors:
        return "behind"
    return "ahead"


async def read_database_revision(db: AsyncSession) -> str | None:
    """``alembic_version.version_num``, or ``None`` if it is missing or unreadable.

    A failed read rolls the session back so the connection goes back to the pool
    clean. Callers should run this last on the session for the same reason.
    """
    try:
        result = await db.execute(text("SELECT version_num FROM alembic_version"))
        rows = result.scalars().all()
    except SQLAlchemyError:
        logger.warning("health check could not read alembic_version", exc_info=True)
        await db.rollback()
        return None
    if len(rows) != 1:
        logger.warning("alembic_version holds %d rows, expected exactly one", len(rows))
        return None
    return str(rows[0])

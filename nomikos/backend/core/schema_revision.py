"""Compare the database's alembic revision with the head the running code ships.

Production never runs alembic: the API image starts uvicorn only and schema SQL is
applied by hand. A deploy whose code needs a migration that was not applied therefore
starts cleanly and then fails every query that touches the new columns. This module
gives ``/health`` the facts to notice that before traffic does.

The revision files are read with ``ast`` instead of alembic: the API image installs
only the ``platform-prod`` dependency group, which has no alembic, and the revision
files themselves import ``alembic.op``, so they are parsed, never imported.
"""

import ast
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

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


def _string_or_none(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
        return node.value
    raise ValueError(f"expected a string or None, got {ast.dump(node)}")


def _read_revision_file(path: Path) -> tuple[str, str | None] | None:
    """``(revision, down_revision)`` from a revision file's module-level assignments.

    ``None`` for a file that assigns no ``revision``. A ``down_revision`` that is not a
    string or ``None`` (a merge revision's tuple) raises ``ValueError``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, ast.expr] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in ("revision", "down_revision"):
                values[target.id] = value
    if "revision" not in values:
        return None
    revision = _string_or_none(values["revision"])
    if revision is None:
        raise ValueError(f"{path.name} sets revision to None")
    down = values.get("down_revision")
    return revision, (None if down is None else _string_or_none(down))


def load_revision_graph(
    script_location: Path = DEFAULT_SCRIPT_LOCATION,
) -> dict[str, str | None]:
    """Map every revision under ``<script_location>/versions`` to its ``down_revision``.

    Raises on a missing directory, an unparsable file, a merge revision or a
    duplicated revision id.
    """
    versions = script_location / "versions"
    if not versions.is_dir():
        raise FileNotFoundError(f"{versions} is not a directory")
    graph: dict[str, str | None] = {}
    for path in sorted(versions.glob("*.py")):
        entry = _read_revision_file(path)
        if entry is None:
            continue
        revision, down = entry
        if revision in graph:
            raise ValueError(f"revision {revision!r} is defined twice")
        graph[revision] = down
    return graph


def _chain_from_graph(graph: dict[str, str | None]) -> tuple[str, list[str]]:
    """The single head and the revisions below it, nearest first."""
    referenced = {down for down in graph.values() if down is not None}
    unknown = referenced - graph.keys()
    if unknown:
        raise ValueError(f"down_revision names unknown revisions: {sorted(unknown)}")
    heads = [rev for rev in graph if rev not in referenced]
    if len(heads) != 1:
        raise ValueError(f"expected exactly one head, found {sorted(heads)}")
    head = heads[0]
    below: list[str] = []
    seen = {head}
    current = graph[head]
    while current is not None:
        if current in seen:
            raise ValueError(f"the revision chain has a cycle at {current!r}")
        seen.add(current)
        below.append(current)
        current = graph[current]
    if len(seen) != len(graph):
        raise ValueError(f"revisions off the head's chain: {sorted(graph.keys() - seen)}")
    return head, below


def head_parent(script_location: Path = DEFAULT_SCRIPT_LOCATION) -> str | None:
    """The head's ``down_revision``, or ``None`` if the head is the base revision."""
    graph = load_revision_graph(script_location)
    head, _ = _chain_from_graph(graph)
    return graph[head]


def load_code_revisions(script_location: Path = DEFAULT_SCRIPT_LOCATION) -> CodeRevisions | None:
    """Read the revision chain from ``script_location``, or ``None`` if it cannot be read.

    ``None`` covers a script directory that was not shipped, a revision file that
    fails to parse, a merge revision, a ``down_revision`` naming an unknown revision,
    a cycle, and a chain with zero or several heads: in each case the code's expected
    revision is not known, so nothing can be compared.
    """
    try:
        head, below = _chain_from_graph(load_revision_graph(script_location))
    except Exception:
        logger.warning(
            "could not read the alembic revision chain at %s: schema checks are disabled",
            script_location,
            exc_info=True,
        )
        return None
    return CodeRevisions(head=head, ancestors=frozenset(below))


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

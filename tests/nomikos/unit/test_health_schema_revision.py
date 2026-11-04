"""``/health`` answers 503 when the database schema is behind the code's alembic head.

Production never runs alembic, so a deploy whose code needs an unapplied migration
used to go healthy and then 500 on every query touching the new columns (the
2026-11-04 ``011_model_output_provenance`` incident). The comparison is tested on its
own against real and synthetic revision chains, and the route through the mounted
application with a fake session, so no Postgres is needed.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import ProgrammingError

from backend.core import schema_revision
from backend.core.app import create_app
from backend.core.schema_revision import (
    CodeRevisions,
    compare_schema_revision,
    head_parent,
    load_code_revisions,
)
from infrastructure.db import get_db

CODE = CodeRevisions(head="003_c", ancestors=frozenset({"001_a", "002_b"}))


# --- comparison ---------------------------------------------------------------


def test_equal_revision_is_ok() -> None:
    assert compare_schema_revision("003_c", CODE) == "ok"


def test_known_older_revision_is_behind() -> None:
    assert compare_schema_revision("002_b", CODE) == "behind"
    assert compare_schema_revision("001_a", CODE) == "behind"


def test_revision_unknown_to_the_code_is_ahead() -> None:
    assert compare_schema_revision("004_d", CODE) == "ahead"


def test_missing_database_revision_is_unknown() -> None:
    assert compare_schema_revision(None, CODE) == "unknown"


def test_unreadable_script_directory_is_unknown() -> None:
    assert compare_schema_revision("003_c", None) == "unknown"


def test_load_code_revisions_reads_the_shipped_chain() -> None:
    code = load_code_revisions()

    assert code is not None
    assert code.head not in code.ancestors
    assert "010_feedback" in code.ancestors
    assert "001_initial_schema" in code.ancestors
    assert compare_schema_revision("010_feedback", code) == "behind"


def test_load_code_revisions_returns_none_for_a_missing_directory(tmp_path: Path) -> None:
    assert load_code_revisions(tmp_path / "not-shipped") is None


def test_default_script_location_does_not_depend_on_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    assert load_code_revisions(schema_revision.DEFAULT_SCRIPT_LOCATION) is not None


def _write_versions(root: Path, files: dict[str, str]) -> Path:
    versions = root / "versions"
    versions.mkdir(parents=True)
    for name, body in files.items():
        (versions / name).write_text(body)
    return root


def _rev(revision: str, down: str | None) -> str:
    return (
        f"import alembic.op\nrevision: str = {revision!r}\ndown_revision: str | None = {down!r}\n"
    )


def test_synthetic_linear_chain(tmp_path: Path) -> None:
    root = _write_versions(
        tmp_path,
        {
            "a.py": _rev("001_a", None),
            "b.py": "revision = '002_b'\ndown_revision = '001_a'\n",
            "c.py": _rev("003_c", "002_b"),
        },
    )

    assert load_code_revisions(root) == CODE
    assert head_parent(root) == "002_b"


def test_synthetic_two_heads_is_none(tmp_path: Path) -> None:
    root = _write_versions(
        tmp_path,
        {
            "a.py": _rev("001_a", None),
            "b.py": _rev("002_b", "001_a"),
            "c.py": _rev("003_c", "001_a"),
        },
    )

    assert load_code_revisions(root) is None


def test_synthetic_unparsable_file_is_none(tmp_path: Path) -> None:
    root = _write_versions(
        tmp_path, {"a.py": _rev("001_a", None), "b.py": "revision = '002_b'\ndef broken(:\n"}
    )

    assert load_code_revisions(root) is None


def test_synthetic_merge_revision_is_none(tmp_path: Path) -> None:
    root = _write_versions(
        tmp_path,
        {
            "a.py": _rev("001_a", None),
            "b.py": _rev("002_b", "001_a"),
            "c.py": "revision = '003_c'\ndown_revision = ('001_a', '002_b')\n",
        },
    )

    assert load_code_revisions(root) is None


def test_synthetic_unknown_down_revision_is_none(tmp_path: Path) -> None:
    root = _write_versions(tmp_path, {"a.py": _rev("002_b", "001_missing")})

    assert load_code_revisions(root) is None


def test_synthetic_file_without_revision_is_skipped(tmp_path: Path) -> None:
    root = _write_versions(
        tmp_path,
        {
            "__init__.py": "",
            "helpers.py": "VALUE = 1\n",
            "a.py": _rev("001_a", None),
            "b.py": _rev("002_b", "001_a"),
            "c.py": _rev("003_c", "002_b"),
        },
    )

    assert load_code_revisions(root) == CODE


def test_shipped_chain_matches_alembic() -> None:
    from alembic.script import ScriptDirectory

    script = ScriptDirectory(str(schema_revision.DEFAULT_SCRIPT_LOCATION))
    head = script.get_current_head()
    ancestors = frozenset(
        rev.revision for rev in script.walk_revisions("base", head) if rev.revision != head
    )

    assert load_code_revisions() == CodeRevisions(head=head, ancestors=ancestors)
    assert head_parent() == script.get_revision(head).down_revision


def test_runtime_import_does_not_load_alembic() -> None:
    """Stand-in for the API image build, whose ``platform-prod`` group has no alembic."""
    repo = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(repo / "nomikos"), str(repo)]),
    }
    code = (
        "import sys; import backend.core.schema_revision; import backend.core.api.health; "
        "assert 'alembic' not in sys.modules, sorted(m for m in sys.modules if 'alembic' in m)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


# --- route --------------------------------------------------------------------


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalar_one_or_none(self) -> Any:
        return self._rows[0] if self._rows else None

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[Any]:
        return list(self._rows)


class _FakeSession:
    """Answers ``SELECT 1``, the queue query (empty) and ``alembic_version``."""

    def __init__(self, version_rows: list[str] | None) -> None:
        # ``None`` means the alembic_version table does not exist.
        self._version_rows = version_rows
        self.rolled_back = False

    async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> _Result:
        sql = str(statement)
        if "alembic_version" in sql:
            if self._version_rows is None:
                raise ProgrammingError(
                    sql, {}, Exception('relation "alembic_version" does not exist')
                )
            return _Result(self._version_rows)
        if sql.strip() == "SELECT 1":
            return _Result([1])
        return _Result([])

    async def rollback(self) -> None:
        self.rolled_back = True


@pytest.fixture
def make_client(monkeypatch):
    def _make(
        version_rows: list[str] | None, code: CodeRevisions | None = CODE
    ) -> tuple[TestClient, _FakeSession]:
        monkeypatch.setattr("backend.core.api.health.code_revisions", lambda: code)
        session = _FakeSession(version_rows)

        async def _get_db():
            yield session

        app = create_app()
        app.dependency_overrides[get_db] = _get_db
        return TestClient(app), session

    return _make


def test_route_is_ok_when_schema_matches(make_client) -> None:
    client, _ = make_client(["003_c"])

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["schema"] == "ok"
    assert body["schema_revision"] == "003_c"
    assert body["expected_schema_revision"] == "003_c"


def test_route_is_503_and_logs_error_when_schema_is_behind(make_client, caplog) -> None:
    client, _ = make_client(["002_b"])

    with caplog.at_level(logging.WARNING):
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "ok"
    assert body["schema"] == "behind"
    assert body["schema_revision"] == "002_b"
    assert body["expected_schema_revision"] == "003_c"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("002_b" in r.getMessage() and "003_c" in r.getMessage() for r in errors)


def test_route_stays_200_and_warns_when_schema_is_ahead(make_client, caplog) -> None:
    client, _ = make_client(["004_d"])

    with caplog.at_level(logging.WARNING):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["schema"] == "ahead"
    assert any(r.levelno == logging.WARNING and "004_d" in r.getMessage() for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_route_stays_200_and_warns_when_alembic_version_is_missing(make_client, caplog) -> None:
    client, session = make_client(None)

    with caplog.at_level(logging.WARNING):
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "unknown"
    assert body["schema_revision"] is None
    assert body["expected_schema_revision"] == "003_c"
    assert session.rolled_back
    assert any(
        r.levelno == logging.WARNING and "alembic_version" in r.getMessage() for r in caplog.records
    )


def test_route_stays_200_when_script_directory_is_unreadable(make_client) -> None:
    client, _ = make_client(["002_b"], code=None)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "unknown"
    assert body["schema_revision"] == "002_b"
    assert body["expected_schema_revision"] is None

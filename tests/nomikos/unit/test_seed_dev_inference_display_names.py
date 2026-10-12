"""Display names for the dev seed of segment catalog rows.

The pickers render ``inference_models.name`` while dispatch parses the
registry id out of ``artifact_ref``, so a segment row is shown as ``seg-v1``
or ``seg-v2`` but must still point at its registry id. These tests cover the
pure name/ref mapping plus the reference-preserving duplicate cleanup, all with
fakes; the database upsert itself is exercised by running the script, so no
database is needed here.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from datetime import datetime
from types import ModuleType, SimpleNamespace

import pytest
from sqlalchemy import Delete, Select, Update

from tests.fixtures.paths import REPO_ROOT

PLATFORM_SCRIPTS = REPO_ROOT / "scripts" / "platform"


@pytest.fixture(scope="module")
def seed_module() -> Iterator[ModuleType]:
    """Import the standalone seed script; it resolves `_bootstrap` as a sibling."""
    sys.path.insert(0, str(PLATFORM_SCRIPTS))
    try:
        yield importlib.import_module("seed_dev_inference")
    finally:
        sys.path.remove(str(PLATFORM_SCRIPTS))


class _FakeScalars:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return list(self._rows)


class _FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._rows)


class _StatementSession:
    """Fake session that records statements and serves rows without a database.

    Selects on ``model_bindings.model_id`` are answered from ``bindings`` keyed
    by model id; every other select returns the catalog ``model_rows`` the
    upsert scans. Updates and deletes are only recorded for assertion.
    """

    def __init__(self, model_rows: list, bindings: dict | None = None) -> None:
        self._model_rows = list(model_rows)
        self._bindings = {key: list(rows) for key, rows in (bindings or {}).items()}
        self.statements: list = []
        self.deleted: list = []

    async def __aenter__(self) -> _StatementSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, stmt: object) -> _FakeResult:
        assert isinstance(stmt, (Select, Update, Delete))
        self.statements.append(stmt)
        if isinstance(stmt, Select):
            where = stmt.whereclause
            table = getattr(getattr(where, "left", None), "table", None)
            if table is not None and table.name == "model_bindings":
                return _FakeResult(list(self._bindings.get(where.right.value, [])))
            return _FakeResult(list(self._model_rows))
        return _FakeResult([])

    async def delete(self, obj: object) -> None:
        self.deleted.append(obj)

    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def refresh(self, obj: object) -> None:
        return None


def _updates(statements: list, table_name: str) -> list:
    return [s for s in statements if isinstance(s, Update) and s.table.name == table_name]


def _deletes(statements: list, table_name: str) -> list:
    return [s for s in statements if isinstance(s, Delete) and s.table.name == table_name]


def _params(stmt: object) -> set:
    assert isinstance(stmt, (Update, Delete))
    return set(stmt.compile().params.values())


def test_artifact_refs_use_registry_ids_never_display_names(
    seed_module: ModuleType,
) -> None:
    assert seed_module.artifact_ref_for("blla-segment") == "registry://blla-segment?tag=stable"
    assert seed_module.artifact_ref_for("ppocr-segment") == "registry://ppocr-segment?tag=stable"
    for registry_id, display_name in seed_module.SEGMENT_MODELS:
        ref = seed_module.artifact_ref_for(registry_id)
        assert ref.split("://", 1)[1].split("?", 1)[0] == registry_id
        assert display_name != registry_id


def test_candidate_names_cover_old_and_new_rows(seed_module: ModuleType) -> None:
    assert seed_module.candidate_names_for("blla-segment", "seg-v1") == ("blla-segment", "seg-v1")
    assert seed_module.candidate_names_for("ppocr-segment", "seg-v2") == (
        "ppocr-segment",
        "seg-v2",
    )
    assert seed_module.candidate_names_for("greek-calamari-v1", "greek-calamari-v1") == (
        "greek-calamari-v1",
    )


def test_duplicate_name_rows_keep_the_oldest(seed_module: ModuleType) -> None:
    old = SimpleNamespace(name="blla-segment", created_at=datetime(2026, 1, 1))
    new = SimpleNamespace(name="kraken", created_at=datetime(2026, 2, 1))
    keep, delete = seed_module.split_duplicate_models([new, old], "seg-v2")
    assert keep is old
    assert delete == [new]


def test_default_segment_ids_offer_both_models(
    seed_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEFAULT_SEGMENT_MODEL", raising=False)
    reloaded = importlib.reload(seed_module)
    assert {registry_id for registry_id, _ in reloaded.SEGMENT_MODELS} == {
        "blla-segment",
        "ppocr-segment",
    }


def test_duplicate_artifact_ref_rows_converge_to_display_name(
    seed_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two rows sharing one artifact_ref (old and new name) seed to one row."""
    import asyncio

    from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask

    ref = seed_module.artifact_ref_for("blla-segment")

    class FakeScalars:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def all(self) -> list:
            return list(self._rows)

    class FakeResult:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def scalars(self) -> FakeScalars:
            return FakeScalars(self._rows)

    class FakeSession:
        def __init__(self, rows: list) -> None:
            self._rows = list(rows)
            self.deleted = []

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def execute(self, stmt: object) -> FakeResult:
            from sqlalchemy import Select

            if isinstance(stmt, Select):
                where = stmt.whereclause
                table = getattr(getattr(where, "left", None), "table", None)
                if table is not None and table.name == "model_bindings":
                    return FakeResult([])
            return FakeResult(self._rows)

        async def delete(self, obj: object) -> None:
            self.deleted.append(obj)
            if obj in self._rows:
                self._rows.remove(obj)

        async def flush(self) -> None:
            return None

        async def commit(self) -> None:
            return None

        async def refresh(self, obj: object) -> None:
            return None

    old = InferenceModel(
        name="blla-segment",
        provider="kraken",
        task=InferenceTask.segment,
        artifact_ref=ref,
        default_params={"device": "cpu"},
        created_at=datetime(2026, 1, 1),
    )
    new = InferenceModel(
        name="seg-v1",
        provider="kraken",
        task=InferenceTask.segment,
        artifact_ref=ref,
        default_params={"device": "cpu"},
        created_at=datetime(2026, 2, 1),
    )
    session = FakeSession([old, new])
    monkeypatch.setattr(seed_module, "system_session", lambda: session)

    model = asyncio.run(
        seed_module._upsert_model(
            registry_id="blla-segment", display_name="seg-v1", task=InferenceTask.segment
        )
    )

    assert model.name == "seg-v1"
    assert model.artifact_ref == ref
    assert session.deleted == [old]
    assert session._rows == [model]
    assert len(session._rows) == 1


def test_duplicate_binding_and_job_move_to_kept_row(seed_module: ModuleType) -> None:
    """A duplicate with a binding and a job repoints both at the kept row."""
    import asyncio
    import uuid

    from backend.ml.infrastructure.orm_models import InferenceTask

    keep_id = uuid.uuid4()
    duplicate_id = uuid.uuid4()
    binding = SimpleNamespace(
        id=uuid.uuid4(),
        task=InferenceTask.segment,
        project_id=uuid.uuid4(),
        document_id=None,
        document_part_id=None,
    )
    session = _StatementSession([], bindings={keep_id: [], duplicate_id: [binding]})

    asyncio.run(seed_module.repoint_model_references(session, duplicate_id, keep_id))

    binding_updates = _updates(session.statements, "model_bindings")
    assert len(binding_updates) == 1
    assert _params(binding_updates[0]) == {keep_id, binding.id}
    job_updates = _updates(session.statements, "jobs")
    assert len(job_updates) == 1
    assert _params(job_updates[0]) == {duplicate_id, keep_id}
    assert _deletes(session.statements, "model_bindings") == []


def test_colliding_binding_is_removed_and_kept_one_survives(
    seed_module: ModuleType,
) -> None:
    """A binding the kept row already covers is removed, not repointed."""
    import asyncio
    import uuid

    from backend.ml.infrastructure.orm_models import InferenceTask

    keep_id = uuid.uuid4()
    duplicate_id = uuid.uuid4()
    scope = {
        "task": InferenceTask.segment,
        "project_id": uuid.uuid4(),
        "document_id": None,
        "document_part_id": None,
    }
    kept_binding = SimpleNamespace(id=uuid.uuid4(), **scope)
    colliding = SimpleNamespace(id=uuid.uuid4(), **scope)
    other = SimpleNamespace(
        id=uuid.uuid4(),
        task=InferenceTask.segment,
        project_id=uuid.uuid4(),
        document_id=None,
        document_part_id=None,
    )
    session = _StatementSession(
        [], bindings={keep_id: [kept_binding], duplicate_id: [colliding, other]}
    )

    asyncio.run(seed_module.repoint_model_references(session, duplicate_id, keep_id))

    deletes = _deletes(session.statements, "model_bindings")
    assert len(deletes) == 1
    assert _params(deletes[0]) == {colliding.id}
    moved = _updates(session.statements, "model_bindings")
    assert len(moved) == 1
    assert _params(moved[0]) == {keep_id, other.id}
    for stmt in session.statements:
        if isinstance(stmt, (Update, Delete)) and stmt.table.name == "model_bindings":
            assert kept_binding.id not in stmt.compile().params.values()
    assert len(_updates(session.statements, "jobs")) == 1


def test_no_duplicates_issues_no_updates(
    seed_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean upsert repoints nothing and deletes nothing."""
    import asyncio

    from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask

    ref = seed_module.artifact_ref_for("blla-segment")
    model = InferenceModel(
        name="seg-v1",
        provider="kraken",
        task=InferenceTask.segment,
        artifact_ref=ref,
        default_params={"device": "cpu"},
        created_at=datetime(2026, 2, 1),
    )
    session = _StatementSession([model])
    monkeypatch.setattr(seed_module, "system_session", lambda: session)

    result = asyncio.run(
        seed_module._upsert_model(
            registry_id="blla-segment", display_name="seg-v1", task=InferenceTask.segment
        )
    )

    assert result is model
    assert session.deleted == []
    assert [s for s in session.statements if isinstance(s, Update)] == []

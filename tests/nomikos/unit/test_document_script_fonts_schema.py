"""A document's per-script font choice is checked against the font registry.

The request model refuses an unknown script or a choice the script does not offer, so
a typo answers 422 instead of being stored and silently rendered in the default font.
The response always carries an object, ``{}`` for a document that overrides nothing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

from backend.core.app import create_app
from backend.core.exceptions import ValidationError
from backend.document.api.responses import document_response
from backend.document.api.schemas import DocumentResponse, DocumentUpdateRequest
from backend.document.application.document_catalog import DocumentCatalog
from backend.document.infrastructure.orm_models import Document, DocumentWorkflow
from backend.users.api.dependencies import get_current_user
from infrastructure.db import get_db


def test_update_request_accepts_a_known_choice() -> None:
    body = DocumentUpdateRequest.model_validate({"script_fonts": {"syriac": "eastern"}})
    assert body.model_dump(exclude_unset=True) == {"script_fonts": {"syriac": "eastern"}}


def test_update_request_accepts_an_empty_map_as_clear() -> None:
    body = DocumentUpdateRequest.model_validate({"script_fonts": {}})
    assert body.model_dump(exclude_unset=True) == {"script_fonts": {}}


@pytest.mark.parametrize(
    "value",
    [{"syriac": "meltho"}, {"klingon": "x"}, None, {"syriac": 3}],
    ids=["unknown-choice", "unknown-script", "null", "non-string"],
)
def test_update_request_rejects_a_bad_map(value) -> None:
    with pytest.raises(PydanticValidationError):
        DocumentUpdateRequest.model_validate({"script_fonts": value})


def _document(script_fonts: dict[str, str] | None) -> Document:
    now = datetime(2026, 11, 22, tzinfo=UTC)
    return Document(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        name="Codex",
        workflow=DocumentWorkflow.draft,
        script_fonts=script_fonts,
        created_at=now,
        updated_at=now,
    )


def test_response_defaults_script_fonts_to_an_empty_object() -> None:
    response = DocumentResponse(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        name="Codex",
        workflow=DocumentWorkflow.draft,
        created_at=datetime(2026, 11, 22, tzinfo=UTC),
        updated_at=datetime(2026, 11, 22, tzinfo=UTC),
    )
    assert response.script_fonts == {}


def test_response_builder_maps_null_to_an_empty_object_and_passes_choices() -> None:
    assert document_response(_document(None), part_count=0).script_fonts == {}
    stored = document_response(_document({"syriac": "western"}), part_count=0)
    assert stored.script_fonts == {"syriac": "western"}


# --- Catalog: what is written to the row ---


class _Access:
    def __init__(self, document: Document) -> None:
        self._document = document

    async def require_document(self, *_args, **_kwargs):
        class _Context:
            project = object()
            document = self._document

        return _Context()


class _Documents:
    def __init__(self) -> None:
        self.written: dict[str, object] | None = None

    async def update(self, _session, document, **fields):
        self.written = fields
        for key, value in fields.items():
            setattr(document, key, value)
        return document


def _catalog(document: Document) -> tuple[DocumentCatalog, _Documents]:
    documents = _Documents()
    catalog = DocumentCatalog(documents=documents, access=_Access(document))
    return catalog, documents


@pytest.mark.asyncio
async def test_catalog_stores_an_empty_map_as_null() -> None:
    document = _document({"syriac": "eastern"})
    catalog, documents = _catalog(document)

    await catalog.update_document(None, object(), document.project_id, document.id, script_fonts={})

    assert documents.written == {"script_fonts": None}
    assert document.script_fonts is None


@pytest.mark.asyncio
async def test_catalog_assigns_a_new_dict_rather_than_the_callers() -> None:
    document = _document(None)
    catalog, _ = _catalog(document)
    choice = {"syriac": "western"}

    await catalog.update_document(
        None, object(), document.project_id, document.id, script_fonts=choice
    )

    assert document.script_fonts == {"syriac": "western"}
    assert document.script_fonts is not choice


@pytest.mark.asyncio
async def test_catalog_rejects_an_unknown_choice_from_a_direct_caller() -> None:
    document = _document(None)
    catalog, documents = _catalog(document)

    with pytest.raises(ValidationError):
        await catalog.update_document(
            None, object(), document.project_id, document.id, script_fonts={"syriac": "x"}
        )
    assert documents.written is None


# --- Route: the mounted app answers 422 before any handler runs ---


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: None
    return TestClient(app)


@pytest.mark.parametrize(
    "script_fonts",
    [{"syriac": "meltho"}, {"klingon": "x"}, None],
    ids=["unknown-choice", "unknown-script", "null"],
)
def test_patch_with_a_bad_script_font_map_is_refused(client, script_fonts) -> None:
    url = f"/projects/{uuid.uuid4()}/documents/{uuid.uuid4()}"
    response = client.patch(url, json={"script_fonts": script_fonts})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"

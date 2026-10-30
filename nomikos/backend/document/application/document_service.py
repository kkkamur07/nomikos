"""The narrow surface other bounded contexts call the document context through.

This is deliberately **not** a sixth module. The document context is five:
:class:`DocumentCatalog`, :class:`DocumentPartService`, :class:`LayoutService`,
:class:`TranscriptionService` and :class:`DocumentJobEnqueueService`, over the
:class:`DocumentAccess` seam, and the routes in ``document/api`` call whichever one they
need directly. What survives here is the handful of methods that ``backend/annotation``
and the dev seed script reach for, kept so that a bounded context outside this one does
not have to know how this one is arranged internally.

Every module it composes shares a single ``DocumentAccess`` and a single repository set,
so authorizing through one and reading through another stays consistent within a request.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.application.document_access import DocumentAccess, PartContext
from backend.document.application.document_catalog import DocumentCatalog
from backend.document.application.ground_truth import GroundTruthText
from backend.document.application.layout_service import LayoutService
from backend.document.application.part_service import DocumentPartService
from backend.document.application.transcription_service import TranscriptionService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import MediaStore, get_media_store
from backend.document.infrastructure.orm_models import Document, DocumentPart, Line
from backend.ml.application.model_service import InferenceModelService
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


class DocumentService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        media: MediaStore | None = None,
        inference_models: InferenceModelService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._media = media or get_media_store()
        self._inference_models = inference_models or InferenceModelService()
        self._access = DocumentAccess(documents=self._documents, projects=self._projects)
        ground_truth = GroundTruthText(documents=self._documents)
        self.catalog = DocumentCatalog(
            documents=self._documents, projects=self._projects, access=self._access
        )
        self.parts = DocumentPartService(
            documents=self._documents,
            projects=self._projects,
            media=self._media,
            access=self._access,
        )
        self.layout = LayoutService(
            documents=self._documents,
            projects=self._projects,
            access=self._access,
            ground_truth=ground_truth,
        )
        self.transcriptions = TranscriptionService(
            documents=self._documents,
            projects=self._projects,
            access=self._access,
            ground_truth=ground_truth,
        )

    # --- Document lifecycle ---

    async def create_document(
        self, session: AsyncSession, user: User, project_id: UUID, *, name: str
    ) -> Document:
        return await self.catalog.create_document(session, user, project_id, name=name)

    async def get_document(
        self, session: AsyncSession, user: User, project_id: UUID, document_id: UUID
    ) -> Document:
        return await self.catalog.get_document(session, user, project_id, document_id)

    async def get_published_part(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> DocumentPart:
        return await self.catalog.get_published_part(
            session, project_id, document_id, part_id, token=token
        )

    async def get_published_part_context(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> PartContext:
        return await self.catalog.get_published_part_context(
            session, project_id, document_id, part_id, token=token
        )

    # --- Parts ---

    async def upload_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        data: bytes,
        filename: str | None = None,
    ) -> DocumentPart:
        return await self.parts.upload_part(
            session, user, project_id, document_id, data=data, filename=filename
        )

    async def backfill_part_dimensions(
        self, session: AsyncSession, parts: list[DocumentPart]
    ) -> None:
        await self.parts.backfill_part_dimensions(session, parts)

    # --- Layout ---

    async def list_part_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> list[Line]:
        return await self.layout.list_part_lines(session, user, project_id, document_id, part_id)

    async def replace_part_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        lines: list[dict],
        allow_new_ids: bool = False,
        restore_origin: bool = False,
    ) -> list[Line]:
        return await self.layout.replace_part_lines(
            session,
            user,
            project_id,
            document_id,
            part_id,
            lines=lines,
            allow_new_ids=allow_new_ids,
            restore_origin=restore_origin,
        )

    # --- Transcription text ---

    async def import_page_transcription(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        text: str,
    ):
        return await self.transcriptions.import_page_transcription(
            session, user, project_id, document_id, part_id, text=text
        )

    async def pair_page_text_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        line_id: UUID,
        text_line_order: int,
    ):
        return await self.transcriptions.pair_page_text_line(
            session,
            user,
            project_id,
            document_id,
            part_id,
            line_id=line_id,
            text_line_order=text_line_order,
        )

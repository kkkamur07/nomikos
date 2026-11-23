"""The documents in a project: their existence, their names, and their workflow state.

One responsibility: a document's life from creation to deletion, plus the reads that
expose it. Page images, geometry and transcription text are three other modules; this one
only ever touches the ``documents`` and ``transcriptions`` rows.

The public (unauthenticated) reads live here rather than in a separate "public service"
because they are the *same* lifecycle reads with a different audience, and the audience is
already a parameter of :class:`DocumentAccess`. Splitting them apart would put the
published-workflow rule in two places again.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from typing import NamedTuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api.pagination import PageCursor
from backend.core.exceptions import AccessDeniedError, ValidationError
from backend.core.fonts import validate_script_fonts
from backend.document.application.document_access import DocumentAccess, PartContext
from backend.document.application.patch_fields import (
    DOCUMENT_UPDATE_FIELDS,
    reject_unknown_fields,
)
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    DocumentWorkflow,
    Line,
    Transcription,
)
from backend.project.domain.access import is_owner
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


class PublicLayoutPage(NamedTuple):
    """One page of anonymous layout, and whether the blocks on it are all of them.

    ``lines`` carries the probe row the caller needs to build a continuation cursor.
    Blocks have no cursor, so their overflow is reported as a flag instead - a client
    that cannot see it has no way to tell a truncated page from a complete one.
    """

    blocks: list[Block]
    blocks_truncated: bool
    lines: list[Line]


class DocumentCatalog:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)

    async def list_documents(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        include_archived: bool = False,
        limit: int = 50,
        cursor=None,
    ) -> list[Document]:
        await self._access.require_project(session, user, project_id)
        return await self._documents.list_for_project(
            session,
            project_id,
            include_archived=include_archived,
            limit=limit,
            cursor=cursor,
        )

    async def create_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        name: str,
    ) -> Document:
        await self._access.require_project(session, user, project_id)
        return await self._documents.create(session, project_id=project_id, name=name)

    async def get_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> Document:
        context = await self._access.require_document(session, user, project_id, document_id)
        return context.document

    async def owns_project(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
    ) -> bool:
        """Whether ``user`` owns ``project_id`` - the bar for seeing the share token.

        Membership is not enough. A collaborator who can read the token can hand an
        anonymous, working link to the whole document to anyone, and the owner has no
        way to notice it happened; the only remedy left is rotation, which breaks every
        link already sent. That is the same reason publishing and rotation are
        owner-only, so reading the secret has to be too.
        """
        project = await self._access.require_project(session, user, project_id)
        return is_owner(project, user.id)

    async def get_document_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        *,
        token: str | None = None,
    ) -> Document:
        # ``None`` is the whole difference: the public router has no authentication
        # dependency, so there is no caller to check membership for, and the seam falls
        # through to the published-workflow-and-token rule.
        context = await self._access.require_document(
            session, None, project_id, document_id, token=token
        )
        return context.document

    async def get_published_part(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> DocumentPart:
        context = await self._access.require_part(
            session, None, project_id, document_id, part_id, token=token
        )
        return context.part

    async def get_published_part_context(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> PartContext:
        """The published part *and* its document, for callers that need both.

        Exports name their files after the document, and a second ``get_document_public``
        would repeat the access check this one already made.
        """
        return await self._access.require_part(
            session, None, project_id, document_id, part_id, token=token
        )

    async def update_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        **fields: object,
    ) -> Document:
        reject_unknown_fields(fields, DOCUMENT_UPDATE_FIELDS, "document update")
        context = await self._access.require_document(session, user, project_id, document_id)
        project, document = context.project, context.document
        if "workflow" in fields and fields["workflow"] is not None:
            workflow = fields["workflow"]
            if not isinstance(workflow, DocumentWorkflow):
                raise ValidationError("Invalid workflow value")
            # Publishing drops the whole document - page images and
            # transcriptions - to unauthenticated readers. That is a decision for
            # whoever owns the project, not for anyone they shared it with.
            if workflow is DocumentWorkflow.published and not is_owner(project, user.id):
                raise AccessDeniedError("Only the project owner can publish a document")
            if workflow is DocumentWorkflow.published and document.public_share_token is None:
                # Minted once, the first time a document goes live. A later draft ->
                # published round trip keeps the same link rather than silently
                # breaking whatever was already shared - rotating is a separate,
                # explicit operation the owner reaches for on purpose.
                fields = {**fields, "public_share_token": secrets.token_urlsafe(32)}
        if "script_fonts" in fields:
            # A fresh dict, never the stored one mutated: SQLAlchemy only notices a JSONB
            # change on reassignment. Empty means "all defaults", which is stored as NULL.
            # Checked again here for callers that skip the request model.
            script_fonts = fields["script_fonts"]
            if not isinstance(script_fonts, Mapping):
                raise ValidationError("Invalid script_fonts value")
            try:
                checked = validate_script_fonts(script_fonts)
            except ValueError as exc:
                raise ValidationError(f"Invalid script_fonts value: {exc}") from exc
            fields = {**fields, "script_fonts": checked or None}
        return await self._documents.update(session, document, **fields)

    async def rotate_share_token(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> Document:
        """Mint a fresh share token, invalidating every link built from the old one.

        Same gate as publishing itself: only the owner may create or destroy a standing
        public link, not just any collaborator. This does not touch ``workflow`` - a
        document can be rotated whether or not it is currently published, since nothing
        stops an owner from pre-arming a link before they flip it live.
        """
        context = await self._access.require_document(session, user, project_id, document_id)
        project, document = context.project, context.document
        if not is_owner(project, user.id):
            raise AccessDeniedError("Only the project owner can rotate the share link")
        return await self._documents.update(
            session, document, public_share_token=secrets.token_urlsafe(32)
        )

    async def delete_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> None:
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
        image_keys = [part.image_key for part in document.parts]
        await self._documents.delete_with_media_intents(session, document, image_keys)

    async def list_transcriptions(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> list[Transcription]:
        document = await self.get_document(session, user, project_id, document_id)
        return await self._documents.list_transcriptions(session, document.id)

    async def list_transcriptions_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        *,
        token: str | None = None,
    ) -> list[Transcription]:
        # Transcription *layers* only - id, name, kind - never a part's lines or their
        # text, so this needs no published-part filtering of its own.
        document = await self.get_document_public(session, project_id, document_id, token=token)
        return await self._documents.list_transcriptions(session, document.id)

    async def list_document_layout_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        token: str | None = None,
    ) -> PublicLayoutPage:
        """Bounded layout read for anonymous callers.

        ``limit`` is the page size. Both reads fetch one row beyond it, which is what
        lets the caller tell "this is everything" from "there is more": for lines the
        extra row becomes the continuation cursor, and for blocks - which have no cursor
        - it becomes ``blocks_truncated``.

        Blocks accompany the first page only, so repeating them per page cannot unbound
        the read. Reporting the truncation matters because a bare ``LIMIT`` returning
        exactly ``limit`` rows is indistinguishable from a document that has exactly
        that many blocks: a published page silently lost the rest of its regions with
        nothing in the response a client could notice.

        Both repository reads filter to the document's *published* parts - the document
        itself being public does not mean every page on it is.
        """
        document = await self.get_document_public(session, project_id, document_id, token=token)
        blocks: list[Block] = []
        blocks_truncated = False
        if cursor is None:
            probed = await self._documents.list_blocks_for_document(
                session, document.id, limit=limit + 1
            )
            blocks_truncated = len(probed) > limit
            blocks = probed[:limit]
        lines = await self._documents.list_lines_for_document(
            session, document.id, limit=limit + 1, cursor=cursor
        )
        return PublicLayoutPage(blocks=blocks, blocks_truncated=blocks_truncated, lines=lines)

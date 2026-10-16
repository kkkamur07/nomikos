"""Shared response builders for document API routers."""

from backend.document.api.schemas import (
    DocumentPartResponse,
    DocumentResponse,
    DocumentWithPartsResponse,
    PublicDocumentWithPartsResponse,
)
from backend.document.infrastructure.orm_models import Document, DocumentPart


def document_response(
    document: Document,
    *,
    part_count: int | None = None,
    share_token: str | None = None,
) -> DocumentResponse:
    count = part_count if part_count is not None else len(document.parts)
    return DocumentResponse(
        id=document.id,
        project_id=document.project_id,
        name=document.name,
        workflow=document.workflow,
        part_count=count,
        # Handed in, never read off ``document``. A flag that has to be remembered
        # fails open when it is forgotten; a secret that has to be passed fails closed.
        # The caller is the only one that knows whether it established *ownership*,
        # which is the bar here: membership is not enough, because a collaborator who
        # can read the token can mint an anonymous link to the whole document and the
        # owner has no way to see that it happened.
        public_share_token=share_token,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def document_with_parts_response(
    document: Document, *, public: bool = False, share_token: str | None = None
) -> DocumentWithPartsResponse:
    parts = sorted(document.parts, key=lambda p: p.order)
    if public:
        # The document being published does not mean every page on it is; an
        # unpublished part must not appear in the body of the public document read.
        parts = [part for part in parts if part.published]
    return DocumentWithPartsResponse(
        **document_response(document, part_count=len(parts), share_token=share_token).model_dump(),
        parts=[part_response(part, public=public) for part in parts],
    )


def part_response(part: DocumentPart, *, public: bool = False) -> DocumentPartResponse:
    media_prefix = "/public/media" if public else "/media"
    return DocumentPartResponse(
        id=part.id,
        document_id=part.document_id,
        order=part.order,
        image_url=f"{media_prefix}/parts/{part.id}",
        width=part.width,
        height=part.height,
        reviewed=part.reviewed,
        published=part.published,
        created_at=part.created_at,
    )


def public_document_with_parts_response(document: Document) -> PublicDocumentWithPartsResponse:
    """The anonymous reader's view: published parts only, and no share token anywhere.

    The token is absent by construction here rather than blanked on the way out, so
    there is no flag to get backwards and nothing for a serialisation setting to have
    to remember.
    """
    parts = [part for part in sorted(document.parts, key=lambda p: p.order) if part.published]
    return PublicDocumentWithPartsResponse(
        id=document.id,
        project_id=document.project_id,
        name=document.name,
        workflow=document.workflow,
        part_count=len(parts),
        parts=[part_response(part, public=True) for part in parts],
        created_at=document.created_at,
        updated_at=document.updated_at,
    )

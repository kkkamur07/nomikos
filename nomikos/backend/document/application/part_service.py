"""Document parts: the page images, their order, their review state, and their bytes.

One responsibility: everything that treats a page as a *scan* rather than as geometry or
as text. That is what keeps the media store out of every other module in this context:
this is the only place that writes to it, reads from it, or has to compensate when a
write to it outlives the transaction that was supposed to record it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import AccessDeniedError, NotFoundError, ValidationError
from backend.document.application.document_access import DocumentAccess
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import (
    DEFAULT_PART_IMAGE_SUFFIX,
    MediaStore,
    PresignUnsupported,
    encode_part_image_with_size,
    encode_part_thumbnail,
    get_media_store,
    persisted_thumbnail_key,
    read_image_size,
    validate_image_key,
)
from backend.document.infrastructure.orm_models import DocumentPart
from backend.project.domain.access import is_owner
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User

logger = logging.getLogger(__name__)

# Parts uploaded before dimensions were persisted are backfilled lazily on read; bound how
# many object-store round trips a single request may trigger.
MAX_DIMENSION_BACKFILLS_PER_REQUEST = 25

# The largest page scan a part upload may carry, on either path. Multipart uploads
# declare it in the request contract; direct-to-storage uploads can PUT whatever they
# like past the API, so finalize re-checks the stored blob against the same bound.
MAX_PART_UPLOAD_BYTES = 100 * 1024 * 1024


class DocumentPartService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        media: MediaStore | None = None,
        access: DocumentAccess | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._injected_media = media
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)

    @property
    def _media(self) -> MediaStore:
        """Resolve the configured store per use, not once at construction.

        Four route modules build this service at import time, so capturing the
        store in ``__init__`` pinned it to whatever ``STORAGE_BACKEND`` said
        before the first request - and left this service the one holder that
        ``reset_settings_caches`` could not reach. Writes then went to the
        import-time backend while readers that call ``get_media_store()`` per
        use went to the current one. An explicitly injected store still wins.
        """
        return self._injected_media or get_media_store()

    async def list_parts(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> list[DocumentPart]:
        context = await self._access.require_document(session, user, project_id, document_id)
        parts = sorted(context.document.parts, key=lambda p: p.order)
        await self.backfill_part_dimensions(session, parts)
        return parts

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
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
        order = await self._documents.next_part_order(session, document.id)
        filename_stem: str | None = None
        if filename and "." in filename:
            filename_stem = filename.rsplit(".", 1)[0]
        # Single bounded decode: it validates the upload, produces the stored WebP, and
        # yields the dimensions the page canvas and PAGE XML export need.
        encoded = await asyncio.to_thread(encode_part_image_with_size, data)
        part = DocumentPart(
            document_id=document.id,
            order=order,
            image_key="pending",
            width=encoded.width,
            height=encoded.height,
        )
        session.add(part)
        await session.flush()
        image_key = self._media.part_image_key(
            part.id,
            suffix=DEFAULT_PART_IMAGE_SUFFIX,
            filename_stem=filename_stem,
        )
        try:
            # Both store calls are synchronous HTTPS round trips on the Supabase backend,
            # uploading up to a 100 MiB WebP. Running either inline would park the event
            # loop - and with it every other in-flight request - for the whole upload.
            await asyncio.to_thread(self._media.write, image_key, encoded.data)
            part.image_key = image_key
            await session.commit()
        except Exception:
            await session.rollback()
            try:
                await asyncio.to_thread(self._media.delete, image_key)
            except Exception:
                try:
                    await self._documents.enqueue_media_deletion_intent(session, image_key)
                except Exception:
                    await session.rollback()
                    logger.exception("Could not persist media compensation intent")
            raise
        await session.refresh(part)
        return part

    async def begin_upload(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        filename: str,
    ) -> tuple[DocumentPart | None, str, str | None, str | None]:
        """Mint a presigned upload URL, or signal multipart with a ``None`` part.

        The part row is created only once the URL exists, so a backend that cannot
        presign (the filesystem) never leaves a pending row behind. When presign is
        impossible the tuple comes back ``(None, image_key, None, None)`` and the
        caller falls back to the multipart upload; otherwise the row is committed with
        ``image_key="pending:<minted key>"`` and the URL/token are returned for the
        direct PUT. The sentinel keeps the minted key on the row so the abandoned-
        upload sweep can reap both the row and the blob of a begin whose browser
        never finalized.

        The key's suffix follows the filename: the direct path stores the browser's
        bytes exactly as sent - no transcode, no loss - so the stored object is
        whatever format the client uploaded, not a server-encoded WebP.
        """
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
        order = await self._documents.next_part_order(session, document.id)
        filename_stem: str | None = None
        suffix = DEFAULT_PART_IMAGE_SUFFIX
        if "." in filename:
            filename_stem, suffix = filename.rsplit(".", 1)
        part = DocumentPart(
            document_id=document.id,
            order=order,
            image_key="pending",
        )
        session.add(part)
        await session.flush()
        image_key = self._media.part_image_key(
            part.id,
            suffix=suffix,
            filename_stem=filename_stem,
        )
        try:
            upload_url, token = self._media.create_upload_url(
                image_key, expires_at=datetime.now(UTC) + timedelta(minutes=5)
            )
        except PresignUnsupported:
            # The local filesystem backend cannot presign. Roll the part back so no
            # pending row is orphaned and signal the caller to use multipart.
            await session.rollback()
            return None, image_key, None, None
        part.image_key = f"pending:{image_key}"
        await session.commit()
        return part, image_key, upload_url, token

    async def finalize_upload(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        image_key: str,
        width: int | None = None,
        height: int | None = None,
    ) -> DocumentPart:
        """Seal a direct upload: verify the blob and persist its key and dimensions.

        The browser-supplied dimensions are trusted only as a hint. The stored blob is
        decoded once here - the same bounded decode multipart uploads run - so a client
        cannot fabricate geometry or smuggle a non-image past the media store. The
        server's read is authoritative; a disagreement is logged rather than rejected,
        because the cost of a mismatch is a wrong thumbnail for one request, not a
        wrong stored image.
        """
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        part = context.part
        if not part.image_key.startswith("pending"):
            raise ValidationError("Part upload has already been finalized")
        # The key is client-supplied. The media store refuses a malformed one with
        # ``ValueError``, and letting that surface through the read below would both
        # misreport it as "not a valid image" and run the delete/compensation ladder
        # against an attacker-controlled string. Reject it before touching storage.
        try:
            validate_image_key(image_key)
        except ValueError as exc:
            raise ValidationError("image_key is not a valid media key") from exc
        # The key must be the one this part's own begin minted. Without this check
        # any project member could seal a *foreign* part's key onto their row, and
        # deleting either row would then destroy the other document's image via the
        # shared blob's deletion intent. Rows begun since the sentinel carry the
        # minted key verbatim; the prefix check covers rows begun before it.
        if part.image_key.startswith("pending:"):
            if part.image_key != f"pending:{image_key}":
                raise ValidationError("image_key does not belong to this part")
        elif not image_key.startswith((f"parts/{part.id}.", f"parts/{part.id}/")):
            raise ValidationError("image_key does not belong to this part")
        try:
            true_width, true_height = await asyncio.to_thread(
                self._read_part_image_size_with_key, image_key
            )
        except FileNotFoundError as exc:
            raise ValidationError("Uploaded image is missing from storage") from exc
        except ValidationError:
            await self._discard_rejected_blob(session, image_key)
            raise
        except Exception as exc:
            await self._discard_rejected_blob(session, image_key)
            raise ValidationError("Uploaded file is not a valid image") from exc

        if (
            width is not None
            and height is not None
            and (width, height)
            != (
                true_width,
                true_height,
            )
        ):
            logger.warning(
                "Part %s dimensions disagreed: client (%d, %d) vs stored (%d, %d)",
                part.id,
                width,
                height,
                true_width,
                true_height,
            )
        part.image_key = image_key
        part.width = true_width
        part.height = true_height
        await session.commit()
        await session.refresh(part)
        return part

    def _read_part_image_size_with_key(self, image_key: str) -> tuple[int, int]:
        # Reject an over-cap blob from storage metadata *before* downloading it.
        # A direct upload PUTs straight to storage past the API's body cap, so
        # without this check any project member could point finalize at a
        # multi-GB object and force the API process to buffer the whole thing
        # into memory. The post-read check stays as a backstop for backends
        # whose metadata size is unavailable.
        if self._media.size(image_key) > MAX_PART_UPLOAD_BYTES:
            raise ValidationError("Uploaded image exceeds the maximum allowed size")
        data = self._media.read(image_key)
        if len(data) > MAX_PART_UPLOAD_BYTES:
            raise ValidationError("Uploaded image exceeds the maximum allowed size")
        return read_image_size(data)

    async def _discard_rejected_blob(self, session: AsyncSession, image_key: str) -> None:
        """Delete a directly-uploaded blob that finalize refused to seal.

        Same compensation ladder as :meth:`upload_part`: without it a rejected
        direct upload stays in storage forever, referenced by nothing.
        """
        try:
            await asyncio.to_thread(self._media.delete, image_key)
        except Exception:
            try:
                await self._documents.enqueue_media_deletion_intent(session, image_key)
            except Exception:
                await session.rollback()
                logger.exception("Could not persist media compensation intent")

    async def backfill_part_dimensions(
        self,
        session: AsyncSession,
        parts: list[DocumentPart],
    ) -> None:
        """Fill width/height for parts stored before dimensions were persisted.

        Dimensions only exist inside the stored blob, so migration 004 cannot backfill
        them in SQL. Read paths that expose dimensions decode the stored image once and
        persist the result; every later read is served from Postgres. Storage failures are
        swallowed: a missing blob must not break listing the rest of the document.
        """
        # A pending row has no stored blob yet (its key is the ``pending`` sentinel),
        # so trying to backfill it would only log a spurious warning per listing.
        pending = [
            part
            for part in parts
            if (part.width is None or part.height is None)
            and not part.image_key.startswith("pending")
        ]
        if not pending:
            return
        changed = False
        for part in pending[:MAX_DIMENSION_BACKFILLS_PER_REQUEST]:
            try:
                size = await asyncio.to_thread(self._read_part_image_size, part)
            except Exception:
                logger.warning("Could not recover dimensions for part %s", part.id, exc_info=True)
                continue
            part.width, part.height = size
            changed = True
        if changed:
            await session.commit()

    def _read_part_image_size(self, part: DocumentPart) -> tuple[int, int]:
        return read_image_size(self._media.read(part.image_key))

    async def reorder_parts(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        ordered_part_ids: list[UUID],
    ) -> list[DocumentPart]:
        context = await self._access.require_document(session, user, project_id, document_id)
        parts = await self._documents.reorder_parts(session, context.document, ordered_part_ids)
        if not parts:
            raise ValidationError("part_ids must match all parts on the document")
        return parts

    async def update_parts_published(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        updates: list[tuple[UUID, bool]],
    ) -> list[DocumentPart]:
        """Bulk per-part publish flag, gated the same way publishing the document is.

        One call for the whole batch rather than N single-part PATCHes: a chapter going
        live is usually "these dozen pages, not those three yet", one UI action and one
        commit, not a request per checkbox. Each id is validated against ``document`` the
        same way any other part write is - via ``part_in_document`` - so naming a
        foreign part's id here cannot flip a page on a document the caller does not own.
        """
        context = await self._access.require_document(session, user, project_id, document_id)
        project, document = context.project, context.document
        # Same rule as publishing the document itself: this is exposure, not editing,
        # so it is not a decision any collaborator gets to make on the owner's behalf.
        if not is_owner(project, user.id):
            raise AccessDeniedError("Only the project owner can change what is published")
        # Resolved against the parts ``require_document`` already eager-loaded, not one
        # ``get_part`` per entry: that reload pulls every line and transcription with it,
        # and the request accepts up to ``MAX_PART_IDS_PER_REQUEST`` entries with no rule
        # against repeating one. Two thousand copies of a single id measured 7.3s and
        # roughly eight thousand round trips on one pooled connection. ``reorder_parts``
        # next door already validates its id set up front; this now does the same.
        by_id = {part.id: part for part in document.parts}
        settled: dict[UUID, bool] = {}
        for part_id, published in updates:
            if part_id not in by_id:
                raise NotFoundError("Part not found")
            # Last value wins, the way a repeated field in any other payload would.
            settled[part_id] = published
        for part_id, published in settled.items():
            by_id[part_id].published = published
        await session.commit()
        return sorted(document.parts, key=lambda p: p.order)

    async def update_part_review_status(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        reviewed: bool,
    ) -> DocumentPart:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        part = context.part
        part.reviewed = reviewed
        # Who and when, written and cleared with the flag itself so the three can
        # never disagree. Unmarking clears them rather than leaving the last
        # reviewer behind: a page that is not reviewed has no reviewer, and keeping
        # a stale name there would read as one.
        part.reviewed_at = datetime.now(UTC) if reviewed else None
        part.reviewed_by = user.id if reviewed else None
        await session.commit()
        await session.refresh(part)
        return part

    async def delete_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> None:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        await self._documents.delete_part(session, context.part)

    async def get_part_for_media(
        self,
        session: AsyncSession,
        user: User,
        part_id: UUID,
    ) -> DocumentPart:
        context = await self._access.require_part_by_id(session, user, part_id)
        return context.part

    async def get_part_for_public_media(
        self,
        session: AsyncSession,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> DocumentPart:
        context = await self._access.require_part_by_id(session, None, part_id, token=token)
        return context.part

    async def read_part_bytes(self, part: DocumentPart, *, width: int | None = None) -> bytes:
        """Read and optionally transform media without blocking the event loop."""
        return await asyncio.to_thread(self._read_part_bytes, part, width)

    def _read_part_bytes(self, part: DocumentPart, width: int | None) -> bytes:
        if width is None:
            return self._read_original(part)

        # A persisted rendering is read *instead of* the original: that is the whole
        # saving. Rendering needs the full page scan, so the first request for a
        # width pays for it once and writes the result back next to the original;
        # every later process reads kilobytes rather than megabytes.
        derived_key = persisted_thumbnail_key(part.image_key, width)
        if derived_key is not None:
            try:
                return self._media.read(derived_key)
            except (ValueError, FileNotFoundError):
                pass
            except Exception:
                # A storage hiccup on the small read must not hide the page: fall
                # through to rendering from the original, which raises on its own.
                logger.warning(
                    "persisted thumbnail unreadable, rendering key=%s", derived_key, exc_info=True
                )

        encoded = encode_part_thumbnail(self._read_original(part), width)
        if derived_key is not None:
            try:
                self._media.write(derived_key, encoded)
            except Exception:
                # Best effort. The response is already in hand; a failed write only
                # means the next process renders again.
                logger.warning("could not persist thumbnail key=%s", derived_key, exc_info=True)
        return encoded

    def _read_original(self, part: DocumentPart) -> bytes:
        try:
            return self._media.read(part.image_key)
        except (ValueError, FileNotFoundError):
            raise NotFoundError("Part image not found") from None

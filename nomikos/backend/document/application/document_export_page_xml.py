"""Every page of one document as PAGE XML plus its full-resolution image, in one zip.

The per-page bundle route already answers "this page as XML next to the pixels it
describes". A chapter export is that, N times, in one archive, so the rendering is
reused rather than restated: :class:`_PositionalPageXmlExport` below is the same service
the per-page route calls, with one method overridden.

Three things differ at document scale, all deliberate:

* entries are named by page position (``0001.xml`` next to ``0001.webp``) rather than by
  document title. Inside one archive the position is what tells pages apart, the title
  is the same on all of them, and PAGE's ``imageFilename`` has to keep naming the
  sibling it actually ships with - which is why the name is changed at the *stem*, where
  the XML, the image and the link between them are all derived from it, and not by
  renaming zip entries afterwards.
* the archive is written to the wire as each page is rendered, not assembled whole and
  then sent. An eighteen-page chapter of manuscript scans is tens of megabytes; holding
  all of it to hand back one ``bytes`` is the difference between a request that costs
  one page of memory and one that costs a chapter.
* the archive carries a ``manifest.json``. This is the archive a training set is built
  from, and a folder of numbered XML files does not say which document it came from,
  which pages were in it, or which of its baselines this platform invented rather than
  measured. The manifest is the last entry written, once every page's counts are known,
  and it states the "corrected" rule it was produced under rather than leaving the
  reader to infer it from the file list.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.page_xml_export_service import (
    PAGE_XML_NAMESPACE,
    PageXmlBundle,
    PageXmlExportService,
)
from backend.annotation.application.page_xml_geometry import BASELINE_HEIGHT_FRACTION
from backend.document.application.document_export_selection import (
    PAGE_NUMBER_DIGITS,
    DocumentExportSelection,
    DocumentExportSelector,
    document_download_name,
)
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Document, DocumentPart
from backend.users.infrastructure.orm_models import User

#: The archive entry that says what the other entries are. Named for what it is rather
#: than for the document, so a script sweeping a folder of these archives looks for one
#: name.
MANIFEST_FILENAME = "manifest.json"

#: Bumped when the manifest's shape changes in a way a reader has to notice.
MANIFEST_VERSION = 1

#: What this export means by a page a human has corrected, stated in the manifest so
#: that a training set carries its own definition.
#:
#: Deliberately the conservative half of what the data model could support. ``reviewed``
#: is a flag somebody set on purpose from the editor, which is the only signal here that
#: means "I have looked at this page and I am done with it"; ground-truth text on at
#: least one line is the evidence that the looking produced something. Neither
#: ``manual_geometry`` nor pairing qualifies on its own: a page can have hand-drawn boxes
#: because somebody started and stopped, and pairing an imported transcription to lines
#: is a step on the way to ground truth, not a signature under it.
CORRECTED_RULE = (
    "part.reviewed is true and the page has at least one line carrying ground-truth text"
)


@dataclass(frozen=True)
class DocumentZipExport:
    """A named archive that has not been rendered yet.

    ``chunks`` is a generator that has not been started, so nothing is read from storage
    until the response body is written. Everything that can legitimately fail with a
    status code - membership, a document that is not there, an export with no pages -
    has already happened by the time this exists.
    """

    filename: str
    chunks: AsyncIterator[bytes]


class DocumentPageXmlExportService:
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
        selector: DocumentExportSelector | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService()
        self._selector = selector or DocumentExportSelector(documents=self._documents)

    async def export_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        reviewed_only: bool = False,
    ) -> DocumentZipExport:
        """Authorize and pick the pages now; render them as the response is written.

        The split matters: once the first chunk is on the wire the status line is spent,
        so a caller who may not read this document, or who asks for reviewed pages when
        none are reviewed, has to be turned away before the stream opens.
        """
        selection = await self._selector.select(
            session, user, project_id, document_id, reviewed_only=reviewed_only
        )
        return DocumentZipExport(
            filename=document_download_name(selection.document.name, "page-xml.zip"),
            chunks=self._stream(session, selection, reviewed_only=reviewed_only),
        )

    async def _stream(
        self,
        session: AsyncSession,
        selection: DocumentExportSelection,
        *,
        reviewed_only: bool,
    ) -> AsyncIterator[bytes]:
        renderer = _PositionalPageXmlExport(
            selection.page_numbers,
            documents=self._documents,
            document_service=self._document_service,
        )
        sink = _ZipSink()
        pages: list[dict[str, object]] = []
        with ZipFile(sink, "w") as archive:
            for part in selection.parts:
                bundle = await renderer.render_bundle(session, selection.document, part)
                # Off the event loop, per page, for the same reason the single-page
                # bundle route does it: deflating the XML and copying a full manuscript
                # scan into the archive is CPU work that would otherwise stall every
                # other in-flight request.
                await asyncio.to_thread(_write_bundle, archive, bundle)
                pages.append(_manifest_page(selection, part, bundle))
                chunk = sink.drain()
                if chunk:
                    yield chunk
            archive.writestr(
                MANIFEST_FILENAME,
                json.dumps(
                    _manifest(selection, pages, reviewed_only=reviewed_only),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                compress_type=ZIP_DEFLATED,
            )
        # The central directory, written when the archive closed.
        yield sink.drain()


class _PositionalPageXmlExport(PageXmlExportService):
    """The per-page bundle renderer, named by page position instead of by title.

    Subclassed rather than copied. Only the stem the files carry differs from the
    single-page download, so overriding the one method that decides it leaves the XML,
    the full-resolution image read and the ``imageFilename`` link between them identical
    to what the per-page route already serves and already has tests for.
    """

    def __init__(
        self,
        page_numbers: dict[UUID, int],
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        super().__init__(documents=documents, document_service=document_service)
        self._page_numbers = page_numbers

    async def render_bundle(
        self, session: AsyncSession, document: Document, part: DocumentPart
    ) -> PageXmlBundle:
        """The bundle for one already-authorized part.

        The inherited entry points authorize a part at a time. A document export has
        authorized the whole document once, and re-deriving the same answer per page
        would be N project loads for one decision that has already been made.
        """
        return await self._render_bundle(session, document, part)

    async def _stem(self, session: AsyncSession, document: Document, part: DocumentPart) -> str:
        return f"{self._page_numbers[part.id]:0{PAGE_NUMBER_DIGITS}d}"


def _manifest_page(
    selection: DocumentExportSelection, part: DocumentPart, bundle: PageXmlBundle
) -> dict[str, object]:
    counts = bundle.line_counts
    return {
        "page_number": selection.page_number(part),
        "part_id": str(part.id),
        "xml": bundle.xml_filename,
        "image": bundle.image_filename,
        "reviewed": bool(part.reviewed),
        "corrected": bool(part.reviewed) and counts.ground_truth > 0,
        "lines": counts.as_dict(),
    }


def _manifest(
    selection: DocumentExportSelection,
    pages: list[dict[str, object]],
    *,
    reviewed_only: bool,
) -> dict[str, object]:
    """What the archive holds, and under which rule.

    ``corrected_pages`` is reported rather than enforced. ``reviewed_only`` is the
    filter three document exports share, and narrowing it here alone would make the
    PAGE XML, the PDF and the text export disagree about what a reviewed chapter is.
    A reader that wants only corrected pages has, in this file, everything it needs
    to drop the rest without a second request.
    """
    return {
        "manifest_version": MANIFEST_VERSION,
        "generator": "Nomikos",
        "created": datetime.now(UTC).isoformat(),
        "page_xml_namespace": PAGE_XML_NAMESPACE,
        "document": {
            "id": str(selection.document.id),
            "name": selection.document.name,
        },
        "reviewed_only": reviewed_only,
        "corrected_rule": CORRECTED_RULE,
        "derived_baseline": {
            "height_fraction": BASELINE_HEIGHT_FRACTION,
            "marked_in_xml_as": "TextLine/@custom contains 'baseline:derived'",
        },
        "page_count": len(pages),
        "corrected_page_count": sum(1 for page in pages if page["corrected"]),
        "pages": pages,
    }


def _write_bundle(archive: ZipFile, bundle: PageXmlBundle) -> None:
    """One page into the open archive: the XML deflated, the image stored.

    The same trade the single-page bundle makes - re-compressing an already-compressed
    raster costs CPU for no size win - held here where it matters more, because a
    document export pays it once per page.
    """
    archive.writestr(bundle.xml_filename, bundle.xml, compress_type=ZIP_DEFLATED)
    archive.writestr(bundle.image_filename, bundle.image, compress_type=ZIP_STORED)


class _ZipSink:
    """A write-only file object :class:`ZipFile` can target, drained between entries.

    ``ZipFile`` needs somewhere to put bytes and a ``tell`` to place its offsets. It does
    not need to seek, as long as it can find that out: with ``tell`` present and ``seek``
    absent it marks itself non-seekable and writes the trailing data descriptors that a
    streamed archive is supposed to carry.

    Holding only the bytes written since the last drain is the whole point. Peak memory
    is one page, not the chapter.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._position = 0

    def write(self, data: bytes) -> int:
        chunk = bytes(data)
        self._chunks.append(chunk)
        self._position += len(chunk)
        return len(chunk)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        """Never called - ``ZipFile`` does not close a file object it was handed.

        Present because the writable-file protocol asks for it, and because a sink that
        silently swallowed a close would be worse than one that says there is nothing
        to release: the bytes have already gone out.
        """
        return None

    def tell(self) -> int:
        return self._position

    def drain(self) -> bytes:
        chunks, self._chunks = self._chunks, []
        return b"".join(chunks)

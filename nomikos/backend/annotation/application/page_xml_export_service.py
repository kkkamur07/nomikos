"""PAGE XML export for page geometry and approved transcription.

Two shapes come out of here. The bare XML is what an API caller who already
holds the page image wants. The *bundle* is what a person clicking "download"
wants: a zip of the XML next to the full-resolution page image it describes,
with the XML's ``imageFilename`` pointing at that sibling file, so Transkribus,
eScriptorium, Aletheia and friends open it as one unit with nothing to relink.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from io import BytesIO
from typing import NamedTuple
from uuid import UUID
from xml.etree.ElementTree import Element, QName, SubElement, register_namespace, tostring
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import read_image_size
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    TranscriptionKind,
)
from backend.users.infrastructure.orm_models import User

PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
PAGE_XML_SCHEMA = (
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
)
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"

register_namespace("", PAGE_XML_NAMESPACE)
register_namespace("xsi", XSI_NAMESPACE)

# Image suffixes a stored part may carry. Anything else is written out as-is but
# under a neutral extension so the zip never pretends to know what it holds.
_IMAGE_SUFFIXES = frozenset({"png", "jpg", "gif", "webp"})


class PageXmlBundle(NamedTuple):
    """A page's PAGE XML and the image it describes, named to sit side by side."""

    stem: str
    xml: bytes
    image_filename: str
    image: bytes

    @property
    def xml_filename(self) -> str:
        return f"{self.stem}.xml"

    @property
    def zip_filename(self) -> str:
        return f"{self.stem}.zip"

    def to_zip(self) -> bytes:
        """Zip both files at the archive root.

        The XML deflates well; the image is already a compressed raster, so it is
        stored rather than deflated - re-compressing a lossless WebP costs CPU for
        no size win, and this runs per download.
        """
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr(self.xml_filename, self.xml, compress_type=ZIP_DEFLATED)
            archive.writestr(self.image_filename, self.image, compress_type=ZIP_STORED)
        return buffer.getvalue()


class PageXmlExportService:
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService()

    # --- Member routes ---

    async def export_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> bytes:
        document, part = await self._member_part(session, user, project_id, document_id, part_id)
        return await self._render_xml(session, document, part)

    async def export_part_bundle(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> PageXmlBundle:
        document, part = await self._member_part(session, user, project_id, document_id, part_id)
        return await self._render_bundle(session, document, part)

    # --- Public routes ---

    async def export_part_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> bytes:
        context = await self._document_service.get_published_part_context(
            session, project_id, document_id, part_id, token=token
        )
        return await self._render_xml(session, context.document, context.part)

    async def export_part_bundle_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> PageXmlBundle:
        context = await self._document_service.get_published_part_context(
            session, project_id, document_id, part_id, token=token
        )
        return await self._render_bundle(session, context.document, context.part)

    # --- Shared ---

    async def _member_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> tuple[Document, DocumentPart]:
        document = await self._document_service.get_document(session, user, project_id, document_id)
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")
        return document, part

    async def _render_xml(
        self, session: AsyncSession, document: Document, part: DocumentPart
    ) -> bytes:
        stem = await self._stem(session, document, part)
        lines = await self._documents.list_part_lines(session, part.id)
        return self._export_part_lines(
            part, lines, image_filename=_image_filename(stem, part), image_size=None
        )

    async def _render_bundle(
        self, session: AsyncSession, document: Document, part: DocumentPart
    ) -> PageXmlBundle:
        stem = await self._stem(session, document, part)
        image_filename = _image_filename(stem, part)
        lines = await self._documents.list_part_lines(session, part.id)
        # The full-resolution stored object, not a thumbnail: ``width=None`` is the
        # same read the editor canvas makes, so the coordinates in the XML land on
        # exactly the pixels in the zip.
        image = await self._document_service.parts.read_part_bytes(part, width=None)
        image_size: tuple[int, int] | None = None
        if part.width is None or part.height is None:
            # PAGE requires imageWidth/imageHeight. A legacy part without persisted
            # dimensions already has its bytes in hand here, so read the header rather
            # than ship an XML that schema validators reject.
            image_size = await asyncio.to_thread(read_image_size, image)
        xml = self._export_part_lines(
            part, lines, image_filename=image_filename, image_size=image_size
        )
        return PageXmlBundle(stem=stem, xml=xml, image_filename=image_filename, image=image)

    async def _stem(self, session: AsyncSession, document: Document, part: DocumentPart) -> str:
        page_number = await self._documents.part_page_number(session, part)
        return export_file_stem(document.name, page_number)

    def _export_part_lines(
        self,
        part: DocumentPart,
        lines: list[Line],
        *,
        image_filename: str,
        image_size: tuple[int, int] | None,
    ) -> bytes:
        root = Element(
            _page_tag("PcGts"),
            {
                _xsi_tag("schemaLocation"): PAGE_XML_SCHEMA,
                "pcGtsId": f"pcgts_{part.id}",
            },
        )
        metadata = SubElement(root, _page_tag("Metadata"))
        SubElement(metadata, _page_tag("Creator")).text = "Nomikos"
        SubElement(metadata, _page_tag("Created")).text = datetime.now(UTC).isoformat()

        width, height = image_size if image_size is not None else (part.width, part.height)
        page_attrs = {"imageFilename": image_filename}
        if width is not None:
            page_attrs["imageWidth"] = str(width)
        if height is not None:
            page_attrs["imageHeight"] = str(height)
        page = SubElement(root, _page_tag("Page"), page_attrs)
        region = SubElement(
            page,
            _page_tag("TextRegion"),
            {
                "id": f"region_{part.id}",
                "type": "paragraph",
            },
        )

        for line in lines:
            self._append_text_line(region, line)

        return tostring(root, encoding="utf-8", xml_declaration=True)

    def _append_text_line(self, region: Element, line: Line) -> None:
        text_line = SubElement(
            region,
            _page_tag("TextLine"),
            {
                "id": f"line_{line.id}",
                "custom": f"source:{line.source.value}; kind:{line.kind.value}",
            },
        )
        SubElement(text_line, _page_tag("Coords"), {"points": _points(line.points)})

        baseline_points = _baseline_export_points(line)
        if baseline_points:
            SubElement(text_line, _page_tag("Baseline"), {"points": _points(baseline_points)})

        text_equiv = SubElement(text_line, _page_tag("TextEquiv"))
        SubElement(text_equiv, _page_tag("Unicode")).text = _ground_truth_text(line)


def export_file_stem(document_name: str, page_number: int) -> str:
    """``<Document_name>_page_<n>``: the stem every file of one page export shares.

    The same recipe the web client uses for the file it saves, so the archive, the XML
    and the image inside it all read as one thing on disk. Characters no common
    filesystem accepts are dropped, runs of whitespace collapse to one underscore, and
    the document part is capped so a long title cannot push the name past path limits.
    Non-ASCII letters stay: a Greek title is a Greek title.
    """
    safe = _UNSAFE_FILENAME_CHARS.sub("", document_name).strip()
    safe = re.sub(r"\s+", "_", safe)[:_MAX_STEM_DOCUMENT_CHARS].rstrip("._")
    return f"{safe or 'document'}_page_{page_number}"


# Path separators, the characters Windows refuses, and the Unicode Cc (control) block.
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f-\x9f]+')
_MAX_STEM_DOCUMENT_CHARS = 80


def _image_filename(stem: str, part: DocumentPart) -> str:
    """The name the page image carries next to the XML.

    A basename, never the storage key: PAGE's ``imageFilename`` is resolved relative
    to the XML file by every consumer, and the storage key is an internal address
    that means nothing outside this platform.
    """
    suffix = part.image_key.rsplit(".", 1)[-1].lower() if "." in part.image_key else ""
    if suffix == "jpeg":
        suffix = "jpg"
    return f"{stem}.{suffix if suffix in _IMAGE_SUFFIXES else 'img'}"


def _page_tag(name: str) -> str:
    return str(QName(PAGE_XML_NAMESPACE, name))


def _xsi_tag(name: str) -> str:
    return str(QName(XSI_NAMESPACE, name))


def _points(points: list[list[float]]) -> str:
    return " ".join(f"{_format_coord(point[0])},{_format_coord(point[1])}" for point in points)


def _format_coord(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _baseline_export_points(line: Line) -> list[list[float]]:
    """Prefer mask polygon for PAGE Baseline; fall back to stored baseline or coords."""
    mask_points = _geometry_points(line.mask)
    if mask_points:
        return mask_points
    baseline_points = _geometry_points(line.baseline)
    if baseline_points:
        return baseline_points
    return [[float(point[0]), float(point[1])] for point in line.points]


def _geometry_points(geometry: dict | None) -> list[list[float]]:
    if not geometry:
        return []
    points = geometry.get("points")
    if not isinstance(points, list):
        return []
    return [
        point
        for point in points
        if isinstance(point, list) and len(point) == 2 and _is_number_pair(point)
    ]


def _is_number_pair(point: list[object]) -> bool:
    return all(isinstance(value, (int, float)) for value in point)


def _ground_truth_text(line: Line) -> str:
    """The approved text with its edge whitespace removed.

    A trainer reads this back as the label for the crop and learns it verbatim,
    edge spaces included: the Greek model emits a trailing space on lines whose
    June export carried one. Nothing in a manuscript line begins or ends with a
    space, so the export is where it stops (ADR 0007).
    """
    for transcription in line.transcriptions:
        if transcription.transcription.kind == TranscriptionKind.ground_truth:
            return transcription.text.strip()
    return ""

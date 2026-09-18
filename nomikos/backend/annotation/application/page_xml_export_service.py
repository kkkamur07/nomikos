"""PAGE XML export for page geometry and approved transcription.

Two shapes come out of here. The bare XML is what an API caller who already
holds the page image wants. The *bundle* is what a person clicking "download"
wants: a zip of the XML next to the full-resolution page image it describes,
with the XML's ``imageFilename`` pointing at that sibling file, so Transkribus,
eScriptorium, Aletheia and friends open it as one unit with nothing to relink.

The geometry the two elements carry is resolved in
:mod:`backend.annotation.application.page_xml_geometry`: ``Coords`` is the line
polygon, ``Baseline`` is a left-to-right polyline, and a line with no stored
baseline gets one derived and says so in its ``custom`` string. Until 2026-09-18
this file wrote the mask polygon into ``Baseline``, which made every export
unusable as segmenter training data (docs/inference/training-data-loop.md).
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

from backend.annotation.application.page_xml_geometry import resolve_line_geometry
from backend.core.exceptions import NotFoundError
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import read_image_size
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    Line,
    LineSource,
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


class PageLineCounts(NamedTuple):
    """What one exported page is made of, for a training set's manifest.

    ``derived_baseline`` is the number that decides whether a page is worth
    training a *segmenter* on: those lines carry a baseline this platform invented
    from the polygon, not one anybody measured or corrected.
    """

    total: int = 0
    kraken: int = 0
    model: int = 0
    manual: int = 0
    manual_geometry: int = 0
    derived_baseline: int = 0
    ground_truth: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self._asdict())


class PageXmlBundle(NamedTuple):
    """A page's PAGE XML and the image it describes, named to sit side by side."""

    stem: str
    xml: bytes
    image_filename: str
    image: bytes
    line_counts: PageLineCounts = PageLineCounts()

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
        blocks = await self._documents.list_part_blocks(session, part.id)
        lines = await self._documents.list_part_lines(session, part.id)
        xml, _ = self._export_part_lines(
            part, blocks, lines, image_filename=_image_filename(stem, part), image_size=None
        )
        return xml

    async def _render_bundle(
        self, session: AsyncSession, document: Document, part: DocumentPart
    ) -> PageXmlBundle:
        stem = await self._stem(session, document, part)
        image_filename = _image_filename(stem, part)
        blocks = await self._documents.list_part_blocks(session, part.id)
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
        xml, counts = self._export_part_lines(
            part, blocks, lines, image_filename=image_filename, image_size=image_size
        )
        return PageXmlBundle(
            stem=stem,
            xml=xml,
            image_filename=image_filename,
            image=image,
            line_counts=counts,
        )

    async def _stem(self, session: AsyncSession, document: Document, part: DocumentPart) -> str:
        page_number = await self._documents.part_page_number(session, part)
        return export_file_stem(document.name, page_number)

    def _export_part_lines(
        self,
        part: DocumentPart,
        blocks: list[Block],
        lines: list[Line],
        *,
        image_filename: str,
        image_size: tuple[int, int] | None,
    ) -> tuple[bytes, PageLineCounts]:
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

        grouped = _group_lines_by_region(part, blocks, lines)
        # PAGE 2019-07-15 fixes the order of ``Page``'s children: ``ReadingOrder``
        # comes before any region, so it is written first even though the regions
        # it refers to do not exist yet.
        if len(grouped) > 1:
            _append_reading_order(page, [region_id for region_id, _ in grouped])

        counts = PageLineCounts()
        for region_id, region_lines in grouped:
            region = SubElement(
                page,
                _page_tag("TextRegion"),
                {"id": region_id, "type": "paragraph"},
            )
            for line in region_lines:
                counts = self._append_text_line(region, line, counts)

        return tostring(root, encoding="utf-8", xml_declaration=True), counts

    def _append_text_line(
        self, region: Element, line: Line, counts: PageLineCounts
    ) -> PageLineCounts:
        geometry = resolve_line_geometry(line)
        custom = f"source:{line.source.value}; kind:{line.kind.value}"
        if geometry.baseline_derived:
            # The one claim in this file a trainer must be able to distrust. A
            # derived baseline is a guess from the polygon, so it is labelled here
            # rather than left to look like a measurement.
            custom = f"{custom}; baseline:derived"
        text_line = SubElement(
            region,
            _page_tag("TextLine"),
            {"id": f"line_{line.id}", "custom": custom},
        )
        SubElement(text_line, _page_tag("Coords"), {"points": _points(geometry.coords)})
        if geometry.baseline:
            SubElement(text_line, _page_tag("Baseline"), {"points": _points(geometry.baseline)})

        text = _ground_truth_text(line)
        text_equiv = SubElement(text_line, _page_tag("TextEquiv"))
        SubElement(text_equiv, _page_tag("Unicode")).text = text

        return PageLineCounts(
            total=counts.total + 1,
            kraken=counts.kraken + (line.source is LineSource.kraken),
            model=counts.model + (line.source is LineSource.model),
            manual=counts.manual + (line.source is LineSource.manual),
            manual_geometry=counts.manual_geometry + bool(getattr(line, "manual_geometry", False)),
            derived_baseline=counts.derived_baseline + geometry.baseline_derived,
            ground_truth=counts.ground_truth + bool(text),
        )


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


def _group_lines_by_region(
    part: DocumentPart, blocks: list[Block], lines: list[Line]
) -> list[tuple[str, list[Line]]]:
    """The ``TextRegion``\\ s to write, each with its lines in reading order.

    One region per block, because a page with two columns held in two blocks is a
    page whose columns a trainer can see. The exception is the single-block page:
    the BLLA adapter emits exactly one block covering the whole page
    (``blla_runtime.build_blla_segment_response``), so splitting there would invent
    a structure nobody drew. One block, or none, means one region.

    Lines attached to no block go into a trailing region of their own rather than
    being folded into the first one: a line without a block is a line whose place
    on the page nobody has decided, and saying so is more useful than guessing.
    """
    ordered_blocks = sorted(blocks, key=lambda block: (block.order, str(block.id)))
    if len(ordered_blocks) <= 1:
        return [(f"region_{part.id}", list(lines))]

    by_block: dict[UUID, list[Line]] = {block.id: [] for block in ordered_blocks}
    orphans: list[Line] = []
    for line in lines:
        bucket = by_block.get(line.block_id) if line.block_id is not None else None
        (bucket if bucket is not None else orphans).append(line)

    grouped = [
        (f"region_{block.id}", by_block[block.id]) for block in ordered_blocks if by_block[block.id]
    ]
    if orphans:
        grouped.append((f"region_{part.id}_unassigned", orphans))
    return grouped or [(f"region_{part.id}", [])]


def _append_reading_order(page: Element, region_ids: list[str]) -> None:
    reading_order = SubElement(page, _page_tag("ReadingOrder"))
    group = SubElement(reading_order, _page_tag("OrderedGroup"), {"id": "reading_order"})
    for index, region_id in enumerate(region_ids):
        SubElement(
            group,
            _page_tag("RegionRefIndexed"),
            {"index": str(index), "regionRef": region_id},
        )


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

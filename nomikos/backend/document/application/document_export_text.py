"""The document's transcription as plain text, page after page.

Neither of the other two document exports serves the reader who wants the words and
nothing else - to paste into an edition, to diff against a printed text, to feed a
concordance. The PDF is a picture of a page and the PAGE XML is geometry; this is the
same ground-truth text both of them carry, in reading order.

A page divided into regions (ADR 0011) prints them in region order, each labelled one
under its label on a line of its own, and the lines in no region after them.

Each page opens with a ``[p.N]`` marker and pages are separated by a blank line, so a
line found here can be traced back to the page it came off without opening the document.
``N`` is the page's position in the whole document, not its position in the export: a
``reviewed_only`` run that covers pages 2 and 5 says 2 and 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.application.document_export_selection import (
    DocumentExportSelector,
    document_download_name,
)
from backend.document.application.regions import RegionHeading, region_text
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Line, TranscriptionKind
from backend.users.infrastructure.orm_models import User

# Under the heading of a page that has no transcribed line, in every document export,
# so "not transcribed yet" cannot be mistaken for a blank folio.
NO_LINES_PLACEHOLDER = "(no transcribed lines)"


@dataclass(frozen=True)
class DocumentTextExport:
    filename: str
    text: str


class DocumentTextExportService:
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        selector: DocumentExportSelector | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._selector = selector or DocumentExportSelector(documents=self._documents)

    async def export_document_text(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        reviewed_only: bool = False,
    ) -> DocumentTextExport:
        selection = await self._selector.select(
            session, user, project_id, document_id, reviewed_only=reviewed_only
        )
        pages: list[str] = []
        for part in selection.parts:
            lines = await self._documents.list_part_lines(session, part.id)
            blocks = await self._documents.list_part_blocks(session, part.id)
            items = region_text(blocks, lines, _ground_truth_text)
            body = [item.label if isinstance(item, RegionHeading) else item for item in items]
            if all(isinstance(item, RegionHeading) for item in items):
                body.append(NO_LINES_PLACEHOLDER)
            # A page with nothing transcribed still gets its marker. The marker is the
            # thing that makes the numbering readable, and a silently absent page 7
            # would read as a document that has no page 7.
            pages.append("\n".join([f"[p.{selection.page_number(part)}]", *body]))
        return DocumentTextExport(
            filename=document_download_name(selection.document.name, "transcription.txt"),
            # Trailing newline: this is a text file, and every tool that reads one line
            # at a time expects the last line to be terminated like the others.
            text="\n\n".join(pages) + "\n",
        )


def _ground_truth_text(line: Line) -> str:
    """The line's approved text, or empty when it has none.

    The same rule the PAGE XML export applies to its ``Unicode`` element: whatever the
    ground-truth layer holds, verbatim. Trailing whitespace is left alone because in a
    diplomatic transcription it can be meaningful; a line that is *only* whitespace is
    dropped by the caller, since a blank line here would read as a page break.
    """
    for transcription in line.transcriptions:
        if transcription.transcription.kind == TranscriptionKind.ground_truth:
            return transcription.text.strip("\n")
    return ""

import { useState } from "react";
import { api, type DocumentWorkflowCounts } from "../../api/client";
import { ApiError } from "../../api/errors";
import { saveBlobAsFile } from "../../utils/downloadBlob";
import { documentExportFileStem } from "../../utils/exportFilename";
import {
  ActionMenu,
  ActionMenuCaption,
  ActionMenuDivider,
  ActionMenuItem,
  ActionMenuSection,
} from "../ui/ActionMenu";
import { toast } from "../ui/toast";
import { pageCountLabel } from "./documentActionCopy";

type ExportKind = "page-xml" | "transcription-pdf" | "text" | "docx";

type DocumentDownloadMenuProps = {
  projectId: string;
  documentId: string;
  documentName: string;
  counts: DocumentWorkflowCounts | null;
  disabled?: boolean;
};

const EXTENSION: Record<ExportKind, string> = {
  "page-xml": "zip",
  "transcription-pdf": "pdf",
  text: "txt",
  docx: "docx",
};

const FAILURE_MESSAGE: Record<ExportKind, string> = {
  "page-xml": "Could not build the PAGE XML archive",
  "transcription-pdf": "Could not build the transcription PDF",
  text: "Could not build the plain text export",
  docx: "Could not build the Word document",
};

// Method names rather than functions, so the client is looked up when the
// download starts, not when this module loads.
const EXPORT_METHOD = {
  "page-xml": "exportDocumentPageXml",
  "transcription-pdf": "exportDocumentTranscriptionPdf",
  text: "exportDocumentText",
  docx: "exportDocumentDocx",
} as const satisfies Record<ExportKind, keyof typeof api>;

/**
 * Whole-document exports.
 *
 * Every route here is Bearer-authenticated, so none of these can be a link: a
 * navigation the browser starts on its own carries no Authorization header.
 * The bytes come back through the API client, which attaches the token and can
 * refresh it, and only then become a file. See `saveBlobAsFile`.
 */
export function DocumentDownloadMenu({
  projectId,
  documentId,
  documentName,
  counts,
  disabled = false,
}: DocumentDownloadMenuProps) {
  const [downloading, setDownloading] = useState<string | null>(null);

  const total = counts?.total ?? 0;
  const reviewed = counts?.reviewed ?? 0;
  const unreviewed = Math.max(total - reviewed, 0);
  const busy = disabled || downloading !== null || counts === null;

  async function download(
    kind: ExportKind,
    reviewedOnly: boolean,
    close: () => void,
  ) {
    const ticket = `${kind}:${reviewedOnly}`;
    setDownloading(ticket);
    try {
      const blob = await api[EXPORT_METHOD[kind]](
        projectId,
        documentId,
        reviewedOnly,
      );
      saveBlobAsFile(
        blob,
        `${documentExportFileStem(documentName, reviewedOnly)}.${EXTENSION[kind]}`,
      );
      close();
    } catch (err) {
      toast.error(
        err instanceof ApiError ? err.message : FAILURE_MESSAGE[kind],
      );
    } finally {
      setDownloading(null);
    }
  }

  function meta(kind: ExportKind, count: number, ticket: string): string {
    if (downloading === ticket) return "preparing…";
    if (kind === "page-xml") return `zip, ${count}`;
    return pageCountLabel(count);
  }

  return (
    <ActionMenu label="Download" menuLabel="Download this document" wide>
      {(close) => (
        <>
          <ActionMenuSection>Whole chapter</ActionMenuSection>
          <ActionMenuItem
            label="PAGE XML + images"
            ariaLabel="PAGE XML + images, whole chapter"
            meta={meta("page-xml", total, "page-xml:false")}
            disabled={busy || total === 0}
            onSelect={() => void download("page-xml", false, close)}
          />
          <ActionMenuItem
            label="Transcription PDF"
            ariaLabel="Transcription PDF, whole chapter"
            meta={meta("transcription-pdf", total, "transcription-pdf:false")}
            disabled={busy || total === 0}
            onSelect={() => void download("transcription-pdf", false, close)}
          />
          <ActionMenuItem
            label="Plain text .txt"
            meta={downloading === "text:false" ? "preparing…" : undefined}
            disabled={busy || total === 0}
            onSelect={() => void download("text", false, close)}
          />
          <ActionMenuItem
            label="Word (.docx)"
            ariaLabel="Word (.docx), whole chapter"
            meta={downloading === "docx:false" ? "preparing…" : undefined}
            disabled={busy || total === 0}
            onSelect={() => void download("docx", false, close)}
          />
          <ActionMenuDivider />
          <ActionMenuSection>Reviewed only</ActionMenuSection>
          <ActionMenuCaption>
            Skips the {unreviewed} {unreviewed === 1 ? "page" : "pages"} nobody
            has checked.
          </ActionMenuCaption>
          <ActionMenuItem
            label="PAGE XML + images"
            ariaLabel="PAGE XML + images, reviewed pages only"
            meta={meta("page-xml", reviewed, "page-xml:true")}
            disabled={busy || reviewed === 0}
            onSelect={() => void download("page-xml", true, close)}
          />
          <ActionMenuItem
            label="Transcription PDF"
            ariaLabel="Transcription PDF, reviewed pages only"
            meta={meta("transcription-pdf", reviewed, "transcription-pdf:true")}
            disabled={busy || reviewed === 0}
            onSelect={() => void download("transcription-pdf", true, close)}
          />
          <ActionMenuItem
            label="Word (.docx)"
            ariaLabel="Word (.docx), reviewed pages only"
            meta={downloading === "docx:true" ? "preparing…" : undefined}
            disabled={busy || reviewed === 0}
            onSelect={() => void download("docx", true, close)}
          />
        </>
      )}
    </ActionMenu>
  );
}

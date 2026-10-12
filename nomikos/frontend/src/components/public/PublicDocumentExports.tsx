import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { toast } from "../ui/toast";
import { exportFileStem } from "../../utils/exportFilename";

type PublicDocumentExportsProps = {
  projectId: string;
  documentId: string;
  documentName: string;
  partId: string;
  partIndex: number;
  token: string | null;
};

export function PublicDocumentExports({
  projectId,
  documentId,
  documentName,
  partId,
  partIndex,
  token,
}: PublicDocumentExportsProps) {
  const downloadFilename = (extension: string) =>
    `${exportFileStem(documentName, partIndex)}.${extension}`;
  const [open, setOpen] = useState(false);
  const [downloading, setDownloading] = useState<"pdf" | "xml" | "docx" | null>(
    null,
  );

  useEffect(() => {
    if (!open) return;
    function handleClick(event: MouseEvent) {
      const target = event.target as HTMLElement | null;
      if (!target?.closest(".pub-exports")) {
        setOpen(false);
      }
    }
    globalThis.document.addEventListener("click", handleClick);
    return () => globalThis.document.removeEventListener("click", handleClick);
  }, [open]);

  async function handleDownloadPdf() {
    setDownloading("pdf");
    try {
      const blob = await api.getPublicTranscriptionPdf(
        projectId,
        documentId,
        partId,
        token,
      );
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement("a");
      anchor.href = url;
      anchor.download = downloadFilename("pdf");
      anchor.click();
      URL.revokeObjectURL(url);
      setOpen(false);
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Failed to download PDF";
      toast.error(message);
    } finally {
      setDownloading(null);
    }
  }

  async function handleDownloadXml() {
    setDownloading("xml");
    try {
      const blob = await api.getPublicPageXmlBundle(
        projectId,
        documentId,
        partId,
        token,
      );
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement("a");
      anchor.href = url;
      anchor.download = downloadFilename("zip");
      anchor.click();
      URL.revokeObjectURL(url);
      setOpen(false);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Failed to download PAGE XML and page image";
      toast.error(message);
    } finally {
      setDownloading(null);
    }
  }

  async function handleDownloadDocx() {
    setDownloading("docx");
    try {
      const blob = await api.getPublicDocx(
        projectId,
        documentId,
        partId,
        token,
      );
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement("a");
      anchor.href = url;
      anchor.download = downloadFilename("docx");
      anchor.click();
      URL.revokeObjectURL(url);
      setOpen(false);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Failed to download the Word document";
      toast.error(message);
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div className="pub-exports">
      <button
        type="button"
        className="btn btn-outline btn-sm pub-exports__trigger"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={(event) => {
          event.stopPropagation();
          setOpen((value) => !value);
        }}
      >
        Export
      </button>
      {open && (
        <div
          className="pub-exports__menu"
          role="menu"
          aria-label="Export options"
        >
          <button
            type="button"
            role="menuitem"
            className="pub-exports__item"
            disabled={downloading !== null}
            onClick={() => void handleDownloadPdf()}
          >
            {downloading === "pdf" ? "Downloading PDF…" : "Transcription PDF"}
          </button>
          <button
            type="button"
            role="menuitem"
            className="pub-exports__item"
            disabled={downloading !== null}
            onClick={() => void handleDownloadXml()}
          >
            {downloading === "xml" ? "Downloading XML…" : "PAGE XML + image"}
          </button>
          <button
            type="button"
            role="menuitem"
            className="pub-exports__item"
            disabled={downloading !== null}
            onClick={() => void handleDownloadDocx()}
          >
            {downloading === "docx" ? "Downloading Word…" : "Word (.docx)"}
          </button>
        </div>
      )}
    </div>
  );
}

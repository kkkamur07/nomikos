import { useState } from "react";
import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { toast } from "../ui/toast";

type PageEditorDocxButtonProps = {
  projectId: string;
  documentId: string;
  partId: string;
  downloadFilename: string;
  /** Lets the button sit in the Workflow menu as well as in a toolbar. */
  className?: string;
  /** A menu needs its children to say they are menu items; a toolbar does not. */
  role?: string;
  /** Fired before the download starts, so a menu can close behind it. */
  onActivate?: () => void;
};

/**
 * Downloads the current part's transcription as a Word file, so the page the
 * researcher is editing in the editor leaves as the document they can keep
 * editing. Unlike the transcription PDF there is nothing to preview, so this
 * is a plain download button rather than a pane toggle.
 */
export function PageEditorDocxButton({
  projectId,
  documentId,
  partId,
  downloadFilename,
  className = "pe-tb-btn",
  role,
  onActivate,
}: PageEditorDocxButtonProps) {
  const [downloading, setDownloading] = useState(false);

  async function handleDownload() {
    setDownloading(true);
    try {
      const blob = await api.exportPartDocx(projectId, documentId, partId);
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement("a");
      anchor.href = url;
      anchor.download = downloadFilename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Could not build the Word document";
      toast.error(message);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <button
      type="button"
      role={role}
      className={className}
      aria-label="Download Word document"
      title="Download the page as Word (.docx)"
      disabled={downloading}
      onClick={() => {
        onActivate?.();
        void handleDownload();
      }}
    >
      Word (.docx)
      {downloading && <span className="pe-dd-meta">preparing…</span>}
    </button>
  );
}

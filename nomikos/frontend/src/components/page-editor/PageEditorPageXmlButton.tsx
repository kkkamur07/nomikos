import { useState } from "react";
import { api } from "../../api/client";
import { userFacingMessage } from "../../api/userFacingError";
import { toast } from "../ui/toast";

type PageEditorPageXmlButtonProps = {
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
 * Downloads the current part's PAGE XML bundle: a zip holding the XML next to
 * the full-resolution page image it describes, so the export opens as one unit
 * in Transkribus, eScriptorium and the like. Unlike the transcription PDF
 * there is nothing to preview, so this is a plain download button rather than
 * a pane toggle.
 */
export function PageEditorPageXmlButton({
  projectId,
  documentId,
  partId,
  downloadFilename,
  className = "pe-tb-btn",
  role,
  onActivate,
}: PageEditorPageXmlButtonProps) {
  const [downloading, setDownloading] = useState(false);
  const label = className.includes("pe-dd-item") ? "PAGE XML + image" : "XML";

  async function handleDownload() {
    setDownloading(true);
    try {
      const blob = await api.getPageXmlBundle(projectId, documentId, partId);
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement("a");
      anchor.href = url;
      anchor.download = downloadFilename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message = userFacingMessage(
        err,
        "Failed to download PAGE XML and page image",
      );
      toast.error(message, err);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <button
      type="button"
      role={role}
      className={className}
      aria-label="Download PAGE XML with page image"
      title="Download PAGE XML with the page image (zip)"
      disabled={downloading}
      onClick={() => {
        onActivate?.();
        void handleDownload();
      }}
    >
      {downloading ? "Preparing XML…" : label}
    </button>
  );
}

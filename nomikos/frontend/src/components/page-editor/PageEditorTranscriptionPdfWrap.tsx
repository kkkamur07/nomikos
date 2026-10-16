import { PageEditorTranscriptionPdfPane } from "./PageEditorTranscriptionPdfPane";
import { usePdfPaneResize } from "./hooks/usePdfPaneResize";

type PageEditorTranscriptionPdfWrapProps = {
  projectId: string;
  documentId: string;
  partId: string;
  downloadFilename: string;
  refreshKey: number;
  onClose: () => void;
  onRefresh: () => void;
};

type PageEditorTranscriptionPdfSlotProps = Omit<
  PageEditorTranscriptionPdfWrapProps,
  "projectId" | "documentId" | "partId"
> & {
  open: boolean;
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
};

/** The PDF pane while it is open, once the page it renders is known. */
export function PageEditorTranscriptionPdfSlot({
  open,
  projectId,
  documentId,
  partId,
  ...pane
}: PageEditorTranscriptionPdfSlotProps) {
  if (!open || !projectId || !documentId || !partId) return null;
  return (
    <PageEditorTranscriptionPdfWrap
      projectId={projectId}
      documentId={documentId}
      partId={partId}
      {...pane}
    />
  );
}

function PageEditorTranscriptionPdfWrap({
  projectId,
  documentId,
  partId,
  downloadFilename,
  refreshKey,
  onClose,
  onRefresh,
}: PageEditorTranscriptionPdfWrapProps) {
  const {
    wrapRef,
    width,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel,
  } = usePdfPaneResize();

  return (
    <div className="pe-pdf-wrap" ref={wrapRef} style={{ width }}>
      <div
        className="pe-pdf-resizer"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize PDF pane"
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
      />
      <PageEditorTranscriptionPdfPane
        projectId={projectId}
        documentId={documentId}
        partId={partId}
        downloadFilename={downloadFilename}
        refreshKey={refreshKey}
        onClose={onClose}
        onRefresh={onRefresh}
      />
    </div>
  );
}

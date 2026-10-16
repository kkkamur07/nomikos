import { useRef, useState } from "react";
import {
  type DocumentWorkflow,
  type DocumentWorkflowCounts,
} from "../../api/client";
import {
  ActionMenu,
  ActionMenuConfirm,
  ActionMenuItem,
} from "../ui/ActionMenu";
import { DocumentDownloadMenu } from "./DocumentDownloadMenu";
import { DocumentPublishMenu } from "./DocumentPublishMenu";
import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

type DocumentActionBarProps = {
  projectId: string;
  documentId: string;
  documentName: string;
  workflow: DocumentWorkflow;
  publicShareToken: string | null;
  /** Null while the counts read is still in flight. */
  counts: DocumentWorkflowCounts | null;
  publishedPageCount: number;
  /**
   * Publish is owner-only in the backend, so a collaborator is not shown a
   * control whose only possible outcome is a red toast.
   */
  isOwner: boolean;
  uploading: boolean;
  /** Narration for a long batch ("Splitting scan.pdf · page 3/12"). */
  uploadProgress: string | null;
  busy: boolean;
  onUpload: (files: File[]) => void;
  onWorkflowChange: (workflow: DocumentWorkflow) => void;
  onShareTokenChange: (token: string | null) => void;
  onJobsQueued: () => void;
  onOpenSettings: () => void;
  onDeleteDocument: () => Promise<void> | void;
};

/**
 * The one action row a chapter gets.
 *
 * An action lives at the level of the thing it changes. Everything here
 * changes the whole document, so it is here and not in the page editor, which
 * keeps only the acts that change the page open in it.
 */
function DeleteConfirmStep({
  deleting,
  documentName,
  total,
  onCancel,
  onConfirm,
}: {
  deleting: boolean;
  documentName: string;
  total: number;
  onCancel: () => void;
  onConfirm: () => void;
}): React.ReactNode {
  return (
    <ActionMenuConfirm
      destructive
      busy={deleting}
      question={`Delete "${documentName}" and its ${total} pages?`}
      detail="Every page image, segmentation and transcription in this document goes with it. There is no undo."
      confirmLabel={deleting ? "Deleting…" : "Yes, delete the document"}
      onCancel={onCancel}
      onConfirm={onConfirm}
    />
  );
}

function OverflowMenuItems({
  busy,
  onSettings,
  onDelete,
}: {
  busy: boolean;
  onSettings: () => void;
  onDelete: () => void;
}): React.ReactNode {
  return (
    <>
      <ActionMenuItem
        label="Document settings"
        detail="Rename the document and see its sharing status."
        onSelect={onSettings}
      />
      <ActionMenuItem
        destructive
        label="Delete document"
        detail="Removes the document and every page in it."
        disabled={busy}
        onSelect={onDelete}
      />
    </>
  );
}

function actionCounts(counts: DocumentWorkflowCounts | null): {
  total: number;
  reviewed: number;
} {
  if (!counts) return { total: 0, reviewed: 0 };
  return { total: counts.total, reviewed: counts.reviewed };
}

export function DocumentActionBar({
  projectId,
  documentId,
  documentName,
  workflow,
  publicShareToken,
  counts,
  publishedPageCount,
  isOwner,
  uploading,
  uploadProgress,
  busy,
  onUpload,
  onWorkflowChange,
  onShareTokenChange,
  onJobsQueued,
  onOpenSettings,
  onDeleteDocument,
}: DocumentActionBarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const { total, reviewed } = actionCounts(counts);

  return (
    <div className="doc-actions" role="group" aria-label="Document actions">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,application/pdf,.pdf"
        multiple
        className="visually-hidden"
        disabled={busy || uploading}
        aria-hidden="true"
        tabIndex={-1}
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length > 0) onUpload(files);
          event.target.value = "";
        }}
      />
      <button
        type="button"
        className="btn btn-primary btn-sm"
        disabled={busy || uploading}
        onClick={() => fileInputRef.current?.click()}
      >
        {uploading ? "Uploading…" : "Upload pages"}
      </button>
      {uploading && uploadProgress && (
        // A PDF of forty sheets takes a while and becomes forty pages. The
        // button can only say that something is happening, so the narration
        // that says which sheet sits beside it.
        <span className="doc-actions__progress" role="status">
          {uploadProgress}
        </span>
      )}

      <DocumentWorkflowMenu
        projectId={projectId}
        documentId={documentId}
        counts={counts}
        disabled={busy}
        onJobsQueued={onJobsQueued}
      />

      <DocumentDownloadMenu
        projectId={projectId}
        documentId={documentId}
        documentName={documentName}
        counts={counts}
        disabled={busy}
      />

      {isOwner && (
        <DocumentPublishMenu
          projectId={projectId}
          documentId={documentId}
          workflow={workflow}
          publicShareToken={publicShareToken}
          publishedPageCount={publishedPageCount}
          totalPageCount={total}
          reviewedPageCount={reviewed}
          disabled={busy}
          onWorkflowChange={onWorkflowChange}
          onShareTokenChange={onShareTokenChange}
        />
      )}

      <ActionMenu
        label="⋯"
        triggerAriaLabel="More document actions"
        menuLabel="More document actions"
        triggerClassName="btn btn-ghost btn-sm doc-actions__overflow"
        wide
        onOpenChange={(open) => {
          if (!open) setConfirmingDelete(false);
        }}
      >
        {(close) =>
          confirmingDelete ? (
            <DeleteConfirmStep
              deleting={deleting}
              documentName={documentName}
              total={total}
              onCancel={() => setConfirmingDelete(false)}
              onConfirm={() => {
                setDeleting(true);
                void Promise.resolve(onDeleteDocument()).finally(() => {
                  setDeleting(false);
                  setConfirmingDelete(false);
                  close();
                });
              }}
            />
          ) : (
            <OverflowMenuItems
              busy={busy}
              onSettings={() => {
                close();
                onOpenSettings();
              }}
              onDelete={() => setConfirmingDelete(true)}
            />
          )
        }
      </ActionMenu>
    </div>
  );
}

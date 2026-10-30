import type { ReactNode } from "react";
import { DocumentActionBar } from "./DocumentActionBar";
import {
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type DocumentWorkflow,
  type DocumentWorkflowCounts,
} from "../../api/client";
import { DocumentSettingsPanel } from "../sharing/DocumentSettingsPanel";
import { PartList } from "./PartList";

export function DocumentHeaderActions(options: {
  projectId: string;
  documentId: string;
  document: DocumentWithPartsResponse;
  workflowCounts: DocumentWorkflowCounts | null;
  shownCount: number;
  isOwner: boolean;
  uploading: boolean;
  uploadProgress: string | null;
  reordering: boolean;
  onUpload: (files: File[]) => void;
  onWorkflowChange: (workflow: DocumentWorkflow) => void;
  onShareTokenChange: (token: string | null) => void;
  onJobsQueued: () => void;
  onOpenSettings: () => void;
  onDeleteDocument: () => void;
}): ReactNode {
  const {
    projectId,
    documentId,
    document,
    workflowCounts,
    shownCount,
    isOwner,
    uploading,
    uploadProgress,
    reordering,
    onUpload,
    onWorkflowChange,
    onShareTokenChange,
    onJobsQueued,
    onOpenSettings,
    onDeleteDocument,
  } = options;
  return (
    <DocumentActionBar
      projectId={projectId}
      documentId={documentId}
      documentName={document.name}
      workflow={document.workflow}
      publicShareToken={document.public_share_token ?? null}
      counts={workflowCounts}
      publishedPageCount={shownCount}
      isOwner={isOwner}
      uploading={uploading}
      uploadProgress={uploadProgress}
      busy={reordering}
      onUpload={(files) => onUpload(files)}
      onWorkflowChange={(workflow: DocumentWorkflow) =>
        onWorkflowChange(workflow)
      }
      onShareTokenChange={(token) => onShareTokenChange(token)}
      onJobsQueued={() => {
        // The jobs are queued, not finished; what is already stale is the
        // count of pages still waiting for one.
        onJobsQueued();
      }}
      onOpenSettings={onOpenSettings}
      onDeleteDocument={onDeleteDocument}
    />
  );
}

export function DocumentTitlePanel(options: {
  projectId: string;
  documentId: string;
  name: string;
  onRenamed: (name: string) => void;
}): ReactNode {
  const { projectId, documentId, name, onRenamed } = options;
  return (
    <DocumentSettingsPanel
      projectId={projectId}
      documentId={documentId}
      name={name}
      onUpdated={(updated) => onRenamed(updated.name)}
    />
  );
}

function pagesAside(
  parts: DocumentPartResponse[],
  shownCount: number,
  workflow: DocumentWorkflow,
): ReactNode {
  if (parts.length === 0) return null;
  return (
    <span className="section-label__aside">
      {shownCount} of {parts.length} shown publicly
      {workflow === "published"
        ? ""
        : workflow === "archived"
          ? ", though archived documents are not public"
          : " once the document is live"}
    </span>
  );
}

export function DocumentPagesSection(options: {
  parts: DocumentPartResponse[];
  projectId: string;
  documentId: string;
  loading: boolean;
  reordering: boolean;
  reviewUpdatingPartId: string | null;
  publishUpdatingPartId: string | null;
  showPublishToggle: boolean;
  workflow: DocumentWorkflow;
  onMoveUp: (index: number) => void;
  onMoveDown: (index: number) => void;
  onDelete: (partId: string) => void;
  onToggleReview: (partId: string, reviewed: boolean) => void;
  onTogglePublished: (partId: string, published: boolean) => void;
}): ReactNode {
  const {
    parts,
    projectId,
    documentId,
    loading,
    reordering,
    reviewUpdatingPartId,
    publishUpdatingPartId,
    showPublishToggle,
    workflow,
    onMoveUp,
    onMoveDown,
    onDelete,
    onToggleReview,
    onTogglePublished,
  } = options;
  const shownCount = parts.filter((part) => part.published).length;
  return (
    <>
      <p className="section-label">
        <span id="pages-label">Pages</span>
        {pagesAside(parts, shownCount, workflow)}
      </p>
      <PartList
        parts={parts}
        projectId={projectId}
        documentId={documentId}
        loading={loading}
        onMoveUp={(i) => onMoveUp(i)}
        onMoveDown={(i) => onMoveDown(i)}
        onDelete={(id) => onDelete(id)}
        onToggleReview={(partId, reviewed) => onToggleReview(partId, reviewed)}
        onTogglePublished={
          showPublishToggle
            ? (partId, published) => onTogglePublished(partId, published)
            : undefined
        }
        reviewUpdatingPartId={reviewUpdatingPartId}
        publishUpdatingPartId={publishUpdatingPartId}
        reordering={reordering}
      />
    </>
  );
}

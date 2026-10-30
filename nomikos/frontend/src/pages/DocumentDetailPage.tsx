import type { ReactNode } from "react";
import type { DocumentWorkflow } from "../api/client";
import { DocumentLiveLinkRow } from "../components/document/DocumentLiveLinkRow";
import {
  DocumentHeaderActions,
  DocumentPagesSection,
  DocumentTitlePanel,
} from "../components/document/DocumentDetailSections";
import { JobsNotice } from "../components/document/JobsNotice";
import { AppPageShell } from "../components/layout/AppPageShell";
import { ContentRegionLoading } from "../components/layout/ContentRegionLoading";
import { WorkflowBadge } from "../components/WorkflowBadge";
import {
  useDocumentDetail,
  type DocumentDetailModel,
} from "./useDocumentDetail";

const ENABLE_TEST_JOBS = process.env.NEXT_PUBLIC_ENABLE_TEST_JOBS === "true";

function PublishedLiveLink(model: DocumentDetailModel): ReactNode {
  const { document, projectId, documentId } = model;
  if (
    !document ||
    document.workflow !== "published" ||
    !projectId ||
    !documentId
  ) {
    return null;
  }
  return (
    <DocumentLiveLinkRow
      projectId={projectId}
      documentId={documentId}
      publicShareToken={document.public_share_token ?? null}
    />
  );
}

function DocumentPagesBlock(model: DocumentDetailModel): ReactNode {
  const { document, projectId, documentId } = model;
  if (!document || !projectId || !documentId) return null;
  return (
    <DocumentPagesSection
      parts={model.parts}
      projectId={projectId}
      documentId={documentId}
      loading={model.loading}
      reordering={model.reordering}
      reviewUpdatingPartId={model.reviewUpdatingPartId}
      publishUpdatingPartId={model.publishUpdatingPartId}
      showPublishToggle={model.isOwner}
      workflow={document.workflow}
      onMoveUp={(i) => model.movePart(i, -1)}
      onMoveDown={(i) => model.movePart(i, 1)}
      onDelete={model.handleDelete}
      onToggleReview={model.handleToggleReview}
      onTogglePublished={model.handleTogglePublished}
    />
  );
}

function DocumentDetailBody(model: DocumentDetailModel): ReactNode {
  const { projectId, documentId, document, loading, error, dragActive } = model;
  if (loading && !document && !error) {
    return <ContentRegionLoading label="Loading document" />;
  }
  return (
    <>
      {error && (
        <div className="notice-banner" role="alert">
          <strong>Document unavailable</strong>
          {error}
        </div>
      )}

      <PublishedLiveLink {...model} />

      {document && projectId && documentId && (
        <JobsNotice
          projectId={projectId}
          documentId={documentId}
          refreshKey={model.jobsQueuedCount}
          enableTestJobs={ENABLE_TEST_JOBS}
        />
      )}

      {dragActive && (
        <div className="drop-overlay" role="presentation">
          <div className="drop-overlay__panel">
            <p>Drop to upload</p>
            <p className="hint">
              Images become pages · a PDF becomes one page per sheet
            </p>
          </div>
        </div>
      )}

      <DocumentPagesBlock {...model} />
    </>
  );
}

function DocumentDetailHeader(model: DocumentDetailModel): ReactNode {
  const { document, projectId, documentId } = model;
  if (!document || !projectId || !documentId) return undefined;
  return (
    <DocumentHeaderActions
      projectId={projectId}
      documentId={documentId}
      document={document}
      workflowCounts={model.workflowCounts}
      shownCount={model.shownCount}
      isOwner={model.isOwner}
      uploading={model.uploading}
      uploadProgress={model.uploadProgress}
      reordering={model.reordering}
      onUpload={model.handleUpload}
      onWorkflowChange={(workflow: DocumentWorkflow) =>
        model.applyDocumentPatch({ workflow })
      }
      onShareTokenChange={(token) =>
        model.applyDocumentPatch({ public_share_token: token })
      }
      onJobsQueued={model.handleJobsQueued}
      onOpenSettings={() => model.setTitlePanelOpen(true)}
      onDeleteDocument={model.handleDeleteDocument}
    />
  );
}

function DocumentDetailTitlePanel(model: DocumentDetailModel): ReactNode {
  const { document, projectId, documentId } = model;
  if (!document || !projectId || !documentId) return null;
  return (
    <DocumentTitlePanel
      projectId={projectId}
      documentId={documentId}
      name={document.name}
      onRenamed={model.handleRenamed}
    />
  );
}

export function DocumentDetailPage() {
  const model = useDocumentDetail();
  const { projectId, document, projectName, username } = model;
  return (
    <AppPageShell
      breadcrumb={[
        { label: "Projects", href: "/projects" },
        {
          label: projectName ?? "Project",
          href: projectId ? `/projects/${projectId}` : undefined,
        },
        { label: document?.name ?? "Document" },
      ]}
      username={username}
      title={document?.name ?? "Document"}
      subtitle={model.subtitle}
      titleExtra={
        document ? <WorkflowBadge workflow={document.workflow} /> : undefined
      }
      titleEditable={Boolean(document && projectId && model.documentId)}
      titlePanelOpen={model.titlePanelOpen}
      onTitlePanelToggle={() => model.setTitlePanelOpen((open) => !open)}
      titlePanelLabel="Document settings"
      // Called as functions, not rendered: the helpers are hook-free, and the
      // shell only skips its containers for `undefined`/`null` props, which a
      // wrapper element would never be.
      headerActions={DocumentDetailHeader(model)}
      titlePanel={DocumentDetailTitlePanel(model)}
    >
      <DocumentDetailBody {...model} />
    </AppPageShell>
  );
}

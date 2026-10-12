import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { toast } from "../components/ui/toast";
import {
  api,
  type DocumentWithPartsResponse,
  type DocumentWorkflow,
  type DocumentWorkflowCounts,
} from "../api/client";
import { ApiError } from "../api/errors";
import { invalidatePartImage } from "../api/imageCache";
import { resourceTags, invalidateAfter } from "../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { DocumentActionBar } from "../components/document/DocumentActionBar";
import { relativeUpdatedLabel } from "../components/document/documentActionCopy";
import { DocumentLiveLinkRow } from "../components/document/DocumentLiveLinkRow";
import { JobsNotice } from "../components/document/JobsNotice";
import { PartList } from "../components/document/PartList";
import { AppPageShell } from "../components/layout/AppPageShell";
import { ContentRegionLoading } from "../components/layout/ContentRegionLoading";
import { DocumentSettingsPanel } from "../components/sharing/DocumentSettingsPanel";
import { WorkflowBadge } from "../components/WorkflowBadge";
import { useFileDrop } from "../hooks/useFileDrop";
import { useServerQuery } from "../hooks/useServerQuery";
import {
  prepareDirectUpload,
  type DirectUploadPayload,
} from "../utils/encodePartImage";
import { renderPdfToPageFiles } from "../utils/pdfToPages";
import { compareFilenames, isPdfFile } from "../utils/uploadBatch";

const ENABLE_TEST_JOBS = process.env.NEXT_PUBLIC_ENABLE_TEST_JOBS === "true";

/**
 * What flipping a page's visibility actually did, in the workflow it happened in.
 *
 * Three states, not two. An archived document is not public and cannot be brought
 * live from here - `DocumentPublishMenu` refuses the toggle outright, and
 * nothing else in the app sets the workflow back - so "when the document goes live"
 * would promise a moment that never arrives.
 */
function partPublishedMessage(
  workflow: DocumentWorkflow | undefined,
  published: boolean,
): string {
  if (workflow === "published") {
    return published
      ? "Page shown on the public page"
      : "Page hidden from the public page";
  }
  if (workflow === "archived") {
    return published
      ? "Page marked shown. Archived documents are not public"
      : "Page marked hidden. Archived documents are not public";
  }
  return published
    ? "Page will be shown when the document goes live"
    : "Page will stay hidden when the document goes live";
}

type DocumentDetailData = {
  username: string;
  projectName: string;
  /** Both already fetched; kept so the page can answer "may I publish?". */
  userId: string;
  /** Null on a project with no recorded owner, which is nobody's match. */
  projectOwnerId: string | null;
  document: DocumentWithPartsResponse;
};

export function DocumentDetailPage() {
  const router = useRouter();
  const { projectId, documentId } =
    useParams<{ projectId: string; documentId: string }>() ?? {};
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [reordering, setReordering] = useState(false);
  const [reviewUpdatingPartId, setReviewUpdatingPartId] = useState<
    string | null
  >(null);
  const [publishUpdatingPartId, setPublishUpdatingPartId] = useState<
    string | null
  >(null);
  const [titlePanelOpen, setTitlePanelOpen] = useState(false);
  // Bumped when the Workflow menu queues jobs, so the Jobs bar re-reads them.
  const [jobsQueuedCount, setJobsQueuedCount] = useState(0);

  const signedIn = hasAccessToken();
  useEffect(() => {
    if (projectId && documentId && !signedIn) navigateToLogin(router);
  }, [projectId, documentId, signedIn, router]);

  const {
    data,
    loading,
    error,
    refetch: reloadDocument,
    patch: patchDocument,
  } = useServerQuery<DocumentDetailData>({
    key:
      projectId && documentId && signedIn
        ? ["document-detail", projectId, documentId]
        : null,
    tags: [
      resourceTags.currentUser,
      resourceTags.project(projectId ?? ""),
      resourceTags.document(projectId ?? "", documentId ?? ""),
    ],
    read: async () => {
      const [me, project, doc] = await Promise.all([
        api.me(),
        api.getProject(projectId!),
        api.getDocument(projectId!, documentId!),
      ]);
      return {
        username: me.username,
        projectName: project.name,
        userId: me.id,
        projectOwnerId: project.owner_id,
        document: doc,
      };
    },
    onError: (err) => {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return null;
      }
      const msg =
        err instanceof ApiError ? err.message : "Failed to load document";
      toast.error(msg);
      return err instanceof ApiError &&
        (err.status === 403 || err.status === 404)
        ? "This document is not available to your account."
        : msg;
    },
  });

  /**
   * The four numbers the action menus label themselves with.
   *
   * A separate read from the document on purpose: the parts list says which
   * pages are reviewed but nothing about which have lines or a pairing, so
   * "unsegmented" and "unpaired" cannot be derived from what this page already
   * holds. It carries the document's own tag, so anything that changes the
   * pages makes these stale too.
   *
   * A failure here is silent. The counts label menu items; they are not the
   * page, and a second red banner over a document that rendered perfectly well
   * would be the loudest thing on screen for the least reason. The menus fall
   * back to disabled items, which is the honest state when nothing is known
   * about what they would act on.
   */
  const { data: workflowCounts, refetch: reloadWorkflowCounts } =
    useServerQuery<DocumentWorkflowCounts>({
      key:
        projectId && documentId && signedIn
          ? ["document-workflow-counts", projectId, documentId]
          : null,
      tags: [resourceTags.document(projectId ?? "", documentId ?? "")],
      read: () => api.getDocumentWorkflowCounts(projectId!, documentId!),
      onError: () => null,
    });

  const document = data?.document ?? null;
  const projectName = data?.projectName ?? null;
  const username = data?.username ?? null;
  // Publishing is owner-only in the backend (DocumentPartService raises for
  // anyone else), so offering the control to a collaborator would only earn
  // them a red toast.
  const isOwner = Boolean(
    data && data.userId && data.projectOwnerId === data.userId,
  );

  const parts = [...(document?.parts ?? [])].sort((a, b) => a.order - b.order);

  const uploadOnePart = async (file: File) => {
    // Prefer the direct-to-storage path: get a presigned Supabase URL, PUT the
    // bytes straight to storage, then finalize. This bypasses Vercel's 4.5 MB
    // function-body cap that a manuscript scan would hit. The payload is never
    // lossy: natively displayable formats upload as the user's original bytes,
    // everything else is transcoded to lossless PNG. If the backend cannot
    // presign (local storage) or the browser cannot decode the file at all,
    // fall back to the legacy multipart upload.
    let payload: DirectUploadPayload | undefined;
    try {
      payload = await prepareDirectUpload(file);
    } catch {
      payload = undefined;
    }

    if (payload) {
      const begin = await api.beginPartUpload(projectId!, documentId!, {
        filename: payload.filename,
        size: payload.blob.size,
      });
      if (begin.upload_url && begin.part_id) {
        const put = await fetch(begin.upload_url, {
          method: "PUT",
          body: payload.blob,
          headers: { "Content-Type": payload.contentType },
        });
        if (!put.ok) {
          throw new Error("Storage upload failed");
        }
        await api.finalizePartUpload(projectId!, documentId!, begin.part_id, {
          image_key: begin.image_key,
          width: payload.width ?? null,
          height: payload.height ?? null,
        });
        return;
      }
      // upload_url is null -> the backend is local storage; use the multipart path.
    }

    await api.uploadPart(projectId!, documentId!, file);
  };

  const handleUpload = async (files: File[]) => {
    if (!projectId || !documentId || files.length === 0) return;
    setUploading(true);
    try {
      // Page order comes from filenames; a PDF holds the slot its name sorts
      // into and expands to its pages in document order.
      const ordered = [...files].sort((a, b) =>
        compareFilenames(a.name, b.name),
      );
      const pages: File[] = [];
      const unreadable: string[] = [];
      for (const file of ordered) {
        if (!isPdfFile(file)) {
          pages.push(file);
          continue;
        }
        try {
          setUploadProgress(`Splitting ${file.name}…`);
          pages.push(
            ...(await renderPdfToPageFiles(file, (done, total) =>
              setUploadProgress(
                `Splitting ${file.name} · page ${done}/${total}`,
              ),
            )),
          );
        } catch {
          unreadable.push(file.name);
        }
      }

      let uploaded = 0;
      const failed: string[] = [];
      let firstFailure: string | null = null;
      for (const [index, file] of pages.entries()) {
        setUploadProgress(
          pages.length > 1
            ? `Uploading page ${index + 1}/${pages.length}`
            : "Uploading…",
        );
        try {
          await uploadOnePart(file);
          uploaded += 1;
        } catch (err) {
          failed.push(file.name);
          firstFailure ??=
            err instanceof ApiError ? err.message : "Upload failed";
        }
      }

      if (uploaded > 0) {
        toast.success(
          uploaded === 1 ? "Part uploaded" : `${uploaded} pages uploaded`,
        );
        invalidateAfter.documentPartsChanged(projectId, documentId);
        await reloadDocument();
      }
      for (const name of unreadable) {
        toast.error(`Could not read ${name} as a PDF`);
      }
      if (failed.length === 1) {
        toast.error(firstFailure ?? "Upload failed");
      } else if (failed.length > 1) {
        toast.error(`Upload failed for ${failed.length} pages`);
      }
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  // Whole-window drop target. The one click-to-pick alternative is the Upload
  // pages button in the action row; there used to be a second, a panel above
  // the page list that opened the same file picker with the same accept list.
  // Two controls doing one thing is not two ways in, it is a question about
  // which one is the real one.
  const dragActive = useFileDrop(
    (files) => void handleUpload(files),
    Boolean(document) && !uploading && !loading,
  );

  const persistOrder = async (partIds: string[]) => {
    if (!projectId || !documentId) return;
    setReordering(true);
    try {
      await api.reorderParts(projectId, documentId, { part_ids: partIds });
      invalidateAfter.documentPartsChanged(projectId, documentId);
      await reloadDocument();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Reorder failed";
      toast.error(msg);
    } finally {
      setReordering(false);
    }
  };

  const movePart = (index: number, direction: -1 | 1) => {
    const next = index + direction;
    if (next < 0 || next >= parts.length) return;
    const ids = parts.map((p) => p.id);
    [ids[index], ids[next]] = [ids[next], ids[index]];
    void persistOrder(ids);
  };

  const handleDelete = async (partId: string) => {
    if (!projectId || !documentId) return;
    try {
      await api.deletePart(projectId, documentId, partId);
      invalidatePartImage(partId);
      toast.success("Part removed");
      invalidateAfter.documentPartsChanged(projectId, documentId);
      await reloadDocument();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Delete failed";
      toast.error(msg);
    }
  };

  const handleToggleReview = async (partId: string, reviewed: boolean) => {
    if (!projectId || !documentId) return;
    setReviewUpdatingPartId(partId);
    try {
      await api.updatePartReviewStatus(projectId, documentId, partId, {
        reviewed,
      });
      toast.success(
        reviewed ? "Part marked reviewed" : "Part marked unreviewed",
      );
      invalidateAfter.documentPartsChanged(projectId, documentId);
      await reloadDocument();
    } catch (err) {
      const msg =
        err instanceof ApiError && err.status === 403
          ? "Only project members can change review status."
          : err instanceof ApiError
            ? err.message
            : "Review status update failed";
      toast.error(msg);
    } finally {
      // Only if this part is still the one in flight. A slower earlier request
      // resolving second would otherwise re-enable the button for a later one.
      setReviewUpdatingPartId((current) =>
        current === partId ? null : current,
      );
    }
  };

  const handleTogglePublished = async (partId: string, published: boolean) => {
    if (!projectId || !documentId) return;
    setPublishUpdatingPartId(partId);
    try {
      await api.updatePartsPublished(projectId, documentId, {
        parts: [{ part_id: partId, published }],
      });
      // On a draft nothing is public yet, and saying otherwise would be the one
      // place in this flow that overstates what just happened.
      toast.success(partPublishedMessage(document?.workflow, published));
      // The public reader reads this flag, and so does the owner-facing parts
      // list the editor holds; neither is the copy this page just wrote.
      invalidateAfter.documentPartsChanged(projectId, documentId);
      await reloadDocument();
    } catch (err) {
      const msg =
        err instanceof ApiError && err.status === 403
          ? "Only the project owner can choose which pages are public."
          : err instanceof ApiError
            ? err.message
            : "Could not change which pages are public";
      toast.error(msg);
    } finally {
      setPublishUpdatingPartId((current) =>
        current === partId ? null : current,
      );
    }
  };

  const shownCount = parts.filter((part) => part.published).length;

  const reviewedCount = parts.filter((part) => part.reviewed).length;

  const subtitle = document
    ? `${parts.length} page${parts.length === 1 ? "" : "s"} · ${reviewedCount} reviewed · ${relativeUpdatedLabel(document.updated_at)}`
    : undefined;

  const applyDocumentPatch = (
    patch: Partial<
      Pick<DocumentWithPartsResponse, "workflow" | "public_share_token">
    >,
  ) => {
    patchDocument((current) => ({
      ...current,
      document: { ...current.document, ...patch },
    }));
    // The document list, the public reader and the copy the page editor holds
    // are all copies of what just changed, and none of them is this one.
    invalidateAfter.documentUpdated(projectId!, documentId!);
  };

  const handleDeleteDocument = async () => {
    if (!projectId || !documentId) return;
    try {
      await api.deleteDocument(projectId, documentId);
      toast.success("Document deleted");
      invalidateAfter.documentDeleted(projectId, documentId);
      router.push(`/projects/${projectId}`);
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "Failed to delete document";
      toast.error(msg);
    }
  };

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
      subtitle={subtitle}
      titleExtra={
        document ? <WorkflowBadge workflow={document.workflow} /> : undefined
      }
      titleEditable={Boolean(document && projectId && documentId)}
      titlePanelOpen={titlePanelOpen}
      onTitlePanelToggle={() => setTitlePanelOpen((open) => !open)}
      titlePanelLabel="Document settings"
      headerActions={
        document && projectId && documentId ? (
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
            onUpload={(files) => void handleUpload(files)}
            onWorkflowChange={(workflow: DocumentWorkflow) =>
              applyDocumentPatch({ workflow })
            }
            onShareTokenChange={(token) =>
              applyDocumentPatch({ public_share_token: token })
            }
            onJobsQueued={() => {
              // The jobs are queued, not finished; what is already stale is the
              // count of pages still waiting for one.
              invalidateAfter.documentPartsChanged(projectId, documentId);
              void reloadWorkflowCounts();
              setJobsQueuedCount((count) => count + 1);
            }}
            onOpenSettings={() => setTitlePanelOpen(true)}
            onDeleteDocument={handleDeleteDocument}
          />
        ) : undefined
      }
      titlePanel={
        document && projectId && documentId ? (
          <DocumentSettingsPanel
            projectId={projectId}
            documentId={documentId}
            name={document.name}
            onUpdated={(updated) => {
              patchDocument((current) => ({
                ...current,
                document: { ...current.document, name: updated.name },
              }));
              // The document list, the public view and the copy the page
              // editor fetches all carry the name; the panel only wrote to the
              // copy this page holds.
              invalidateAfter.documentUpdated(projectId!, documentId!);
            }}
          />
        ) : null
      }
    >
      {loading && !document && !error ? (
        <ContentRegionLoading label="Loading document" />
      ) : (
        <>
          {error && (
            <div className="notice-banner" role="alert">
              <strong>Document unavailable</strong>
              {error}
            </div>
          )}

          {document &&
            document.workflow === "published" &&
            projectId &&
            documentId && (
              <DocumentLiveLinkRow
                projectId={projectId}
                documentId={documentId}
                publicShareToken={document.public_share_token ?? null}
              />
            )}

          {document && projectId && documentId && (
            <JobsNotice
              projectId={projectId}
              documentId={documentId}
              refreshKey={jobsQueuedCount}
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

          {document && (
            <>
              <p className="section-label">
                <span id="pages-label">Pages</span>
                {parts.length > 0 && (
                  <span className="section-label__aside">
                    {shownCount} of {parts.length} shown publicly
                    {document.workflow === "published"
                      ? ""
                      : document.workflow === "archived"
                        ? ", though archived documents are not public"
                        : " once the document is live"}
                  </span>
                )}
              </p>
              <PartList
                parts={parts}
                projectId={projectId!}
                documentId={documentId!}
                loading={loading}
                onMoveUp={(i) => movePart(i, -1)}
                onMoveDown={(i) => movePart(i, 1)}
                onDelete={(id) => void handleDelete(id)}
                onToggleReview={(partId, reviewed) =>
                  void handleToggleReview(partId, reviewed)
                }
                onTogglePublished={
                  isOwner
                    ? (partId, published) =>
                        void handleTogglePublished(partId, published)
                    : undefined
                }
                reviewUpdatingPartId={reviewUpdatingPartId}
                publishUpdatingPartId={publishUpdatingPartId}
                reordering={reordering}
              />
            </>
          )}
        </>
      )}
    </AppPageShell>
  );
}

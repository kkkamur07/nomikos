import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { toast } from "../components/ui/toast";
import {
  api,
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type DocumentWorkflowCounts,
} from "../api/client";
import { ApiError } from "../api/errors";
import { invalidateAfter, resourceTags } from "../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { relativeUpdatedLabel } from "../components/document/documentActionCopy";
import { useFileDrop } from "../hooks/useFileDrop";
import { useServerQuery } from "../hooks/useServerQuery";
import { usePartMutations } from "./usePartMutations";
import { usePartUpload } from "./usePartUpload";

type DocumentDetailData = {
  username: string;
  projectName: string;
  /** Both already fetched; kept so the page can answer "may I publish?". */
  userId: string;
  /** Null on a project with no recorded owner, which is nobody's match. */
  projectOwnerId: string | null;
  document: DocumentWithPartsResponse;
};

export interface DocumentDetailModel {
  projectId: string | undefined;
  documentId: string | undefined;
  username: string | null;
  projectName: string | null;
  document: DocumentWithPartsResponse | null;
  parts: DocumentPartResponse[];
  subtitle: string | undefined;
  shownCount: number;
  isOwner: boolean;
  loading: boolean;
  error: string | null;
  workflowCounts: DocumentWorkflowCounts | null;
  titlePanelOpen: boolean;
  /** Bumped when the Workflow menu queues jobs, so the Jobs bar re-reads them. */
  jobsQueuedCount: number;
  uploading: boolean;
  uploadProgress: string | null;
  reordering: boolean;
  reviewUpdatingPartId: string | null;
  publishUpdatingPartId: string | null;
  dragActive: boolean;
  setTitlePanelOpen: (open: boolean | ((open: boolean) => boolean)) => void;
  handleUpload: (files: File[]) => void;
  movePart: (index: number, direction: -1 | 1) => void;
  handleDelete: (partId: string) => void;
  handleToggleReview: (partId: string, reviewed: boolean) => void;
  handleTogglePublished: (partId: string, published: boolean) => void;
  applyDocumentPatch: (
    patch: Partial<
      Pick<DocumentWithPartsResponse, "workflow" | "public_share_token">
    >,
  ) => void;
  handleDeleteDocument: () => void;
  handleJobsQueued: () => void;
  handleRenamed: (name: string) => void;
}

function documentDetailSubtitle(
  document: DocumentWithPartsResponse,
  parts: DocumentPartResponse[],
): string {
  const reviewedCount = parts.filter((part) => part.reviewed).length;
  return `${parts.length} page${parts.length === 1 ? "" : "s"} · ${reviewedCount} reviewed · ${relativeUpdatedLabel(document.updated_at)}`;
}

function documentLoadError(err: unknown): string | null {
  if (isUnauthorized(err)) return null;
  return err instanceof ApiError ? err.message : "Failed to load document";
}

function documentBannerText(err: unknown): string {
  const msg = documentLoadError(err) ?? "Failed to load document";
  return err instanceof ApiError && (err.status === 403 || err.status === 404)
    ? "This document is not available to your account."
    : msg;
}

// Publishing is owner-only in the backend (DocumentPartService raises for
// anyone else), so offering the control to a collaborator would only earn
// them a red toast.
function isDocumentOwner(data: DocumentDetailData): boolean {
  return Boolean(
    data.userId && data.projectOwnerId && data.projectOwnerId === data.userId,
  );
}

function countShownParts(parts: DocumentPartResponse[]): number {
  return parts.filter((part) => part.published).length;
}

function selectDocumentDetail(data: DocumentDetailData | null): {
  username: string | null;
  projectName: string | null;
  document: DocumentWithPartsResponse | null;
  parts: DocumentPartResponse[];
  subtitle: string | undefined;
  shownCount: number;
  isOwner: boolean;
} {
  if (!data) {
    return {
      username: null,
      projectName: null,
      document: null,
      parts: [],
      subtitle: undefined,
      shownCount: 0,
      isOwner: false,
    };
  }
  const { document } = data;
  const parts = [...(document.parts ?? [])].sort((a, b) => a.order - b.order);
  return {
    username: data.username,
    projectName: data.projectName,
    document,
    parts,
    subtitle: documentDetailSubtitle(document, parts),
    shownCount: countShownParts(parts),
    isOwner: isDocumentOwner(data),
  };
}

function useDocumentDetailQueries(
  projectId: string | undefined,
  documentId: string | undefined,
  signedIn: boolean,
  onAuthFailure: () => void,
): {
  data: DocumentDetailData | null;
  loading: boolean;
  error: string | null;
  workflowCounts: DocumentWorkflowCounts | null;
  patchDocument: (
    update: (current: DocumentDetailData) => DocumentDetailData,
  ) => void;
  reloadWorkflowCounts: () => Promise<void>;
} {
  const enabled = Boolean(projectId && documentId && signedIn);
  const queries = useServerQuery<DocumentDetailData>({
    key: enabled ? ["document-detail", projectId!, documentId!] : null,
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
        onAuthFailure();
        return null;
      }
      const msg = documentLoadError(err) ?? "Failed to load document";
      toast.error(msg);
      return documentBannerText(err);
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
  const counts = useServerQuery<DocumentWorkflowCounts>({
    key: enabled ? ["document-workflow-counts", projectId!, documentId!] : null,
    tags: [resourceTags.document(projectId ?? "", documentId ?? "")],
    read: () => api.getDocumentWorkflowCounts(projectId!, documentId!),
    onError: () => null,
  });

  return {
    data: queries.data,
    loading: queries.loading,
    error: queries.error,
    workflowCounts: counts.data,
    patchDocument: queries.patch,
    reloadWorkflowCounts: counts.refetch,
  };
}

async function deleteDocumentAndLeave(
  projectId: string,
  documentId: string,
  leave: () => void,
): Promise<void> {
  try {
    await api.deleteDocument(projectId, documentId);
    toast.success("Document deleted");
    invalidateAfter.documentDeleted(projectId, documentId);
    leave();
  } catch (err) {
    const msg =
      err instanceof ApiError ? err.message : "Failed to delete document";
    toast.error(msg);
  }
}

export function useDocumentDetail(): DocumentDetailModel {
  const router = useRouter();
  const { projectId, documentId } =
    useParams<{ projectId: string; documentId: string }>() ?? {};
  const [titlePanelOpen, setTitlePanelOpen] = useState(false);
  const [jobsQueuedCount, setJobsQueuedCount] = useState(0);

  const signedIn = hasAccessToken();
  useEffect(() => {
    if (projectId && documentId && !signedIn) navigateToLogin(router);
  }, [projectId, documentId, signedIn, router]);

  const {
    data,
    loading,
    error,
    workflowCounts,
    patchDocument,
    reloadWorkflowCounts,
  } = useDocumentDetailQueries(projectId, documentId, signedIn, () =>
    navigateToLogin(router),
  );

  const selected = selectDocumentDetail(data);
  const { document, parts } = selected;

  // Invalidation refetches the mounted queries, so the hooks await it instead
  // of firing a second explicit refetch at the same key.
  const refreshAfterPartsChanged = async () => {
    if (!projectId || !documentId) return;
    await invalidateAfter.documentPartsChanged(projectId, documentId);
  };

  const { uploading, uploadProgress, handleUpload } = usePartUpload({
    projectId,
    documentId,
    onUploaded: refreshAfterPartsChanged,
  });

  const {
    reordering,
    reviewUpdatingPartId,
    publishUpdatingPartId,
    movePart,
    handleDelete,
    handleToggleReview,
    handleTogglePublished,
  } = usePartMutations({
    projectId,
    documentId,
    parts,
    workflow: document?.workflow,
    onChanged: refreshAfterPartsChanged,
  });

  // Whole-window drop target. The one click-to-pick alternative is the Upload
  // pages button in the action row; there used to be a second, a panel above
  // the page list that opened the same file picker with the same accept list.
  // Two controls doing one thing is not two ways in, it is a question about
  // which one is the real one.
  const dragActive = useFileDrop(
    (files) => void handleUpload(files),
    Boolean(document) && !uploading && !loading,
  );

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
    void invalidateAfter.documentUpdated(projectId!, documentId!);
  };

  return {
    projectId,
    documentId,
    ...selected,
    loading,
    error,
    workflowCounts,
    titlePanelOpen,
    jobsQueuedCount,
    uploading,
    uploadProgress,
    reordering,
    reviewUpdatingPartId,
    publishUpdatingPartId,
    dragActive,
    setTitlePanelOpen,
    handleUpload: (files) => void handleUpload(files),
    movePart,
    handleDelete: (partId) => void handleDelete(partId),
    handleToggleReview: (partId, reviewed) =>
      void handleToggleReview(partId, reviewed),
    handleTogglePublished: (partId, published) =>
      void handleTogglePublished(partId, published),
    applyDocumentPatch,
    handleDeleteDocument: () => {
      if (projectId && documentId) {
        void deleteDocumentAndLeave(projectId, documentId, () =>
          router.push(`/projects/${projectId}`),
        );
      }
    },
    handleJobsQueued: () => {
      // The jobs are queued, not finished; what is already stale is the
      // count of pages still waiting for one.
      void refreshAfterPartsChanged();
      void reloadWorkflowCounts();
      setJobsQueuedCount((count) => count + 1);
    },
    handleRenamed: (name) => {
      patchDocument((current) => ({
        ...current,
        document: { ...current.document, name },
      }));
      // The document list, the public view and the copy the page
      // editor fetches all carry the name; the panel only wrote to the
      // copy this page holds.
      void invalidateAfter.documentUpdated(projectId!, documentId!);
    },
  };
}

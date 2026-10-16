import { useEffect, useState, type FormEvent } from "react";
import { useParams, useRouter } from "next/navigation";
import { toast } from "../components/ui/toast";
import {
  api,
  type DocumentResponse,
  type ProjectResponse,
  type UserResponse,
} from "../api/client";
import { ApiError } from "../api/errors";
import { invalidateAfter, resourceTags } from "../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { useServerQuery } from "../hooks/useServerQuery";

export type ProjectDashboardData = {
  me: UserResponse;
  project: ProjectResponse;
  documents: DocumentResponse[];
};

export interface ProjectDashboardModel {
  projectId: string | undefined;
  project: ProjectResponse | null;
  documents: DocumentResponse[];
  username: string | null;
  isOwner: boolean;
  loading: boolean;
  error: string | null;
  includeArchived: boolean;
  createModalOpen: boolean;
  titlePanelOpen: boolean;
  shareFocusToken: number;
  creating: boolean;
  deletingProject: boolean;
  deletingDocumentId: string | null;
  newDocName: string;
  docCountLabel: string;
  setIncludeArchived: (value: boolean) => void;
  setCreateModalOpen: (open: boolean) => void;
  setTitlePanelOpen: (open: boolean | ((open: boolean) => boolean)) => void;
  setNewDocName: (name: string) => void;
  openSharePanel: () => void;
  handleCreate: (event: FormEvent<HTMLFormElement>) => void;
  handleDeleteProject: () => void;
  handleDeleteDocument: (documentId: string) => void;
  handleProjectUpdated: (updated: {
    name: string;
    guidelines: string | null;
  }) => void;
}

function dashboardBannerText(err: unknown): string {
  const msg = err instanceof ApiError ? err.message : "Failed to load project";
  // 403 and 404 both mean "not yours to see", which reads better as the
  // feature sentence than as the raw API message. Note the toast still
  // carries the raw message.
  return err instanceof ApiError && (err.status === 403 || err.status === 404)
    ? "This project is not available to your account."
    : msg;
}

async function createDocumentAndOpen(
  projectId: string,
  name: string,
  open: (documentId: string) => void,
): Promise<boolean> {
  try {
    const doc = await api.createDocument(projectId, { name: name.trim() });
    toast.success("Document created");
    invalidateAfter.documentCreated(projectId);
    open(doc.id);
    return true;
  } catch (err) {
    const msg =
      err instanceof ApiError ? err.message : "Failed to create document";
    toast.error(msg);
    return false;
  }
}

function dashboardQueryKey(
  projectId: string | undefined,
  signedIn: boolean,
  includeArchived: boolean,
): readonly [string, string, boolean] | null {
  return projectId && signedIn
    ? ["project-dashboard", projectId, includeArchived]
    : null;
}

function selectDashboardView(data: ProjectDashboardData | null): {
  project: ProjectResponse | null;
  documents: DocumentResponse[];
  username: string | null;
  isOwner: boolean;
  docCountLabel: string;
} {
  if (!data) {
    return {
      project: null,
      documents: [],
      username: null,
      isOwner: false,
      docCountLabel: "0 documents",
    };
  }
  const { project, documents, me } = data;
  const count = documents.length;
  return {
    project,
    documents,
    username: me.username,
    isOwner: Boolean(me.id && project.owner_id === me.id),
    docCountLabel: count === 1 ? "1 document" : `${count} documents`,
  };
}

export function useProjectDashboard(): ProjectDashboardModel {
  const router = useRouter();
  const { projectId } = useParams<{ projectId: string }>() ?? {};
  const [includeArchived, setIncludeArchived] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [titlePanelOpen, setTitlePanelOpen] = useState(false);
  // Incremented by the Share button so the panel puts the cursor in the box.
  const [shareFocusToken, setShareFocusToken] = useState(0);
  const [creating, setCreating] = useState(false);
  const [deletingProject, setDeletingProject] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(
    null,
  );
  const [newDocName, setNewDocName] = useState("");

  const signedIn = hasAccessToken();
  useEffect(() => {
    if (projectId && !signedIn) navigateToLogin(router);
  }, [projectId, signedIn, router]);

  const {
    data,
    loading,
    error,
    refetch: reloadDashboard,
    patch: patchDashboard,
  } = useServerQuery<ProjectDashboardData>({
    key: dashboardQueryKey(projectId, signedIn, includeArchived),
    tags: [
      resourceTags.currentUser,
      resourceTags.project(projectId ?? ""),
      resourceTags.documents(projectId ?? ""),
    ],
    read: async () => {
      const [me, project, documents] = await Promise.all([
        api.me(),
        api.getProject(projectId!),
        api.listDocuments(projectId!, includeArchived),
      ]);
      return { me, project, documents };
    },
    onError: (err) => {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return null;
      }
      const msg =
        err instanceof ApiError ? err.message : "Failed to load project";
      toast.error(msg);
      return dashboardBannerText(err);
    },
  });

  const selected = selectDashboardView(data);
  const { project, documents } = selected;

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!projectId || !newDocName.trim()) return;
    setCreating(true);
    try {
      const created = await createDocumentAndOpen(projectId, newDocName, (id) =>
        router.push(`/projects/${projectId}/documents/${id}`),
      );
      if (created) {
        setCreateModalOpen(false);
        setNewDocName("");
      }
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteProject = async () => {
    if (!projectId || !project) return;
    const confirmed = window.confirm(
      `Delete project "${project.name}"? All documents in this project will be removed.`,
    );
    if (!confirmed) return;

    setDeletingProject(true);
    try {
      await api.deleteProject(projectId);
      toast.success("Project deleted");
      invalidateAfter.projectDeleted(projectId);
      router.push("/projects");
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "Failed to delete project";
      toast.error(msg);
    } finally {
      setDeletingProject(false);
    }
  };

  const handleDeleteDocument = async (documentId: string) => {
    if (!projectId) return;
    const document = documents.find((item) => item.id === documentId);
    if (!document) return;
    const confirmed = window.confirm(
      `Delete document "${document.name}"? All parts and transcriptions will be removed.`,
    );
    if (!confirmed) return;

    setDeletingDocumentId(documentId);
    try {
      await api.deleteDocument(projectId, documentId);
      toast.success("Document deleted");
      invalidateAfter.documentDeleted(projectId, documentId);
      await reloadDashboard();
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "Failed to delete document";
      toast.error(msg);
    } finally {
      setDeletingDocumentId(null);
    }
  };

  return {
    projectId,
    ...selected,
    loading,
    error,
    includeArchived,
    createModalOpen,
    titlePanelOpen,
    shareFocusToken,
    creating,
    deletingProject,
    deletingDocumentId,
    newDocName,
    setIncludeArchived,
    setCreateModalOpen,
    setTitlePanelOpen,
    setNewDocName,
    // The panel closes on any click outside itself, and this button is
    // outside it. Stop the event and set open rather than toggling, so
    // clicking Share twice re-focuses the box instead of flickering shut.
    openSharePanel: () => {
      setTitlePanelOpen(true);
      setShareFocusToken((token) => token + 1);
    },
    handleCreate: (event) => void handleCreate(event),
    handleDeleteProject: () => void handleDeleteProject(),
    handleDeleteDocument: (documentId) => void handleDeleteDocument(documentId),
    handleProjectUpdated: (updated) => {
      patchDashboard((current) => ({
        ...current,
        project: {
          ...current.project,
          name: updated.name,
          guidelines: updated.guidelines,
        },
      }));
      // The patch shows the new name at once; the invalidation is what
      // reaches the reads the patch cannot. `includeArchived` is part
      // of this query's key, so the variant the researcher is not
      // looking at is a second cache entry, and the project list is a
      // third.
      if (projectId) void invalidateAfter.projectUpdated(projectId);
    },
  };
}

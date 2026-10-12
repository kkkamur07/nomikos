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
import { resourceTags, invalidateAfter } from "../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { AppPageShell } from "../components/layout/AppPageShell";
import { ContentRegionLoading } from "../components/layout/ContentRegionLoading";
import { DocumentsTable } from "../components/projects/DocumentsTable";
import { ProjectJobsPanel } from "../components/projects/ProjectJobsPanel";
import { ProjectDefaultModelsPanel } from "../components/sharing/ProjectDefaultModelsPanel";
import { ProjectSettingsPanel } from "../components/sharing/ProjectSettingsPanel";
import { ProjectSharingPanel } from "../components/sharing/ProjectSharingPanel";
import { FormModal } from "../components/ui/FormModal";
import { useServerQuery } from "../hooks/useServerQuery";

type ProjectDashboardData = {
  me: UserResponse;
  project: ProjectResponse;
  documents: DocumentResponse[];
};

export function ProjectDashboardPage() {
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
    key:
      projectId && signedIn
        ? ["project-dashboard", projectId, includeArchived]
        : null,
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
      // 403 and 404 both mean "not yours to see", which reads better as the
      // feature sentence than as the raw API message. Note the toast still
      // carries the raw message.
      return err instanceof ApiError &&
        (err.status === 403 || err.status === 404)
        ? "This project is not available to your account."
        : msg;
    },
  });

  const project = data?.project ?? null;
  const documents = data?.documents ?? [];
  const userId = data?.me.id ?? null;
  const username = data?.me.username ?? null;

  const isOwner = Boolean(project && userId && project.owner_id === userId);

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!projectId || !newDocName.trim()) return;
    setCreating(true);
    try {
      const doc = await api.createDocument(projectId, {
        name: newDocName.trim(),
      });
      toast.success("Document created");
      setCreateModalOpen(false);
      setNewDocName("");
      invalidateAfter.documentCreated(projectId);
      router.push(`/projects/${projectId}/documents/${doc.id}`);
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "Failed to create document";
      toast.error(msg);
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

  const docCountLabel =
    documents.length === 1 ? "1 document" : `${documents.length} documents`;

  return (
    <AppPageShell
      breadcrumb={[
        { label: "Projects", href: "/projects" },
        { label: project?.name ?? "Project" },
      ]}
      username={username}
      title={project?.name ?? "Project"}
      subtitle={project ? docCountLabel : undefined}
      titleEditable={Boolean(isOwner && project && projectId)}
      titlePanelOpen={titlePanelOpen}
      onTitlePanelToggle={() => setTitlePanelOpen((open) => !open)}
      titlePanelLabel="Project settings"
      titlePanel={
        project && projectId ? (
          <>
            <ProjectSettingsPanel
              projectId={projectId}
              name={project.name}
              guidelines={project.guidelines ?? null}
              onUpdated={(updated) => {
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
                invalidateAfter.projectUpdated(projectId);
              }}
            />
            {/* Every member may set the project defaults, so this one is not
                gated on ownership the way the collaborator list below is. */}
            <ProjectDefaultModelsPanel projectId={projectId} />
            {/* The panel only opens for the owner (titleEditable), and the
                collaborator list is owner-only on the API too. */}
            {isOwner && (
              <ProjectSharingPanel
                projectId={projectId}
                focusToken={shareFocusToken}
              />
            )}
          </>
        ) : null
      }
      headerActions={
        project ? (
          <>
            <label className="field-check">
              <input
                type="checkbox"
                id="show-archived"
                checked={includeArchived}
                onChange={(e) => setIncludeArchived(e.target.checked)}
              />
              Show archived
            </label>
            {isOwner && (
              <button
                type="button"
                className="btn btn-outline btn-sm"
                aria-haspopup="dialog"
                aria-expanded={titlePanelOpen}
                onClick={(event) => {
                  // The panel closes on any click outside itself, and this
                  // button is outside it. Stop the event and set open rather
                  // than toggling, so clicking Share twice re-focuses the box
                  // instead of flickering the panel shut.
                  event.stopPropagation();
                  setTitlePanelOpen(true);
                  setShareFocusToken((token) => token + 1);
                }}
              >
                Share
              </button>
            )}
            {isOwner && (
              <button
                type="button"
                className="btn btn-ghost btn-sm btn--danger-ghost"
                disabled={deletingProject}
                onClick={() => void handleDeleteProject()}
              >
                Delete project
              </button>
            )}
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => setCreateModalOpen(true)}
            >
              New document
            </button>
          </>
        ) : undefined
      }
    >
      {loading && !project && !error ? (
        <ContentRegionLoading label="Loading project" />
      ) : (
        <>
          {error && (
            <div className="notice-banner" role="alert">
              <strong>Project unavailable</strong>
              {error}
            </div>
          )}

          {project && (
            <>
              <DocumentsTable
                projectId={projectId!}
                documents={documents}
                loading={loading}
                emptyText="No documents yet"
                onDelete={(documentId) => void handleDeleteDocument(documentId)}
                deletingDocumentId={deletingDocumentId}
              />

              <ProjectJobsPanel projectId={projectId!} documents={documents} />
            </>
          )}
        </>
      )}

      <FormModal
        open={createModalOpen}
        title="New document"
        onClose={() => setCreateModalOpen(false)}
        onSubmit={handleCreate}
        submitLabel="Create"
        loading={creating}
      >
        <div className="field">
          <label htmlFor="document-name">Name</label>
          <input
            id="document-name"
            required
            value={newDocName}
            onChange={(e) => setNewDocName(e.target.value)}
          />
        </div>
      </FormModal>
    </AppPageShell>
  );
}

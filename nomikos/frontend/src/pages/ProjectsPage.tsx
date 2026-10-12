import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { toast } from "../components/ui/toast";
import {
  api,
  type InferenceModelResponse,
  type InferenceTask,
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
import {
  apiBindingStore,
  saveProjectDefault,
} from "../components/page-editor/projectModelDefaults";
import { ProjectsTable } from "../components/projects/ProjectsTable";
import { FormModal } from "../components/ui/FormModal";
import { ModelSelectRow } from "../components/ui/ModelSelectRow";
import { useServerQuery } from "../hooks/useServerQuery";
import { slugify } from "../utils/slugify";

type ProjectsPageData = {
  me: UserResponse;
  projects: ProjectResponse[];
};

export function ProjectsPage() {
  const router = useRouter();
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(
    null,
  );
  const [newName, setNewName] = useState("");
  // Both start on "No default": a new project binds a model only if the
  // researcher picks one here.
  const [newSegmentModelId, setNewSegmentModelId] = useState("");
  const [newTranscribeModelId, setNewTranscribeModelId] = useState("");
  const [catalog, setCatalog] = useState<InferenceModelResponse[]>([]);
  const bindingStore = useMemo(() => apiBindingStore(), []);

  // Redirecting before any request goes out leaves the page in its loading
  // state rather than flashing an empty list on the way to login.
  const signedIn = hasAccessToken();
  useEffect(() => {
    if (!signedIn) navigateToLogin(router);
  }, [signedIn, router]);

  // Read once the dialog is open, so the projects list never waits on it. A
  // failed read leaves the selects empty and disabled; the name still works.
  useEffect(() => {
    if (!createModalOpen) return;
    let cancelled = false;
    void (async () => {
      try {
        const models = await api.listInferenceModels();
        if (!cancelled) setCatalog(models);
      } catch {
        if (!cancelled) setCatalog([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [createModalOpen]);

  const segmentModels = useMemo(
    () => catalog.filter((model) => model.task === "segment"),
    [catalog],
  );
  const transcribeModels = useMemo(
    () => catalog.filter((model) => model.task === "transcribe"),
    [catalog],
  );

  const {
    data,
    loading,
    error,
    refetch: reloadProjects,
  } = useServerQuery<ProjectsPageData>({
    key: signedIn ? ["projects-page"] : null,
    tags: [resourceTags.currentUser, resourceTags.projects],
    read: async () => {
      const [me, projects] = await Promise.all([api.me(), api.listProjects()]);
      return { me, projects };
    },
    onError: (err) => {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return null;
      }
      // Unlike its sibling pages this one has no 403/404 rewrite: a project list
      // is never "not yours to see", it is only ever empty.
      const msg =
        err instanceof ApiError ? err.message : "Failed to load projects";
      toast.error(msg);
      return msg;
    },
  });

  const projects = data?.projects ?? [];
  const userId = data?.me.id ?? null;
  const username = data?.me.username ?? null;

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const project = await api.createProject({
        name: newName.trim(),
        slug: slugify(newName),
      });
      // The project exists from here on. A default that will not save is worth
      // a different toast, never a failed creation.
      const chosen: [InferenceTask, string][] = [
        ["segment", newSegmentModelId],
        ["transcribe", newTranscribeModelId],
      ];
      let defaultsFailed = false;
      for (const [task, modelId] of chosen) {
        if (!modelId) continue;
        try {
          await saveProjectDefault(bindingStore, project.id, task, modelId);
        } catch {
          defaultsFailed = true;
        }
      }
      if (defaultsFailed) {
        toast.error(
          "Project created, but its default models could not be saved. You can set them on the project page.",
        );
      } else {
        toast.success("Project created");
      }
      setCreateModalOpen(false);
      setNewName("");
      setNewSegmentModelId("");
      setNewTranscribeModelId("");
      invalidateAfter.projectCreated();
      await reloadProjects();
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "Failed to create project";
      toast.error(msg);
    } finally {
      setCreating(false);
    }
  };

  const owned = projects.filter((p) => p.owner_id === userId);
  const shared = projects.filter((p) => p.owner_id !== userId);

  const handleDeleteProject = async (projectId: string) => {
    const project = projects.find((item) => item.id === projectId);
    if (!project) return;
    const confirmed = window.confirm(
      `Delete project "${project.name}"? All documents in this project will be removed.`,
    );
    if (!confirmed) return;

    setDeletingProjectId(projectId);
    try {
      await api.deleteProject(projectId);
      toast.success("Project deleted");
      invalidateAfter.projectDeleted(projectId);
      await reloadProjects();
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "Failed to delete project";
      toast.error(msg);
    } finally {
      setDeletingProjectId(null);
    }
  };

  return (
    <AppPageShell
      currentLabel="Projects"
      username={username}
      title="Projects"
      subtitle="Owned and shared"
      headerActions={
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={() => setCreateModalOpen(true)}
        >
          New project
        </button>
      }
    >
      {error && (
        <div className="notice-banner" role="alert">
          <strong>Projects unavailable</strong>
          {error}
        </div>
      )}

      <p className="section-label" id="owned-label">
        Owned
      </p>
      <ProjectsTable
        id="owned-label"
        caption="Owned projects"
        projects={owned}
        userId={userId}
        loading={loading}
        emptyText="No owned projects yet"
        onDelete={(projectId) => void handleDeleteProject(projectId)}
        deletingProjectId={deletingProjectId}
      />

      <p className="section-label" id="shared-label">
        Shared
      </p>
      <ProjectsTable
        id="shared-label"
        caption="Shared projects"
        projects={shared}
        userId={userId}
        loading={loading}
        emptyText="No shared projects"
        showOwner
      />

      <FormModal
        open={createModalOpen}
        title="New project"
        onClose={() => setCreateModalOpen(false)}
        onSubmit={handleCreate}
        submitLabel="Create"
        loading={creating}
      >
        <div className="field">
          <label htmlFor="project-name">Name</label>
          <input
            id="project-name"
            required
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
        </div>
        <p className="model-row-hint">
          Optional. New jobs in this project start with these models. You can
          change them later.
        </p>
        <div className="model-row-group">
          <ModelSelectRow
            id="new-project-segment-model"
            label="Segmentation model"
            value={newSegmentModelId}
            options={segmentModels}
            emptyLabel="No default"
            disabled={creating}
            onChange={setNewSegmentModelId}
          />
          <ModelSelectRow
            id="new-project-transcribe-model"
            label="Transcription model"
            value={newTranscribeModelId}
            options={transcribeModels}
            emptyLabel="No default"
            disabled={creating}
            onChange={setNewTranscribeModelId}
          />
        </div>
      </FormModal>
    </AppPageShell>
  );
}

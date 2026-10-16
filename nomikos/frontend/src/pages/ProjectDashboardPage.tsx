import type { ReactNode } from "react";
import { AppPageShell } from "../components/layout/AppPageShell";
import { ContentRegionLoading } from "../components/layout/ContentRegionLoading";
import { DocumentsTable } from "../components/projects/DocumentsTable";
import { ProjectJobsPanel } from "../components/projects/ProjectJobsPanel";
import { ProjectSettingsPanel } from "../components/sharing/ProjectSettingsPanel";
import { ProjectSharingPanel } from "../components/sharing/ProjectSharingPanel";
import { FormModal } from "../components/ui/FormModal";
import {
  useProjectDashboard,
  type ProjectDashboardModel,
} from "./useProjectDashboard";

function DashboardTitlePanel(model: ProjectDashboardModel): ReactNode {
  const { project, projectId, shareFocusToken, isOwner } = model;
  if (!project || !projectId) return null;
  return (
    <>
      <ProjectSettingsPanel
        projectId={projectId}
        name={project.name}
        guidelines={project.guidelines ?? null}
        onUpdated={model.handleProjectUpdated}
      />
      {/* The panel only opens for the owner (titleEditable), and the
          collaborator list is owner-only on the API too. */}
      {isOwner && (
        <ProjectSharingPanel
          projectId={projectId}
          focusToken={shareFocusToken}
        />
      )}
    </>
  );
}

function DashboardHeaderActions(model: ProjectDashboardModel): ReactNode {
  const { project, isOwner, includeArchived, deletingProject } = model;
  if (!project) return undefined;
  return (
    <>
      <label className="field-check">
        <input
          type="checkbox"
          id="show-archived"
          checked={includeArchived}
          onChange={(e) => model.setIncludeArchived(e.target.checked)}
        />
        Show archived
      </label>
      {isOwner && (
        <button
          type="button"
          className="btn btn-outline btn-sm"
          aria-haspopup="dialog"
          aria-expanded={model.titlePanelOpen}
          onClick={(event) => {
            // The panel closes on any click outside itself, and this
            // button is outside it. Stop the event and set open rather
            // than toggling, so clicking Share twice re-focuses the box
            // instead of flickering the panel shut.
            event.stopPropagation();
            model.openSharePanel();
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
          onClick={model.handleDeleteProject}
        >
          Delete project
        </button>
      )}
      <button
        type="button"
        className="btn btn-primary btn-sm"
        onClick={() => model.setCreateModalOpen(true)}
      >
        New document
      </button>
    </>
  );
}

function DashboardContent(model: ProjectDashboardModel): ReactNode {
  const { project, projectId, documents, loading, error, deletingDocumentId } =
    model;
  if (loading && !project && !error) {
    return <ContentRegionLoading label="Loading project" />;
  }
  if (!project) {
    return error ? (
      <div className="notice-banner" role="alert">
        <strong>Project unavailable</strong>
        {error}
      </div>
    ) : null;
  }
  return (
    <>
      {error && (
        <div className="notice-banner" role="alert">
          <strong>Project unavailable</strong>
          {error}
        </div>
      )}

      <DocumentsTable
        projectId={projectId!}
        documents={documents}
        loading={loading}
        emptyText="No documents yet"
        onDelete={model.handleDeleteDocument}
        deletingDocumentId={deletingDocumentId}
      />

      <ProjectJobsPanel projectId={projectId!} documents={documents} />
    </>
  );
}

export function ProjectDashboardPage() {
  const model = useProjectDashboard();
  const { project, projectId, username } = model;
  return (
    <AppPageShell
      breadcrumb={[
        { label: "Projects", href: "/projects" },
        { label: project?.name ?? "Project" },
      ]}
      username={username}
      title={project?.name ?? "Project"}
      subtitle={project ? model.docCountLabel : undefined}
      titleEditable={Boolean(model.isOwner && project && projectId)}
      titlePanelOpen={model.titlePanelOpen}
      onTitlePanelToggle={() => model.setTitlePanelOpen((open) => !open)}
      titlePanelLabel="Project settings"
      // Called as functions, not rendered: the helpers are hook-free, and the
      // shell only skips its containers for `undefined`/`null` props, which a
      // wrapper element would never be.
      titlePanel={DashboardTitlePanel(model)}
      headerActions={DashboardHeaderActions(model)}
    >
      <DashboardContent {...model} />

      <FormModal
        open={model.createModalOpen}
        title="New document"
        onClose={() => model.setCreateModalOpen(false)}
        onSubmit={model.handleCreate}
        submitLabel="Create"
        loading={model.creating}
      >
        <div className="field">
          <label htmlFor="document-name">Name</label>
          <input
            id="document-name"
            required
            value={model.newDocName}
            onChange={(e) => model.setNewDocName(e.target.value)}
          />
        </div>
      </FormModal>
    </AppPageShell>
  );
}

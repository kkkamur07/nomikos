import { useRouter } from "next/navigation";
import type { ProjectResponse } from "../../api/client";

type ProjectsTableProps = {
  id: string;
  caption: string;
  projects: ProjectResponse[];
  userId: string | null;
  loading: boolean;
  emptyText: string;
  showOwner?: boolean;
  onDelete?: (projectId: string) => void;
  deletingProjectId?: string | null;
};

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    month: "short",
    year: "numeric",
  });
}

function formatDocumentCount(count: number): string {
  return count === 1 ? "1 document" : `${count} documents`;
}

export function ProjectsTable({
  id,
  caption,
  projects,
  userId,
  loading,
  emptyText,
  showOwner = false,
  onDelete,
  deletingProjectId = null,
}: ProjectsTableProps) {
  const router = useRouter();

  const openProject = (projectId: string) => {
    router.push(`/projects/${projectId}`);
  };

  const onRowKeyDown = (event: React.KeyboardEvent, projectId: string) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openProject(projectId);
    }
  };

  return (
    <div className="data-panel mb-4">
      <table className="data-list" aria-labelledby={id}>
        <thead>
          <tr>
            <th scope="col">Project</th>
            {showOwner ? (
              <th scope="col">Owner</th>
            ) : (
              <th scope="col">Documents</th>
            )}
            <th scope="col">
              <span className="text-muted">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr className="data-list-empty">
              <td colSpan={3}>Loading…</td>
            </tr>
          ) : projects.length === 0 ? (
            <tr className="data-list-empty">
              <td colSpan={3}>{emptyText}</td>
            </tr>
          ) : (
            projects.map((project) => {
              const isShared = project.owner_id !== userId;
              return (
                <tr
                  key={project.id}
                  className="data-list-row--clickable"
                  tabIndex={0}
                  aria-label={`Open project ${project.name}`}
                  onClick={() => openProject(project.id)}
                  onKeyDown={(event) => onRowKeyDown(event, project.id)}
                >
                  <td>
                    {isShared ? (
                      <div className="flex items-center gap-2">
                        <span className="row-title">{project.name}</span>
                        <span className="badge badge-shared">shared</span>
                      </div>
                    ) : (
                      <span className="row-title">{project.name}</span>
                    )}
                    <span className="row-sub">
                      {isShared
                        ? formatDocumentCount(project.document_count)
                        : `${project.slug} · ${formatDate(project.created_at)}`}
                    </span>
                  </td>
                  <td className="col-muted">
                    {showOwner
                      ? (project.owner_username ?? "-")
                      : project.document_count}
                  </td>
                  <td
                    className="col-action"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="data-list-actions">
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => openProject(project.id)}
                      >
                        Open
                      </button>
                      {onDelete && !isShared && (
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm btn--danger-ghost"
                          disabled={deletingProjectId === project.id}
                          aria-label={`Delete project ${project.name}`}
                          onClick={() => onDelete(project.id)}
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
      <span className="visually-hidden">{caption}</span>
    </div>
  );
}

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { api, type ProjectCollaboratorResponse } from "../../api/client";
import { ApiError } from "../../api/errors";
import { userFacingMessage } from "../../api/userFacingError";
import { toast } from "../ui/toast";

type ProjectSharingPanelProps = {
  projectId: string;
  /**
   * Bumped by the header's Share button. Each new value puts the cursor in the
   * box, so "Share" lands you ready to type rather than merely somewhere the
   * box happens to be. A counter rather than a boolean because opening the
   * panel twice in a row has to focus twice.
   */
  focusToken?: number;
};

/**
 * Owner-only section of the project settings panel: who else can open the
 * project, plus a box to add someone by the email they registered with (a
 * username works too, for people who know one).
 *
 * Sharing does not send an invitation; the other person needs an account
 * under that email first. The 404 from the API says so, and it is surfaced
 * verbatim.
 */
export function ProjectSharingPanel({
  projectId,
  focusToken = 0,
}: ProjectSharingPanelProps) {
  const [collaborators, setCollaborators] = useState<
    ProjectCollaboratorResponse[] | null
  >(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [adding, setAdding] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (focusToken > 0) inputRef.current?.focus();
  }, [focusToken]);

  const load = useCallback(async () => {
    try {
      const listed = await api.listProjectCollaborators(projectId);
      setCollaborators(listed);
      setLoadError(null);
    } catch (err) {
      setLoadError(
        err instanceof ApiError ? err.message : "Failed to load collaborators",
      );
    }
  }, [projectId]);

  useEffect(() => {
    void load();
  }, [load]);

  async function handleAdd(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = draft.trim();
    if (!value || adding) return;
    setAdding(true);
    try {
      // Deliberately not branching on "does it contain an @": a username may
      // legally contain one, so that guess sends a real username as an email
      // and the owner gets a validation error they cannot act on. The server
      // resolves it against actual accounts instead.
      await api.shareProject(projectId, { identifier: value });
      toast.success(`Shared with ${value}`);
      setDraft("");
      await load();
    } catch (err) {
      const message = userFacingMessage(err, "Failed to share project");
      toast.error(message, err);
    } finally {
      setAdding(false);
    }
  }

  async function handleRemove(person: ProjectCollaboratorResponse) {
    if (removing) return;
    setRemoving(person.id);
    try {
      await api.unshareProject(projectId, person.id);
      toast.success(`Removed ${person.username}`);
      setCollaborators((current) =>
        current ? current.filter((item) => item.id !== person.id) : current,
      );
    } catch (err) {
      const message = userFacingMessage(err, "Failed to remove access");
      toast.error(message, err);
    } finally {
      setRemoving(null);
    }
  }

  return (
    <div className="entity-panel__section">
      <h2 className="entity-panel__heading">Sharing</h2>
      <p className="entity-panel__hint">
        People you share with can open this project and edit its documents. They
        need a Nomikos account under that email first.
      </p>
      <form className="entity-panel__share-form" onSubmit={handleAdd}>
        <label className="visually-hidden" htmlFor="entity-project-share">
          Email or username
        </label>
        <input
          id="entity-project-share"
          ref={inputRef}
          type="text"
          autoComplete="off"
          placeholder="Email or username"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          disabled={adding}
        />
        <button
          type="submit"
          className="btn btn-primary btn-sm"
          disabled={!draft.trim() || adding}
        >
          {adding ? "Sharing…" : "Share"}
        </button>
      </form>
      {loadError ? (
        <p className="entity-panel__meta" role="alert">
          {loadError}
        </p>
      ) : collaborators === null ? (
        <p className="entity-panel__meta">Loading collaborators…</p>
      ) : collaborators.length === 0 ? (
        <p className="entity-panel__meta">Not shared with anyone yet.</p>
      ) : (
        <ul className="entity-panel__people" aria-label="Collaborators">
          {collaborators.map((person) => (
            <li key={person.id} className="entity-panel__person">
              <div className="entity-panel__person-name">
                <strong>{person.username}</strong>
                <span>{person.email}</span>
              </div>
              <button
                type="button"
                className="btn btn-ghost btn-sm btn--danger-ghost"
                disabled={removing !== null}
                aria-label={`Remove ${person.username}`}
                onClick={() => void handleRemove(person)}
              >
                {removing === person.id ? "Removing…" : "Remove"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

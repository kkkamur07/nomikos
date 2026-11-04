import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { userFacingMessage } from "../../api/userFacingError";
import { toast } from "../ui/toast";
import { slugify } from "../../utils/slugify";

type ProjectSettingsPanelProps = {
  projectId: string;
  name: string;
  guidelines: string | null;
  onUpdated: (patch: { name: string; guidelines: string | null }) => void;
};

export function ProjectSettingsPanel({
  projectId,
  name,
  guidelines,
  onUpdated,
}: ProjectSettingsPanelProps) {
  const [draftName, setDraftName] = useState(name);
  const [draftGuidelines, setDraftGuidelines] = useState(guidelines ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDraftName(name);
    setDraftGuidelines(guidelines ?? "");
  }, [name, guidelines]);

  async function handleSave() {
    const trimmedName = draftName.trim();
    if (!trimmedName) return;
    setSaving(true);
    try {
      const updated = await api.updateProject(projectId, {
        name: trimmedName,
        slug: slugify(trimmedName),
        guidelines: draftGuidelines.trim() || null,
      });
      onUpdated({ name: updated.name, guidelines: updated.guidelines ?? null });
      toast.success("Project updated");
    } catch (err) {
      const message = userFacingMessage(err, "Failed to update project");
      toast.error(message, err);
    } finally {
      setSaving(false);
    }
  }

  const hasChanges =
    draftName.trim() !== name ||
    (draftGuidelines.trim() || null) !== (guidelines?.trim() || null);

  return (
    <div className="entity-panel__section">
      <h2 className="entity-panel__heading">Project</h2>
      <div className="field">
        <label htmlFor="entity-project-name">Name</label>
        <input
          id="entity-project-name"
          value={draftName}
          onChange={(event) => setDraftName(event.target.value)}
        />
      </div>
      <div className="field">
        <label htmlFor="entity-project-guidelines">Guidelines</label>
        <textarea
          id="entity-project-guidelines"
          rows={4}
          value={draftGuidelines}
          placeholder="Optional notes for collaborators"
          onChange={(event) => setDraftGuidelines(event.target.value)}
        />
      </div>
      <button
        type="button"
        className="btn btn-primary btn-sm"
        disabled={!hasChanges || !draftName.trim() || saving}
        onClick={() => void handleSave()}
      >
        {saving ? "Saving…" : "Save changes"}
      </button>
    </div>
  );
}

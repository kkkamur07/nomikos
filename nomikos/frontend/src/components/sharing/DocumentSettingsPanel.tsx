import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { userFacingMessage } from "../../api/userFacingError";
import { toast } from "../ui/toast";

type DocumentSettingsPanelProps = {
  projectId: string;
  documentId: string;
  name: string;
  onUpdated: (patch: { name: string }) => void;
};

/**
 * The document's own settings, behind the title.
 *
 * Publishing used to live here too, as a bare "Publish document" button beside
 * the public URL. It has moved to the Publish menu in the action row, which is
 * the only place that can say how many pages are about to go live and how many
 * of them nobody has checked. Two buttons that publish the same document, one
 * of them without a confirm step or a count, is worse than one.
 */
export function DocumentSettingsPanel({
  projectId,
  documentId,
  name,
  onUpdated,
}: DocumentSettingsPanelProps) {
  const [draftName, setDraftName] = useState(name);
  const [savingName, setSavingName] = useState(false);

  useEffect(() => {
    setDraftName(name);
  }, [name]);

  async function handleSaveName() {
    const trimmed = draftName.trim();
    if (!trimmed || trimmed === name) return;
    setSavingName(true);
    try {
      const updated = await api.updateDocument(projectId, documentId, {
        name: trimmed,
      });
      onUpdated({ name: updated.name });
      toast.success("Document renamed");
    } catch (err) {
      const message = userFacingMessage(err, "Failed to rename document");
      toast.error(message, err);
    } finally {
      setSavingName(false);
    }
  }

  const nameChanged = draftName.trim() !== name && draftName.trim().length > 0;

  return (
    <div className="entity-panel__section">
      <h2 className="entity-panel__heading">Document</h2>
      <div className="field">
        <label htmlFor="entity-document-name">Name</label>
        <input
          id="entity-document-name"
          value={draftName}
          onChange={(event) => setDraftName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void handleSaveName();
            }
          }}
        />
      </div>
      <button
        type="button"
        className="btn btn-outline btn-sm"
        disabled={!nameChanged || savingName}
        onClick={() => void handleSaveName()}
      >
        {savingName ? "Saving…" : "Save name"}
      </button>
    </div>
  );
}

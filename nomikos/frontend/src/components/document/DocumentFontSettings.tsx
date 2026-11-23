import { useState } from "react";
import { api } from "../../api/client";
import { userFacingMessage } from "../../api/userFacingError";
import { toast } from "../ui/toast";
import {
  SCRIPT_FONTS,
  choosableScripts,
  effectiveChoice,
  type ScriptFonts,
} from "./scriptFonts";

type DocumentFontSettingsProps = {
  projectId: string;
  documentId: string;
  scriptFonts: ScriptFonts | null | undefined;
  onUpdated: (scriptFonts: ScriptFonts) => void;
};

function scriptLabel(script: string): string {
  return script.charAt(0).toUpperCase() + script.slice(1);
}

/** The mapping to send: the default for a script is stored as no entry. */
function nextScriptFonts(
  current: ScriptFonts | null | undefined,
  script: string,
  choice: string,
): ScriptFonts {
  const next: ScriptFonts = { ...(current ?? {}) };
  if (choice === SCRIPT_FONTS[script].defaultChoice) delete next[script];
  else next[script] = choice;
  return next;
}

/**
 * One select per script that has more than one bundled face. The editor and
 * the PDF and Word exports all follow the document's choice.
 */
export function DocumentFontSettings({
  projectId,
  documentId,
  scriptFonts,
  onUpdated,
}: DocumentFontSettingsProps) {
  const [saving, setSaving] = useState(false);
  const scripts = choosableScripts();
  if (scripts.length === 0) return null;

  async function handleChange(script: string, choice: string) {
    setSaving(true);
    try {
      const updated = await api.updateDocument(projectId, documentId, {
        script_fonts: nextScriptFonts(scriptFonts, script, choice),
      });
      onUpdated(updated.script_fonts ?? {});
      toast.success("Font updated");
    } catch (err) {
      toast.error(userFacingMessage(err, "Failed to update the font"), err);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="entity-panel__section">
      <h2 className="entity-panel__heading">Fonts</h2>
      {scripts.map((script) => {
        const id = `entity-document-font-${script}`;
        return (
          <div className="field" key={script}>
            <label htmlFor={id}>{scriptLabel(script)}</label>
            <select
              id={id}
              value={effectiveChoice(script, scriptFonts)}
              disabled={saving}
              onChange={(event) =>
                void handleChange(script, event.target.value)
              }
            >
              {SCRIPT_FONTS[script].choices.map((choice) => (
                <option key={choice.id} value={choice.id}>
                  {choice.label}
                </option>
              ))}
            </select>
          </div>
        );
      })}
    </div>
  );
}

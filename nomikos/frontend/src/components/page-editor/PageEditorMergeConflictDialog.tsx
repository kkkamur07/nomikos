import { useEffect, useState, type FormEvent } from "react";

import type { LineResponse } from "../../api/client";
import { FormModal } from "../ui/FormModal";
import { approvedText } from "./hooks/utils";
import { segmentNumbersById } from "./segmentNumbering";

/**
 * The one question a merge can ask back.
 *
 * Every other merge goes through without a prompt, because the server can work
 * out which row should keep its id: the one carrying somebody's text or
 * pairing. When more than one does there is no such answer, and guessing would
 * delete approved ground truth. So the server refuses, names the segments, and
 * this is where the reviewer picks.
 *
 * It shows the text rather than only the segment numbers. "Keep 4 or 7?" is
 * unanswerable without scrolling back to the page; the words are the thing
 * being chosen between.
 */

type PageEditorMergeConflictDialogProps = {
  open: boolean;
  lineIds: string[];
  lines: LineResponse[];
  busy: boolean;
  onCancel: () => void;
  onKeep: (lineId: string) => void;
};

export function PageEditorMergeConflictDialog({
  open,
  lineIds,
  lines,
  busy,
  onCancel,
  onKeep,
}: PageEditorMergeConflictDialogProps) {
  const [keepId, setKeepId] = useState<string | null>(null);

  // A fresh conflict is a fresh question: never carry the previous answer over,
  // or a second merge silently keeps whatever the first one did.
  useEffect(() => {
    setKeepId(open ? (lineIds[0] ?? null) : null);
  }, [open, lineIds]);

  if (!open) return null;

  const numbers = segmentNumbersById(lines);
  const choices = lineIds.map((id) => {
    const line = lines.find((candidate) => candidate.id === id) ?? null;
    return {
      id,
      number: numbers.get(id) ?? null,
      text: line ? (approvedText(line) ?? "") : "",
    };
  });

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (keepId) onKeep(keepId);
  }

  return (
    <FormModal
      open={open}
      title="Which text should the merged segment keep?"
      onClose={onCancel}
      onSubmit={handleSubmit}
      submitLabel="Merge and keep this text"
      loading={busy}
    >
      <p>
        More than one of the selected segments carries text or a pairing.
        Merging keeps one of them; the others are removed and their pairings are
        released.
      </p>
      <fieldset>
        <legend className="sr-only">Text to keep</legend>
        {choices.map((choice) => (
          <label key={choice.id} className="pe-merge-choice">
            <input
              type="radio"
              name="merge-keep-text"
              value={choice.id}
              checked={keepId === choice.id}
              onChange={() => setKeepId(choice.id)}
            />
            <span>
              {choice.number === null ? "Segment" : `Segment ${choice.number}`}
              {choice.text ? `: ${choice.text}` : " (paired, no approved text)"}
            </span>
          </label>
        ))}
      </fieldset>
    </FormModal>
  );
}

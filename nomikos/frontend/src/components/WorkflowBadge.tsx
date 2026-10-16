import type { DocumentWorkflow } from "../api/client";

const WORKFLOW_LABEL: Record<DocumentWorkflow, string> = {
  draft: "Draft",
  published: "Live",
  archived: "Archived",
};

const WORKFLOW_CLASS: Record<DocumentWorkflow, string> = {
  draft: "badge-draft",
  published: "badge-live",
  archived: "badge-archived",
};

export function WorkflowBadge({ workflow }: { workflow: DocumentWorkflow }) {
  return (
    <span className={`badge ${WORKFLOW_CLASS[workflow]}`}>
      {WORKFLOW_LABEL[workflow]}
    </span>
  );
}

export function ReviewBadge({ reviewed }: { reviewed: boolean }) {
  return (
    <span
      className={`badge ${reviewed ? "badge-reviewed" : "badge-unreviewed"}`}
    >
      {reviewed ? "reviewed" : "unreviewed"}
    </span>
  );
}

/**
 * Whether one page of a live document is reachable by the public reader.
 *
 * Deliberately not the same words as {@link WorkflowBadge}: a document is
 * "live", a page within it is "shown" or "hidden". Calling a page "draft" would
 * collide with the document workflow of that name, which is a different switch
 * with different consequences.
 */
export function PublishBadge({ published }: { published: boolean }) {
  return (
    <span className={`badge ${published ? "badge-live" : "badge-hidden"}`}>
      {/* Read in a row that already says "Part 1, unreviewed", a bare "shown"
          has no subject. Sighted readers get that from the column it sits in. */}
      <span className="sr-only">Publicly&nbsp;</span>
      {published ? "shown" : "hidden"}
    </span>
  );
}

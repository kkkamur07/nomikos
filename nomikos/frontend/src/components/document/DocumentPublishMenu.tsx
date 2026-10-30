import { useState } from "react";
import { api, type DocumentWorkflow } from "../../api/client";
import { ApiError } from "../../api/errors";
import { publicDocumentUrl } from "../../utils/publicDocumentUrl";
import {
  ActionMenu,
  ActionMenuConfirm,
  ActionMenuDivider,
  ActionMenuItem,
  ActionMenuSection,
  ActionMenuWarning,
} from "../ui/ActionMenu";
import { toast } from "../ui/toast";
import { pageCountLabel, publishConfirmSummary } from "./documentActionCopy";

type PendingStep = "publish" | "rotate" | null;

type DocumentPublishMenuProps = {
  projectId: string;
  documentId: string;
  workflow: DocumentWorkflow;
  /** Owner-only on the wire: a collaborator's document never carries it. */
  publicShareToken: string | null;
  /** Pages flagged to show, out of the pages the document has. */
  publishedPageCount: number;
  totalPageCount: number;
  reviewedPageCount: number;
  disabled?: boolean;
  onWorkflowChange: (workflow: DocumentWorkflow) => void;
  onShareTokenChange: (token: string | null) => void;
};

/**
 * Publishing, owner-only.
 *
 * The control this replaces was called "Publish live page" and sat in the page
 * editor: a document-level act wearing a page-level name, one click away from
 * the page a person was reading, with nothing on it saying how many pages it
 * was about to expose. Here it is named for the document it publishes, it
 * carries the page count in its own label, and the confirm step spells out how
 * many of those pages nobody has checked yet.
 */
export function DocumentPublishMenu({
  projectId,
  documentId,
  workflow,
  publicShareToken,
  publishedPageCount,
  totalPageCount,
  reviewedPageCount,
  disabled = false,
  onWorkflowChange,
  onShareTokenChange,
}: DocumentPublishMenuProps) {
  const [pending, setPending] = useState<PendingStep>(null);
  const [working, setWorking] = useState(false);

  const isPublished = workflow === "published";
  const isArchived = workflow === "archived";
  const busy = disabled || working || isArchived;
  const publicUrl = publicShareToken
    ? publicDocumentUrl(projectId, documentId, publicShareToken)
    : null;

  async function setWorkflow(next: DocumentWorkflow, close: () => void) {
    setWorking(true);
    try {
      const updated = await api.updateDocument(projectId, documentId, {
        workflow: next,
      });
      onWorkflowChange(updated.workflow);
      toast.success(
        next === "published"
          ? "Document published. The secret link is live"
          : "Document unpublished. Every link to it now 404s",
      );
      close();
    } catch (err) {
      toast.error(
        err instanceof ApiError
          ? err.message
          : "Could not change the document status",
      );
    } finally {
      setWorking(false);
      setPending(null);
    }
  }

  async function copyLink(close: () => void) {
    if (!publicUrl) return;
    try {
      await navigator.clipboard.writeText(publicUrl);
      toast.success("Secret link copied");
      close();
    } catch {
      toast.error("Could not copy the link");
    }
  }

  async function rotateLink(close: () => void) {
    setWorking(true);
    try {
      const updated = await api.rotateDocumentShareToken(projectId, documentId);
      onShareTokenChange(updated.public_share_token ?? null);
      toast.success("Link rotated. Every link already sent now 404s");
      close();
    } catch (err) {
      toast.error(
        err instanceof ApiError ? err.message : "Could not rotate the link",
      );
    } finally {
      setWorking(false);
      setPending(null);
    }
  }

  function PublishConfirmStep({
    working,
    totalPageCount,
    reviewedPageCount,
    publishedPageCount,
    onCancel,
    onConfirm,
  }: {
    working: boolean;
    totalPageCount: number;
    reviewedPageCount: number;
    publishedPageCount: number;
    onCancel: () => void;
    onConfirm: () => void;
  }): React.ReactNode {
    return (
      <ActionMenuConfirm
        busy={working}
        question={`Publish ${publishConfirmSummary(totalPageCount, reviewedPageCount)}.`}
        detail={`${publishedPageCount} of ${totalPageCount} pages are set to show. Anyone holding the secret link can read them.`}
        confirmLabel={working ? "Publishing…" : "Publish document"}
        onCancel={onCancel}
        onConfirm={onConfirm}
      />
    );
  }

  function RotateConfirmStep({
    working,
    onCancel,
    onConfirm,
  }: {
    working: boolean;
    onCancel: () => void;
    onConfirm: () => void;
  }): React.ReactNode {
    return (
      <ActionMenuConfirm
        destructive
        busy={working}
        question="Rotate the secret link?"
        detail="Every link already sent stops working. Anyone still reading through the old one loses access with no warning."
        confirmLabel={working ? "Rotating…" : "Yes, rotate the link"}
        onCancel={onCancel}
        onConfirm={onConfirm}
      />
    );
  }

  function PublishMenuItems({
    isPublished,
    isArchived,
    busy,
    totalPageCount,
    publishedPageCount,
    publicUrl,
    publicShareToken,
    onPublish,
    onUnpublish,
    onCopyLink,
    onRotate,
  }: {
    isPublished: boolean;
    isArchived: boolean;
    busy: boolean;
    totalPageCount: number;
    publishedPageCount: number;
    publicUrl: string | null;
    publicShareToken: string | null;
    onPublish: () => void;
    onUnpublish: () => void;
    onCopyLink: () => void;
    onRotate: () => void;
  }): React.ReactNode {
    return (
      <>
        {!isPublished && (
          <ActionMenuItem
            label={`Publish document (${pageCountLabel(totalPageCount)})`}
            detail={`Sets ${publishedPageCount} of ${totalPageCount} pages live. Readers reach them through one secret link.`}
            disabled={busy || totalPageCount === 0}
            onSelect={onPublish}
          />
        )}
        {isPublished && (
          <ActionMenuItem
            label={`Unpublish (${pageCountLabel(totalPageCount)})`}
            detail="Takes the public page down. Every link to it starts answering 404."
            disabled={busy}
            onSelect={onUnpublish}
          />
        )}
        {isArchived && (
          <ActionMenuWarning>
            This document is archived. Restore it before publishing.
          </ActionMenuWarning>
        )}
        {isPublished && (
          <>
            <ActionMenuDivider />
            <ActionMenuSection>Link</ActionMenuSection>
            <ActionMenuItem
              label="Copy secret link"
              disabled={busy || !publicUrl}
              onSelect={onCopyLink}
            />
            <ActionMenuItem
              destructive
              label="Rotate link"
              disabled={busy || !publicShareToken}
              onSelect={onRotate}
            />
            <ActionMenuWarning>
              Anyone holding the link can read it. Rotating breaks every link
              already sent.
            </ActionMenuWarning>
            {!publicShareToken && (
              <ActionMenuWarning>
                Only the project owner can get the secret link.
              </ActionMenuWarning>
            )}
          </>
        )}
      </>
    );
  }

  return (
    <ActionMenu
      label="Publish"
      menuLabel="Publish this document"
      wide
      onOpenChange={(open) => {
        if (!open) setPending(null);
      }}
    >
      {(close) => {
        if (pending === "publish") {
          return (
            <PublishConfirmStep
              working={working}
              totalPageCount={totalPageCount}
              reviewedPageCount={reviewedPageCount}
              publishedPageCount={publishedPageCount}
              onCancel={() => setPending(null)}
              onConfirm={() => void setWorkflow("published", close)}
            />
          );
        }
        if (pending === "rotate") {
          return (
            <RotateConfirmStep
              working={working}
              onCancel={() => setPending(null)}
              onConfirm={() => void rotateLink(close)}
            />
          );
        }
        return (
          <PublishMenuItems
            isPublished={isPublished}
            isArchived={isArchived}
            busy={busy}
            totalPageCount={totalPageCount}
            publishedPageCount={publishedPageCount}
            publicUrl={publicUrl}
            publicShareToken={publicShareToken}
            onPublish={() => setPending("publish")}
            onUnpublish={() => void setWorkflow("draft", close)}
            onCopyLink={() => void copyLink(close)}
            onRotate={() => setPending("rotate")}
          />
        );
      }}
    </ActionMenu>
  );
}

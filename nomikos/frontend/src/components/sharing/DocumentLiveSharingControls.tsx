import { useState } from "react";
import Link from "next/link";
import { api, type DocumentWorkflow } from "../../api/client";
import { ApiError } from "../../api/errors";
import { toast } from "../ui/toast";
import { WorkflowBadge } from "../WorkflowBadge";
import {
  publicDocumentPath,
  publicDocumentUrl,
} from "../../utils/publicDocumentUrl";

type DocumentLiveSharingControlsProps = {
  projectId: string;
  documentId: string;
  workflow: DocumentWorkflow;
  /**
   * Set only in the owner-facing document response - a collaborator's copy
   * carries `null` here even once the document is published, since the token
   * is what makes the link work and only the owner may hand it out.
   */
  publicShareToken: string | null;
  onWorkflowChange: (workflow: DocumentWorkflow) => void;
  disabled?: boolean;
};

/**
 * Publishing, as it appears inside the page editor's dropdown.
 *
 * This is the page editor's only route to publishing, which is why it stays.
 * The document page has its own Publish menu in the action row and no longer
 * renders this; the wider, panel-shaped variant this component used to grow
 * when `compact` was false went with it.
 */
function shareLinkUrls(
  projectId: string,
  documentId: string,
  publicShareToken: string | null,
): { publicPath: string | null; publicUrl: string | null } {
  if (!publicShareToken) return { publicPath: null, publicUrl: null };
  return {
    publicPath: publicDocumentPath(projectId, documentId, publicShareToken),
    publicUrl: publicDocumentUrl(projectId, documentId, publicShareToken),
  };
}

export function DocumentLiveSharingControls({
  projectId,
  documentId,
  workflow,
  publicShareToken,
  onWorkflowChange,
  disabled = false,
}: DocumentLiveSharingControlsProps) {
  const [publishing, setPublishing] = useState(false);
  const isPublished = workflow === "published";
  const isArchived = workflow === "archived";
  const { publicPath, publicUrl } = shareLinkUrls(
    projectId,
    documentId,
    publicShareToken,
  );
  const busy = disabled || publishing;

  async function handlePublishToggle() {
    if (isArchived) return;
    setPublishing(true);
    try {
      const nextWorkflow: DocumentWorkflow = isPublished
        ? "draft"
        : "published";
      const updated = await api.updateDocument(projectId, documentId, {
        workflow: nextWorkflow,
      });
      onWorkflowChange(updated.workflow);
      toast.success(
        nextWorkflow === "published"
          ? "Document published. Public link is live"
          : "Document returned to draft",
      );
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Failed to update document status";
      toast.error(message);
    } finally {
      setPublishing(false);
    }
  }

  async function handleCopyPublicLink() {
    if (!publicUrl) return;
    try {
      await navigator.clipboard.writeText(publicUrl);
      toast.success("Public link copied");
    } catch {
      toast.error("Could not copy link");
    }
  }

  function toggleLabel(publishing: boolean, isPublished: boolean): string {
    if (publishing) return "Updating…";
    return isPublished ? "Unpublish document" : "Publish document";
  }

  function PublicLinkBlock({
    documentId,
    publicUrl,
    publicPath,
    busy,
    onCopy,
  }: {
    documentId: string;
    publicUrl: string;
    publicPath: string;
    busy: boolean;
    onCopy: () => void;
  }): React.ReactNode {
    return (
      <div className="pe-dd-share">
        <label
          className="entity-panel__label"
          htmlFor={`public-url-${documentId}`}
        >
          Public page
        </label>
        <input
          id={`public-url-${documentId}`}
          className="pe-dd-share__url"
          type="text"
          readOnly
          value={publicUrl}
          aria-label="Public document URL"
        />
        <div className="pe-dd-share__actions">
          <button
            type="button"
            className="btn btn-outline btn-xs"
            disabled={busy}
            onClick={onCopy}
          >
            Copy link
          </button>
          <Link
            href={publicPath}
            className="btn btn-ghost btn-xs"
            target="_blank"
            rel="noopener noreferrer"
          >
            Open public view
          </Link>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="entity-panel__meta">
        <WorkflowBadge workflow={workflow} />
      </div>
      {isPublished && publicUrl && publicPath && (
        <PublicLinkBlock
          documentId={documentId}
          publicUrl={publicUrl}
          publicPath={publicPath}
          busy={busy}
          onCopy={() => void handleCopyPublicLink()}
        />
      )}
      {/*
        A collaborator's document response never carries the token - only the
        owner may hand out the link - so there is nothing here to build. Say
        so plainly rather than showing a link that 404s for whoever opens it.
      */}
      {isPublished && !publicShareToken && (
        <p className="pe-dd-share">
          Only the project owner can get the public share link.
        </p>
      )}
      {!isArchived && (
        <button
          type="button"
          role="menuitem"
          className="pe-dd-item"
          disabled={busy}
          onClick={() => void handlePublishToggle()}
        >
          {toggleLabel(publishing, isPublished)}
        </button>
      )}
    </>
  );
}

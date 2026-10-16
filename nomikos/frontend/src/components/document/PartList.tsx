import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import type { DocumentPartResponse } from "../../api/client";
import { prefetchPartImage } from "../../api/imageCache";
import { PublishBadge, ReviewBadge } from "../WorkflowBadge";
import { AuthenticatedImage } from "../AuthenticatedImage";

type PartListProps = {
  parts: DocumentPartResponse[];
  projectId: string;
  documentId: string;
  loading?: boolean;
  onMoveUp?: (index: number) => void;
  onMoveDown?: (index: number) => void;
  onDelete?: (partId: string) => void;
  onToggleReview?: (partId: string, reviewed: boolean) => void;
  onTogglePublished?: (partId: string, published: boolean) => void;
  reviewUpdatingPartId?: string | null;
  publishUpdatingPartId?: string | null;
  reordering?: boolean;
};

export function PartList({
  parts,
  projectId,
  documentId,
  loading = false,
  onMoveUp,
  onMoveDown,
  onDelete,
  onToggleReview,
  onTogglePublished,
  reviewUpdatingPartId = null,
  publishUpdatingPartId = null,
  reordering = false,
}: PartListProps) {
  if (!loading && parts.length === 0) {
    // The only place worth naming both ways in. Once there are pages the
    // action row is enough, but an empty document has nothing to point at, and
    // dropping onto the window works without anything on screen saying so.
    return (
      <p className="text-muted text-sm">
        No pages yet. Use Upload pages, or drop images and PDFs anywhere on this
        page.
      </p>
    );
  }

  return (
    <div className="part-list" role="list" aria-labelledby="pages-label">
      {parts.map((part, index) => (
        <PartRow
          key={part.id}
          part={part}
          index={index}
          total={parts.length}
          projectId={projectId}
          documentId={documentId}
          onMoveUp={onMoveUp ? () => onMoveUp(index) : undefined}
          onMoveDown={onMoveDown ? () => onMoveDown(index) : undefined}
          onDelete={onDelete ? () => onDelete(part.id) : undefined}
          onToggleReview={
            onToggleReview
              ? (reviewed) => onToggleReview(part.id, reviewed)
              : undefined
          }
          onTogglePublished={
            onTogglePublished
              ? (published) => onTogglePublished(part.id, published)
              : undefined
          }
          reviewUpdating={reviewUpdatingPartId === part.id}
          publishUpdating={publishUpdatingPartId === part.id}
          reordering={reordering}
        />
      ))}
    </div>
  );
}

type PartRowProps = {
  part: DocumentPartResponse;
  index: number;
  total: number;
  projectId: string;
  documentId: string;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onDelete?: () => void;
  onToggleReview?: (reviewed: boolean) => void;
  onTogglePublished?: (published: boolean) => void;
  reviewUpdating?: boolean;
  publishUpdating?: boolean;
  reordering?: boolean;
};

function thumbnailUrlFor(imageUrl: string | null): string | null {
  if (!imageUrl) return null;
  return `${imageUrl}${imageUrl.includes("?") ? "&" : "?"}w=200`;
}

function PartThumbnail({
  part,
  index,
}: {
  part: DocumentPartResponse;
  index: number;
}): React.ReactNode {
  const thumbnailUrl = thumbnailUrlFor(part.image_url);
  return (
    <div className="part-thumb" aria-hidden={!!part.image_url}>
      {thumbnailUrl ? (
        <AuthenticatedImage
          compact
          src={thumbnailUrl}
          alt={`Part ${index + 1}`}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "block",
          }}
        />
      ) : (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth="1.5"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Z"
          />
        </svg>
      )}
    </div>
  );
}

function PartInfo({
  part,
  index,
  total,
}: {
  part: DocumentPartResponse;
  index: number;
  total: number;
}): React.ReactNode {
  const dim =
    part.width && part.height
      ? `${part.width} × ${part.height}`
      : "Dimensions pending";
  return (
    <div className="part-info">
      <div className="part-num-row">
        <span className="part-num">Part {index + 1}</span>
        <ReviewBadge reviewed={part.reviewed} />
        <PublishBadge published={part.published} />
      </div>
      <div className="part-desc">
        Page {index + 1} of {total}
      </div>
      <div className="part-dim">{dim}</div>
    </div>
  );
}

function publishToggleLabel(
  index: number,
  published: boolean,
  publishUpdating: boolean,
): string {
  if (publishUpdating) return `Updating part ${index + 1}`;
  return published
    ? `Hide part ${index + 1} from the public page`
    : `Show part ${index + 1} on the public page`;
}

function PublishToggle({
  part,
  index,
  publishUpdating,
  onTogglePublished,
}: {
  part: DocumentPartResponse;
  index: number;
  publishUpdating: boolean;
  onTogglePublished: (published: boolean) => void;
}): React.ReactNode {
  return (
    <button
      type="button"
      className="btn btn-ghost btn-sm"
      disabled={publishUpdating}
      aria-busy={publishUpdating}
      onClick={() => onTogglePublished(!part.published)}
      aria-label={publishToggleLabel(index, part.published, publishUpdating)}
    >
      {publishUpdating ? "…" : part.published ? "Hide" : "Show"}
    </button>
  );
}

function ReviewToggle({
  part,
  index,
  reviewUpdating,
  onToggleReview,
}: {
  part: DocumentPartResponse;
  index: number;
  reviewUpdating: boolean;
  onToggleReview: (reviewed: boolean) => void;
}): React.ReactNode {
  return (
    <button
      type="button"
      className="btn btn-ghost btn-sm"
      disabled={reviewUpdating}
      onClick={() => onToggleReview(!part.reviewed)}
      aria-label={
        part.reviewed
          ? `Mark part ${index + 1} unreviewed`
          : `Mark part ${index + 1} reviewed`
      }
    >
      {part.reviewed ? "Unreview" : "Review"}
    </button>
  );
}
function MoveButtons({
  index,
  total,
  reordering,
  onMoveUp,
  onMoveDown,
}: {
  index: number;
  total: number;
  reordering: boolean | undefined;
  onMoveUp: (() => void) | undefined;
  onMoveDown: (() => void) | undefined;
}): React.ReactNode {
  return (
    <>
      {onMoveUp && (
        <button
          type="button"
          className="btn btn-ghost btn-xs"
          disabled={index === 0 || reordering}
          onClick={onMoveUp}
          aria-label={`Move part ${index + 1} up`}
        >
          ↑
        </button>
      )}
      {onMoveDown && (
        <button
          type="button"
          className="btn btn-ghost btn-xs"
          disabled={index === total - 1 || reordering}
          onClick={onMoveDown}
          aria-label={`Move part ${index + 1} down`}
        >
          ↓
        </button>
      )}
    </>
  );
}

function PartRowActions({
  part,
  index,
  total,
  onMoveUp,
  onMoveDown,
  onDelete,
  onToggleReview,
  onTogglePublished,
  reviewUpdating,
  publishUpdating,
  reordering,
}: {
  part: DocumentPartResponse;
  index: number;
  total: number;
  onMoveUp: (() => void) | undefined;
  onMoveDown: (() => void) | undefined;
  onDelete: (() => void) | undefined;
  onToggleReview: ((reviewed: boolean) => void) | undefined;
  onTogglePublished: ((published: boolean) => void) | undefined;
  reviewUpdating: boolean;
  publishUpdating: boolean;
  reordering: boolean | undefined;
}): React.ReactNode {
  return (
    <div
      className="part-actions"
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => e.stopPropagation()}
    >
      {onTogglePublished && (
        <PublishToggle
          part={part}
          index={index}
          publishUpdating={publishUpdating}
          onTogglePublished={onTogglePublished}
        />
      )}
      {onToggleReview && (
        <ReviewToggle
          part={part}
          index={index}
          reviewUpdating={reviewUpdating}
          onToggleReview={onToggleReview}
        />
      )}
      <MoveButtons
        index={index}
        total={total}
        reordering={reordering}
        onMoveUp={onMoveUp}
        onMoveDown={onMoveDown}
      />
      {onDelete && (
        <button
          type="button"
          className="btn btn-ghost btn-xs"
          onClick={onDelete}
          aria-label={`Remove part ${index + 1}`}
        >
          ✕
        </button>
      )}
    </div>
  );
}

function PartRow({
  part,
  index,
  total,
  projectId,
  documentId,
  onMoveUp,
  onMoveDown,
  onDelete,
  onToggleReview,
  onTogglePublished,
  reviewUpdating = false,
  publishUpdating = false,
  reordering,
}: PartRowProps) {
  const router = useRouter();
  const prefetchTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const editorPath = `/projects/${projectId}/documents/${documentId}/parts/${part.id}`;

  const cancelPrefetch = () => {
    if (prefetchTimer.current !== undefined) {
      clearTimeout(prefetchTimer.current);
      prefetchTimer.current = undefined;
    }
  };

  const scheduleFullImagePrefetch = () => {
    if (!part.image_url || prefetchTimer.current !== undefined) return;
    prefetchTimer.current = setTimeout(() => {
      prefetchTimer.current = undefined;
      prefetchPartImage(part.image_url);
    }, 100);
  };

  useEffect(() => () => clearTimeout(prefetchTimer.current), []);

  const openEditor = () => {
    router.push(editorPath);
  };

  const onRowKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openEditor();
    }
  };

  return (
    <article
      className="part-row part-row--clickable"
      role="listitem"
      tabIndex={0}
      aria-label={`Open part ${index + 1} in editor`}
      onClick={openEditor}
      onKeyDown={onRowKeyDown}
      onMouseEnter={scheduleFullImagePrefetch}
      onMouseLeave={cancelPrefetch}
      onFocus={scheduleFullImagePrefetch}
      onBlur={cancelPrefetch}
    >
      <PartThumbnail part={part} index={index} />
      <PartInfo part={part} index={index} total={total} />
      <PartRowActions
        part={part}
        index={index}
        total={total}
        onMoveUp={onMoveUp}
        onMoveDown={onMoveDown}
        onDelete={onDelete}
        onToggleReview={onToggleReview}
        onTogglePublished={onTogglePublished}
        reviewUpdating={reviewUpdating}
        publishUpdating={publishUpdating}
        reordering={reordering}
      />
    </article>
  );
}

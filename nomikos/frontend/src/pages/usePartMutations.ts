import { useState } from "react";
import { toast } from "../components/ui/toast";
import {
  api,
  type DocumentPartResponse,
  type DocumentWorkflow,
} from "../api/client";
import { ApiError } from "../api/errors";
import { invalidatePartImage } from "../api/imageCache";
import { partPublishedMessage } from "../components/document/documentActionCopy";

export function usePartMutations(options: {
  projectId: string | undefined;
  documentId: string | undefined;
  parts: DocumentPartResponse[];
  workflow: DocumentWorkflow | undefined;
  onChanged: () => Promise<void>;
}): {
  reordering: boolean;
  reviewUpdatingPartId: string | null;
  publishUpdatingPartId: string | null;
  movePart: (index: number, direction: -1 | 1) => void;
  handleDelete: (partId: string) => Promise<void>;
  handleToggleReview: (partId: string, reviewed: boolean) => Promise<void>;
  handleTogglePublished: (partId: string, published: boolean) => Promise<void>;
} {
  const { projectId, documentId, parts, workflow, onChanged } = options;
  const [reordering, setReordering] = useState(false);
  const [reviewUpdatingPartId, setReviewUpdatingPartId] = useState<
    string | null
  >(null);
  const [publishUpdatingPartId, setPublishUpdatingPartId] = useState<
    string | null
  >(null);

  const persistOrder = async (partIds: string[]) => {
    if (!projectId || !documentId) return;
    setReordering(true);
    try {
      await api.reorderParts(projectId, documentId, { part_ids: partIds });
      await onChanged();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Reorder failed";
      toast.error(msg);
    } finally {
      setReordering(false);
    }
  };

  const movePart = (index: number, direction: -1 | 1) => {
    const next = index + direction;
    if (next < 0 || next >= parts.length) return;
    const ids = parts.map((p) => p.id);
    [ids[index], ids[next]] = [ids[next], ids[index]];
    void persistOrder(ids);
  };

  const handleDelete = async (partId: string) => {
    if (!projectId || !documentId) return;
    try {
      await api.deletePart(projectId, documentId, partId);
      invalidatePartImage(partId);
      toast.success("Part removed");
      await onChanged();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Delete failed";
      toast.error(msg);
    }
  };

  const handleToggleReview = async (partId: string, reviewed: boolean) => {
    if (!projectId || !documentId) return;
    setReviewUpdatingPartId(partId);
    try {
      await api.updatePartReviewStatus(projectId, documentId, partId, {
        reviewed,
      });
      toast.success(
        reviewed ? "Part marked reviewed" : "Part marked unreviewed",
      );
      await onChanged();
    } catch (err) {
      const msg =
        err instanceof ApiError && err.status === 403
          ? "Only project members can change review status."
          : err instanceof ApiError
            ? err.message
            : "Review status update failed";
      toast.error(msg);
    } finally {
      // Only if this part is still the one in flight. A slower earlier request
      // resolving second would otherwise re-enable the button for a later one.
      setReviewUpdatingPartId((current) =>
        current === partId ? null : current,
      );
    }
  };

  const handleTogglePublished = async (partId: string, published: boolean) => {
    if (!projectId || !documentId) return;
    setPublishUpdatingPartId(partId);
    try {
      await api.updatePartsPublished(projectId, documentId, {
        parts: [{ part_id: partId, published }],
      });
      // On a draft nothing is public yet, and saying otherwise would be the one
      // place in this flow that overstates what just happened.
      toast.success(partPublishedMessage(workflow, published));
      await onChanged();
    } catch (err) {
      const msg =
        err instanceof ApiError && err.status === 403
          ? "Only the project owner can choose which pages are public."
          : err instanceof ApiError
            ? err.message
            : "Could not change which pages are public";
      toast.error(msg);
    } finally {
      setPublishUpdatingPartId((current) =>
        current === partId ? null : current,
      );
    }
  };

  return {
    reordering,
    reviewUpdatingPartId,
    publishUpdatingPartId,
    movePart,
    handleDelete,
    handleToggleReview,
    handleTogglePublished,
  };
}

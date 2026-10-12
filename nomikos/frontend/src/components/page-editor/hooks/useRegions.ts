import {
  useCallback,
  useEffect,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  api,
  type LinePoint,
  type LineResponse,
  type PartLayoutResponse,
} from "../../../api/client";
import { ApiError } from "../../../api/errors";
import { invalidateAfter } from "../../../api/resources";
import { toast } from "../../ui/toast";
import type { RegionEdit } from "./useLayoutMutations";

type RegionsInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  layout: PartLayoutResponse;
  setLayout: Dispatch<SetStateAction<PartLayoutResponse>>;
  lines: LineResponse[];
  setLines: Dispatch<SetStateAction<LineResponse[]>>;
  /** Puts a region create or delete on the editor's undo stack. */
  onRecordEdit?: (edit: RegionEdit) => void;
};

function regionErrorMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return "Only project members can edit regions.";
  }
  return error instanceof Error ? error.message : "Region update failed.";
}

/**
 * Regions (ADR 0011): draw, name, delete, and move Segments between them.
 *
 * Every one of these writes renumbers the Page on the server (Segments are
 * numbered region by region), so each is followed by a reload of the layout
 * and the Segments rather than a local patch: the server's order is the one
 * the transcript and the exports show.
 */
export function useRegions({
  projectId,
  documentId,
  partId,
  layout,
  setLayout,
  lines,
  setLines,
  onRecordEdit,
}: RegionsInput) {
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [regionError, setRegionError] = useState<string | null>(null);

  useEffect(() => {
    setSelectedRegionId(null);
    setRegionError(null);
  }, [projectId, documentId, partId]);

  const reload = useCallback(async () => {
    if (!projectId || !documentId || !partId) return;
    const [nextLayout, nextLines] = await Promise.all([
      api.getPartLayout(projectId, documentId, partId),
      api.listPartLines(projectId, documentId, partId),
    ]);
    setLayout(nextLayout);
    setLines(nextLines);
    invalidateAfter.partContentChanged(projectId, documentId);
  }, [projectId, documentId, partId, setLayout, setLines]);

  const run = useCallback(
    async (write: () => Promise<unknown>): Promise<boolean> => {
      setRegionError(null);
      try {
        await write();
        await reload();
        return true;
      } catch (error) {
        const message = regionErrorMessage(error);
        // The panel shows it only while a region or Segment is selected, which
        // a first draw never is, so the toast is what makes a failure visible.
        setRegionError(message);
        toast.error(message);
        return false;
      }
    },
    [reload],
  );

  /**
   * Draw a region; the Segments inside it that have no region join it. No
   * `order`: the server ranks it by where it sits on the Page.
   */
  const createRegion = useCallback(
    async (box: LinePoint[]) => {
      if (!projectId || !documentId || !partId) return;
      let createdId: string | null = null;
      const ok = await run(async () => {
        const created = await api.createPartBlock(
          projectId,
          documentId,
          partId,
          { box: { points: box }, assign_contained: true },
        );
        createdId = created.id;
        onRecordEdit?.({
          kind: "region-create",
          region: { id: created.id, box: created.box, label: null },
        });
      });
      if (ok) setSelectedRegionId(createdId);
    },
    [projectId, documentId, partId, run, onRecordEdit],
  );

  const renameRegion = useCallback(
    async (regionId: string, label: string) => {
      if (!projectId || !documentId || !partId) return;
      const next = label.trim();
      const current = layout.blocks.find((block) => block.id === regionId);
      if ((current?.label ?? "") === next) return;
      await run(() =>
        api.patchPartBlock(projectId, documentId, partId, regionId, {
          label: next || null,
        }),
      );
    },
    [projectId, documentId, partId, layout.blocks, run],
  );

  const deleteRegion = useCallback(
    async (regionId: string) => {
      if (!projectId || !documentId || !partId) return;
      const region = layout.blocks.find((block) => block.id === regionId);
      const lineIds = lines
        .filter((line) => line.block_id === regionId)
        .map((line) => line.id);
      if (
        lineIds.length > 0 &&
        !window.confirm(
          `Delete this region? Its ${lineIds.length} segment${lineIds.length === 1 ? "" : "s"} stay on the page and lose their region.`,
        )
      ) {
        return;
      }
      const ok = await run(async () => {
        await api.deletePartBlock(projectId, documentId, partId, regionId);
        if (region?.box) {
          onRecordEdit?.({
            kind: "region-delete",
            region: {
              id: region.id,
              box: region.box,
              label: region.label ?? null,
            },
            lineIds,
          });
        }
      });
      if (ok) setSelectedRegionId(null);
    },
    [projectId, documentId, partId, layout.blocks, lines, run, onRecordEdit],
  );

  /** Move one Segment into a region, or out of every region with null. */
  const assignSegmentToRegion = useCallback(
    async (segmentId: string, regionId: string | null) => {
      if (!projectId || !documentId || !partId) return;
      await run(() =>
        api.patchPartLine(projectId, documentId, partId, segmentId, {
          block_id: regionId,
        }),
      );
    },
    [projectId, documentId, partId, run],
  );

  return {
    selectedRegionId,
    setSelectedRegionId,
    regionError,
    createRegion,
    renameRegion,
    deleteRegion,
    assignSegmentToRegion,
  };
}

import { useEffect, useState } from "react";
import {
  api,
  type LineResponse,
  type PartLayoutResponse,
} from "../../../api/client";
import { INFERENCE_JOB_WAIT_CEILING_MS } from "../jobProgress";
import { statusMessage, type StatusMessage } from "../statusMessage";
import {
  reportJobFailure,
  showPairing,
  throwIfCancelled,
  type PairingProgressSetter,
  type SetState,
  type TextLinesSetter,
  type TrackJobAndWait,
} from "./editorHookTypes";

type AutoSegmentInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  lineCount: number;
  setLines: SetState<LineResponse[]>;
  setLayout: SetState<PartLayoutResponse>;
  setTextLines: TextLinesSetter;
  setPairingProgress: PairingProgressSetter;
  setPairingError: SetState<string | null>;
  setSubmissionRefusal: SetState<string | null>;
  selectedSegmentModelId: string | null;
  trackJobAndWait: TrackJobAndWait;
  notePartContentChanged: () => void;
  /** Empties the undo and redo stacks, whose line ids the run replaced. */
  clearEditStacks: () => void;
  /** Drops the selections and the draft box once the new Segments show. */
  onSegmentsReloaded: () => void;
};

/**
 * The sentence a finished segmentation reports.
 *
 * It names neither a host nor a model: the job does that itself, on the job,
 * which is the entire user interface for **execution target** (ADR 0002), and
 * the picker can choose a segmenter other than Kraken.
 */
function segmentationMessage(segmentCount: number): string {
  return `Line segmentation completed (${segmentCount} Segment(s)).`;
}

/** Line segmentation of the page: enqueue, wait, reload the new Segments. */
export function useAutoSegment(input: AutoSegmentInput) {
  const { projectId, documentId, partId } = input;
  // A count, not a flag: a superseded run unwinds while its successor is still
  // going, and must not report the page as idle on the way out.
  const [segmentRunCount, setSegmentRunCount] = useState(0);
  const [segmentMessage, setSegmentMessage] = useState<StatusMessage | null>(
    null,
  );

  useEffect(() => {
    setSegmentMessage(null);
  }, [projectId, documentId, partId]);

  /**
   * The page state that follows a finished segmentation. Returns the Segment
   * count for the sentence it reports.
   *
   * The segmentation already replaced every Segment on the server, so any edit
   * still on the undo/redo stack names a line id this reload is about to make
   * up. The stacks are cleared first and unconditionally: the segmentation is
   * stored by the time a caller reaches this function, whether or not the
   * reload below succeeds.
   */
  async function reloadAfterSegmentation(
    projectId: string,
    documentId: string,
    partId: string,
  ): Promise<number> {
    input.clearEditStacks();
    const [reloadedLines, reloadedLayout, pairing] = await Promise.all([
      api.listPartLines(projectId, documentId, partId),
      api.getPartLayout(projectId, documentId, partId),
      api.getPagePairing(projectId, documentId, partId),
    ]);
    input.setLines(reloadedLines);
    input.setLayout(reloadedLayout ?? { blocks: [], lines: [] });
    input.onSegmentsReloaded();
    showPairing(pairing, input.setTextLines, input.setPairingProgress);
    return reloadedLines.length;
  }

  async function runAutoSegment(modelId?: string | null) {
    if (!projectId || !documentId || !partId) return;
    const chosenModelId =
      modelId === undefined ? input.selectedSegmentModelId : modelId;
    if (
      input.lineCount > 0 &&
      !window.confirm(
        "Run line segmentation? Existing machine Segments on this Page will be replaced.",
      )
    ) {
      return;
    }
    setSegmentRunCount((count) => count + 1);
    setSegmentMessage(null);
    input.setPairingError(null);
    input.setSubmissionRefusal(null);

    try {
      const enqueued = await api.segmentPart(
        projectId,
        documentId,
        partId,
        chosenModelId ? { model_id: chosenModelId } : {},
      );
      const job = await input.trackJobAndWait(
        enqueued,
        { label: "Line segmentation", kind: "segmentation" },
        { timeoutMs: INFERENCE_JOB_WAIT_CEILING_MS },
      );
      throwIfCancelled(job);
      input.notePartContentChanged();

      // The segmentation is stored by now and only the reload is left. A
      // failure here is a stale view of saved Segments, so it surfaces as an
      // error banner and stops: the write is already finished, and there is
      // nothing left for it to re-run.
      const count = await reloadAfterSegmentation(
        projectId,
        documentId,
        partId,
      );
      setSegmentMessage(statusMessage(segmentationMessage(count)));
    } catch (err) {
      reportJobFailure(err, "Auto segment failed.", {
        setSubmissionRefusal: input.setSubmissionRefusal,
        setError: input.setPairingError,
      });
    } finally {
      setSegmentRunCount((count) => Math.max(0, count - 1));
    }
  }

  return {
    segmenting: segmentRunCount > 0,
    segmentMessage,
    runAutoSegment,
  };
}

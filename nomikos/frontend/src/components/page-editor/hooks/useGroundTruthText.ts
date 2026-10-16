import type { RefObject } from "react";
import {
  api,
  type LineResponse,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import { statusMessage, type StatusMessage } from "../statusMessage";
import {
  errorMessage,
  showPairing,
  type PairingProgressSetter,
  type SetState,
  type TextLinesSetter,
} from "./editorHookTypes";
import {
  lineTextForLayer,
  modelLayerIdForPromotion,
  withLocalGroundTruth,
} from "./utils";

type GroundTruthTextInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  setLines: SetState<LineResponse[]>;
  groundTruthTranscriptionId: string | null;
  setSelectedTranscriptionLayerId: SetState<string | null>;
  selectedTranscriptionLayer: TranscriptionLayerResponse | null;
  selectedSegmentId: string | null;
  selectedSegment: LineResponse | null;
  /** The segment open now, read after an await instead of the render's copy. */
  selectedSegmentIdRef: RefObject<string | null>;
  approvedTextDraft: string;
  setApprovedTextDraft: SetState<string>;
  setTranscriptionSaveMessage: SetState<StatusMessage | null>;
  setTextLines: TextLinesSetter;
  setPairingProgress: PairingProgressSetter;
  setPairingError: SetState<string | null>;
  notePartContentChanged: () => void;
};

type GroundTruthWrite = {
  /** Why this write may not run, or null when it may. */
  refusal: string | null;
  fallback: string;
  /** Shown once the write lands; a write without one leaves the line alone. */
  successMessage?: string;
};

/**
 * Every write of a segment's Ground truth text: pairing a text line, saving
 * the draft, and promoting a model line. A write that resolves after the
 * researcher opened another segment never touches the new segment's draft.
 */
export function useGroundTruthText(input: GroundTruthTextInput) {
  const {
    projectId,
    documentId,
    partId,
    setLines,
    groundTruthTranscriptionId,
    selectedSegmentId,
    selectedSegmentIdRef,
    setApprovedTextDraft,
    setTranscriptionSaveMessage,
    setPairingError,
    notePartContentChanged,
  } = input;

  function stillSelected(segmentId: string): boolean {
    return selectedSegmentIdRef.current === segmentId;
  }

  function foldGroundTruth(segmentId: string, text: string) {
    // Read-modify-write after an await: fold into the current segments, not
    // into the render-time snapshot, or a concurrent edit is reverted.
    setLines((current) =>
      withLocalGroundTruth(
        current,
        groundTruthTranscriptionId,
        segmentId,
        text,
      ),
    );
  }

  async function pairTextLine(order: number) {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    const segmentId = selectedSegmentId;
    try {
      const pairing = await api.pairTextLine(projectId, documentId, partId, {
        line_id: segmentId,
        text_line_order: order,
      });
      const candidate = pairing.text_lines.find(
        (textLine) => textLine.order === order,
      );
      if (candidate) {
        foldGroundTruth(segmentId, candidate.text);
        if (stillSelected(segmentId)) setApprovedTextDraft(candidate.text);
      }
      showPairing(pairing, input.setTextLines, input.setPairingProgress);
      notePartContentChanged();
      setPairingError(null);
    } catch (err) {
      setPairingError(errorMessage(err, "Failed to pair Text line."));
    }
  }

  /** Resolves true when the text was saved, false when the save failed. */
  async function writeGroundTruthText({
    refusal,
    fallback,
    successMessage,
  }: GroundTruthWrite): Promise<boolean> {
    if (!projectId || !documentId || !partId || !selectedSegmentId) {
      return false;
    }
    if (refusal !== null || !groundTruthTranscriptionId) {
      setPairingError(refusal);
      return false;
    }
    try {
      const updated = await api.updateGroundTruthLineText(
        projectId,
        documentId,
        groundTruthTranscriptionId,
        selectedSegmentId,
        { text: input.approvedTextDraft },
      );
      foldGroundTruth(selectedSegmentId, updated.text);
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      showPairing(pairing, input.setTextLines, input.setPairingProgress);
      notePartContentChanged();
      setPairingError(null);
      if (successMessage) {
        setTranscriptionSaveMessage(statusMessage(successMessage));
      }
      return true;
    } catch (err) {
      if (successMessage) setTranscriptionSaveMessage(null);
      setPairingError(errorMessage(err, fallback));
      return false;
    }
  }

  function saveApprovedText(): Promise<boolean> {
    return writeGroundTruthText({
      refusal: groundTruthTranscriptionId
        ? null
        : "Ground truth transcription layer is not available.",
      fallback: "Failed to save approved text.",
    });
  }

  function saveGroundTruthText(): Promise<boolean> {
    const editable =
      groundTruthTranscriptionId &&
      input.selectedTranscriptionLayer?.kind === "ground_truth";
    return writeGroundTruthText({
      refusal: editable ? null : "Only Ground truth can be edited.",
      fallback: "Failed to save Ground truth text.",
      successMessage: "Ground truth text saved",
    });
  }

  async function promoteSelectedSegmentToGroundTruth() {
    const segment = input.selectedSegment;
    if (!projectId || !documentId || !partId || !segment) return;
    const modelLayerId = modelLayerIdForPromotion(
      segment,
      input.selectedTranscriptionLayer,
    );
    if (!modelLayerId) {
      setPairingError(
        "Model transcription is not available to save as Ground truth.",
      );
      return;
    }
    try {
      await api.copyToGroundTruth(projectId, documentId, modelLayerId, {
        line_ids: [segment.id],
      });
      const [reloadedLines, pairing] = await Promise.all([
        api.listPartLines(projectId, documentId, partId),
        api.getPagePairing(projectId, documentId, partId),
      ]);
      setLines(reloadedLines);
      showPairing(pairing, input.setTextLines, input.setPairingProgress);
      notePartContentChanged();
      setPairingError(null);
      if (groundTruthTranscriptionId) {
        input.setSelectedTranscriptionLayerId(groundTruthTranscriptionId);
        const reloaded = reloadedLines.find((line) => line.id === segment.id);
        if (reloaded && stillSelected(segment.id)) {
          setApprovedTextDraft(
            lineTextForLayer(reloaded, groundTruthTranscriptionId),
          );
        }
      }
      setTranscriptionSaveMessage(statusMessage("Saved to Ground truth"));
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(errorMessage(err, "Failed to save to Ground truth."));
    }
  }

  return {
    pairTextLine,
    saveApprovedText,
    saveGroundTruthText,
    promoteSelectedSegmentToGroundTruth,
  };
}

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
  approvedText,
  initialDraftFor,
  modelLayerIdForPromotion,
  modelReadingFor,
  segmentHasGroundTruth,
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

/**
 * Every write of a segment's Ground truth text: pairing a text line, saving
 * the draft, and approving a model line. A write that resolves after the
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

  /**
   * Saves one Segment's Ground truth text. The one write path for approved
   * text: the strip saves the open Segment through it, the transcript column
   * saves any row. Throws when the save fails and leaves reporting it to the
   * caller: a transcript row says it in the row, the strip in the page alert.
   * ``previousText`` is the text the edit started from; the server answers
   * 409 when it no longer holds it. ``seededFromTranscriptionId`` names the
   * model layer a corrected reading came from; the server records it on a
   * Ground truth row that has no seed yet, in the same write.
   */
  async function commitLineText(
    lineId: string,
    text: string,
    previousText?: string,
    seededFromTranscriptionId?: string,
  ): Promise<void> {
    if (!projectId || !documentId || !partId) {
      throw new Error("Page context is missing. Reload and try again.");
    }
    if (!groundTruthTranscriptionId) {
      throw new Error("Ground truth transcription layer is not available.");
    }
    const updated = await api.updateGroundTruthLineText(
      projectId,
      documentId,
      groundTruthTranscriptionId,
      lineId,
      {
        text,
        ...(previousText === undefined ? {} : { previous_text: previousText }),
        ...(seededFromTranscriptionId === undefined
          ? {}
          : { seeded_from_transcription_id: seededFromTranscriptionId }),
      },
    );
    foldGroundTruth(lineId, updated.text);
    const pairing = await api.getPagePairing(projectId, documentId, partId);
    showPairing(pairing, input.setTextLines, input.setPairingProgress);
    notePartContentChanged();
  }

  /**
   * Copies the model's reading of one Segment into Ground truth. Recording the
   * copy is what keeps the link from the approved text to the model layer it
   * came from (``seeded_from_transcription_id``), which a later correction of
   * the same row keeps.
   */
  async function copyModelReading(segmentId: string, modelLayerId: string) {
    if (!projectId || !documentId) return;
    await api.copyToGroundTruth(projectId, documentId, modelLayerId, {
      line_ids: [segmentId],
    });
  }

  /**
   * Saves the strip's draft as the open Segment's Ground truth. A Segment with
   * no Ground truth yet whose draft started as the model's reading is saved in
   * one checked write that names the model layer it corrected, so the saved
   * text records the model output it came from. A failed write leaves no
   * Ground truth behind, and text another browser wrote meanwhile is a 409,
   * not an overwrite. Resolves true when the text was saved, false when the
   * save failed.
   */
  async function saveSelectedSegmentText(): Promise<boolean> {
    const segment = input.selectedSegment;
    if (!projectId || !documentId || !partId || !segment) return false;
    if (!groundTruthTranscriptionId) {
      setPairingError("Ground truth transcription layer is not available.");
      return false;
    }
    const reading = segmentHasGroundTruth(segment)
      ? null
      : modelReadingFor(segment, input.selectedTranscriptionLayer);
    try {
      if (reading?.text) {
        // No Ground truth to speak of yet: the edit started from whatever
        // the row holds now (usually nothing), and the server refuses the
        // save if someone else wrote text there since.
        await commitLineText(
          segment.id,
          input.approvedTextDraft,
          approvedText(segment) ?? "",
          reading.transcription_id,
        );
      } else {
        await commitLineText(segment.id, input.approvedTextDraft);
      }
      setPairingError(null);
      setTranscriptionSaveMessage(statusMessage("Ground truth text saved"));
      return true;
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(errorMessage(err, "Failed to save Ground truth text."));
      return false;
    }
  }

  /** Resolves true when the model output was approved, false otherwise. */
  async function promoteSelectedSegmentToGroundTruth(): Promise<boolean> {
    const segment = input.selectedSegment;
    if (!projectId || !documentId || !partId || !segment) return false;
    const modelLayerId = modelLayerIdForPromotion(
      segment,
      input.selectedTranscriptionLayer,
    );
    if (!modelLayerId) {
      setPairingError(
        "Model transcription is not available to save as Ground truth.",
      );
      return false;
    }
    try {
      await copyModelReading(segment.id, modelLayerId);
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
            initialDraftFor(reloaded, groundTruthTranscriptionId),
          );
        }
      }
      setTranscriptionSaveMessage(statusMessage("Saved to Ground truth"));
      return true;
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(errorMessage(err, "Failed to save to Ground truth."));
      return false;
    }
  }

  return {
    pairTextLine,
    commitLineText,
    saveSelectedSegmentText,
    promoteSelectedSegmentToGroundTruth,
  };
}

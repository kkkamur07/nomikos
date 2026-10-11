import {
  useEffect,
  useState,
  type ChangeEvent,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  api,
  type EnqueueJobResponse,
  type JobResponse,
  type LineResponse,
  type TranscribeJobResult,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import { isAbortError } from "../../../api/errors";
import { invalidateAfter } from "../../../api/resources";
import { submissionRefusalExplanation } from "../../../inference";
import {
  INFERENCE_JOB_WAIT_CEILING_MS,
  type PageEditorJobKind,
} from "../jobProgress";
import { segmentNumberFor, segmentsInNumberOrder } from "../segmentNumbering";
import { statusMessage, type StatusMessage } from "../statusMessage";
import { groundTruthLayer } from "../../../utils/transcriptionLayerLabel";
import {
  lineTextForLayer,
  modelLayerIdForPromotion,
  withLocalGroundTruth,
} from "./utils";

type PairingStateInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  lines: LineResponse[];
  setLines: Dispatch<SetStateAction<LineResponse[]>>;
  transcriptionLayers: TranscriptionLayerResponse[];
  setTranscriptionLayers: Dispatch<
    SetStateAction<TranscriptionLayerResponse[]>
  >;
  selectedTranscriptionLayerId: string | null;
  setSelectedTranscriptionLayerId: Dispatch<SetStateAction<string | null>>;
  groundTruthTranscriptionId: string | null;
  setTextLines: Dispatch<
    SetStateAction<
      { order: number; text: string; paired_line_id: string | null }[]
    >
  >;
  setPairingProgress: Dispatch<
    SetStateAction<{
      paired_lines: number;
      total_lines: number;
      percent: number;
    }>
  >;
  setPairingError: Dispatch<SetStateAction<string | null>>;
  selectedTranscribeModelId: string | null;
  /**
   * Where a refused submission is explained. It is a standing line rather than
   * the error toast, because "no inference host had capacity" is something the
   * researcher has to act on, and a toast is gone before they can.
   */
  setSubmissionRefusal: Dispatch<SetStateAction<string | null>>;
  trackJobAndWait: (
    enqueued: EnqueueJobResponse,
    meta: { label: string; kind: PageEditorJobKind },
    options?: { timeoutMs?: number },
  ) => Promise<JobResponse>;
};

export function usePairingState({
  projectId,
  documentId,
  partId,
  lines,
  setLines,
  transcriptionLayers,
  setTranscriptionLayers,
  selectedTranscriptionLayerId,
  setSelectedTranscriptionLayerId,
  groundTruthTranscriptionId,
  setTextLines,
  setPairingProgress,
  setPairingError,
  selectedTranscribeModelId,
  setSubmissionRefusal,
  trackJobAndWait,
}: PairingStateInput) {
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(
    null,
  );
  const [approvedTextDraft, setApprovedTextDraft] = useState("");
  const [transcriptionSaveMessage, setTranscriptionSaveMessage] =
    useState<StatusMessage | null>(null);
  const [ocrRunning, setOcrRunning] = useState(false);
  const [ocrScope, setOcrScope] = useState<"segment" | "page" | null>(null);
  const [ocrMessage, setOcrMessage] = useState<StatusMessage | null>(null);

  useEffect(() => {
    setSelectedSegmentId(null);
    setApprovedTextDraft("");
    setTranscriptionSaveMessage(null);
    setOcrMessage(null);
  }, [projectId, documentId, partId]);

  /**
   * Called once per committed write, after the server has accepted it.
   *
   * Text written here shows on the page from this hook's own state, so this
   * invalidates the other cached reads of it (the document and the
   * published page a reader sees).
   */
  function notePartContentChanged() {
    if (!projectId || !documentId) return;
    invalidateAfter.partContentChanged(projectId, documentId);
  }

  const selectedSegmentNumber = segmentNumberFor(lines, selectedSegmentId);

  const selectedTranscriptionLayer =
    selectedTranscriptionLayerId === null
      ? null
      : (transcriptionLayers.find(
          (layer) => layer.id === selectedTranscriptionLayerId,
        ) ?? null);

  const selectedSegment =
    selectedSegmentId === null
      ? null
      : (lines.find((line) => line.id === selectedSegmentId) ?? null);

  async function pairTextLine(order: number) {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    try {
      const pairing = await api.pairTextLine(projectId, documentId, partId, {
        line_id: selectedSegmentId,
        text_line_order: order,
      });
      const candidate = pairing.text_lines.find(
        (textLine) => textLine.order === order,
      );
      if (candidate) {
        // Read-modify-write after an await: fold into the current segments, not
        // into the render-time snapshot, or a concurrent edit is reverted.
        setLines((current) =>
          withLocalGroundTruth(
            current,
            groundTruthTranscriptionId,
            selectedSegmentId,
            candidate.text,
          ),
        );
        setApprovedTextDraft(candidate.text);
      }
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      notePartContentChanged();
      setPairingError(null);
    } catch (err) {
      setPairingError(
        err instanceof Error ? err.message : "Failed to pair Text line.",
      );
    }
  }

  /**
   * Saves one Segment's Ground truth text. The one write path for approved
   * text: the strip saves the open Segment through it, the transcript column
   * saves any row. Throws when the save fails, after recording the error.
   */
  async function commitLineText(lineId: string, text: string): Promise<void> {
    if (!projectId || !documentId || !partId) {
      setPairingError("Page context is missing. Reload and try again.");
      throw new Error("Page context is missing.");
    }
    if (!groundTruthTranscriptionId) {
      setPairingError("Ground truth transcription layer is not available.");
      throw new Error("Ground truth transcription layer is not available.");
    }
    try {
      const updated = await api.updateGroundTruthLineText(
        projectId,
        documentId,
        groundTruthTranscriptionId,
        lineId,
        { text },
      );
      // Read-modify-write after an await: fold into the current segments, not
      // into the render-time snapshot, or a concurrent edit is reverted.
      setLines((current) =>
        withLocalGroundTruth(
          current,
          groundTruthTranscriptionId,
          lineId,
          updated.text,
        ),
      );
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      notePartContentChanged();
      setPairingError(null);
    } catch (err) {
      setPairingError(
        err instanceof Error ? err.message : "Failed to save approved text.",
      );
      throw err;
    }
  }

  /** Resolves true when the text was saved, false when the save failed. */
  async function saveApprovedText(): Promise<boolean> {
    if (!selectedSegmentId) return false;
    try {
      await commitLineText(selectedSegmentId, approvedTextDraft);
      return true;
    } catch {
      return false;
    }
  }

  function selectTranscriptionLayer(event: ChangeEvent<HTMLSelectElement>) {
    const nextLayerId = event.target.value;
    setSelectedTranscriptionLayerId(nextLayerId);
    setTranscriptionSaveMessage(null);
    setPairingError(null);
    if (selectedSegment) {
      setApprovedTextDraft(lineTextForLayer(selectedSegment, nextLayerId));
    }
  }

  /** Resolves true when the text was saved, false when the save failed. */
  async function saveGroundTruthText(): Promise<boolean> {
    if (!projectId || !documentId || !partId || !selectedSegmentId) {
      return false;
    }
    if (
      !groundTruthTranscriptionId ||
      selectedTranscriptionLayer?.kind !== "ground_truth"
    ) {
      setPairingError("Only Ground truth can be edited.");
      return false;
    }
    try {
      const updated = await api.updateGroundTruthLineText(
        projectId,
        documentId,
        groundTruthTranscriptionId,
        selectedSegmentId,
        { text: approvedTextDraft },
      );
      // Read-modify-write after an await: fold into the current segments, not
      // into the render-time snapshot, or a concurrent edit is reverted.
      setLines((current) =>
        withLocalGroundTruth(
          current,
          groundTruthTranscriptionId,
          selectedSegmentId,
          updated.text,
        ),
      );
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      notePartContentChanged();
      setPairingError(null);
      setTranscriptionSaveMessage(statusMessage("Ground truth text saved"));
      return true;
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(
        err instanceof Error
          ? err.message
          : "Failed to save Ground truth text.",
      );
      return false;
    }
  }

  /** Keeps the draft box on the open segment showing the layer just written. */
  function syncApprovedTextDraft(
    reloadedLines: LineResponse[],
    layerId: string,
  ) {
    if (!selectedSegmentId) return;
    const segment = reloadedLines.find((line) => line.id === selectedSegmentId);
    if (segment) {
      setApprovedTextDraft(lineTextForLayer(segment, layerId));
    }
  }

  /**
   * The reload a finished OCR run needs, without touching the open layer.
   *
   * A model run produces a suggestion, never a replacement: switching the
   * dropdown to the new model layer, and the draft box with it, read as "the
   * model overwrote my Ground truth" even though the layers are separate rows.
   * So the selection stays where the researcher left it, and the draft box is
   * re-synced to that same layer. With nothing selected yet, Ground truth is
   * the layer to open, never the model output. A layer deleted while the run
   * was in flight is the exception: an id the reload no longer carries would
   * resolve to no layer at all, so it is dropped and Ground truth opens.
   */
  async function refreshAfterOcr() {
    if (!projectId || !documentId || !partId) return;
    const [reloadedLines, layers] = await Promise.all([
      api.listPartLines(projectId, documentId, partId),
      api.listTranscriptions(projectId, documentId),
    ]);
    setLines(reloadedLines);
    setTranscriptionLayers(layers);
    const stillPresent = (layerId: string | null) =>
      layerId !== null && layers.some((layer) => layer.id === layerId)
        ? layerId
        : null;
    const layerToShow =
      stillPresent(selectedTranscriptionLayerId) ??
      stillPresent(groundTruthTranscriptionId) ??
      groundTruthLayer(layers)?.id ??
      null;
    if (layerToShow === null) return;
    if (layerToShow !== selectedTranscriptionLayerId) {
      setSelectedTranscriptionLayerId(layerToShow);
    }
    syncApprovedTextDraft(reloadedLines, layerToShow);
  }

  /**
   * The page state a finished transcription leaves behind, wherever it was
   * produced: the new layer is folded into the segments already on screen, then
   * the page reloads to pick up what the server actually stored.
   */
  async function applyTranscribeResult(result: TranscribeJobResult) {
    setLines((current) =>
      current.map((line) => {
        const output = result.lines.find((entry) => entry.line_id === line.id);
        if (!output) return line;
        const withoutLayer = line.line_transcriptions.filter(
          (transcription) =>
            transcription.transcription_id !== result.transcription_id,
        );
        return {
          ...line,
          line_transcriptions: [
            ...withoutLayer,
            {
              id: `ocr-${line.id}-${result.transcription_id}`,
              transcription_id: result.transcription_id,
              transcription_kind: "model" as const,
              text: output.text,
              confidence: output.confidence,
            },
          ],
        };
      }),
    );
    await refreshAfterOcr();
    notePartContentChanged();
    return result;
  }

  async function applyTranscribeJob(
    job: JobResponse,
  ): Promise<TranscribeJobResult | null> {
    const result = job.result as TranscribeJobResult | null;
    if (result?.transcription_id) {
      return applyTranscribeResult(result);
    }
    // The layer is committed before the job reports done, so a result this
    // client cannot read must not strand the page on stale segments: find the
    // layer the job created and reload, exactly what a manual refresh does.
    if (projectId && documentId) {
      const layers = await api.listTranscriptions(projectId, documentId);
      const created = layers.find(
        (layer) => layer.created_by_job_id === job.id,
      );
      if (created) {
        await refreshAfterOcr();
        notePartContentChanged();
        return null;
      }
    }
    throw new Error("Transcribe job returned no result.");
  }

  async function runSegmentOcr() {
    if (!projectId || !documentId || !partId) {
      setPairingError("Page context is missing. Reload and try again.");
      return;
    }
    if (!selectedSegmentId) {
      setPairingError("Select a segment on the canvas first.");
      return;
    }
    if (!selectedTranscribeModelId) {
      setPairingError("Select an HTR model before running OCR.");
      return;
    }
    setOcrRunning(true);
    setOcrScope("segment");
    setOcrMessage(null);
    setPairingError(null);
    setSubmissionRefusal(null);
    try {
      const enqueued = await api.enqueueTranscribePart(
        projectId,
        documentId,
        partId,
        {
          model_id: selectedTranscribeModelId,
          line_ids: [selectedSegmentId],
        },
      );
      const job = await trackJobAndWait(
        enqueued,
        {
          label: selectedSegmentNumber
            ? `Segment ${selectedSegmentNumber}`
            : "Selected segment",
          kind: "transcription-segment",
        },
        { timeoutMs: INFERENCE_JOB_WAIT_CEILING_MS },
      );
      const result = await applyTranscribeJob(job);
      const hasAnyText =
        result === null || result.lines.some((line) => line.text?.trim());
      setOcrMessage(
        statusMessage(
          hasAnyText
            ? "OCR prediction completed for selected Segment."
            : "OCR finished with no text for this segment.",
        ),
      );
    } catch (err) {
      // The jobs panel already reports a user cancellation.
      if (isAbortError(err)) return;
      const refusal = submissionRefusalExplanation(err);
      if (refusal) {
        setSubmissionRefusal(refusal);
        return;
      }
      setPairingError(
        err instanceof Error ? err.message : "Segment OCR failed.",
      );
    } finally {
      setOcrRunning(false);
      setOcrScope(null);
    }
  }

  async function runPageOcr() {
    if (!projectId || !documentId || !partId) {
      setPairingError("Page context is missing. Reload and try again.");
      return;
    }
    if (!selectedTranscribeModelId) {
      setPairingError("Select an HTR model before running OCR.");
      return;
    }
    setOcrRunning(true);
    setOcrScope("page");
    setOcrMessage(null);
    setPairingError(null);
    setSubmissionRefusal(null);
    try {
      const enqueued = await api.enqueueTranscribePart(
        projectId,
        documentId,
        partId,
        {
          model_id: selectedTranscribeModelId,
        },
      );
      const job = await trackJobAndWait(
        enqueued,
        { label: "Full page", kind: "transcription-page" },
        { timeoutMs: INFERENCE_JOB_WAIT_CEILING_MS },
      );
      const result = await applyTranscribeJob(job);
      const withText =
        result === null
          ? null
          : result.lines.filter((line) => line.text?.trim()).length;
      setOcrMessage(
        statusMessage(
          withText === null
            ? "OCR prediction completed for the page."
            : withText > 0
              ? `OCR prediction completed for ${withText} Segment(s).`
              : "OCR finished with no text for the selected segments.",
        ),
      );
    } catch (err) {
      // The jobs panel already reports a user cancellation.
      if (isAbortError(err)) return;
      const refusal = submissionRefusalExplanation(err);
      if (refusal) {
        setSubmissionRefusal(refusal);
        return;
      }
      setPairingError(err instanceof Error ? err.message : "Page OCR failed.");
    } finally {
      setOcrRunning(false);
      setOcrScope(null);
    }
  }

  async function promoteSelectedSegmentToGroundTruth() {
    if (
      !projectId ||
      !documentId ||
      !partId ||
      !selectedSegmentId ||
      !selectedSegment
    )
      return;
    const modelLayerId = modelLayerIdForPromotion(
      selectedSegment,
      selectedTranscriptionLayer,
    );
    if (!modelLayerId) {
      setPairingError(
        "Model transcription is not available to save as Ground truth.",
      );
      return;
    }
    try {
      await api.copyToGroundTruth(projectId, documentId, modelLayerId, {
        line_ids: [selectedSegmentId],
      });
      const [reloadedLines, pairing] = await Promise.all([
        api.listPartLines(projectId, documentId, partId),
        api.getPagePairing(projectId, documentId, partId),
      ]);
      setLines(reloadedLines);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
      notePartContentChanged();
      setPairingError(null);
      if (groundTruthTranscriptionId) {
        setSelectedTranscriptionLayerId(groundTruthTranscriptionId);
        syncApprovedTextDraft(reloadedLines, groundTruthTranscriptionId);
      }
      setTranscriptionSaveMessage(statusMessage("Saved to Ground truth"));
    } catch (err) {
      setTranscriptionSaveMessage(null);
      setPairingError(
        err instanceof Error ? err.message : "Failed to save to Ground truth.",
      );
    }
  }

  function selectSegment(lineId: string) {
    const selected = lines.find((line) => line.id === lineId) ?? null;
    setSelectedSegmentId(lineId);
    setTranscriptionSaveMessage(null);
    setApprovedTextDraft(
      selected ? lineTextForLayer(selected, selectedTranscriptionLayerId) : "",
    );
  }

  function navigateSegment(direction: -1 | 1) {
    const sorted = segmentsInNumberOrder(lines);
    if (sorted.length === 0) return;

    const currentIndex = selectedSegmentId
      ? sorted.findIndex((line) => line.id === selectedSegmentId)
      : -1;

    let nextIndex: number;
    if (currentIndex < 0) {
      nextIndex = direction === 1 ? 0 : sorted.length - 1;
    } else {
      nextIndex = Math.min(
        Math.max(currentIndex + direction, 0),
        sorted.length - 1,
      );
    }

    if (nextIndex !== currentIndex) {
      selectSegment(sorted[nextIndex].id);
    }
  }

  return {
    selectedSegmentId,
    setSelectedSegmentId,
    approvedTextDraft,
    setApprovedTextDraft,
    transcriptionSaveMessage,
    ocrRunning,
    ocrScope,
    ocrMessage,
    selectedSegment,
    selectedSegmentNumber,
    selectedTranscriptionLayer,
    pairTextLine,
    commitLineText,
    saveApprovedText,
    selectTranscriptionLayer,
    saveGroundTruthText,
    runSegmentOcr,
    runPageOcr,
    promoteSelectedSegmentToGroundTruth,
    selectSegment,
    navigateSegment,
  };
}

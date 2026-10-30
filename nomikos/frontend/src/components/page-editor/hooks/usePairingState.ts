import { useEffect, useRef, useState, type ChangeEvent } from "react";
import {
  api,
  type LineResponse,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import { invalidateAfter } from "../../../api/resources";
import { useUnsavedTranscriptGuard } from "../PageEditorTranscriptEditor";
import { segmentNumberFor, segmentsInNumberOrder } from "../segmentNumbering";
import type { StatusMessage } from "../statusMessage";
import type {
  PairingProgressSetter,
  SetState,
  TextLinesSetter,
  TrackJobAndWait,
} from "./editorHookTypes";
import { useGroundTruthText } from "./useGroundTruthText";
import { useOcrRun } from "./useOcrRun";
import { lineTextForLayer } from "./utils";

type PairingStateInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  lines: LineResponse[];
  setLines: SetState<LineResponse[]>;
  transcriptionLayers: TranscriptionLayerResponse[];
  setTranscriptionLayers: SetState<TranscriptionLayerResponse[]>;
  selectedTranscriptionLayerId: string | null;
  setSelectedTranscriptionLayerId: SetState<string | null>;
  groundTruthTranscriptionId: string | null;
  setTextLines: TextLinesSetter;
  setPairingProgress: PairingProgressSetter;
  setPairingError: SetState<string | null>;
  selectedTranscribeModelId: string | null;
  /**
   * Where a refused submission is explained. It is a standing line rather than
   * the error toast, because "no inference host had capacity" is something the
   * researcher has to act on, and a toast is gone before they can.
   */
  setSubmissionRefusal: SetState<string | null>;
  trackJobAndWait: TrackJobAndWait;
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
  const selectedSegmentIdRef = useRef(selectedSegmentId);
  selectedSegmentIdRef.current = selectedSegmentId;
  const selectedTranscriptionLayerIdRef = useRef(selectedTranscriptionLayerId);
  selectedTranscriptionLayerIdRef.current = selectedTranscriptionLayerId;

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

  const groundTruthText = useGroundTruthText({
    projectId,
    documentId,
    partId,
    setLines,
    groundTruthTranscriptionId,
    setSelectedTranscriptionLayerId,
    selectedTranscriptionLayer,
    selectedSegmentId,
    selectedSegment,
    selectedSegmentIdRef,
    approvedTextDraft,
    setApprovedTextDraft,
    setTranscriptionSaveMessage,
    setTextLines,
    setPairingProgress,
    setPairingError,
    notePartContentChanged,
  });

  const ocrRun = useOcrRun({
    projectId,
    documentId,
    partId,
    setLines,
    setTranscriptionLayers,
    selectedTranscriptionLayerIdRef,
    setSelectedTranscriptionLayerId,
    groundTruthTranscriptionId,
    selectedSegmentId,
    selectedSegmentIdRef,
    selectedSegmentNumber,
    setApprovedTextDraft,
    setPairingError,
    selectedTranscribeModelId,
    setSubmissionRefusal,
    trackJobAndWait,
    notePartContentChanged,
  });
  const { setOcrMessage } = ocrRun;

  useEffect(() => {
    setSelectedSegmentId(null);
    setApprovedTextDraft("");
    setTranscriptionSaveMessage(null);
    setOcrMessage(null);
  }, [projectId, documentId, partId, setOcrMessage]);

  function selectTranscriptionLayer(event: ChangeEvent<HTMLSelectElement>) {
    const nextLayerId = event.target.value;
    setSelectedTranscriptionLayerId(nextLayerId);
    setTranscriptionSaveMessage(null);
    setPairingError(null);
    if (selectedSegment) {
      setApprovedTextDraft(lineTextForLayer(selectedSegment, nextLayerId));
    }
  }

  const lineIds = lines.map((line) => line.id);
  useUnsavedTranscriptGuard(lineIds);

  /**
   * Folds the server's current texts into the segments on screen, leaving
   * geometry alone so an in-flight layout edit is not reverted.
   */
  function mergeLineTexts(fresh: LineResponse[]) {
    const byId = new Map(fresh.map((line) => [line.id, line]));
    setLines((current) =>
      current.map((line) => {
        const match = byId.get(line.id);
        return match
          ? { ...line, line_transcriptions: match.line_transcriptions }
          : line;
      }),
    );
  }

  /** One row's "Reload row": its current text, or null when it is gone. */
  async function reloadLineText(lineId: string): Promise<LineResponse | null> {
    if (!projectId || !documentId || !partId) return null;
    const fresh = await api.listPartLines(projectId, documentId, partId);
    const match = fresh.find((line) => line.id === lineId) ?? null;
    if (match) mergeLineTexts([match]);
    return match;
  }

  // Back in a tab after a while: pick up text saved from elsewhere, so a row
  // is edited from what the server holds now. Rows with a draft keep it.
  const refreshTextsRef = useRef<() => void>(() => {});
  refreshTextsRef.current = () => {
    if (!projectId || !documentId || !partId) return;
    api
      .listPartLines(projectId, documentId, partId)
      .then(mergeLineTexts)
      .catch(() => {
        // A failed background refresh changes nothing on screen.
      });
  };
  useEffect(() => {
    function onVisibilityChange() {
      if (document.visibilityState === "visible") refreshTextsRef.current();
    }
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () =>
      document.removeEventListener("visibilitychange", onVisibilityChange);
  }, []);

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
    ocrRunning: ocrRun.ocrRunning,
    ocrScope: ocrRun.ocrScope,
    ocrMessage: ocrRun.ocrMessage,
    selectedSegment,
    selectedSegmentNumber,
    selectedTranscriptionLayer,
    pairTextLine: groundTruthText.pairTextLine,
    commitLineText: groundTruthText.commitLineText,
    reloadLineText,
    saveApprovedText: groundTruthText.saveApprovedText,
    selectTranscriptionLayer,
    saveGroundTruthText: groundTruthText.saveGroundTruthText,
    runSegmentOcr: ocrRun.runSegmentOcr,
    runPageOcr: ocrRun.runPageOcr,
    promoteSelectedSegmentToGroundTruth:
      groundTruthText.promoteSelectedSegmentToGroundTruth,
    selectSegment,
    navigateSegment,
  };
}

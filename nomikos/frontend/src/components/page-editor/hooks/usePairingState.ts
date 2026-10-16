import { useEffect, useRef, useState, type ChangeEvent } from "react";
import type {
  LineResponse,
  TranscriptionLayerResponse,
} from "../../../api/client";
import { invalidateAfter } from "../../../api/resources";
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

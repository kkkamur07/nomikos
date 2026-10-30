import { useState, type RefObject } from "react";
import {
  api,
  type JobResponse,
  type LineResponse,
  type TranscribeJobResult,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import { INFERENCE_JOB_WAIT_CEILING_MS } from "../jobProgress";
import { statusMessage, type StatusMessage } from "../statusMessage";
import { groundTruthLayer } from "../../../utils/transcriptionLayerLabel";
import {
  reportJobFailure,
  type SetState,
  type TrackJobAndWait,
} from "./editorHookTypes";
import { lineTextForLayer } from "./utils";

type OcrScope = "segment" | "page";

type OcrRunInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  setLines: SetState<LineResponse[]>;
  setTranscriptionLayers: SetState<TranscriptionLayerResponse[]>;
  /** The open layer now, read after an await instead of the render's copy. */
  selectedTranscriptionLayerIdRef: RefObject<string | null>;
  setSelectedTranscriptionLayerId: SetState<string | null>;
  groundTruthTranscriptionId: string | null;
  selectedSegmentId: string | null;
  /** The segment open now, read after an await instead of the render's copy. */
  selectedSegmentIdRef: RefObject<string | null>;
  selectedSegmentNumber: number | null;
  setApprovedTextDraft: SetState<string>;
  setPairingError: SetState<string | null>;
  selectedTranscribeModelId: string | null;
  setSubmissionRefusal: SetState<string | null>;
  trackJobAndWait: TrackJobAndWait;
  notePartContentChanged: () => void;
};

/** The model output a finished run folds into the segments already on screen. */
function withTranscribeResult(
  lines: LineResponse[],
  result: TranscribeJobResult,
): LineResponse[] {
  return lines.map((line) => {
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
  });
}

function linesWithText(result: TranscribeJobResult): number {
  return result.lines.filter((line) => line.text?.trim()).length;
}

/** The sentence a finished run reports; a null result is an unread layer. */
function ocrCompletionMessage(
  scope: OcrScope,
  result: TranscribeJobResult | null,
): string {
  if (scope === "segment") {
    return result === null || linesWithText(result) > 0
      ? "OCR prediction completed for selected Segment."
      : "OCR finished with no text for this segment.";
  }
  if (result === null) return "OCR prediction completed for the page.";
  const withText = linesWithText(result);
  return withText > 0
    ? `OCR prediction completed for ${withText} Segment(s).`
    : "OCR finished with no text for the selected segments.";
}

/** Why a run may not start, or null when it may. */
function ocrPrecondition(input: OcrRunInput, scope: OcrScope): string | null {
  if (!input.projectId || !input.documentId || !input.partId) {
    return "Page context is missing. Reload and try again.";
  }
  if (scope === "segment" && !input.selectedSegmentId) {
    return "Select a segment on the canvas first.";
  }
  if (!input.selectedTranscribeModelId) {
    return "Select an HTR model before running OCR.";
  }
  return null;
}

/**
 * The layer a finished OCR run leaves open: the one that was open if the
 * reload still has it, else Ground truth. See refreshAfterOcr.
 */
function layerToShowAfterOcr(
  layers: TranscriptionLayerResponse[],
  openLayerId: string | null,
  groundTruthId: string | null,
): string | null {
  const stillPresent = (layerId: string | null) =>
    layerId !== null && layers.some((layer) => layer.id === layerId)
      ? layerId
      : null;
  return (
    stillPresent(openLayerId) ??
    stillPresent(groundTruthId) ??
    groundTruthLayer(layers)?.id ??
    null
  );
}

/**
 * Segment and page OCR: enqueue, wait on the job, then fold the new model
 * layer into the page. A run can take half an hour, so everything it does
 * after the wait reads the selection as it is then, not as it was at the start.
 */
export function useOcrRun(input: OcrRunInput) {
  const { projectId, documentId, partId, setLines, notePartContentChanged } =
    input;
  // A count, not a flag: overlapping runs must not report idle while one is
  // still going.
  const [ocrRunCount, setOcrRunCount] = useState(0);
  const [lastScope, setLastScope] = useState<OcrScope | null>(null);
  const [ocrMessage, setOcrMessage] = useState<StatusMessage | null>(null);

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
   *
   * The draft box is re-synced only when the segment open now is the one that
   * was open when the run started: the researcher may have moved on, and the
   * draft box then belongs to another segment.
   */
  async function refreshAfterOcr(segmentIdAtStart: string | null) {
    if (!projectId || !documentId || !partId) return;
    const [reloadedLines, layers] = await Promise.all([
      api.listPartLines(projectId, documentId, partId),
      api.listTranscriptions(projectId, documentId),
    ]);
    setLines(reloadedLines);
    input.setTranscriptionLayers(layers);
    const openLayerId = input.selectedTranscriptionLayerIdRef.current;
    const layerToShow = layerToShowAfterOcr(
      layers,
      openLayerId,
      input.groundTruthTranscriptionId,
    );
    if (layerToShow === null) return;
    if (layerToShow !== openLayerId) {
      input.setSelectedTranscriptionLayerId(layerToShow);
    }
    const segmentId = input.selectedSegmentIdRef.current;
    if (!segmentId || segmentId !== segmentIdAtStart) return;
    const segment = reloadedLines.find((line) => line.id === segmentId);
    if (segment) {
      input.setApprovedTextDraft(lineTextForLayer(segment, layerToShow));
    }
  }

  /**
   * The page state a finished transcription leaves behind. A result this
   * client cannot read still has its layer committed before the job reports
   * done, so the page reloads when the job's layer exists, exactly what a
   * manual refresh does.
   */
  async function applyTranscribeJob(
    job: JobResponse,
    segmentIdAtStart: string | null,
  ): Promise<TranscribeJobResult | null> {
    const result = job.result as TranscribeJobResult | null;
    if (result?.transcription_id) {
      setLines((current) => withTranscribeResult(current, result));
    } else {
      const layers =
        projectId && documentId
          ? await api.listTranscriptions(projectId, documentId)
          : [];
      if (!layers.some((layer) => layer.created_by_job_id === job.id)) {
        throw new Error("Transcribe job returned no result.");
      }
    }
    await refreshAfterOcr(segmentIdAtStart);
    notePartContentChanged();
    return result?.transcription_id ? result : null;
  }

  async function runOcr(scope: OcrScope) {
    const refusal = ocrPrecondition(input, scope);
    if (refusal) {
      input.setPairingError(refusal);
      return;
    }
    const segmentIdAtStart = input.selectedSegmentId;
    setOcrRunCount((count) => count + 1);
    setLastScope(scope);
    setOcrMessage(null);
    input.setPairingError(null);
    input.setSubmissionRefusal(null);
    try {
      const enqueued = await api.enqueueTranscribePart(
        projectId!,
        documentId!,
        partId!,
        scope === "segment"
          ? {
              model_id: input.selectedTranscribeModelId!,
              line_ids: [segmentIdAtStart!],
            }
          : { model_id: input.selectedTranscribeModelId! },
      );
      const job = await input.trackJobAndWait(
        enqueued,
        scope === "segment"
          ? {
              label: input.selectedSegmentNumber
                ? `Segment ${input.selectedSegmentNumber}`
                : "Selected segment",
              kind: "transcription-segment",
            }
          : { label: "Full page", kind: "transcription-page" },
        { timeoutMs: INFERENCE_JOB_WAIT_CEILING_MS },
      );
      // `waitForJob` resolves on a cancel, and a cancel from the jobs panel or
      // another tab is not an AbortError: it is a choice, not a failure.
      if (job.status === "cancelled") {
        setOcrMessage(statusMessage("Job cancelled"));
        return;
      }
      const result = await applyTranscribeJob(job, segmentIdAtStart);
      setOcrMessage(statusMessage(ocrCompletionMessage(scope, result)));
    } catch (err) {
      reportJobFailure(
        err,
        scope === "segment" ? "Segment OCR failed." : "Page OCR failed.",
        {
          setSubmissionRefusal: input.setSubmissionRefusal,
          setError: input.setPairingError,
        },
      );
    } finally {
      setOcrRunCount((count) => Math.max(0, count - 1));
    }
  }

  return {
    ocrRunning: ocrRunCount > 0,
    ocrScope: ocrRunCount > 0 ? lastScope : null,
    ocrMessage,
    setOcrMessage,
    runSegmentOcr: () => runOcr("segment"),
    runPageOcr: () => runOcr("page"),
  };
}

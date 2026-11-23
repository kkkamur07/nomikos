import type {
  LineResponse,
  LineTranscriptionResponse,
  PartLayoutResponse,
  TranscriptionLayerResponse,
} from "../../../api/client";

export function approvedText(line: LineResponse): string | null {
  return (
    line.line_transcriptions.find(
      (transcription) => transcription.transcription_kind === "ground_truth",
    )?.text ?? null
  );
}

export function segmentHasGroundTruth(line: LineResponse): boolean {
  const text = approvedText(line);
  return Boolean(text?.trim());
}

export function segmentIdsWithGroundTruth(lines: LineResponse[]): Set<string> {
  return new Set(lines.filter(segmentHasGroundTruth).map((line) => line.id));
}

export function lineTranscriptionForLayer(
  line: LineResponse,
  transcriptionLayerId: string | null,
): LineTranscriptionResponse | null {
  if (!transcriptionLayerId) return null;
  return (
    line.line_transcriptions.find(
      (transcription) =>
        transcription.transcription_id === transcriptionLayerId,
    ) ?? null
  );
}

export function modelTranscriptionForLine(
  line: LineResponse,
  preferredLayerId?: string | null,
): LineTranscriptionResponse | null {
  const modelTranscriptions = line.line_transcriptions.filter(
    (transcription) => transcription.transcription_kind === "model",
  );
  if (modelTranscriptions.length === 0) return null;
  if (preferredLayerId) {
    const preferred = modelTranscriptions.find(
      (transcription) => transcription.transcription_id === preferredLayerId,
    );
    if (preferred) return preferred;
  }
  return modelTranscriptions[modelTranscriptions.length - 1] ?? null;
}

/**
 * What the strip's edit box opens with for a Segment: its Ground truth when it
 * has any, else the model's newest reading (the text the Model output block
 * shows), else nothing. The researcher corrects the reading in place instead
 * of retyping the line. ``selectedLayerId`` picks the model reading when a
 * model layer is the open one.
 */
export function initialDraftFor(
  line: LineResponse,
  selectedLayerId: string | null,
): string {
  const groundTruth = approvedText(line);
  if (groundTruth?.trim()) return groundTruth;
  return modelTranscriptionForLine(line, selectedLayerId)?.text ?? "";
}

/**
 * The model reading the strip offers for a Segment: the open model layer's
 * when it has one for this line, else the newest. Approve copies this one.
 */
export function modelReadingFor(
  line: LineResponse,
  selectedLayer: TranscriptionLayerResponse | null,
): LineTranscriptionResponse | null {
  return modelTranscriptionForLine(
    line,
    selectedLayer?.kind === "model" ? selectedLayer.id : null,
  );
}

export function modelLayerIdForPromotion(
  line: LineResponse,
  selectedLayer: TranscriptionLayerResponse | null,
): string | null {
  return modelReadingFor(line, selectedLayer)?.transcription_id ?? null;
}

export function mergeSavedLine(
  lines: LineResponse[],
  saved: LineResponse,
): LineResponse[] {
  const index = lines.findIndex((line) => line.id === saved.id);
  if (index === -1) {
    return [...lines, saved].sort((a, b) => a.order - b.order);
  }
  return lines.map((line) => (line.id === saved.id ? saved : line));
}

export function withLocalGroundTruth(
  lines: LineResponse[],
  groundTruthTranscriptionId: string | null,
  lineId: string,
  text: string,
): LineResponse[] {
  if (!groundTruthTranscriptionId) return lines;
  return lines.map((line) => {
    if (line.id !== lineId) return line;
    const existing = line.line_transcriptions.filter(
      (transcription) => transcription.transcription_kind !== "ground_truth",
    );
    const nextTranscription: LineTranscriptionResponse = {
      id:
        line.line_transcriptions.find(
          (transcription) =>
            transcription.transcription_kind === "ground_truth",
        )?.id ?? `ground-truth-${lineId}`,
      transcription_id: groundTruthTranscriptionId,
      transcription_kind: "ground_truth",
      text,
      confidence: null,
    };
    return { ...line, line_transcriptions: [...existing, nextTranscription] };
  });
}

export function syncLayoutLinesFromSegments(
  layout: PartLayoutResponse,
  segments: LineResponse[],
): PartLayoutResponse {
  const byId = new Map(segments.map((line) => [line.id, line]));
  const synced = layout.lines.map((layoutLine) => {
    const segment = byId.get(layoutLine.id);
    if (!segment) return layoutLine;
    return {
      ...layoutLine,
      block_id: segment.block_id,
      baseline: segment.baseline,
      mask: segment.mask,
      manual_geometry: segment.manual_geometry,
    };
  });

  for (const segment of segments) {
    if (synced.some((line) => line.id === segment.id)) continue;
    synced.push({
      id: segment.id,
      block_id: segment.block_id,
      baseline: segment.baseline,
      mask: segment.mask,
      manual_geometry: segment.manual_geometry,
    });
  }

  return { ...layout, lines: synced };
}

export function applyLayoutLineGeometryToSegments(
  segments: LineResponse[],
  layoutLines: PartLayoutResponse["lines"],
): LineResponse[] {
  const byId = new Map(layoutLines.map((line) => [line.id, line]));
  return segments.map((segment) => {
    const layoutLine = byId.get(segment.id);
    if (!layoutLine) return segment;
    return {
      ...segment,
      block_id: layoutLine.block_id ?? segment.block_id,
      baseline: layoutLine.baseline ?? segment.baseline,
      mask: layoutLine.mask ?? segment.mask,
      manual_geometry: layoutLine.manual_geometry ?? segment.manual_geometry,
    };
  });
}

/**
 * Page-level endpoints: layout, lines, pairing, segment health, OCR models
 * and ground-truth text.
 */
import { apiRequest, fetchBinaryApi } from "./http";
import { documentPath, partPath } from "./paths";
import type {
  BlockCreateRequest,
  BlockPatchRequest,
  BlockResponse,
  CopyToGroundTruthRequest,
  CopyToGroundTruthResponse,
  EnqueueJobResponse,
  InferenceModelResponse,
  InferenceTask,
  LineCreateRequest,
  LinePatchRequest,
  LineResponse,
  LinesReplaceRequest,
  LineTranscriptionPatchRequest,
  LineTranscriptionResponse,
  PagePairingResponse,
  PageTranscriptionImportRequest,
  PairTextLineRequest,
  PartLayoutResponse,
  ResolvedModelBindingResponse,
  ResetPartLayoutRequest,
  SegmentHealthResponse,
  SegmentPartRequest,
  TranscribePartRequest,
  UpdateLineGeometryRequest,
} from "./types";

export const partsApi = {
  getPartLayout: (projectId: string, documentId: string, partId: string) =>
    apiRequest<PartLayoutResponse>(
      `${partPath(projectId, documentId, partId)}/layout`,
    ),

  listPartLines: (projectId: string, documentId: string, partId: string) =>
    apiRequest<LineResponse[]>(
      `${partPath(projectId, documentId, partId)}/lines`,
    ),

  /**
   * A region (ADR 0011). With `assign_contained` (the server default) every
   * unassigned line whose centre lies inside the box joins it, and the page is
   * renumbered region-first: reload the lines after.
   */
  createPartBlock: (
    projectId: string,
    documentId: string,
    partId: string,
    body: BlockCreateRequest,
  ) =>
    apiRequest<BlockResponse>(
      `${partPath(projectId, documentId, partId)}/blocks`,
      { method: "POST", body },
    ),

  patchPartBlock: (
    projectId: string,
    documentId: string,
    partId: string,
    blockId: string,
    body: BlockPatchRequest,
  ) =>
    apiRequest<BlockResponse>(
      `${partPath(projectId, documentId, partId)}/blocks/${blockId}`,
      { method: "PATCH", body },
    ),

  /** The region's lines stay on the page, in no region. */
  deletePartBlock: (
    projectId: string,
    documentId: string,
    partId: string,
    blockId: string,
  ) =>
    apiRequest<void>(
      `${partPath(projectId, documentId, partId)}/blocks/${blockId}`,
      { method: "DELETE" },
    ),

  createPartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    body: LineCreateRequest,
  ) =>
    apiRequest<LineResponse>(
      `${partPath(projectId, documentId, partId)}/lines`,
      { method: "POST", body },
    ),

  replacePartLines: (
    projectId: string,
    documentId: string,
    partId: string,
    body: LinesReplaceRequest,
  ) =>
    apiRequest<LineResponse[]>(
      `${partPath(projectId, documentId, partId)}/lines`,
      { method: "PUT", body },
    ),

  deletePartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
  ) =>
    apiRequest<void>(
      `${partPath(projectId, documentId, partId)}/lines/${encodeURIComponent(lineId)}`,
      { method: "DELETE" },
    ),

  patchPartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
    body: LinePatchRequest,
  ) =>
    apiRequest<LineResponse>(
      `${partPath(projectId, documentId, partId)}/lines/${encodeURIComponent(lineId)}`,
      { method: "PATCH", body },
    ),

  /**
   * The geometry-only slice of `patchPartLine`: same URL, same method, a
   * narrower body. Kept under its own name because the page editor's manual
   * line edit reads clearer as a geometry write than as a generic patch.
   */
  updateLineGeometry: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
    body: UpdateLineGeometryRequest,
  ) => partsApi.patchPartLine(projectId, documentId, partId, lineId, body),

  resetPartLayout: (
    projectId: string,
    documentId: string,
    partId: string,
    body: ResetPartLayoutRequest,
  ) =>
    apiRequest<PartLayoutResponse>(
      `${partPath(projectId, documentId, partId)}/layout/reset`,
      { method: "POST", body },
    ),

  getPagePairing: (projectId: string, documentId: string, partId: string) =>
    apiRequest<PagePairingResponse>(
      `${partPath(projectId, documentId, partId)}/pairing`,
    ),

  importPageTranscription: (
    projectId: string,
    documentId: string,
    partId: string,
    body: PageTranscriptionImportRequest,
  ) =>
    apiRequest<PagePairingResponse>(
      `${partPath(projectId, documentId, partId)}/page-transcription`,
      { method: "PUT", body },
    ),

  pairTextLine: (
    projectId: string,
    documentId: string,
    partId: string,
    body: PairTextLineRequest,
  ) =>
    apiRequest<PagePairingResponse>(
      `${partPath(projectId, documentId, partId)}/pairings`,
      { method: "POST", body },
    ),

  generateTranscriptionPdf: (
    projectId: string,
    documentId: string,
    partId: string,
  ) =>
    fetchBinaryApi(
      `${partPath(projectId, documentId, partId)}/transcription-pdf`,
      { method: "POST" },
    ),

  exportPartDocx: (projectId: string, documentId: string, partId: string) =>
    fetchBinaryApi(`${partPath(projectId, documentId, partId)}/docx`),

  getPageXml: (projectId: string, documentId: string, partId: string) =>
    fetchBinaryApi(`${partPath(projectId, documentId, partId)}/page-xml`),

  /** Zip of the PAGE XML next to the full-resolution page image it describes. */
  getPageXmlBundle: (projectId: string, documentId: string, partId: string) =>
    fetchBinaryApi(
      `${partPath(projectId, documentId, partId)}/page-xml-bundle`,
    ),

  segmentPart: (
    projectId: string,
    documentId: string,
    partId: string,
    body?: SegmentPartRequest,
  ) =>
    apiRequest<EnqueueJobResponse>(
      `${partPath(projectId, documentId, partId)}/segment`,
      { method: "POST", body: body ?? {} },
    ),

  /**
   * Fold the segments a human picked into one. Not a segment-health fix: the
   * server is told which lines, never which polygon, and it offers nothing of
   * its own. A 409 here means several of them carry text or a pairing, and its
   * `error.line_ids` is what the caller shows before asking again with
   * `keepTextFrom`.
   */
  mergePartLines: (
    projectId: string,
    documentId: string,
    partId: string,
    lineIds: string[],
    keepTextFrom?: string,
  ) =>
    apiRequest<LineResponse[]>(
      `${partPath(projectId, documentId, partId)}/lines/merge`,
      {
        method: "POST",
        body: keepTextFrom
          ? { line_ids: lineIds, keep_text_from: keepTextFrom }
          : { line_ids: lineIds },
      },
    ),

  getSegmentHealth: (projectId: string, documentId: string, partId: string) =>
    apiRequest<SegmentHealthResponse>(
      `${partPath(projectId, documentId, partId)}/segment-health`,
    ),

  // Every apply takes ids and nothing else. The server re-derives the outline
  // from the rows as they stand, so a panel left open while somebody else
  // edited the page cannot write geometry measured against the old one.
  splitSpanningSegment: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
  ) =>
    apiRequest<LineResponse[]>(
      `${partPath(projectId, documentId, partId)}/segment-health/splits`,
      { method: "POST", body: { line_id: lineId } },
    ),

  mergeSegmentFragment: (
    projectId: string,
    documentId: string,
    partId: string,
    primaryId: string,
    fragmentId: string,
  ) =>
    apiRequest<LineResponse[]>(
      `${partPath(projectId, documentId, partId)}/segment-health/merges`,
      {
        method: "POST",
        body: { primary_id: primaryId, fragment_id: fragmentId },
      },
    ),

  trimSegmentOverlap: (
    projectId: string,
    documentId: string,
    partId: string,
    upperId: string,
    lowerId: string,
  ) =>
    apiRequest<LineResponse[]>(
      `${partPath(projectId, documentId, partId)}/segment-health/trims`,
      { method: "POST", body: { upper_id: upperId, lower_id: lowerId } },
    ),

  deleteSegmentSuspect: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
  ) =>
    apiRequest<LineResponse[]>(
      `${partPath(projectId, documentId, partId)}/segment-health/deletions`,
      { method: "POST", body: { line_id: lineId } },
    ),

  enqueueTranscribePart: (
    projectId: string,
    documentId: string,
    partId: string,
    body?: TranscribePartRequest,
  ) =>
    apiRequest<EnqueueJobResponse>(
      `${partPath(projectId, documentId, partId)}/transcribe`,
      { method: "POST", body: body ?? {} },
    ),

  listInferenceModels: () =>
    apiRequest<InferenceModelResponse[]>("/inference/models"),

  resolvePartModelBinding: (
    projectId: string,
    documentId: string,
    partId: string,
    task: InferenceTask,
  ) =>
    apiRequest<ResolvedModelBindingResponse>(
      `${partPath(projectId, documentId, partId)}/model-bindings/resolve?task=${task}`,
    ),

  updateGroundTruthLineText: (
    projectId: string,
    documentId: string,
    transcriptionId: string,
    lineId: string,
    body: LineTranscriptionPatchRequest,
  ) =>
    apiRequest<LineTranscriptionResponse>(
      `${documentPath(projectId, documentId)}/transcriptions/${encodeURIComponent(transcriptionId)}/lines/${encodeURIComponent(lineId)}`,
      { method: "PATCH", body },
    ),

  copyToGroundTruth: (
    projectId: string,
    documentId: string,
    transcriptionId: string,
    body: CopyToGroundTruthRequest,
  ) =>
    apiRequest<CopyToGroundTruthResponse>(
      `${documentPath(projectId, documentId)}/transcriptions/${encodeURIComponent(transcriptionId)}/copy-to-ground-truth`,
      { method: "POST", body },
    ),
};

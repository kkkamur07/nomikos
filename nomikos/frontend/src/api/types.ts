/**
 * API wire types - schema aliases plus the one hand-narrowed geometry claim.
 *
 * Every response and request shape here is `components["schemas"][...]` from
 * the generated `schema.d.ts`, so a backend change reaches the frontend
 * through `codegen:api` and fails the typecheck instead of being shadowed by
 * a second copy of the type. The single exception is the geometry narrowing
 * (`LinePoint`, `GeometryValue`, `NarrowGeometry`): the server types a
 * geometry column as a bare `dict`, so the schema can only say "an object",
 * and the canvas needs "`[x, y]` pairs". That claim is checked where it is
 * consumed (`normalizeGeometryPoints`), not where it arrives.
 */
import type { CursorPage, CursorPageOptions } from "./cursorPagination";
import type { components } from "./schema";

export type TokenResponse = components["schemas"]["TokenResponse"];
export type UserResponse = components["schemas"]["UserResponse"];
export type LoginRequest = components["schemas"]["LoginRequest"];
export type RegisterRequest = components["schemas"]["RegisterRequest"];
export type ProjectResponse = components["schemas"]["ProjectResponse"];
export type ProjectCreateRequest =
  components["schemas"]["ProjectCreateRequest"];
export type ProjectUpdateRequest =
  components["schemas"]["ProjectUpdateRequest"];
export type ShareUserRequest = components["schemas"]["ShareUserRequest"];
export type ProjectCollaboratorResponse =
  components["schemas"]["ProjectCollaboratorResponse"];
export type DocumentResponse = components["schemas"]["DocumentResponse"];
export type DocumentWithPartsResponse =
  components["schemas"]["DocumentWithPartsResponse"];
export type DocumentCreateRequest =
  components["schemas"]["DocumentCreateRequest"];
export type DocumentUpdateRequest =
  components["schemas"]["DocumentUpdateRequest"];
export type DocumentPartResponse =
  components["schemas"]["DocumentPartResponse"];
export type DocumentPartUpdateRequest =
  components["schemas"]["DocumentPartUpdateRequest"];
export type PartsPublishedUpdateRequest =
  components["schemas"]["PartsPublishedUpdateRequest"];
export type DocumentWorkflow = components["schemas"]["DocumentWorkflow"];
export type ReorderPartsRequest = components["schemas"]["ReorderPartsRequest"];

export type PartUploadBeginRequest =
  components["schemas"]["PartUploadBeginRequest"];
export type PartUploadBeginResponse =
  components["schemas"]["PartUploadBeginResponse"];
export type PartUploadFinalizeRequest =
  components["schemas"]["PartUploadFinalizeRequest"];
export type PublicLayoutResponse =
  components["schemas"]["PublicLayoutResponse"];
export type PublicLineResponse = NonNullable<
  PublicLayoutResponse["lines"]
>[number];
export type PublicTranscriptionLayerResponse =
  components["schemas"]["PublicTranscriptionLayerResponse"];
export type TranscriptionLayerResponse =
  components["schemas"]["TranscriptionLayerResponse"];
export type LineTranscriptionResponse =
  components["schemas"]["LineTranscriptionResponse"];
export type CharacterConfidence = components["schemas"]["CharacterConfidence"];
export type JobResponse = components["schemas"]["JobResponse"];
export type JobStatus = components["schemas"]["JobStatus"];
/** The **inference host** one job runs on. Fixed at submission (ADR 0002). */
export type ExecutionTarget = components["schemas"]["ExecutionTarget"];
export type ExecutionPreferenceResponse =
  components["schemas"]["ExecutionPreferenceResponse"];
export type EnqueueJobResponse = components["schemas"]["EnqueueJobResponse"];
/** Every tuning field is server-defaulted, so any subset is a valid body. */
export type SegmentPartRequest = Partial<
  components["schemas"]["SegmentPartRequest"]
>;
export type TranscribePartRequest =
  components["schemas"]["TranscribePartRequest"];
export type InferenceModelResponse =
  components["schemas"]["InferenceModelResponse"];
export type InferenceTask = components["schemas"]["InferenceTask"];
export type ResolvedModelBindingResponse =
  components["schemas"]["ResolvedModelBindingResponse"];
export type EnqueueTestJobRequest =
  components["schemas"]["EnqueueTestJobRequest"];
export type EnqueueTestJobResponse =
  components["schemas"]["EnqueueTestJobResponse"];
/**
 * What a finished transcribe carries back. The server types it as a bare
 * `dict` on the job (`JobResponse.result`), so the shape is the client's
 * claim, made once here and read by `applyTranscribeResult`.
 */
export type TranscribeJobResult = {
  transcription_id: string;
  lines: Array<{ line_id: string; text: string; confidence: number }>;
};
/**
 * The one place the generated wire types are narrowed by hand.
 *
 * The server types a geometry column as a bare `dict` and a point list as
 * `list[list[float]]`, so the schema can only say "an object" and
 * "number[][]". The canvas needs "an `[x, y]` pair" and "a point list, or an
 * object wrapping one", and every helper in `canvasGeometry.ts` is written
 * against those. The claim is checked where it's consumed, not where it
 * arrives: `normalizeGeometryPoints` unwraps the object forms and drops
 * anything that isn't a pair of numbers.
 *
 * Nothing else about these payloads is asserted here. Every field below comes
 * from `schema.d.ts`, so a backend change reaches the frontend through
 * `codegen:api` and fails the typecheck instead of being shadowed by a
 * second copy of the type.
 */
export type LinePoint = [number, number];
/** The same pair, under the name the layout overlay uses. */
export type LayoutPoint = LinePoint;
type GeometryValue =
  | LayoutPoint[]
  | {
      points?: LayoutPoint[];
      type?: string;
      coordinates?: LayoutPoint[];
    };

/** A geometry column as the schema renders it: an object with no shape. */
type WireGeometry = { [key: string]: unknown };

/** Applies the narrowing above to a generated shape, field by field. */
type NarrowGeometry<T> = {
  [K in keyof T]: K extends "points" | "kraken_ceiling"
    ? Exclude<T[K], number[][]> | LinePoint[]
    : K extends "baseline" | "mask" | "box"
      ? Exclude<T[K], WireGeometry> | GeometryValue
      : T[K];
};

type BlockResponse = NarrowGeometry<components["schemas"]["BlockResponse"]>;
export type LineResponse = Omit<
  NarrowGeometry<components["schemas"]["LineResponse"]>,
  "line_transcriptions"
> & {
  /**
   * Generated as optional only because the server field has a
   * `default_factory` and so never reaches the schema's `required` list.
   * FastAPI serializes it on every response.
   */
  line_transcriptions: LineTranscriptionResponse[];
};
export type LineCreateRequest = NarrowGeometry<
  components["schemas"]["LineCreateRequest"]
>;
export type LinePatchRequest = NarrowGeometry<
  components["schemas"]["LinePatchRequest"]
>;
export type LineUpsertRequest = NarrowGeometry<
  components["schemas"]["LineUpsertRequest"]
>;
/** `LinesReplaceRequest`, with the narrowing carried into its lines. */
export type LinesReplaceRequest = {
  lines?: LineUpsertRequest[];
};
/** The geometry-only slice of `LinePatchRequest` a manual edit sends. */
export type UpdateLineGeometryRequest = Pick<
  LinePatchRequest,
  "baseline" | "mask"
>;
export type ResetPartLayoutRequest =
  components["schemas"]["LayoutResetRequest"];

/**
 * The editor's working copy of a layout, not a response shape.
 *
 * `GET .../layout` answers with whole blocks and lines, but the editor only
 * reads and rewrites their geometry, and `syncLayoutLinesFromSegments` folds
 * a segment edit back in as a line carrying geometry and nothing else. Field
 * names and types are the generated ones; the optionality is the editor's own.
 */
export type LayoutBlockResponse = Pick<BlockResponse, "id"> &
  Partial<Pick<BlockResponse, "box" | "manual_geometry">>;
export type SegmentHealthResponse =
  components["schemas"]["SegmentHealthResponse"];
export type SegmentSuspectResponse =
  components["schemas"]["SegmentSuspectResponse"];
export type SegmentSplitResponse =
  components["schemas"]["SegmentSplitResponse"];
export type SegmentMergeResponse =
  components["schemas"]["SegmentMergeResponse"];
export type SegmentOverlapResponse =
  components["schemas"]["SegmentOverlapResponse"];

export type LayoutLineResponse = Pick<LineResponse, "id"> &
  Partial<
    Pick<LineResponse, "block_id" | "baseline" | "mask" | "manual_geometry">
  >;
export type PartLayoutResponse = {
  blocks: LayoutBlockResponse[];
  lines: LayoutLineResponse[];
};

export type PagePairingResponse = components["schemas"]["PagePairingResponse"];
export type PageTranscriptionImportRequest =
  components["schemas"]["PageTranscriptionImportRequest"];
export type PairTextLineRequest = components["schemas"]["PairTextLineRequest"];
export type LineTranscriptionPatchRequest =
  components["schemas"]["LineTranscriptionPatchRequest"];
export type CopyToGroundTruthRequest =
  components["schemas"]["CopyToGroundTruthRequest"];
export type CopyToGroundTruthResponse =
  components["schemas"]["CopyToGroundTruthResponse"];

/** What the Workflow and Download menus put in their count badges. */
export type DocumentWorkflowCounts =
  components["schemas"]["DocumentWorkflowCountsResponse"];

/**
 * Which pages to segment, and with which model. Null means "whatever the
 * document resolves to", which is all the UI offers; the field itself is
 * optional on the wire, so a caller that has nothing to say omits it.
 */
export type DocumentSegmentJobRequest =
  components["schemas"]["SegmentDocumentRequest"];

export type DocumentTranscribeJobRequest =
  components["schemas"]["TranscribeDocumentRequest"];

/**
 * A 202 from either batch route. `skipped` is the count the scope excluded,
 * so "queued 0, skipped 18" is a complete answer rather than a silent no-op.
 * Each entry of `jobs` is the same shape a single enqueue returns, host
 * included.
 */
export type DocumentBatchJobResponse =
  components["schemas"]["DocumentBatchJobsResponse"];

export type PageResponse<T> = CursorPage<T>;

export type ListPageOptions = CursorPageOptions;

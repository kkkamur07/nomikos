/**
 * Document and page endpoints: metadata, workflow counts, exports, batch
 * jobs, transcriptions and page (part) CRUD.
 */
import { collectCursorPages } from "./cursorPagination";
import { apiRequest, fetchBinaryApi } from "./http";
import {
  cursorQuery,
  documentPath,
  partPath,
  reviewedOnlyQuery,
} from "./paths";
import type {
  DocumentBatchJobResponse,
  DocumentCreateRequest,
  DocumentPartResponse,
  DocumentPartUpdateRequest,
  DocumentResponse,
  DocumentSegmentJobRequest,
  DocumentTranscribeJobRequest,
  DocumentUpdateRequest,
  DocumentWithPartsResponse,
  DocumentWorkflowCounts,
  ListPageOptions,
  PageResponse,
  PartsPublishedUpdateRequest,
  PartUploadBeginRequest,
  PartUploadBeginResponse,
  PartUploadFinalizeRequest,
  ReorderPartsRequest,
  TranscriptionLayerResponse,
} from "./types";

export const documentsApi = {
  listDocumentsPage: (
    projectId: string,
    includeArchived = false,
    options: ListPageOptions = {},
  ) => {
    const query = cursorQuery(options, (params) =>
      params.set("include_archived", String(includeArchived)),
    );
    return apiRequest<PageResponse<DocumentResponse>>(
      `/projects/${encodeURIComponent(projectId)}/documents?${query}`,
      { signal: options.signal },
    );
  },

  listDocuments: (
    projectId: string,
    includeArchived = false,
    options?: { maxPages?: number; signal?: AbortSignal },
  ) =>
    collectCursorPages(
      (pageOptions) =>
        documentsApi.listDocumentsPage(projectId, includeArchived, pageOptions),
      options,
    ),

  createDocument: (projectId: string, body: DocumentCreateRequest) =>
    apiRequest<DocumentResponse>(
      `/projects/${encodeURIComponent(projectId)}/documents`,
      {
        method: "POST",
        body,
      },
    ),

  getDocument: (projectId: string, documentId: string) =>
    apiRequest<DocumentWithPartsResponse>(documentPath(projectId, documentId)),

  updateDocument: (
    projectId: string,
    documentId: string,
    body: DocumentUpdateRequest,
  ) =>
    apiRequest<DocumentResponse>(documentPath(projectId, documentId), {
      method: "PATCH",
      body,
    }),

  deleteDocument: (projectId: string, documentId: string) =>
    apiRequest<void>(documentPath(projectId, documentId), {
      method: "DELETE",
    }),

  /**
   * Mint a fresh share token. Owner-only, and every link built from the old
   * token stops resolving the moment this returns.
   */
  rotateDocumentShareToken: (projectId: string, documentId: string) =>
    apiRequest<DocumentResponse>(
      `${documentPath(projectId, documentId)}/share-token/rotate`,
      { method: "POST" },
    ),

  /**
   * The four numbers the document action menus label themselves with.
   *
   * Deliberately a separate read from the document: the parts list carries
   * `reviewed` but says nothing about whether a page has lines or a pairing,
   * so the counts a Workflow menu needs cannot be derived from what this page
   * already holds.
   */
  getDocumentWorkflowCounts: (projectId: string, documentId: string) =>
    apiRequest<DocumentWorkflowCounts>(
      `${documentPath(projectId, documentId)}/workflow-counts`,
    ),

  /** Zip of the PAGE XML for every page next to the image it describes. */
  exportDocumentPageXml: (
    projectId: string,
    documentId: string,
    reviewedOnly = false,
  ) =>
    fetchBinaryApi(
      `${documentPath(projectId, documentId)}/export/page-xml${reviewedOnlyQuery(reviewedOnly)}`,
    ),

  exportDocumentTranscriptionPdf: (
    projectId: string,
    documentId: string,
    reviewedOnly = false,
  ) =>
    fetchBinaryApi(
      `${documentPath(projectId, documentId)}/export/transcription-pdf${reviewedOnlyQuery(reviewedOnly)}`,
    ),

  exportDocumentText: (
    projectId: string,
    documentId: string,
    reviewedOnly = false,
  ) =>
    fetchBinaryApi(
      `${documentPath(projectId, documentId)}/export/text${reviewedOnlyQuery(reviewedOnly)}`,
    ),

  exportDocumentDocx: (
    projectId: string,
    documentId: string,
    reviewedOnly = false,
  ) =>
    fetchBinaryApi(
      `${documentPath(projectId, documentId)}/export/docx${reviewedOnlyQuery(reviewedOnly)}`,
    ),

  enqueueDocumentSegment: (
    projectId: string,
    documentId: string,
    body: DocumentSegmentJobRequest,
  ) =>
    apiRequest<DocumentBatchJobResponse>(
      `${documentPath(projectId, documentId)}/jobs/segment`,
      { method: "POST", body },
    ),

  enqueueDocumentTranscribe: (
    projectId: string,
    documentId: string,
    body: DocumentTranscribeJobRequest,
  ) =>
    apiRequest<DocumentBatchJobResponse>(
      `${documentPath(projectId, documentId)}/jobs/transcribe`,
      { method: "POST", body },
    ),

  listTranscriptions: (projectId: string, documentId: string) =>
    apiRequest<TranscriptionLayerResponse[]>(
      `${documentPath(projectId, documentId)}/transcriptions`,
    ),

  uploadPart: (projectId: string, documentId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return apiRequest<DocumentPartResponse>(
      `${documentPath(projectId, documentId)}/parts`,
      { method: "POST", body: form },
    );
  },

  beginPartUpload: (
    projectId: string,
    documentId: string,
    body: PartUploadBeginRequest,
  ) =>
    apiRequest<PartUploadBeginResponse>(
      `${documentPath(projectId, documentId)}/parts/upload`,
      { method: "POST", body },
    ),

  finalizePartUpload: (
    projectId: string,
    documentId: string,
    partId: string,
    body: PartUploadFinalizeRequest,
  ) =>
    apiRequest<DocumentPartResponse>(
      `${partPath(projectId, documentId, partId)}/finalize`,
      { method: "POST", body },
    ),

  reorderParts: (
    projectId: string,
    documentId: string,
    body: ReorderPartsRequest,
  ) =>
    apiRequest<DocumentPartResponse[]>(
      `${documentPath(projectId, documentId)}/parts/reorder`,
      { method: "PATCH", body },
    ),

  deletePart: (projectId: string, documentId: string, partId: string) =>
    apiRequest<void>(partPath(projectId, documentId, partId), {
      method: "DELETE",
    }),

  updatePartReviewStatus: (
    projectId: string,
    documentId: string,
    partId: string,
    body: DocumentPartUpdateRequest,
  ) =>
    apiRequest<DocumentPartResponse>(partPath(projectId, documentId, partId), {
      method: "PATCH",
      body,
    }),

  /**
   * Which pages of a published document the public reader can actually reach.
   *
   * Takes a batch because the endpoint does. The only caller today sends one
   * page per click, deliberately: the response is authoritative for the whole
   * document, so a request carrying rows this tab loaded minutes ago could
   * overwrite a flag someone else has since changed.
   */
  updatePartsPublished: (
    projectId: string,
    documentId: string,
    body: PartsPublishedUpdateRequest,
  ) =>
    apiRequest<DocumentPartResponse[]>(
      `${documentPath(projectId, documentId)}/parts/published`,
      { method: "PATCH", body },
    ),
};

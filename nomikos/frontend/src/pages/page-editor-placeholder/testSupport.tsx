import { act, fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";

import {
  api,
  type DocumentWithPartsResponse,
  type ExecutionPreferenceResponse,
  type JobResponse,
} from "../../api/client";
import { BackgroundJobsProvider } from "../../context/BackgroundJobsContext";
import { BackgroundJobsPanel } from "../../components/BackgroundJobsPanel";
import { PageEditorPlaceholderPage } from "../PageEditorPlaceholderPage";

vi.mock("../../auth/session", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../auth/session")>();
  return {
    ...actual,
    hasAccessToken: vi.fn(() => true),
    redirectToLogin: vi.fn(),
  };
});

vi.mock("../../components/AuthenticatedImage", () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  const mockedApi = {
    ...actual.api,
    getDocument: vi.fn(),
    getPartLayout: vi.fn(),
    listPartLines: vi.fn(),
    listTranscriptions: vi.fn(),
    getPagePairing: vi.fn(),
    importPageTranscription: vi.fn(),
    pairTextLine: vi.fn(),
    updateGroundTruthLineText: vi.fn(),
    copyToGroundTruth: vi.fn(),
    updatePartReviewStatus: vi.fn(),
    replacePartLines: vi.fn(),
    createPartLine: vi.fn(),
    patchPartLine: vi.fn(),
    deletePartLine: vi.fn(),
    createPartBlock: vi.fn(),
    patchPartBlock: vi.fn(),
    deletePartBlock: vi.fn(),
    updateLineGeometry: vi.fn(),
    resetPartLayout: vi.fn(),
    generateTranscriptionPdf: vi.fn(),
    updateDocument: vi.fn(),
    listInferenceModels: vi.fn(),
    resolvePartModelBinding: vi.fn(),
    enqueueTranscribePart: vi.fn(),
    segmentPart: vi.fn(),
    getExecutionPreference: vi.fn(),
    setExecutionPreference: vi.fn(),
    getJob: vi.fn(),
  };

  async function waitForJob(
    jobId: string,
    options?: { timeoutMs?: number; onUpdate?: (job: JobResponse) => void },
  ): Promise<JobResponse> {
    const timeoutMs = options?.timeoutMs ?? 120_000;
    const deadline = Date.now() + timeoutMs;
    let lastJob: JobResponse | null = null;
    while (Date.now() < deadline) {
      const job = await mockedApi.getJob(jobId);
      if (
        !lastJob ||
        lastJob.status !== job.status ||
        lastJob.updated_at !== job.updated_at
      ) {
        options?.onUpdate?.(job);
      }
      lastJob = job;
      if (job.status === "done") return job;
      if (job.status === "failed") {
        throw new Error(job.error ?? "Job failed.");
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    throw new Error("Job timed out.");
  }

  return {
    ...actual,
    api: mockedApi,
    waitForJob: vi.fn(waitForJob),
  };
});

export type MockedEditorApi = {
  getDocument: ReturnType<typeof vi.fn>;
  getPartLayout: ReturnType<typeof vi.fn>;
  listPartLines: ReturnType<typeof vi.fn>;
  listTranscriptions: ReturnType<typeof vi.fn>;
  getPagePairing: ReturnType<typeof vi.fn>;
  importPageTranscription: ReturnType<typeof vi.fn>;
  pairTextLine: ReturnType<typeof vi.fn>;
  updateGroundTruthLineText: ReturnType<typeof vi.fn>;
  copyToGroundTruth: ReturnType<typeof vi.fn>;
  updatePartReviewStatus: ReturnType<typeof vi.fn>;
  replacePartLines: ReturnType<typeof vi.fn>;
  createPartLine: ReturnType<typeof vi.fn>;
  patchPartLine: ReturnType<typeof vi.fn>;
  deletePartLine: ReturnType<typeof vi.fn>;
  createPartBlock: ReturnType<typeof vi.fn>;
  patchPartBlock: ReturnType<typeof vi.fn>;
  deletePartBlock: ReturnType<typeof vi.fn>;
  updateLineGeometry: ReturnType<typeof vi.fn>;
  resetPartLayout: ReturnType<typeof vi.fn>;
  generateTranscriptionPdf: ReturnType<typeof vi.fn>;
  updateDocument: ReturnType<typeof vi.fn>;
  listInferenceModels: ReturnType<typeof vi.fn>;
  resolvePartModelBinding: ReturnType<typeof vi.fn>;
  enqueueTranscribePart: ReturnType<typeof vi.fn>;
  segmentPart: ReturnType<typeof vi.fn>;
  getExecutionPreference: ReturnType<typeof vi.fn>;
  setExecutionPreference: ReturnType<typeof vi.fn>;
  getJob: ReturnType<typeof vi.fn>;
};

export const mockedApi = api as unknown as MockedEditorApi;

export const DOCUMENT: DocumentWithPartsResponse = {
  id: "doc-1",
  project_id: "project-1",
  name: "Grec 1360",
  workflow: "draft",
  created_at: "2026-06-16T10:00:00Z",
  updated_at: "2026-06-16T10:00:00Z",
  part_count: 1,
  parts: [
    {
      id: "part-1",
      document_id: "doc-1",
      order: 0,
      image_url: "/media/parts/part-1",
      width: 640,
      height: 900,
      reviewed: false,
      created_at: "2026-06-16T10:00:00Z",
    },
  ],
};

/**
 * The same document, `count` pages long, in page order.
 *
 * `DOCUMENT` holds one page because most of these tests only need a canvas to
 * put a Segment on. Paging needs a document long enough for a page turn to
 * mean something, and long enough that "the page rail fetched a thumbnail for
 * every page in the manuscript" is a mistake a test can see.
 */
export function documentWithPages(count: number): DocumentWithPartsResponse {
  return {
    ...DOCUMENT,
    part_count: count,
    parts: Array.from({ length: count }, (_, index) => ({
      ...DOCUMENT.parts[0],
      id: `part-${index + 1}`,
      order: index,
      image_url: `/media/parts/part-${index + 1}`,
    })),
  };
}

/**
 * A Line as `listPartLines` answers with one.
 *
 * The default is the plain manual rectangle-ish polygon most of these tests
 * only need to exist so that a Segment is on the canvas; anything a test
 * actually asserts on - its id, order, kind, points, source, or its
 * transcriptions - is passed as an override, so no call site loses a value it
 * cared about. `baseline` and `mask` are deliberately absent: a segment line
 * carries neither, and adding them here would put baselines on the canvas of
 * every test that never asked for one.
 */
export function line(overrides: Record<string, unknown> = {}) {
  return {
    id: "line-1",
    part_id: "part-1",
    block_id: null,
    order: 0,
    kind: "polygon",
    points: [
      [10, 10],
      [50, 10],
      [50, 30],
      [10, 30],
    ],
    source: "manual",
    source_metadata: null,
    kraken_ceiling: null,
    manual_geometry: true,
    line_transcriptions: [],
    created_at: "2026-06-16T10:00:00Z",
    ...overrides,
  };
}

/** The baseline the layout fixtures are drawn around, fresh on every call. */
export function baselinePoints() {
  return [
    [60, 140],
    [300, 150],
  ];
}

/** The line outline the layout fixtures are drawn around, fresh every call. */
export function maskPoints() {
  return [
    [55, 110],
    [305, 118],
    [300, 178],
    [50, 168],
  ];
}

/**
 * The same Line as `listPartLines` answers with it once Kraken has laid the
 * page out: machine geometry, carrying the baseline and mask the layout editor
 * moves. Pass `baseline`/`mask` as `{ points }` to drive the wrapped form.
 */
export function layoutLine(overrides: Record<string, unknown> = {}) {
  return line({
    kind: "polygon",
    points: maskPoints(),
    baseline: baselinePoints(),
    mask: maskPoints(),
    source: "kraken",
    manual_geometry: false,
    ...overrides,
  });
}

/**
 * `getPartLayout`'s answer for a part holding one machine-detected Line.
 *
 * The overrides land on the line, which is the only part of this payload the
 * layout tests vary - `manual_geometry`, and whether the geometry arrives as a
 * bare point list or wrapped in `{ points }`.
 */
export function layoutWith(overrides: Record<string, unknown> = {}) {
  return {
    blocks: [],
    lines: [
      {
        id: "line-1",
        baseline: baselinePoints(),
        mask: maskPoints(),
        manual_geometry: false,
        ...overrides,
      },
    ],
  };
}

function pageEditorTree() {
  return (
    <BackgroundJobsProvider>
      <PageEditorPlaceholderPage />
      <BackgroundJobsPanel />
    </BackgroundJobsProvider>
  );
}

export function renderPageEditor() {
  window.history.replaceState(
    {},
    "",
    "/projects/project-1/documents/doc-1/parts/part-1",
  );
  const view = render(pageEditorTree());
  return {
    ...view,
    /**
     * Re-render the same editor against whatever the address bar now says.
     *
     * `useParams` is stubbed to read `window.location` (see vitest.setup.ts),
     * so this is how a test plays back a route change the editor did not
     * make - browser Back, above all.
     */
    rerenderFromUrl: () => view.rerender(pageEditorTree()),
  };
}

export async function enableBaselinesOnCanvas() {
  fireEvent.click(
    await screen.findByRole("button", { name: /editor settings/i }),
  );
  const checkbox = screen.getByRole("checkbox", {
    name: /show line baselines/i,
  });
  if (!(checkbox as HTMLInputElement).checked) {
    fireEvent.click(checkbox);
  }
}

export async function flushPageEditorEffects() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 100));
  });
}

/**
 * The account-level **host preference**, stored the way the platform stores it:
 * a PUT changes it and every later GET reflects the change. Canned responses
 * would let a "setting" that never persisted still pass its own test.
 */
export function seedExecutionPreference(preferLocalInference = false): void {
  let stored = preferLocalInference;
  const current = (): ExecutionPreferenceResponse => ({
    prefer_local_inference: stored,
    preferred_execution_target: stored ? "local" : "cloud",
    available_targets: ["cloud"],
  });
  mockedApi.getExecutionPreference.mockImplementation(async () => current());
  mockedApi.setExecutionPreference.mockImplementation(
    async (preferLocal: boolean) => {
      stored = preferLocal;
      return current();
    },
  );
}

export function resetPageEditorApiMocks() {
  vi.clearAllMocks();
  seedExecutionPreference();
  mockedApi.getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
  mockedApi.listPartLines.mockReset();
  mockedApi.listPartLines.mockResolvedValue([]);
  mockedApi.listTranscriptions.mockResolvedValue([
    {
      id: "ground-truth-1",
      document_id: "doc-1",
      name: "Ground truth",
      kind: "ground_truth",
      created_by_job_id: null,
      created_at: "2026-06-16T10:00:00Z",
    },
  ]);
  mockedApi.getPagePairing.mockResolvedValue({
    text_lines: [],
    pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
  });
  mockedApi.listInferenceModels.mockResolvedValue([]);
  mockedApi.resolvePartModelBinding.mockResolvedValue({
    binding: {
      id: "binding-1",
      task: "transcribe",
      model_id: "model-1",
      overrides: {},
    },
    model: {
      id: "model-1",
      name: "greek-v1",
      provider: "kraken",
      task: "transcribe",
      artifact_ref: "x",
      default_params: {},
      created_at: "2026-06-16T10:00:00Z",
    },
    effective_params: {},
  });
  mockedApi.updateDocument.mockResolvedValue({
    ...DOCUMENT,
    workflow: "published",
  });
  mockedApi.importPageTranscription.mockResolvedValue({
    text_lines: [],
    pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
  });
  mockedApi.pairTextLine.mockResolvedValue({
    text_lines: [],
    pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
  });
  mockedApi.updateGroundTruthLineText.mockResolvedValue({
    id: "line-tx-1",
    transcription_id: "ground-truth-1",
    transcription_kind: "ground_truth",
    text: "typed approved text",
    confidence: null,
    character_confidences: null,
  });
  mockedApi.copyToGroundTruth.mockResolvedValue({
    copied_line_ids: ["line-1"],
  });
  mockedApi.updatePartReviewStatus.mockResolvedValue({
    ...DOCUMENT.parts[0],
    reviewed: true,
  });
  mockedApi.deletePartLine.mockResolvedValue(undefined);
  mockedApi.createPartBlock.mockImplementation(
    async (_projectId, _documentId, _partId, body) => ({
      id: "block-new",
      part_id: "part-1",
      manual_geometry: true,
      created_at: "2026-10-11T10:00:00Z",
      label: null,
      ...body,
    }),
  );
  mockedApi.patchPartBlock.mockImplementation(
    async (_projectId, _documentId, _partId, blockId, body) => ({
      id: blockId,
      part_id: "part-1",
      order: 0,
      box: { points: [] },
      manual_geometry: true,
      created_at: "2026-10-11T10:00:00Z",
      label: null,
      ...body,
    }),
  );
  mockedApi.deletePartBlock.mockResolvedValue(undefined);
  mockedApi.createPartLine.mockImplementation(
    async (_projectId, _documentId, _partId, body) => ({
      id: `line-${(body.order ?? 0) + 1}`,
      part_id: "part-1",
      block_id: body.block_id ?? null,
      order: body.order ?? 0,
      kind: body.kind,
      points: body.points,
      baseline: body.baseline ?? { points: body.points },
      mask: body.mask ?? { points: body.points },
      source: "manual",
      source_metadata: null,
      kraken_ceiling: null,
      manual_geometry: true,
      line_transcriptions: [],
      created_at: "2026-06-16T10:00:00Z",
    }),
  );
  mockedApi.patchPartLine.mockImplementation(
    async (_projectId, _documentId, _partId, lineId, body) => {
      const existing = (
        (await mockedApi.listPartLines.mock.results.at(-1)?.value) as
          Array<Record<string, unknown>> | undefined
      )?.find((line) => line.id === lineId) ?? {
        id: lineId,
        part_id: "part-1",
        block_id: null,
        order: 0,
        kind: "polygon",
        points: body.points ?? [],
        baseline: { points: [] },
        mask: null,
        source: "manual",
        source_metadata: null,
        kraken_ceiling: null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      };
      return {
        ...existing,
        ...body,
        manual_geometry: true,
        source: "manual",
      };
    },
  );
  mockedApi.replacePartLines.mockImplementation(
    async (_projectId, _documentId, _partId, body) =>
      body.lines.map((line: Record<string, unknown>, index: number) => ({
        id: (line.id as string | undefined) ?? `line-${index + 1}`,
        part_id: "part-1",
        block_id: line.block_id ?? null,
        order: line.order,
        kind: line.kind,
        points: line.points,
        baseline: line.baseline ?? { points: [] },
        mask: line.mask ?? null,
        source: line.source,
        source_metadata: line.source_metadata ?? null,
        kraken_ceiling: line.kraken_ceiling ?? null,
        manual_geometry: true,
        line_transcriptions: [],
        created_at: "2026-06-16T10:00:00Z",
      })),
  );
}

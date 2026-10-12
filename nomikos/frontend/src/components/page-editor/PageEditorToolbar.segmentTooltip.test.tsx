import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type {
  DocumentWithPartsResponse,
  InferenceModelResponse,
} from "../../api/client";
import { DEFAULT_PAGE_EDITOR_SETTINGS } from "./pageEditorSettings";
import { PageEditorToolbar } from "./PageEditorToolbar";

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

function segmentModel(id: string, name: string): InferenceModelResponse {
  return {
    id,
    name,
    task: "segment",
    provider: "test",
    artifact_ref: `registry://${id === "seg-kraken" ? "blla-segment" : "ppocr-segment"}?tag=stable`,
    created_at: "2026-01-01T00:00:00Z",
    default_params: {},
  };
}

const SEGMENT_MODELS = [
  segmentModel("seg-kraken", "kraken"),
  segmentModel("seg-ppocr", "ppocr"),
];

function toolbar(
  segmentModels: InferenceModelResponse[],
  selectedSegmentModelId: string | null,
) {
  render(
    <PageEditorToolbar
      projectId="project-1"
      documentId="document-1"
      document={{ name: "page" } as unknown as DocumentWithPartsResponse}
      partIndex={1}
      pageCount={3}
      hasPreviousPart={false}
      hasNextPart
      onPreviousPart={() => {}}
      onNextPart={() => {}}
      lines={[]}
      pairingProgress={{ paired_lines: 0, total_lines: 0, percent: 0 }}
      partId="part-1"
      selectedSegmentId={null}
      textLines={[]}
      onPairTextLine={() => {}}
      onDocumentWorkflowChange={() => {}}
      actionsOpen={false}
      onActionsOpenChange={() => {}}
      segmenting={false}
      ocrRunning={false}
      transcribeModels={[]}
      selectedTranscribeModelId={null}
      onSelectedTranscribeModelIdChange={() => {}}
      segmentModels={segmentModels}
      selectedSegmentModelId={selectedSegmentModelId}
      onSelectedSegmentModelIdChange={() => {}}
      onRunAutoSegment={() => {}}
      onRunSegmentOcr={() => {}}
      onRunPageOcr={() => {}}
      transcriptionPdfOpen={false}
      onOpenTranscriptionPdf={() => {}}
      onCloseTranscriptionPdf={() => {}}
      settingsOpen={false}
      onSettingsOpenChange={() => {}}
      segmentHealthOpen={false}
      onSegmentHealthOpenChange={() => {}}
      segmentHealth={{
        report: null,
        loading: false,
        error: null,
        pending: null,
        apply: () => {},
        refresh: () => {},
      }}
      canvasSettings={DEFAULT_PAGE_EDITOR_SETTINGS}
      onCanvasSettingsChange={() => {}}
      preferLocalInference={false}
      onPreferLocalInferenceChange={() => {}}
      preferenceSaving={false}
      hasLocalCapacity={false}
      hostPreferenceLoading={false}
    />,
  );
  return screen.getByRole("button", { name: "Segment" });
}

describe("PageEditorToolbar project defaults", () => {
  // Setting a default happens on the project page now. The editor still
  // starts on it, but nothing here offers to change it.
  it("shows no project default text and no make-default button", () => {
    toolbar(SEGMENT_MODELS, "seg-kraken");
    expect(screen.queryByText(/project default/i)).toBeNull();
    expect(screen.queryByRole("button", { name: /default/i })).toBeNull();
  });
});

describe("PageEditorToolbar segment tooltip", () => {
  it("names the selected segment model", () => {
    expect(toolbar(SEGMENT_MODELS, "seg-kraken").getAttribute("title")).toBe(
      "Segment this page with kraken",
    );
  });

  it("follows the selection instead of a hard-coded model", () => {
    expect(toolbar(SEGMENT_MODELS, "seg-ppocr").getAttribute("title")).toBe(
      "Segment this page with ppocr",
    );
  });

  it("says only Segment this page when the catalog is empty", () => {
    expect(toolbar([], null).getAttribute("title")).toBe("Segment this page");
  });
});

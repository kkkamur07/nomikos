import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api, type DocumentWithPartsResponse } from "../../api/client";
import type * as ClientModule from "../../api/client";
import { PageEditorToolbar } from "./PageEditorToolbar";
import { DEFAULT_PAGE_EDITOR_SETTINGS } from "./pageEditorSettings";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<ClientModule>();
  return {
    ...actual,
    api: { ...actual.api, sendFeedback: vi.fn() },
  };
});

function renderToolbar() {
  const document = { id: "d-1", name: "Grec 1360" };
  render(
    <PageEditorToolbar
      projectId="p-1"
      documentId="d-1"
      document={document as unknown as DocumentWithPartsResponse}
      partIndex={1}
      pageCount={2}
      hasPreviousPart={false}
      hasNextPart
      onPreviousPart={vi.fn()}
      onNextPart={vi.fn()}
      lines={[]}
      pairingProgress={{ paired_lines: 0, total_lines: 0, percent: 0 }}
      partId="part-1"
      selectedSegmentId={null}
      textLines={[]}
      onPairTextLine={vi.fn()}
      onDocumentWorkflowChange={vi.fn()}
      actionsOpen={false}
      onActionsOpenChange={vi.fn()}
      segmenting={false}
      ocrRunning={false}
      transcribeModels={[]}
      selectedTranscribeModelId={null}
      onSelectedTranscribeModelIdChange={vi.fn()}
      segmentModels={[]}
      selectedSegmentModelId={null}
      onSelectedSegmentModelIdChange={vi.fn()}
      onRunAutoSegment={vi.fn()}
      onRunSegmentOcr={vi.fn()}
      onRunPageOcr={vi.fn()}
      transcriptionPdfOpen={false}
      onOpenTranscriptionPdf={vi.fn()}
      onCloseTranscriptionPdf={vi.fn()}
      settingsOpen={false}
      onSettingsOpenChange={vi.fn()}
      segmentHealthOpen={false}
      onSegmentHealthOpenChange={vi.fn()}
      segmentHealth={{
        report: null,
        loading: false,
        error: null,
        pending: null,
        apply: vi.fn(),
        refresh: vi.fn(),
      }}
      canvasSettings={DEFAULT_PAGE_EDITOR_SETTINGS}
      onCanvasSettingsChange={vi.fn()}
      preferLocalInference={false}
      onPreferLocalInferenceChange={vi.fn()}
      preferenceSaving={false}
      hasLocalCapacity={false}
      hostPreferenceLoading={false}
    />,
  );
}

beforeEach(() => {
  vi.mocked(api.sendFeedback).mockReset();
});

describe("PageEditorToolbar feedback", () => {
  it("renders the Feedback control beside the settings gear", () => {
    renderToolbar();

    expect(screen.getByRole("button", { name: "Feedback" })).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Editor settings" }),
    ).toBeTruthy();
  });

  it("opens the dialog and shows Thanks after sending", async () => {
    vi.mocked(api.sendFeedback).mockResolvedValue({
      id: "f-1",
      created_at: "2026-10-11T12:00:00Z",
      message: "nice",
      page_path: window.location.pathname,
    });
    renderToolbar();

    fireEvent.click(screen.getByRole("button", { name: "Feedback" }));
    fireEvent.change(screen.getByLabelText("Feedback"), {
      target: { value: "nice" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("button", { name: "Thanks" })).toBeTruthy();
    expect(api.sendFeedback).toHaveBeenCalledWith({
      message: "nice",
      page_path: window.location.pathname,
    });
  });
});

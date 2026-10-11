import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  api,
  type DocumentWithPartsResponse,
  type LineResponse,
  type SegmentHealthResponse,
} from "../../api/client";
import type * as ClientModule from "../../api/client";
import { PageEditorCanvasIsland } from "./PageEditorCanvasIsland";
import { PageEditorToolbar } from "./PageEditorToolbar";
import { DEFAULT_PAGE_EDITOR_SETTINGS } from "./pageEditorSettings";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<ClientModule>();
  return {
    ...actual,
    api: { ...actual.api, sendFeedback: vi.fn() },
  };
});

type ToolbarProps = Parameters<typeof PageEditorToolbar>[0];

function renderToolbar(overrides: Partial<ToolbarProps> = {}) {
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
      transcriptOpen
      onToggleTranscript={vi.fn()}
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
      {...overrides}
    />,
  );
}

beforeEach(() => {
  vi.mocked(api.sendFeedback).mockReset();
});

describe("PageEditorToolbar feedback", () => {
  it("renders the icon-only Feedback control beside the settings gear", () => {
    renderToolbar();

    expect(screen.getByRole("button", { name: "Send feedback" })).toBeTruthy();
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

    fireEvent.click(screen.getByRole("button", { name: "Send feedback" }));
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

const twoLines = [{ id: "l-1" }, { id: "l-2" }] as unknown as LineResponse[];
const healthWithFindings = {
  report: { finding_count: 3 } as unknown as SegmentHealthResponse,
  loading: false,
  error: null,
  pending: null,
  apply: vi.fn(),
  refresh: vi.fn(),
};

describe("PageEditorToolbar transcript toggle and Check page", () => {
  it("shows a pressed Transcript toggle in place of Check page", () => {
    const onToggleTranscript = vi.fn();
    renderToolbar({ onToggleTranscript });

    const toggle = screen.getByRole("button", { name: "Transcript" });
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    expect(toggle.getAttribute("aria-keyshortcuts")).toBe("T");
    expect(toggle.getAttribute("title")).toBe("Hide transcript pane (T)");
    expect(screen.queryByRole("button", { name: /check page/i })).toBeNull();

    fireEvent.click(toggle);
    expect(onToggleTranscript).toHaveBeenCalledTimes(1);
  });

  it("reads unpressed once the pane is closed", () => {
    renderToolbar({ transcriptOpen: false });
    const toggle = screen.getByRole("button", { name: "Transcript" });
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    expect(toggle.getAttribute("title")).toBe("Show transcript pane (T)");
  });

  it("puts Check page with its count first in the Workflow menu", () => {
    const onActionsOpenChange = vi.fn();
    const onSegmentHealthOpenChange = vi.fn();
    renderToolbar({
      actionsOpen: true,
      lines: twoLines,
      segmentHealth: healthWithFindings,
      onActionsOpenChange,
      onSegmentHealthOpenChange,
    });

    const menu = screen.getByRole("menu", { name: "Workflow" });
    const first = menu.querySelector("[role^='menuitem']");
    expect(first?.textContent).toBe("Check page · 3");

    fireEvent.click(first as HTMLElement);
    expect(onActionsOpenChange).toHaveBeenCalledWith(false);
    expect(onSegmentHealthOpenChange).toHaveBeenCalledWith(true);
  });

  it("disables Check page on a page with no lines", () => {
    renderToolbar({ actionsOpen: true });
    const item = screen.getByRole("menuitem", { name: "Check page" });
    expect((item as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("PageEditorCanvasIsland", () => {
  it("no longer carries a Transcript button", () => {
    render(
      <PageEditorCanvasIsland
        tool="none"
        onSelectTool={vi.fn()}
        onPickDrawMode={vi.fn()}
        canDelete={false}
        onDeleteSelected={vi.fn()}
        zoomPercent={100}
        onZoomIn={vi.fn()}
        onZoomOut={vi.fn()}
        onFitToView={vi.fn()}
        onResetZoom={vi.fn()}
        panOverride={false}
      />,
    );
    expect(screen.queryByRole("button", { name: /transcript/i })).toBeNull();
  });
});

describe("PageEditorToolbar fold", () => {
  // jsdom lays nothing out, so the bar's widths are faked: the content asks
  // for less room at each fold level, the way the real buttons do.
  const required = { "0": 1400, "1": 1150, "2": 900 } as Record<string, number>;
  let available = 2000;
  const restore: (() => void)[] = [];

  function fakeWidth(name: "clientWidth" | "scrollWidth") {
    const original = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      name,
    );
    Object.defineProperty(HTMLElement.prototype, name, {
      configurable: true,
      get(this: HTMLElement) {
        if (!this.classList.contains("pe-toolbar")) return 0;
        const need = required[this.dataset.collapse ?? "0"];
        return name === "clientWidth" ? available : Math.max(available, need);
      },
    });
    restore.push(() => {
      if (original)
        Object.defineProperty(HTMLElement.prototype, name, original);
    });
  }

  beforeEach(() => {
    fakeWidth("clientWidth");
    fakeWidth("scrollWidth");
  });
  afterEach(() => {
    restore.splice(0).forEach((undo) => undo());
  });

  it("keeps labels when everything fits", () => {
    available = 2000;
    renderToolbar();
    expect(screen.getByRole("button", { name: /^segment$/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /workflow ▾/i })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "More" })).toBeNull();
  });

  it("drops to icon-only buttons first", () => {
    available = 1200;
    renderToolbar();
    const segment = screen.getByRole("button", { name: "Segment" });
    expect(segment.textContent).toBe("");
    expect(segment.className).toContain("pe-tb-btn--icon");
    expect(screen.getByRole("button", { name: "Workflow" })).toBeTruthy();
    const transcript = screen.getByRole("button", { name: "Transcript" });
    expect(transcript.textContent).toBe("");
    expect(transcript.getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByRole("button", { name: "More" })).toBeNull();
  });

  it("folds the labelled buttons into More when icons still do not fit", () => {
    available = 1000;
    renderToolbar();
    expect(screen.queryByRole("button", { name: "Segment" })).toBeNull();
    expect(screen.queryByRole("button", { name: /workflow/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /check page/i })).toBeNull();
    expect(screen.queryByRole("button", { name: "Transcript" })).toBeNull();
    expect(screen.getByRole("button", { name: "More" })).toBeTruthy();
    // The always-visible controls stay put.
    expect(screen.getByRole("button", { name: "Send feedback" })).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Editor settings" }),
    ).toBeTruthy();
  });

  it("offers Transcript as a checkable item and Check page in More", () => {
    available = 1000;
    const onToggleTranscript = vi.fn();
    renderToolbar({
      actionsOpen: true,
      lines: twoLines,
      segmentHealth: healthWithFindings,
      onToggleTranscript,
    });
    const menu = screen.getByRole("menu", { name: "More" });
    expect(menu).toBeTruthy();
    const transcript = screen.getByRole("menuitemcheckbox", {
      name: /^transcript(?!ion)/i,
    });
    expect(transcript.getAttribute("aria-checked")).toBe("true");
    expect(
      screen.getByRole("menuitem", { name: "Check page · 3" }),
    ).toBeTruthy();

    fireEvent.click(transcript);
    expect(onToggleTranscript).toHaveBeenCalledTimes(1);
  });
});

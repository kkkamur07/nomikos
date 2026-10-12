import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  api,
  type DocumentWithPartsResponse,
  type PublicLayoutResponse,
} from "../api/client";
import { PublicDocumentPage } from "./PublicDocumentPage";

vi.mock("../components/public/PublicPageCanvas", () => ({
  PublicPageCanvas: ({
    regions,
    selectedRegionId,
    onSelectRegion,
  }: {
    regions: Array<{ id: number }>;
    selectedRegionId: number | null;
    onSelectRegion: (id: number | null) => void;
  }) => (
    <div data-testid="public-page-canvas">
      <span>Regions: {regions.length}</span>
      <span>Selected: {selectedRegionId ?? "none"}</span>
      <button type="button" onClick={() => onSelectRegion(1)}>
        Select line 1
      </button>
    </div>
  ),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getPublicDocument: vi.fn(),
      getPublicLayout: vi.fn(),
      getPublicTranscriptionPdf: vi.fn(),
      getPublicPageXmlBundle: vi.fn(),
      getPublicDocx: vi.fn(),
    },
  };
});

const DOCUMENT: DocumentWithPartsResponse = {
  id: "doc-1",
  project_id: "project-1",
  name: "MS Or. 1445 - Genesis",
  workflow: "published",
  created_at: "2026-06-16T10:00:00Z",
  updated_at: "2026-06-16T10:00:00Z",
  part_count: 1,
  parts: [
    {
      id: "part-1",
      document_id: "doc-1",
      order: 0,
      image_url: "/public/media/parts/part-1",
      width: 640,
      height: 900,
      reviewed: true,
      created_at: "2026-06-16T10:00:00Z",
    },
  ],
};

const LAYOUT: PublicLayoutResponse = {
  blocks: [
    {
      id: "block-1",
      part_id: "part-1",
      order: 0,
      box: { coordinates: [0, 0, 640, 900] },
    },
  ],
  lines: [
    {
      id: "line-1",
      part_id: "part-1",
      order: 0,
      points: [
        [10, 10],
        [50, 10],
        [50, 30],
        [10, 30],
      ],
      line_transcriptions: [
        {
          id: "lt-1",
          transcription_id: "layer-1",
          transcription_kind: "ground_truth",
          text: "alpha beta",
          confidence: null,
        },
      ],
    },
  ],
};

function renderPublicPage() {
  window.history.replaceState(
    {},
    "",
    "/public/projects/project-1/documents/doc-1?t=share-token-1",
  );
  return render(<PublicDocumentPage />);
}

function defer<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("PublicDocumentPage", () => {
  beforeEach(() => {
    vi.mocked(api.getPublicDocument).mockResolvedValue(DOCUMENT);
    vi.mocked(api.getPublicLayout).mockResolvedValue(LAYOUT);
    vi.mocked(api.getPublicTranscriptionPdf).mockResolvedValue(
      new Blob(["pdf"], { type: "application/pdf" }),
    );
    vi.mocked(api.getPublicPageXmlBundle).mockResolvedValue(
      new Blob(["PK"], { type: "application/zip" }),
    );
    vi.mocked(api.getPublicDocx).mockResolvedValue(
      new Blob(["PK"], {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      }),
    );
  });

  it("keeps public chrome and spins only in the content region while loading", async () => {
    const documentFetch = defer<DocumentWithPartsResponse>();
    const layoutFetch = defer<PublicLayoutResponse>();
    vi.mocked(api.getPublicDocument).mockReturnValue(documentFetch.promise);
    vi.mocked(api.getPublicLayout).mockReturnValue(layoutFetch.promise);

    renderPublicPage();

    expect(
      screen.getByRole("navigation", { name: "Main navigation" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Public view")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Document" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "Loading document" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Document not available"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("No page image available"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("This published document has no page images yet."),
    ).not.toBeInTheDocument();

    documentFetch.resolve(DOCUMENT);
    layoutFetch.resolve(LAYOUT);

    expect(
      await screen.findByRole("heading", { name: "MS Or. 1445 - Genesis" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("status", { name: "Loading document" }),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("public-page-canvas")).toBeInTheDocument();
  });

  it("syncs canvas selection with the transcript panel", async () => {
    renderPublicPage();
    await screen.findByRole("heading", { name: "MS Or. 1445 - Genesis" });

    fireEvent.click(screen.getByRole("button", { name: "Select line 1" }));

    await waitFor(() => {
      expect(screen.getByText("Line 1")).toBeInTheDocument();
    });
    expect(screen.getByTestId("public-page-canvas")).toHaveTextContent(
      "Selected: 1",
    );
  });

  it("sends the token from its own URL on every public request", async () => {
    renderPublicPage();
    await screen.findByRole("heading", { name: "MS Or. 1445 - Genesis" });

    expect(api.getPublicDocument).toHaveBeenCalledWith(
      "project-1",
      "doc-1",
      "share-token-1",
    );
    expect(api.getPublicLayout).toHaveBeenCalledWith(
      "project-1",
      "doc-1",
      "share-token-1",
    );
  });

  it("downloads the page as a Word file with the share token", async () => {
    const saved: string[] = [];
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        saved.push(this.download);
      });
    URL.createObjectURL = vi.fn(() => "blob:docx");
    URL.revokeObjectURL = vi.fn();

    renderPublicPage();
    await screen.findByRole("heading", { name: "MS Or. 1445 - Genesis" });

    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Word (.docx)" }),
    );

    await waitFor(() => {
      expect(api.getPublicDocx).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        "share-token-1",
      );
    });
    await waitFor(() => {
      expect(saved).toEqual(["MS_Or._1445_-_Genesis_page_1.docx"]);
    });
    click.mockRestore();
  });

  it("shows the not-found state rather than spinning forever with no token", () => {
    window.history.replaceState(
      {},
      "",
      "/public/projects/project-1/documents/doc-1",
    );
    render(<PublicDocumentPage />);

    expect(screen.getByText("Document not available")).toBeInTheDocument();
    expect(
      screen.queryByRole("status", { name: "Loading document" }),
    ).not.toBeInTheDocument();
    expect(api.getPublicDocument).not.toHaveBeenCalled();
  });
});

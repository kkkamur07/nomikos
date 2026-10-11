import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  DOCUMENT,
  flushPageEditorEffects,
  line,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

/**
 * Regions (ADR 0011): a spread split into its folios. The server numbers the
 * Segments region by region; the editor draws the regions, names them, moves
 * Segments between them, and shows where each region starts in the transcript.
 */

const VERSO = {
  id: "block-verso",
  part_id: "part-1",
  order: 0,
  label: "4v",
  box: {
    points: [
      [0, 0],
      [300, 0],
      [300, 400],
      [0, 400],
    ],
  },
  manual_geometry: true,
  created_at: "2026-10-11T10:00:00Z",
};
const RECTO = {
  ...VERSO,
  id: "block-recto",
  order: 1,
  label: null,
  box: {
    points: [
      [320, 0],
      [620, 0],
      [620, 400],
      [320, 400],
    ],
  },
};

function spreadLines() {
  return [
    line({ id: "line-v", order: 0, block_id: VERSO.id }),
    line({
      id: "line-r",
      order: 1,
      block_id: RECTO.id,
      points: [
        [330, 10],
        [600, 10],
        [600, 30],
        [330, 30],
      ],
    }),
    line({
      id: "line-x",
      order: 2,
      block_id: null,
      points: [
        [10, 500],
        [50, 500],
        [50, 520],
        [10, 520],
      ],
    }),
  ];
}

function seedSpread() {
  mockedApi.getDocument.mockResolvedValue(DOCUMENT);
  mockedApi.getPartLayout.mockResolvedValue({
    blocks: [VERSO, RECTO],
    lines: [],
  });
  mockedApi.listPartLines.mockResolvedValue(spreadLines());
}

describe("PageEditorPlaceholderPage regions", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await flushPageEditorEffects();
  });

  it("offers a Region tool in the canvas toolbar, armed by its button or G", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    const island = await screen.findByRole("toolbar", {
      name: /canvas tools/i,
    });
    const region = within(island).getByRole("button", {
      name: /region \(g\)/i,
    });
    expect(region).toHaveAttribute("aria-keyshortcuts", "G");
    expect(region).toHaveAttribute("aria-pressed", "false");

    fireEvent.keyDown(window, { key: "g" });
    expect(region).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(region);
    expect(region).toHaveAttribute("aria-pressed", "false");
  });

  it("draws the regions with their names and heads each one's rows in the transcript", async () => {
    seedSpread();

    renderPageEditor();

    expect(
      await screen.findByRole("button", { name: "Region 4v" }),
    ).toBeInTheDocument();
    // Unlabelled: named by position.
    expect(
      screen.getByRole("button", { name: "Region Region 2" }),
    ).toBeInTheDocument();

    const headings = screen
      .getAllByRole("heading", { level: 3 })
      .filter((heading) => heading.classList.contains("pe-transcript-region"))
      .map((heading) => heading.textContent);
    expect(headings).toEqual(["4v", "Region 2", "Unassigned"]);
  });

  it("names a region from its panel and deletes it", async () => {
    seedSpread();

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: "Region 4v" }));
    const label = await screen.findByRole("textbox", { name: /region label/i });
    expect(label).toHaveValue("4v");
    expect(label).toHaveAttribute("placeholder", "Folio label, e.g. 4v");

    fireEvent.change(label, { target: { value: "5r" } });
    fireEvent.keyDown(label, { key: "Enter" });
    await waitFor(() =>
      expect(mockedApi.patchPartBlock).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        VERSO.id,
        { label: "5r" },
      ),
    );

    fireEvent.click(screen.getByRole("button", { name: /delete region/i }));
    await waitFor(() =>
      expect(mockedApi.deletePartBlock).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        VERSO.id,
      ),
    );
  });

  it("assigns the selected segment to a region by its block id, or to none", async () => {
    seedSpread();

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^segment 3/i }));
    const select = await screen.findByRole("combobox", {
      name: /assign to region/i,
    });
    expect(select).toHaveValue("");
    expect(
      within(select)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["4v", "Region 2", "No region"]);

    fireEvent.change(select, { target: { value: VERSO.id } });
    await waitFor(() =>
      expect(mockedApi.patchPartLine).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        "line-x",
        { block_id: VERSO.id },
      ),
    );
    // The server renumbers region-first, so the editor reads the page back.
    await waitFor(() =>
      expect(mockedApi.listPartLines.mock.calls.length).toBeGreaterThan(1),
    );
  });
});

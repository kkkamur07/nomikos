import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import { toast } from "../components/ui/toast";
import {
  DOCUMENT,
  openMenu,
  renderDocumentPage,
  seedActionMocks,
} from "./DocumentDetailPageActionsSupport";

vi.mock("../components/ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}));

vi.mock("../components/AuthenticatedImage", () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

vi.mock("../components/document/JobsNotice", () => ({
  JobsNotice: () => <div>Jobs panel</div>,
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getDocument: vi.fn(),
      me: vi.fn(),
      getProject: vi.fn(),
      updateDocument: vi.fn(),
      deleteDocument: vi.fn(),
      updatePartReviewStatus: vi.fn(),
      updatePartsPublished: vi.fn(),
      getDocumentWorkflowCounts: vi.fn(),
      listInferenceModels: vi.fn(),
      enqueueDocumentSegment: vi.fn(),
      enqueueDocumentTranscribe: vi.fn(),
      exportDocumentPageXml: vi.fn(),
      exportDocumentTranscriptionPdf: vi.fn(),
      exportDocumentText: vi.fn(),
      rotateDocumentShareToken: vi.fn(),
    },
  };
});

describe("DocumentDetailPage toolbar", () => {
  beforeEach(seedActionMocks);

  it("offers one way to pick files, not the same picker twice", async () => {
    // The failure this catches: the page used to render a large upload panel
    // above the page list as well as this button, and both opened the same
    // hidden multi-file input with the same accept list. Not two affordances,
    // one action drawn twice, which leaves a reader working out which is the
    // real one. Counting the file inputs is the check that bites, because a
    // second panel that looks different still has to have its own.
    const { container } = renderDocumentPage();

    await screen.findByRole("heading", { name: "Chapter 4" });
    expect(screen.getByRole("button", { name: "Upload pages" })).toBeTruthy();
    expect(container.querySelectorAll('input[type="file"]')).toHaveLength(1);
  });

  it("names both ways in only while the document has no pages", async () => {
    vi.mocked(api.getDocument).mockResolvedValue({
      ...DOCUMENT,
      part_count: 0,
      parts: [],
    });
    renderDocumentPage();

    // Nothing on screen says the whole window takes a drop, so the empty state
    // is where that gets said. It stops being worth the room once there are
    // pages and the action row is visible above them.
    expect(
      await screen.findByText(/drop images and PDFs anywhere on this page/i),
    ).toBeTruthy();
  });

  it("puts the page counts in the header line", async () => {
    renderDocumentPage();

    await screen.findByRole("heading", { name: "Chapter 4" });
    expect(screen.getByText(/3 pages · 1 reviewed · updated/)).toBeTruthy();
  });

  it("opens, walks, activates and closes a menu with the keyboard alone", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const trigger = screen.getByRole("button", { name: "Workflow" });
    trigger.focus();
    expect(trigger.getAttribute("aria-haspopup")).toBe("menu");
    expect(trigger.getAttribute("aria-expanded")).toBe("false");

    // Enter opens rather than doing the trigger's other job. An earlier
    // publish control in this repo navigated here instead of toggling.
    fireEvent.keyDown(trigger, { key: "Enter", code: "Enter" });
    const menu = await screen.findByRole("menu", { name: "Document workflow" });
    expect(trigger.getAttribute("aria-expanded")).toBe("true");

    const items = within(menu).getAllByRole("menuitem");
    await waitFor(() => {
      expect(document.activeElement).toBe(items[0]);
    });

    fireEvent.keyDown(items[0], { key: "ArrowDown" });
    expect(document.activeElement).toBe(items[1]);
    fireEvent.keyDown(items[1], { key: "ArrowUp" });
    expect(document.activeElement).toBe(items[0]);
    fireEvent.keyDown(items[0], { key: "End" });
    expect(document.activeElement).toBe(items[items.length - 1]);
    fireEvent.keyDown(items[items.length - 1], { key: "Home" });
    expect(document.activeElement).toBe(items[0]);

    // Escape closes and hands focus back, so the next Tab carries on from the
    // control the person actually pressed rather than from the page top.
    fireEvent.keyDown(items[0], { key: "Escape" });
    await waitFor(() => {
      expect(
        screen.queryByRole("menu", { name: "Document workflow" }),
      ).toBeNull();
    });
    expect(document.activeElement).toBe(trigger);
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
  });

  it("survives the click a real browser fires after Enter on the trigger", async () => {
    // The test above cannot see this. jsdom does not synthesize a click from
    // Enter on a button, so a keyDown-only test passes whether or not the
    // handler calls preventDefault, and the guard it is meant to protect can be
    // deleted without turning anything red. A browser does synthesize it, and
    // the click would run the toggle a second time and shut the menu the key
    // press just opened.
    //
    // So do what the browser does: dispatch the keydown, and fire the click
    // only if nothing cancelled it. `fireEvent` returns false when the handler
    // called preventDefault, which is exactly the browser's own condition for
    // suppressing the synthesized click.
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const trigger = screen.getByRole("button", { name: "Workflow" });
    trigger.focus();

    const notCancelled = fireEvent.keyDown(trigger, {
      key: "Enter",
      code: "Enter",
    });
    if (notCancelled) fireEvent.click(trigger);

    expect(
      await screen.findByRole("menu", { name: "Document workflow" }),
    ).toBeInTheDocument();
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
  });

  it("keeps the menu shut and reports the failure when a batch job is refused", async () => {
    vi.mocked(api.enqueueDocumentSegment).mockRejectedValue(
      new ApiError("Only project members can run jobs", 403),
    );

    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Workflow", "Document workflow");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: /Segment unsegmented pages/,
      }),
    );

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        "Only project members can run jobs",
      );
    });
  });

  it("disables the items a count says have nothing to act on", async () => {
    vi.mocked(api.getDocumentWorkflowCounts).mockResolvedValue({
      total: 3,
      reviewed: 0,
      unsegmented: 0,
      unpaired: 3,
    });

    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const workflow = await openMenu("Workflow", "Document workflow");
    expect(
      within(workflow).getByRole("menuitem", {
        name: /Segment unsegmented pages\s*0/,
      }),
    ).toBeDisabled();
    fireEvent.keyDown(
      within(workflow).getByRole("menuitem", { name: /Re-segment/ }),
      { key: "Escape" },
    );

    const download = await openMenu("Download", "Download this document");
    // Nothing is reviewed, so the reviewed-only exports would be empty files.
    expect(
      within(download).getByRole("menuitem", {
        name: "PAGE XML + images, reviewed pages only",
      }),
    ).toBeDisabled();
  });
});

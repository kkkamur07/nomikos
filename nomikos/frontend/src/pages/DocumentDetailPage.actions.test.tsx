import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { testRouter } from "../../vitest.setup";

import {
  api,
  type DocumentResponse,
  type DocumentWithPartsResponse,
} from "../api/client";
import { ApiError } from "../api/errors";
import * as session from "../auth/session";
import { toast } from "../components/ui/toast";
import { DocumentDetailPage } from "./DocumentDetailPage";

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
      exportDocumentDocx: vi.fn(),
      rotateDocumentShareToken: vi.fn(),
    },
  };
});

/** Three pages, one of them checked by a human: the numbers the menus quote. */
const DOCUMENT: DocumentWithPartsResponse = {
  id: "doc-1",
  project_id: "project-1",
  name: "Chapter 4",
  workflow: "draft",
  created_at: "2026-06-16T10:00:00Z",
  updated_at: "2026-06-16T10:00:00Z",
  part_count: 3,
  parts: [0, 1, 2].map((order) => ({
    id: `part-${order + 1}`,
    document_id: "doc-1",
    order,
    image_url: `/media/parts/part-${order + 1}`,
    width: 800,
    height: 1000,
    reviewed: order === 0,
    published: true,
    created_at: "2026-06-16T10:00:00Z",
  })),
};

const COUNTS = { total: 3, reviewed: 1, unsegmented: 2, unpaired: 3 };

/** The anchor `saveBlobAsFile` builds, captured instead of navigated to. */
let lastDownload: { href: string; download: string } | null = null;

function renderDocumentPage() {
  window.history.replaceState({}, "", "/projects/project-1/documents/doc-1");
  return render(<DocumentDetailPage />);
}

async function openMenu(triggerName: string, menuName: string) {
  fireEvent.click(screen.getByRole("button", { name: triggerName }));
  return await screen.findByRole("menu", { name: menuName });
}

function seedProject(ownerId: string) {
  vi.mocked(api.getProject).mockResolvedValue({
    id: "project-1",
    name: "East Syriac",
    slug: "east-syriac",
    owner_id: ownerId,
    guidelines: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    document_count: 1,
  });
}

/** One job as a batch route announces it: id plus the host fixed for it. */
function queuedJob(jobId: string) {
  return {
    job_id: jobId,
    execution_target: "cloud" as const,
    preferred_execution_target: "cloud" as const,
    execution_target_substituted: false,
  };
}

describe("DocumentDetailPage action toolbar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    lastDownload = null;
    vi.spyOn(session, "hasAccessToken").mockReturnValue(true);
    vi.spyOn(session, "navigateToLogin").mockImplementation(() => {});
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      lastDownload = { href: this.href, download: this.download };
    });
    Object.assign(URL, {
      createObjectURL: vi.fn(() => "blob:document-export"),
      revokeObjectURL: vi.fn(),
    });

    vi.mocked(api.me).mockResolvedValue({
      id: "user-1",
      email: "dev@example.com",
      username: "dev",
      created_at: "2026-01-01T00:00:00Z",
    });
    seedProject("user-1");
    vi.mocked(api.getDocument).mockResolvedValue(DOCUMENT);
    vi.mocked(api.getDocumentWorkflowCounts).mockResolvedValue(COUNTS);
    vi.mocked(api.listInferenceModels).mockResolvedValue([]);
    vi.mocked(api.enqueueDocumentSegment).mockResolvedValue({
      jobs: [queuedJob("job-1")],
      queued: 3,
      skipped: 0,
    });
    vi.mocked(api.enqueueDocumentTranscribe).mockResolvedValue({
      jobs: [queuedJob("job-2")],
      queued: 3,
      skipped: 0,
    });
    vi.mocked(api.exportDocumentPageXml).mockResolvedValue(
      new Blob(["PK"], { type: "application/zip" }),
    );
    vi.mocked(api.exportDocumentTranscriptionPdf).mockResolvedValue(
      new Blob(["%PDF"], { type: "application/pdf" }),
    );
    vi.mocked(api.exportDocumentText).mockResolvedValue(
      new Blob(["text"], { type: "text/plain" }),
    );
    vi.mocked(api.exportDocumentDocx).mockResolvedValue(
      new Blob(["PK"], {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      }),
    );
  });

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

  it("opens the Workflow menu with both segment scopes, the model captions and the warning", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Workflow", "Document workflow");

    // The counts come from `workflow-counts`, not from the parts list: the
    // parts list has no idea which pages have lines or a pairing.
    expect(api.getDocumentWorkflowCounts).toHaveBeenCalledWith(
      "project-1",
      "doc-1",
    );
    expect(
      within(menu).getByRole("menuitem", {
        name: /Segment unsegmented pages\s*2/,
      }),
    ).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", { name: /Re-segment every page\s*3/ }),
    ).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", {
        name: /Transcribe unpaired pages\s*3/,
      }),
    ).toBeTruthy();
    expect(within(menu).getByText("Segment")).toBeTruthy();
    expect(within(menu).getByText("Transcribe")).toBeTruthy();
    // The segment engine caption became a picker: Default plus the catalog.
    expect(
      within(menu).getByRole("combobox", { name: "Segmentation model" }),
    ).toBeTruthy();
    expect(within(menu).getByRole("option", { name: "Default" })).toBeTruthy();
    expect(within(menu).getByText(/Model/)).toHaveTextContent(
      "Model blla-greek-v2",
    );
    expect(
      within(menu).getByText(/Re-segmenting discards unapproved machine text/),
    ).toBeTruthy();
  });

  it("queues the safe segment scope straight from the menu", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Workflow", "Document workflow");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: /Segment unsegmented pages/,
      }),
    );

    await waitFor(() => {
      expect(api.enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        { scope: "unsegmented", model_id: null },
      );
    });
  });

  it("does nothing when Re-segment every page is chosen, until it is confirmed", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Workflow", "Document workflow");
    fireEvent.click(
      within(menu).getByRole("menuitem", { name: /Re-segment every page/ }),
    );

    // The whole point of the confirm: the click that reaches the destructive
    // item must not be the click that discards the transcriptions.
    expect(api.enqueueDocumentSegment).not.toHaveBeenCalled();
    expect(
      screen.getByText(/the model's unapproved text on it is discarded/),
    ).toBeTruthy();
    expect(screen.getByText("Re-segment every page (3 pages)?")).toBeTruthy();

    // Backing out leaves the document exactly as it was.
    fireEvent.click(screen.getByRole("menuitem", { name: "Cancel" }));
    expect(api.enqueueDocumentSegment).not.toHaveBeenCalled();

    fireEvent.click(
      screen.getByRole("menuitem", { name: /Re-segment every page/ }),
    );
    fireEvent.click(
      screen.getByRole("menuitem", { name: /Yes, re-segment 3 pages/ }),
    );

    await waitFor(() => {
      expect(api.enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        { scope: "all", model_id: null },
      );
    });
  });

  it("lists both download scopes with their counts", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Download", "Download this document");

    expect(within(menu).getByText("Whole chapter")).toBeTruthy();
    expect(within(menu).getByText("Reviewed only")).toBeTruthy();
    expect(within(menu).getByText("zip, 3")).toBeTruthy();
    expect(within(menu).getByText("zip, 1")).toBeTruthy();
    expect(within(menu).getByText("3 pages")).toBeTruthy();
    expect(within(menu).getByText("1 page")).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", { name: /Plain text/ }),
    ).toBeTruthy();
    // 3 pages, 1 of them checked, so 2 are about to be left out.
    expect(
      within(menu).getByText("Skips the 2 pages nobody has checked."),
    ).toBeTruthy();
  });

  it("downloads the reviewed-only archive through the authenticated client", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Download", "Download this document");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: "PAGE XML + images, reviewed pages only",
      }),
    );

    // A plain link cannot reach these routes: a browser-initiated request
    // carries no Authorization header. The bytes come back through the client
    // and only then become a file.
    await waitFor(() => {
      expect(api.exportDocumentPageXml).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        true,
      );
    });
    await waitFor(() => {
      expect(lastDownload?.download).toBe("Chapter_4_reviewed.zip");
    });
  });

  it("names the whole-chapter text export without the reviewed marker", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Download", "Download this document");
    fireEvent.click(within(menu).getByRole("menuitem", { name: /Plain text/ }));

    await waitFor(() => {
      expect(api.exportDocumentText).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        false,
      );
    });
    await waitFor(() => {
      expect(lastDownload?.download).toBe("Chapter_4.txt");
    });
  });

  it("downloads the Word export for either scope through the authenticated client", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    let menu = await openMenu("Download", "Download this document");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: "Word (.docx), whole chapter",
      }),
    );
    await waitFor(() => {
      expect(api.exportDocumentDocx).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        false,
      );
    });
    await waitFor(() => {
      expect(lastDownload?.download).toBe("Chapter_4.docx");
    });

    menu = await openMenu("Download", "Download this document");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: "Word (.docx), reviewed pages only",
      }),
    );
    await waitFor(() => {
      expect(api.exportDocumentDocx).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        true,
      );
    });
    await waitFor(() => {
      expect(lastDownload?.download).toBe("Chapter_4_reviewed.docx");
    });
  });

  it("names the page count in the publish label and confirms with the review split", async () => {
    vi.mocked(api.updateDocument).mockResolvedValue({
      ...DOCUMENT,
      workflow: "published",
    } as DocumentResponse);

    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Publish", "Publish this document");

    // "Publish live page" was a document-level act wearing a page-level name.
    // The count in the label is what stops it being read as a page control.
    const publishItem = within(menu).getByRole("menuitem", {
      name: /Publish document \(3 pages\)/,
    });
    expect(
      within(menu).getByText(
        "Sets 3 of 3 pages live. Readers reach them through one secret link.",
      ),
    ).toBeTruthy();

    fireEvent.click(publishItem);
    expect(api.updateDocument).not.toHaveBeenCalled();
    expect(
      screen.getByText("Publish 3 pages, 1 reviewed, 2 not."),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("menuitem", { name: "Publish document" }));
    await waitFor(() => {
      expect(api.updateDocument).toHaveBeenCalledWith("project-1", "doc-1", {
        workflow: "published",
      });
    });
  });

  it("offers the secret link only once the document is published", async () => {
    vi.mocked(api.getDocument).mockResolvedValue({
      ...DOCUMENT,
      workflow: "published",
      public_share_token: "secret-token",
    });

    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    expect(
      (screen.getByLabelText("Public document link") as HTMLInputElement).value,
    ).toContain("/public/projects/project-1/documents/doc-1?t=secret-token");

    const menu = await openMenu("Publish", "Publish this document");
    expect(
      within(menu).getByRole("menuitem", { name: "Copy secret link" }),
    ).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", { name: "Rotate link" }),
    ).toBeTruthy();
    expect(
      within(menu).getByText(/Rotating breaks every link already sent/),
    ).toBeTruthy();
    expect(
      within(menu).queryByRole("menuitem", { name: /Publish document/ }),
    ).toBeNull();
  });

  it("offers no publish menu to a member who does not own the project", async () => {
    seedProject("someone-else");

    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    // The backend refuses a non-owner outright, so the control would only ever
    // earn a red toast. Workflow and Download are member-level and stay.
    expect(screen.queryByRole("button", { name: "Publish" })).toBeNull();
    expect(screen.getByRole("button", { name: "Workflow" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Download" })).toBeTruthy();
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

  it("opens the settings panel from the overflow menu and keeps it open", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu(
      "More document actions",
      "More document actions",
    );
    fireEvent.click(
      within(menu).getByRole("menuitem", { name: /Document settings/ }),
    );

    // The panel closes itself on a click outside it, and the menu that opens
    // it sits outside it. The listener must not see the click that opened it.
    expect(
      await screen.findByRole("dialog", {
        name: "Document settings",
      }),
    ).toBeTruthy();
  });

  it("leaves publishing to the Publish menu alone, with no second control in the settings panel", async () => {
    // The failure this catches: the settings panel used to carry its own
    // "Publish document" button, next to the public URL. It published with no
    // confirm step and no page count, so a document could go live by a route
    // that never said how many pages it was exposing or how many of them
    // nobody had checked. Deleting the panel's copy is only worth anything if
    // nothing puts it back, and the panel is a dialog nothing else asserts on.
    vi.mocked(api.getDocument).mockResolvedValue({
      ...DOCUMENT,
      workflow: "published",
      public_share_token: "tok-visible",
    });
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu(
      "More document actions",
      "More document actions",
    );
    fireEvent.click(
      within(menu).getByRole("menuitem", { name: /Document settings/ }),
    );
    const panel = await screen.findByRole("dialog", {
      name: "Document settings",
    });

    // Renaming is what the panel is for, so this asserts the panel really
    // rendered rather than passing because the query found an empty node.
    expect(
      within(panel).getByRole("button", { name: "Save name" }),
    ).toBeTruthy();
    expect(
      within(panel).queryByRole("button", { name: /publish/i }),
    ).toBeNull();
    expect(within(panel).queryByLabelText("Public document URL")).toBeNull();
  });

  it("deletes the document only after the confirm names what goes with it", async () => {
    vi.mocked(api.deleteDocument).mockResolvedValue(undefined);

    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu(
      "More document actions",
      "More document actions",
    );
    fireEvent.click(
      within(menu).getByRole("menuitem", { name: /Delete document/ }),
    );

    expect(api.deleteDocument).not.toHaveBeenCalled();
    expect(
      screen.getByText('Delete "Chapter 4" and its 3 pages?'),
    ).toBeTruthy();

    fireEvent.click(
      screen.getByRole("menuitem", { name: "Yes, delete the document" }),
    );
    await waitFor(() => {
      expect(api.deleteDocument).toHaveBeenCalledWith("project-1", "doc-1");
    });
    await waitFor(() => {
      expect(testRouter().push).toHaveBeenCalledWith("/projects/project-1");
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

import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { testRouter } from "../../vitest.setup";

import { api, type DocumentResponse } from "../api/client";
import {
  DOCUMENT,
  openMenu,
  renderDocumentPage,
  seedActionMocks,
  seedProject,
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

describe("DocumentDetailPage publish and settings", () => {
  beforeEach(seedActionMocks);

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
});

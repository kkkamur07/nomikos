import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DocumentDownloadMenu } from "./DocumentDownloadMenu";

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}));

function openDownloadMenu() {
  render(
    <DocumentDownloadMenu
      projectId="project-1"
      documentId="doc-1"
      documentName="Chapter 4"
      counts={{ total: 12, reviewed: 4, unpaired: 0, unsegmented: 0 }}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /download/i }));
  return screen.getByRole("menu", { name: "Download this document" });
}

describe("DocumentDownloadMenu Word item", () => {
  it("says how many of the chapter's pages are reviewed", () => {
    const menu = openDownloadMenu();
    const item = within(menu).getByRole("menuitem", {
      name: "Word (.docx), whole chapter",
    });
    expect(item).toHaveTextContent("12 pages, 4 reviewed");
  });

  it("leaves the reviewed-only item without a count", () => {
    const menu = openDownloadMenu();
    const item = within(menu).getByRole("menuitem", {
      name: "Word (.docx), reviewed pages only",
    });
    expect(item).toHaveTextContent(/^Word \(\.docx\)$/);
  });
});

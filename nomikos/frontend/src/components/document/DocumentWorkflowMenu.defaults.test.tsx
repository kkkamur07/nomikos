import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), info: vi.fn(), error: vi.fn() },
}));

const CATALOG = [
  {
    id: "seg-a",
    name: "kraken",
    provider: "kraken",
    task: "segment",
    artifact_ref: "registry://blla-segment?tag=stable",
    default_params: {},
    created_at: "2026-09-21T00:00:00Z",
  },
  {
    id: "htr-1",
    name: "htr",
    provider: "calamari",
    task: "transcribe",
    artifact_ref: "registry://htr-1?tag=stable",
    default_params: {},
    created_at: "2026-09-21T00:00:00Z",
  },
];

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Queuing a job used to store the chosen model as the project default when
 * the project had none, from inside the API client. Setting the default is
 * now an explicit act: picking a model in the menu writes the binding, and
 * running a job writes nothing at all. With the real API client underneath,
 * the only request a run may make is the job itself.
 */
describe("DocumentWorkflowMenu and the project defaults", () => {
  let calls: string[];

  beforeEach(() => {
    calls = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        calls.push(`${method} ${url.replace(/^https?:\/\/[^/]+/, "")}`);
        if (method === "GET" && url.includes("/inference/models")) {
          return jsonResponse(CATALOG);
        }
        if (method === "GET" && url.includes("/model-bindings")) {
          return jsonResponse([]);
        }
        if (method === "POST" && url.includes("/jobs/transcribe")) {
          return jsonResponse({ queued: 1, skipped: 0, jobs: [] });
        }
        throw new Error(`unexpected fetch ${method} ${url}`);
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("writes no project binding when a job is queued", async () => {
    render(
      <DocumentWorkflowMenu
        projectId="proj-r2"
        documentId="document-1"
        counts={{ total: 3, reviewed: 0, unsegmented: 1, unpaired: 2 }}
        onJobsQueued={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));

    const select = await screen.findByRole("combobox", {
      name: "HTR transcription model",
    });
    await waitFor(() => expect(select).toHaveValue("htr-1"));

    fireEvent.click(
      screen.getByRole("menuitem", { name: /transcribe unpaired pages/i }),
    );

    // A successful run closes the menu.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /workflow/i })).toHaveAttribute(
        "aria-expanded",
        "false",
      ),
    );

    expect(
      calls.filter(
        (call) => call.includes("/model-bindings") && !call.startsWith("GET "),
      ),
    ).toEqual([]);
    expect(screen.queryByText(/project default/i)).toBeNull();
  });
});

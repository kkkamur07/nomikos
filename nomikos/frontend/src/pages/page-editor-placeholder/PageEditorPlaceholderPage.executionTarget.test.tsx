import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { MessageInstance } from "antd/es/message/interface";

import { registerToastApi } from "../../components/ui/toast";
import {
  DOCUMENT,
  flushPageEditorEffects,
  line,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

/**
 * Records what `toast` would have put on screen.
 *
 * The themed `message` instance is normally handed to `toast` at mount by
 * `ToastBridge`, which these tests do not render - so without a stand-in every
 * toast is silently dropped and "this did not become a toast" would pass for
 * the wrong reason. The returned array is the assertion's evidence.
 */
function recordedToasts(): string[] {
  const shown: string[] = [];
  const record = (content: unknown) => {
    shown.push(String(content));
    return (() => {}) as unknown as ReturnType<MessageInstance["success"]>;
  };
  registerToastApi({
    success: record,
    error: record,
    info: record,
    warning: record,
    loading: record,
    open: record,
    destroy: () => {},
  } as unknown as MessageInstance);
  return shown;
}

afterEach(() => registerToastApi(null));

const TRANSCRIBE_MODEL = {
  id: "model-1",
  name: "kraken-transcribe-default",
  provider: "kraken",
  task: "transcribe",
  artifact_ref: "registry://greek-calamari-v1",
  default_params: {},
  created_at: "2026-06-16T10:00:00Z",
};

/** A job as `GET /jobs/{id}` returns it, including its **execution target**. */
function jobResponse(overrides: Record<string, unknown> = {}) {
  return {
    id: "job-ocr-1",
    type: "transcribe",
    status: "done",
    payload: {},
    result: { transcription_id: "model-2", lines: [] },
    error: null,
    document_id: "doc-1",
    document_part_id: "part-1",
    created_at: "2026-06-16T10:00:00Z",
    updated_at: "2026-06-16T10:00:00Z",
    started_at: "2026-06-16T10:00:00Z",
    completed_at: "2026-06-16T10:00:00Z",
    execution_target: "cloud",
    preferred_execution_target: "cloud",
    execution_target_substituted: false,
    execution: "cloud",
    ...overrides,
  };
}

function loadedPageWithOneSegment() {
  mockedApi.getDocument.mockResolvedValue(DOCUMENT);
  mockedApi.listInferenceModels.mockResolvedValue([TRANSCRIBE_MODEL]);
  mockedApi.listPartLines.mockResolvedValue([line()]);
}

async function runOcrOnTheSelectedSegment() {
  fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
  fireEvent.click(
    screen.getByRole("button", { name: /re-run ocr on segment 1/i }),
  );
}

async function openBackgroundJobs() {
  fireEvent.click(
    await screen.findByRole("button", { name: /background job/i }),
  );
}

describe("the account-level host preference", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
    // These tests exercise the segment strip, which only opens when the transcript column is off.
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ sideBySide: false }),
    );
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("persists and round-trips through the account, not the browser", async () => {
    renderPageEditor();

    fireEvent.click(
      await screen.findByRole("button", { name: /editor settings/i }),
    );
    const setting = screen.getByRole("checkbox", {
      name: /use my computer when it is available/i,
    });
    expect(setting).not.toBeChecked();

    fireEvent.click(setting);

    await waitFor(() => {
      expect(mockedApi.setExecutionPreference).toHaveBeenCalledWith(true);
    });
    await waitFor(() => expect(setting).toBeChecked());

    // A fresh mount reads the account again. Were the setting held in this
    // browser rather than on the account, this is where it would be lost.
    await flushPageEditorEffects();
    cleanup();
    renderPageEditor();
    fireEvent.click(
      await screen.findByRole("button", { name: /editor settings/i }),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("checkbox", {
          name: /use my computer when it is available/i,
        }),
      ).toBeChecked(),
    );
  });
});

/** The 202 from an enqueue route: the id and the host fixed at submission. */
function enqueued(overrides: Record<string, unknown> = {}) {
  return {
    job_id: "job-ocr-1",
    execution_target: "cloud",
    preferred_execution_target: "cloud",
    execution_target_substituted: false,
    ...overrides,
  };
}

describe("the announcement on a job", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
    // These tests exercise the segment strip, which only opens when the transcript column is off.
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ sideBySide: false }),
    );
    loadedPageWithOneSegment();
    mockedApi.enqueueTranscribePart.mockResolvedValue(enqueued());
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("renders from the enqueue response, before any job update arrives", async () => {
    // The platform has answered the enqueue and nothing else: no SSE event, no
    // poll result. Between the click and the first status update the job must
    // already say where it went, and with the substitution spelled out, since
    // this is exactly the moment a researcher is looking at the screen.
    mockedApi.enqueueTranscribePart.mockResolvedValue(
      enqueued({
        execution_target: "cloud",
        preferred_execution_target: "local",
        execution_target_substituted: true,
      }),
    );
    mockedApi.getJob.mockReturnValue(new Promise(() => {}));

    renderPageEditor();
    await runOcrOnTheSelectedSegment();
    await openBackgroundJobs();

    const announcement = await screen.findByText(
      /running in the cloud\. you asked for your computer, which had no capacity/i,
    );
    expect(announcement.getAttribute("data-execution-target")).toBe("cloud");
    expect(announcement.getAttribute("data-execution-substituted")).toBe(
      "true",
    );
  });

  it("states a substituted host on the job rather than in a toast", async () => {
    const toasts = recordedToasts();
    mockedApi.getJob.mockResolvedValue(
      jobResponse({
        execution_target: "cloud",
        preferred_execution_target: "local",
        execution_target_substituted: true,
      }),
    );

    renderPageEditor();
    await runOcrOnTheSelectedSegment();
    await openBackgroundJobs();

    const announcement = await screen.findByText(
      /you asked for your computer, which had no capacity/i,
    );
    // It stays on the job. A toast would be gone before a researcher who
    // looked away could read where their work went.
    expect(announcement).toBeTruthy();
    expect(toasts.filter((shown) => /had no capacity/i.test(shown))).toEqual(
      [],
    );
  });
});

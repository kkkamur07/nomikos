import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import { toast } from "../../components/ui/toast";
import {
  DOCUMENT,
  flushPageEditorEffects,
  line,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

describe("PageEditorPlaceholderPage transcription", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
    // These tests exercise the segment strip, which only opens when the transcript column is off.
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ sideBySide: false }),
    );
  });

  afterEach(async () => {
    await flushPageEditorEffects();
  });

  it("pairs a selected Segment to imported text lines and updates Pairing progress", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      line(),
      line({
        id: "line-2",
        order: 1,
        points: [
          [80, 20],
          [120, 20],
          [120, 50],
          [80, 50],
        ],
      }),
    ]);
    mockedApi.getPagePairing.mockResolvedValue({
      text_lines: [
        { order: 0, text: "alpha", paired_line_id: null },
        { order: 1, text: "beta", paired_line_id: null },
      ],
      pairing_progress: { paired_lines: 0, total_lines: 2, percent: 0 },
    });
    mockedApi.pairTextLine.mockResolvedValue({
      text_lines: [
        { order: 0, text: "alpha", paired_line_id: null },
        { order: 1, text: "beta", paired_line_id: "line-1" },
      ],
      pairing_progress: { paired_lines: 1, total_lines: 2, percent: 50 },
    });

    renderPageEditor();

    expect(
      await screen.findByText("Pairing progress: 0/2 Lines paired"),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /^Segment 1/ }));
    fireEvent.click(screen.getByRole("button", { name: /pair text line 2/i }));

    await waitFor(() => {
      expect(mockedApi.pairTextLine).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        { line_id: "line-1", text_line_order: 1 },
      );
    });
    expect(
      await screen.findByText("Pairing progress: 1/2 Lines paired"),
    ).toBeTruthy();
  });

  it("saves typed approved text for the selected Segment and refreshes Pairing progress", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([line()]);
    mockedApi.getPagePairing
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 0, total_lines: 2, percent: 0 },
      })
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 1, total_lines: 2, percent: 50 },
      });
    mockedApi.updateGroundTruthLineText.mockResolvedValue({
      id: "line-tx-1",
      transcription_id: "ground-truth-1",
      transcription_kind: "ground_truth",
      text: "typed approved text",
      confidence: null,
      character_confidences: null,
    });

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
    fireEvent.change(
      screen.getByLabelText(/approved text for selected segment/i),
      {
        target: { value: "typed approved text" },
      },
    );
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-1",
        { text: "typed approved text" },
      );
    });
    expect(
      await screen.findByText("Pairing progress: 1/2 Lines paired"),
    ).toBeTruthy();
    // The typing panel closes once the text is saved; the segment stays selected.
    await waitFor(() => {
      expect(
        screen.queryByLabelText(/approved text for selected segment/i),
      ).toBeNull();
    });
    expect(screen.getByRole("button", { name: /^Segment 1/ })).toHaveAttribute(
      "aria-current",
      "true",
    );
  });

  it("saves Ground truth text for the selected Segment", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      line({
        line_transcriptions: [
          {
            id: "line-tx-1",
            transcription_id: "ground-truth-1",
            transcription_kind: "ground_truth",
            text: "old approved text",
            confidence: null,
            character_confidences: null,
          },
          {
            id: "line-tx-2",
            transcription_id: "model-1",
            transcription_kind: "model",
            text: "model suggestion",
            confidence: 0.91,
            character_confidences: null,
          },
        ],
      }),
    ]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: "ground-truth-1",
        document_id: "doc-1",
        name: "Ground truth",
        kind: "ground_truth",
        created_by_job_id: null,
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "model-1",
        document_id: "doc-1",
        name: "Kraken run",
        kind: "model",
        created_by_job_id: "job-1",
        created_at: "2026-06-16T10:01:00Z",
      },
    ]);
    mockedApi.getPagePairing
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 1, total_lines: 1, percent: 100 },
      })
      .mockResolvedValueOnce({
        text_lines: [],
        pairing_progress: { paired_lines: 1, total_lines: 1, percent: 100 },
      });

    renderPageEditor();

    expect(await screen.findByText("ANNOTE PAGE WORKSPACE")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /^Segment 1/ }));

    const textArea = screen.getByLabelText(
      /ground truth text for selected segment/i,
    );
    expect(textArea).toHaveValue("old approved text");
    fireEvent.change(textArea, { target: { value: "corrected ground truth" } });
    const toastSuccess = vi.spyOn(toast, "success");
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-1",
        { text: "corrected ground truth" },
      );
    });
    await waitFor(() => {
      expect(toastSuccess).toHaveBeenCalledWith("Ground truth text saved");
    });
  });

  it("re-runs OCR on the selected segment from the pairing strip", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listInferenceModels.mockResolvedValue([
      {
        id: "model-1",
        name: "kraken-transcribe-default",
        provider: "kraken",
        task: "transcribe",
        artifact_ref: "registry://greek-calamari-v1",
        default_params: {},
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: "ground-truth-1",
        document_id: "doc-1",
        name: "Ground truth",
        kind: "ground_truth",
        created_by_job_id: null,
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "model-1",
        document_id: "doc-1",
        name: "Model layer",
        kind: "model",
        created_by_job_id: "job-old",
        created_at: "2026-06-16T10:00:00Z",
      },
    ]);
    const groundTruthOnLine = {
      id: "line-tx-gt-1",
      transcription_id: "ground-truth-1",
      transcription_kind: "ground_truth",
      text: "hand-checked text",
      confidence: null,
      character_confidences: null,
    };
    const oldModelOutput = {
      id: "line-tx-model-1",
      transcription_id: "model-1",
      transcription_kind: "model",
      text: "old ocr",
      confidence: 0.8,
      character_confidences: null,
    };
    const freshModelOutput = {
      id: "line-tx-model-2",
      transcription_id: "model-2",
      transcription_kind: "model",
      text: "fresh ocr",
      confidence: 0.92,
      character_confidences: null,
    };
    // Only the reload that follows the run can see the new layer, and the job
    // has to have been polled for that layer to exist at all.
    mockedApi.listPartLines.mockImplementation(async () => [
      line({
        line_transcriptions:
          mockedApi.getJob.mock.calls.length > 0
            ? [groundTruthOnLine, oldModelOutput, freshModelOutput]
            : [groundTruthOnLine, oldModelOutput],
      }),
    ]);
    mockedApi.enqueueTranscribePart.mockResolvedValue({
      job_id: "job-ocr-1",
      execution_target: "cloud",
      preferred_execution_target: "cloud",
      execution_target_substituted: false,
    });
    mockedApi.getJob.mockResolvedValue({
      id: "job-ocr-1",
      type: "transcribe",
      status: "done",
      payload: {},
      result: {
        transcription_id: "model-2",
        lines: [{ line_id: "line-1", text: "fresh ocr", confidence: 0.92 }],
      },
      error: null,
      document_id: "doc-1",
      document_part_id: "part-1",
      created_at: "2026-06-16T10:00:00Z",
      updated_at: "2026-06-16T10:00:00Z",
      started_at: "2026-06-16T10:00:00Z",
      completed_at: "2026-06-16T10:00:00Z",
    });

    renderPageEditor();
    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
    fireEvent.click(
      screen.getByRole("button", { name: /re-run ocr on segment 1/i }),
    );

    await waitFor(() => {
      expect(mockedApi.enqueueTranscribePart).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        { model_id: "model-1", line_ids: ["line-1"] },
      );
    });
    const jobsButton = await screen.findByRole("button", {
      name: /1 background job finished/i,
    });
    expect(jobsButton).toHaveTextContent("1 job finished");

    await waitFor(() => {
      expect(mockedApi.getJob).toHaveBeenCalledWith("job-ocr-1");
    });

    // The run produced a suggestion, so the fresh output shows as model output
    // while the researcher's Ground truth stays exactly as they left it.
    await waitFor(() => {
      expect(
        screen.getByLabelText(/ocr model output for segment 1/i),
      ).toHaveTextContent("fresh ocr");
    });
    expect(
      screen.getByLabelText(/ground truth text for selected segment/i),
    ).toHaveValue("hand-checked text");
  });

  it("offers the model output as a suggestion to accept while Ground truth is open", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listTranscriptions.mockResolvedValue([
      {
        id: "ground-truth-1",
        document_id: "doc-1",
        name: "Ground truth",
        kind: "ground_truth",
        created_by_job_id: null,
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "model-1",
        document_id: "doc-1",
        name: "Model layer",
        kind: "model",
        created_by_job_id: "job-old",
        created_at: "2026-06-16T10:01:00Z",
      },
    ]);
    mockedApi.listPartLines.mockResolvedValue([
      line({
        line_transcriptions: [
          {
            id: "line-tx-model-1",
            transcription_id: "model-1",
            transcription_kind: "model",
            text: "model suggestion",
            confidence: 0.88,
            character_confidences: null,
          },
        ],
      }),
    ]);

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
    // Ground truth is the open layer and the segment has none yet, so the
    // model output is there to be accepted rather than merely read.
    fireEvent.click(await screen.findByRole("button", { name: /^accept$/i }));

    await waitFor(() => {
      expect(mockedApi.copyToGroundTruth).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "model-1",
        { line_ids: ["line-1"] },
      );
    });
  });

  it("saves what the researcher typed instead of accepting the suggestion", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      line({
        line_transcriptions: [
          {
            id: "line-tx-model-1",
            transcription_id: "model-1",
            transcription_kind: "model",
            text: "model suggestion",
            confidence: 0.88,
            character_confidences: null,
          },
        ],
      }),
    ]);

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
    fireEvent.change(
      screen.getByLabelText(/approved text for selected segment/i),
      { target: { value: "my own reading" } },
    );
    // Accepting here would throw the typing away, so the button saves it.
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-1",
        { text: "my own reading" },
      );
    });
    expect(mockedApi.copyToGroundTruth).not.toHaveBeenCalled();
  });

  it("surfaces Ground truth save API errors and keeps the typed text visible", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      line({
        line_transcriptions: [
          {
            id: "line-tx-1",
            transcription_id: "ground-truth-1",
            transcription_kind: "ground_truth",
            text: "old approved text",
            confidence: null,
            character_confidences: null,
          },
        ],
      }),
    ]);
    mockedApi.updateGroundTruthLineText.mockRejectedValue(
      new ApiError("Only Ground truth transcriptions can be edited.", 400),
    );

    renderPageEditor();

    expect(await screen.findByText("ANNOTE PAGE WORKSPACE")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /^Segment 1/ }));
    const textArea = screen.getByLabelText(
      /ground truth text for selected segment/i,
    );
    fireEvent.change(textArea, { target: { value: "typed but rejected" } });
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    expect(
      await screen.findByText(
        "Only Ground truth transcriptions can be edited.",
      ),
    ).toBeTruthy();
    expect(
      screen.getByLabelText(/ground truth text for selected segment/i),
    ).toHaveValue("typed but rejected");
  });
});

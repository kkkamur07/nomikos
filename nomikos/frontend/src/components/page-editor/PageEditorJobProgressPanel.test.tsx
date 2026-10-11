import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { TrackedBackgroundJob } from "../../context/BackgroundJobsContext";
import { PageEditorJobProgressPanel } from "./PageEditorJobProgressPanel";

function job(
  overrides: Partial<TrackedBackgroundJob> = {},
): TrackedBackgroundJob {
  return {
    id: "job-1",
    label: "Segment page",
    kind: "segmentation",
    status: "running",
    error: null,
    progressLabel: "Starting…",
    finishedAt: null,
    execution: {
      execution_target: "local",
      preferred_execution_target: "local",
      execution_target_substituted: false,
      status: "running",
    },
    ...overrides,
  };
}

describe("PageEditorJobProgressPanel", () => {
  it("renders nothing when there are no jobs", () => {
    const { container } = render(
      <PageEditorJobProgressPanel
        jobs={[]}
        activeCount={0}
        expanded={false}
        onExpandedChange={() => {}}
        onDismissCompleted={() => {}}
        onCancelJob={() => {}}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("collapses to a pill that expands on click", () => {
    const onExpandedChange = vi.fn();
    render(
      <PageEditorJobProgressPanel
        jobs={[job()]}
        activeCount={1}
        expanded={false}
        onExpandedChange={onExpandedChange}
        onDismissCompleted={() => {}}
        onCancelJob={() => {}}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "1 background job running" }),
    );
    expect(onExpandedChange).toHaveBeenCalledWith(true);
  });

  it("lets a finished job be cleared and a running job be cancelled", () => {
    const onDismissCompleted = vi.fn();
    const onCancelJob = vi.fn();
    render(
      <PageEditorJobProgressPanel
        jobs={[
          job({ id: "done-1", status: "done", progressLabel: "Complete" }),
          job({ id: "run-1", label: "OCR segment" }),
        ]}
        activeCount={1}
        expanded
        onExpandedChange={() => {}}
        onDismissCompleted={onDismissCompleted}
        onCancelJob={onCancelJob}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(onDismissCompleted).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancelJob).toHaveBeenCalledWith("run-1");
  });
});

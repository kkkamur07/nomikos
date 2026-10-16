import type { TrackedBackgroundJob } from "../../context/BackgroundJobsContext";
import { JobExecutionAnnouncement } from "../JobExecutionAnnouncement";
import { isTerminalJobStatus, pageEditorJobKindLabel } from "./jobProgress";

type PageEditorJobProgressPanelProps = {
  jobs: TrackedBackgroundJob[];
  activeCount: number;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  onDismissCompleted: () => void;
  onCancelJob: (jobId: string) => void;
};

function statusClass(status: TrackedBackgroundJob["status"]): string {
  if (status === "done") return "pe-job-item--done";
  if (status === "failed") return "pe-job-item--failed";
  if (status === "cancelled") return "pe-job-item--cancelled";
  return "pe-job-item--active";
}

function counted(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function CollapsedJobPanel({
  activeCount,
  completedCount,
  onExpand,
}: {
  activeCount: number;
  completedCount: number;
  onExpand: () => void;
}) {
  const count = activeCount > 0 ? activeCount : completedCount;
  const state = activeCount > 0 ? "running" : "finished";
  return (
    <button
      type="button"
      className="pe-job-panel pe-job-panel--collapsed"
      aria-label={`${counted(count, "background job")} ${state}`}
      onClick={onExpand}
    >
      <span className="pe-job-panel__pulse" aria-hidden="true" />
      {`${counted(count, "job")} ${state}`}
    </button>
  );
}

function JobItem({
  job,
  onCancel,
}: {
  job: TrackedBackgroundJob;
  onCancel: (jobId: string) => void;
}) {
  const running = !isTerminalJobStatus(job.status);
  return (
    <li className={`pe-job-item ${statusClass(job.status)}`}>
      {running && <span className="pe-job-item__spinner" aria-hidden="true" />}
      <div className="pe-job-item__body">
        <div className="pe-job-item__row">
          <span className="pe-job-item__kind">
            {pageEditorJobKindLabel(job.kind)}
          </span>
          <span className="pe-job-item__status">{job.progressLabel}</span>
        </div>
        <strong className="pe-job-item__label">{job.label}</strong>
        <JobExecutionAnnouncement
          execution={job.execution}
          className="pe-job-item__host"
        />
        {job.error && <p className="pe-job-item__error">{job.error}</p>}
        {running && (
          <div className="pe-job-item__actions">
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={() => onCancel(job.id)}
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </li>
  );
}

export function PageEditorJobProgressPanel({
  jobs,
  activeCount,
  expanded,
  onExpandedChange,
  onDismissCompleted,
  onCancelJob,
}: PageEditorJobProgressPanelProps) {
  if (jobs.length === 0) return null;

  const completedCount = jobs.filter((job) =>
    isTerminalJobStatus(job.status),
  ).length;

  if (!expanded) {
    return (
      <CollapsedJobPanel
        activeCount={activeCount}
        completedCount={completedCount}
        onExpand={() => onExpandedChange(true)}
      />
    );
  }

  return (
    <section
      className="pe-job-panel"
      role="dialog"
      aria-label="Background jobs"
      aria-live="polite"
    >
      <header className="pe-job-panel__head">
        <div>
          <h2 className="pe-job-panel__title">Background jobs</h2>
          <p className="pe-job-panel__subtitle">
            {activeCount > 0
              ? `${activeCount} in progress. You can keep editing other segments.`
              : "Recent jobs"}
          </p>
        </div>
        <div className="pe-job-panel__head-actions">
          {completedCount > 0 && (
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={onDismissCompleted}
            >
              Clear
            </button>
          )}
          <button
            type="button"
            className="pe-tb-btn"
            aria-label="Minimize jobs panel"
            title="Minimize"
            onClick={() => onExpandedChange(false)}
          >
            −
          </button>
        </div>
      </header>
      <ul className="pe-job-panel__list">
        {jobs.map((job) => (
          <JobItem key={job.id} job={job} onCancel={onCancelJob} />
        ))}
      </ul>
    </section>
  );
}

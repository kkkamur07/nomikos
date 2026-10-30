import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Popconfirm } from "antd";
import {
  api,
  type DocumentResponse,
  type JobResponse,
  type JobStatus,
} from "../../api/client";
import { ApiError } from "../../api/errors";
import { useJobPolling } from "../../hooks/useJobPolling";
import { toast } from "../ui/toast";
import {
  isTerminalJobStatus,
  jobStatusLabel,
} from "../page-editor/jobProgress";
import {
  executionAnnouncement,
  INFERENCE_HOST_LABEL,
  jobExecution,
} from "../../inference/executionTarget";

type ProjectJobsPanelProps = {
  projectId: string;
  documents: DocumentResponse[];
};

const VISIBLE_JOB_LIMIT = 8;

const JOB_TYPE_LABELS: Record<string, string> = {
  segment: "Segmentation",
  transcribe: "Transcription",
  binarize: "Binarize",
  pipeline: "Pipeline",
};

function formatJobType(type: JobResponse["type"]): string {
  return JOB_TYPE_LABELS[type] ?? type;
}

function formatWhen(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function executionClass(job: JobResponse): string {
  return job.execution_target === "local"
    ? "project-jobs-panel__host--local"
    : "project-jobs-panel__host--cloud";
}

const STATUS_CLASSES: Record<JobStatus, string> = {
  pending: "project-jobs-panel__status--pending",
  waiting: "project-jobs-panel__status--pending",
  running: "project-jobs-panel__status--active",
  done: "project-jobs-panel__status--done",
  failed: "project-jobs-panel__status--failed",
  cancelled: "project-jobs-panel__status--cancelled",
};

function statusClass(status: JobStatus): string {
  return STATUS_CLASSES[status] ?? "project-jobs-panel__status--active";
}

function mergeJobPage(
  current: JobResponse[],
  page: { items: JobResponse[] },
  isFirstPage: boolean,
): JobResponse[] {
  const next = isFirstPage ? page.items : [...current, ...page.items];
  return Array.from(new Map(next.map((job) => [job.id, job])).values()).sort(
    (left, right) =>
      new Date(right.created_at).getTime() -
      new Date(left.created_at).getTime(),
  );
}

function loadErrorMessage(err: unknown): string {
  return err instanceof ApiError ? err.message : "Failed to load jobs";
}

function jobsSummary(
  loading: boolean,
  activeCount: number,
  total: number,
): string {
  if (loading) return "Loading…";
  if (activeCount > 0) return `${activeCount} running`;
  if (total > 0) return `${total} recent`;
  return "No jobs yet";
}

function useProjectJobs(projectId: string): {
  jobs: JobResponse[];
  loading: boolean;
  loadingMore: boolean;
  nextCursor: string | null;
  error: string | null;
  activeCount: number;
  clearing: boolean;
  loadMore: () => void;
  clearHistory: () => void;
} {
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const loadMoreControllerRef = useRef<AbortController | null>(null);

  const setPageLoading = (isFirstPage: boolean, value: boolean) => {
    if (isFirstPage) setLoading(value);
    else setLoadingMore(value);
  };

  const load = useCallback(
    async (cursor: string | null, signal: AbortSignal) => {
      const isFirstPage = cursor === null;
      setPageLoading(isFirstPage, true);
      setError(null);
      try {
        const page = await api.listProjectJobsPage(projectId, {
          cursor,
          limit: VISIBLE_JOB_LIMIT,
          signal,
        });
        if (signal.aborted) return;
        setJobs((current) => mergeJobPage(current, page, isFirstPage));
        setNextCursor(page.next_cursor);
      } catch (err) {
        if (signal.aborted) return;
        setError(loadErrorMessage(err));
        if (isFirstPage) setJobs([]);
      } finally {
        if (!signal.aborted) setPageLoading(isFirstPage, false);
      }
    },
    [projectId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(null, controller.signal);
    return () => {
      controller.abort();
      loadMoreControllerRef.current?.abort();
    };
  }, [load]);

  const activeJobIds = jobs
    .filter((job) => !isTerminalJobStatus(job.status))
    .map((job) => job.id);

  useJobPolling(activeJobIds, (updates) => {
    setJobs((current) => {
      const next = [...current];
      for (const update of updates) {
        const index = next.findIndex((job) => job.id === update.id);
        if (index >= 0) {
          next[index] = update;
        }
      }
      return next.sort(
        (left, right) =>
          new Date(right.created_at).getTime() -
          new Date(left.created_at).getTime(),
      );
    });
  });

  const loadMore = () => {
    if (!nextCursor || loadingMore) return;
    const controller = new AbortController();
    loadMoreControllerRef.current?.abort();
    loadMoreControllerRef.current = controller;
    void load(nextCursor, controller.signal).finally(() => {
      if (loadMoreControllerRef.current === controller) {
        loadMoreControllerRef.current = null;
      }
    });
  };

  const clearHistory = async () => {
    if (clearing) return;
    setClearing(true);
    try {
      const { deleted } = await api.clearProjectJobHistory(projectId);
      toast.success(
        deleted === 1
          ? "1 finished job removed"
          : `${deleted} finished jobs removed`,
      );
      setNextCursor(null);
      const controller = new AbortController();
      await load(null, controller.signal);
    } catch (err) {
      toast.error(
        err instanceof ApiError ? err.message : "Could not clear job history",
      );
    } finally {
      setClearing(false);
    }
  };

  return {
    jobs,
    loading,
    loadingMore,
    nextCursor,
    error,
    activeCount: jobs.filter((job) => !isTerminalJobStatus(job.status)).length,
    clearing,
    loadMore,
    clearHistory: () => void clearHistory(),
  };
}

function documentDisplayName(
  job: JobResponse,
  documentNames: Map<string, string>,
): string {
  if (!job.document_id) return "-";
  return documentNames.get(job.document_id) ?? job.document_id.slice(0, 8);
}

function JobRow({
  job,
  projectId,
  documentName,
}: {
  job: JobResponse;
  projectId: string;
  documentName: string;
}): React.ReactNode {
  return (
    <tr key={job.id}>
      <td>
        <span className="row-title">{formatJobType(job.type)}</span>
        <span className="row-sub">{job.id.slice(0, 8)}</span>
      </td>
      <td>
        <span className={`project-jobs-panel__host ${executionClass(job)}`}>
          {INFERENCE_HOST_LABEL[job.execution_target]}
        </span>
        {/* The host is stated in full on the row itself, not
            only as a badge: a substituted host has to be
            readable long after the submission that caused it. */}
        <span className="row-sub">
          {executionAnnouncement(jobExecution(job))}
        </span>
      </td>
      <td className="col-muted">
        {job.document_id ? (
          <Link href={`/projects/${projectId}/documents/${job.document_id}`}>
            {documentName}
          </Link>
        ) : (
          documentName
        )}
      </td>
      <td>
        <span
          className={`project-jobs-panel__status ${statusClass(job.status)}`}
        >
          {jobStatusLabel(job)}
        </span>
        {job.error && (
          <span className="row-sub" title={job.error}>
            {job.error}
          </span>
        )}
      </td>
      <td className="col-muted">{formatWhen(job.created_at)}</td>
    </tr>
  );
}

function ClearHistoryButton({
  finishedCount,
  clearing,
  onClear,
}: {
  finishedCount: number;
  clearing: boolean;
  onClear: () => void;
}): React.ReactNode {
  if (finishedCount === 0) return null;
  return (
    <Popconfirm
      title="Clear job history?"
      description="Only finished jobs (done, failed and cancelled) are removed. Jobs that are still running stay in the list."
      okText="Clear history"
      cancelText="Keep"
      okButtonProps={{ loading: clearing }}
      onConfirm={onClear}
    >
      <button
        type="button"
        className="btn btn-outline btn-sm project-jobs-panel__clear"
        disabled={clearing}
      >
        {clearing ? "Clearing…" : "Clear history"}
      </button>
    </Popconfirm>
  );
}

function JobsPanelBody({
  projectId,
  jobs,
  documentNames,
  loading,
  error,
  expanded,
  nextCursor,
  loadingMore,
  loadMore,
}: {
  projectId: string;
  jobs: JobResponse[];
  documentNames: Map<string, string>;
  loading: boolean;
  error: string | null;
  expanded: boolean;
  nextCursor: string | null;
  loadingMore: boolean;
  loadMore: () => void;
}): React.ReactNode {
  if (!expanded || loading || error || jobs.length === 0) return null;
  return (
    <div className="project-jobs-panel__body" id="project-jobs-panel-body">
      <div className="data-panel">
        <table className="data-list" aria-label="Project jobs">
          <thead>
            <tr>
              <th scope="col">Job</th>
              <th scope="col">Host</th>
              <th scope="col">Document</th>
              <th scope="col">Status</th>
              <th scope="col">Started</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <JobRow
                key={job.id}
                job={job}
                projectId={projectId}
                documentName={documentDisplayName(job, documentNames)}
              />
            ))}
          </tbody>
        </table>
      </div>
      {nextCursor && (
        <div className="project-jobs-panel__more">
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={loadMore}
            disabled={loadingMore}
          >
            {loadingMore ? "Loading jobs…" : "Load more jobs"}
          </button>
        </div>
      )}
    </div>
  );
}

export function ProjectJobsPanel({
  projectId,
  documents,
}: ProjectJobsPanelProps) {
  const {
    jobs,
    loading,
    loadingMore,
    nextCursor,
    error,
    activeCount,
    clearing,
    loadMore,
    clearHistory,
  } = useProjectJobs(projectId);
  const [expanded, setExpanded] = useState(false);

  const documentNames = useMemo(
    () => new Map(documents.map((document) => [document.id, document.name])),
    [documents],
  );

  useEffect(() => {
    if (activeCount > 0) {
      setExpanded(true);
    }
  }, [activeCount]);

  if (!loading && !error && jobs.length === 0) {
    return (
      <section
        className="project-jobs-panel project-jobs-panel--empty"
        aria-labelledby="project-jobs-heading"
      >
        <h2 className="project-jobs-panel__heading" id="project-jobs-heading">
          Jobs
        </h2>
        <p className="project-jobs-panel__hint">
          No jobs yet. Segment or transcribe from a document page - completed,
          failed, and cancelled runs show up here.
        </p>
      </section>
    );
  }

  return (
    <section
      className="project-jobs-panel"
      aria-labelledby="project-jobs-heading"
    >
      <div className="project-jobs-panel__bar">
        <button
          type="button"
          className="project-jobs-panel__toggle"
          aria-expanded={expanded}
          aria-controls="project-jobs-panel-body"
          onClick={() => setExpanded((value) => !value)}
        >
          <span className="project-jobs-panel__toggle-main">
            <span
              className="project-jobs-panel__heading"
              id="project-jobs-heading"
            >
              Jobs
            </span>
            <span className="project-jobs-panel__summary">
              {jobsSummary(loading, activeCount, jobs.length)}
            </span>
          </span>
          <span className="project-jobs-panel__chevron" aria-hidden="true">
            {expanded ? "▴" : "▾"}
          </span>
        </button>
        <ClearHistoryButton
          finishedCount={jobs.length - activeCount}
          clearing={clearing}
          onClear={clearHistory}
        />
      </div>

      {error && (
        <div className="notice-banner" role="alert">
          <strong>Jobs unavailable</strong>
          {error}
        </div>
      )}

      <JobsPanelBody
        projectId={projectId}
        jobs={jobs}
        documentNames={documentNames}
        loading={loading}
        error={error}
        expanded={expanded}
        nextCursor={nextCursor}
        loadingMore={loadingMore}
        loadMore={loadMore}
      />
    </section>
  );
}

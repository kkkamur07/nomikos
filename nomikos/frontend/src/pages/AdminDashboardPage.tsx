import { useEffect, useState, type ReactNode } from "react";
import { notFound, useRouter } from "next/navigation";
import { Segmented, Table } from "antd";
import { api, type AdminOverview } from "../api/client";
import { ApiError } from "../api/errors";
import { userFacingMessage } from "../api/userFacingError";
import { useAuthSession } from "../auth/AuthProvider";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { AppPageShell } from "../components/layout/AppPageShell";
import { ContentRegionLoading } from "../components/layout/ContentRegionLoading";
import { useServerQuery } from "../hooks/useServerQuery";
import "../styles/admin.css";

type Users = AdminOverview["users"];
type UserRow = Users["top_users"][number];
type ModelRow = AdminOverview["models"][number];
type TableRow = AdminOverview["database"]["busiest_tables"][number];
type RouteRow = AdminOverview["requests"]["routes"][number];
type FailureRow = AdminOverview["client_failures"]["recent"][number];

const DAY_CHOICES = [7, 30, 90] as const;
/** Above this many bars the per-bar value labels collide, so all are hidden. */
const MAX_LABELLED_BARS = 31;

/** One stacked or grouped series; `color` is a CSS custom property name. */
type Series = { key: string; label: string; color: string };
type DayValues = Record<string, Record<string, number>>;

const JOB_STATUS_SERIES: Series[] = [
  { key: "done", label: "Done", color: "--admin-done" },
  { key: "failed", label: "Failed", color: "--admin-failed" },
  { key: "running", label: "Running", color: "--admin-running" },
  { key: "waiting", label: "Waiting", color: "--admin-waiting" },
  { key: "pending", label: "Pending", color: "--admin-pending" },
  { key: "cancelled", label: "Cancelled", color: "--admin-cancelled" },
];

/** Every UTC day of the window, oldest first, as `YYYY-MM-DD`. */
export function windowDays(days: number, now: Date = new Date()): string[] {
  const today = Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate(),
  );
  return Array.from({ length: days }, (_, index) =>
    new Date(today - (days - 1 - index) * 86_400_000)
      .toISOString()
      .slice(0, 10),
  );
}

function formatNumber(value: number): string {
  return value.toLocaleString("en-US");
}

function formatBytes(bytes: number): string {
  const units = ["B", "kB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) return "never";
  return new Date(value).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });
}

function percent(part: number, whole: number): string {
  if (whole === 0) return "n/a";
  return `${((part / whole) * 100).toFixed(1)}%`;
}

function DayBars({
  title,
  days,
  values,
  series,
  mode = "stacked",
}: {
  title: string;
  days: string[];
  values: DayValues;
  series: Series[];
  mode?: "stacked" | "grouped";
}) {
  const totals = days.map((day) =>
    series.reduce((sum, item) => sum + (values[day]?.[item.key] ?? 0), 0),
  );
  const peak = Math.max(
    1,
    ...days.flatMap((day) =>
      mode === "stacked"
        ? [series.reduce((sum, s) => sum + (values[day]?.[s.key] ?? 0), 0)]
        : series.map((s) => values[day]?.[s.key] ?? 0),
    ),
  );
  const labelled = days.length <= MAX_LABELLED_BARS;
  const grandTotal = totals.reduce((sum, value) => sum + value, 0);

  return (
    <figure className="admin-chart" aria-label={title}>
      <figcaption className="admin-chart-head">
        <span className="admin-chart-title">{title}</span>
        <span className="admin-legend">
          {series.map((item) => (
            <span key={item.key} className="admin-legend-item">
              <span
                className="admin-swatch"
                style={{ background: `var(${item.color})` }}
                aria-hidden="true"
              />
              {item.label}
            </span>
          ))}
        </span>
      </figcaption>
      {grandTotal === 0 ? (
        <p className="admin-empty">Nothing in this window.</p>
      ) : (
        <div
          className={`admin-bars admin-bars--${mode}`}
          style={{ gridTemplateColumns: `repeat(${days.length}, 1fr)` }}
        >
          {days.map((day, index) => {
            const tooltip = [
              day,
              ...series.map((s) => `${s.label}: ${values[day]?.[s.key] ?? 0}`),
            ].join("\n");
            return (
              <div key={day} className="admin-bar-slot" title={tooltip}>
                {mode === "stacked" ? (
                  <div className="admin-bar-column">
                    {labelled && totals[index] > 0 && (
                      <span className="admin-bar-value">{totals[index]}</span>
                    )}
                    <div
                      className="admin-bar-stack"
                      style={{ height: `${(totals[index] / peak) * 100}%` }}
                    >
                      {series.map((s) => {
                        const value = values[day]?.[s.key] ?? 0;
                        if (value === 0) return null;
                        return (
                          <div
                            key={s.key}
                            className="admin-bar-segment"
                            style={{
                              flexGrow: value,
                              background: `var(${s.color})`,
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="admin-bar-group">
                    {series.map((s) => {
                      const value = values[day]?.[s.key] ?? 0;
                      return (
                        <div key={s.key} className="admin-bar-column">
                          {labelled && value > 0 && (
                            <span className="admin-bar-value">{value}</span>
                          )}
                          <div
                            className="admin-bar-stack"
                            style={{
                              height: `${(value / peak) * 100}%`,
                              background: `var(${s.color})`,
                            }}
                          />
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      {grandTotal > 0 && (
        <div className="admin-axis" aria-hidden="true">
          <span>{days[0]}</span>
          <span>{days[days.length - 1]}</span>
        </div>
      )}
    </figure>
  );
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="admin-stat">
      <span className="admin-stat-label">{label}</span>
      <span className="admin-stat-value">{value}</span>
      {hint && <span className="admin-stat-hint">{hint}</span>}
    </div>
  );
}

function Card({
  id,
  title,
  note,
  children,
}: {
  id: string;
  title: string;
  note?: string;
  children: ReactNode;
}) {
  return (
    <section className="admin-card" aria-labelledby={id} id={`card-${id}`}>
      <header className="admin-card-head">
        <h2 id={id}>{title}</h2>
        {note && <p className="admin-card-note">{note}</p>}
      </header>
      {children}
    </section>
  );
}

function byDay<T extends { day: string }>(
  rows: T[],
  pick: (row: T) => Record<string, number>,
): DayValues {
  const out: DayValues = {};
  for (const row of rows) {
    const bucket = (out[row.day] ??= {});
    for (const [key, value] of Object.entries(pick(row))) {
      bucket[key] = (bucket[key] ?? 0) + value;
    }
  }
  return out;
}

function tableColumns(): {
  title: string;
  dataIndex: keyof TableRow;
  align?: "right";
  render?: (value: number) => string;
}[] {
  const n = (value: number) => formatNumber(value);
  return [
    { title: "Table", dataIndex: "name" },
    { title: "Inserts", dataIndex: "inserts", align: "right", render: n },
    { title: "Updates", dataIndex: "updates", align: "right", render: n },
    { title: "Deletes", dataIndex: "deletes", align: "right", render: n },
    { title: "Seq scans", dataIndex: "seq_scan", align: "right", render: n },
    { title: "Index scans", dataIndex: "idx_scan", align: "right", render: n },
    { title: "Rows", dataIndex: "live_rows", align: "right", render: n },
    {
      title: "Size",
      dataIndex: "total_bytes",
      align: "right",
      render: formatBytes,
    },
  ];
}

export function AdminOverviewView({ overview }: { overview: AdminOverview }) {
  const days = windowDays(overview.days);
  const { users, activity, models, database, requests } = overview;
  const failures = overview.client_failures;

  const jobsByStatus = byDay(activity.jobs, (row) => ({
    [row.status]: row.count,
  }));
  const jobsByKind = activity.jobs.reduce<Record<string, number>>(
    (acc, row) => ({ ...acc, [row.kind]: (acc[row.kind] ?? 0) + row.count }),
    {},
  );
  const totalJobs = activity.jobs.reduce((sum, row) => sum + row.count, 0);
  const failedJobs = activity.jobs
    .filter((row) => row.status === "failed")
    .reduce((sum, row) => sum + row.count, 0);
  const pagesPerDay = byDay(activity.pages, (row) => ({ pages: row.count }));
  const linesPerDay = byDay(activity.lines, (row) => ({ lines: row.count }));
  const annotations = byDay(activity.annotations, (row) => ({
    corrections: row.corrections,
    approved: row.approved,
  }));
  const failuresPerDay = byDay(failures.per_day, (row) => ({
    failures: row.count,
  }));
  const cacheHit = percent(
    database.blks_hit,
    database.blks_hit + database.blks_read,
  );
  const transactions = database.xact_commit + database.xact_rollback;

  return (
    <div className="admin-dashboard">
      <div className="admin-stats" aria-label="Headline numbers">
        <Stat label="Users" value={formatNumber(users.total)} />
        <Stat label="Active, 7 days" value={formatNumber(users.active_7d)} />
        <Stat label="Active, 30 days" value={formatNumber(users.active_30d)} />
        <Stat
          label={`New, ${overview.days} days`}
          value={formatNumber(users.new_in_window)}
        />
        <Stat
          label={`Jobs, ${overview.days} days`}
          value={formatNumber(totalJobs)}
          hint={`${formatNumber(failedJobs)} failed`}
        />
        <Stat
          label="Client failures"
          value={formatNumber(
            failures.per_day.reduce((sum, row) => sum + row.count, 0),
          )}
          hint="since API start"
        />
      </div>

      <Card
        id="activity"
        title="Activity per day"
        note="UTC days. Corrections are page saves; approved lines are the rise in lines with approved text since the page's previous save."
      >
        <div className="admin-chart-grid">
          <DayBars
            title="Jobs by status"
            days={days}
            values={jobsByStatus}
            series={JOB_STATUS_SERIES}
          />
          <DayBars
            title="Corrections and approved lines"
            days={days}
            values={annotations}
            mode="grouped"
            series={[
              {
                key: "corrections",
                label: "Corrections",
                color: "--admin-series-a",
              },
              {
                key: "approved",
                label: "Lines approved",
                color: "--admin-series-b",
              },
            ]}
          />
          <DayBars
            title="Pages uploaded"
            days={days}
            values={pagesPerDay}
            series={[
              { key: "pages", label: "Pages", color: "--admin-series-a" },
            ]}
          />
          <DayBars
            title="Lines created"
            days={days}
            values={linesPerDay}
            series={[
              { key: "lines", label: "Lines", color: "--admin-series-b" },
            ]}
          />
        </div>
        {totalJobs > 0 && (
          <p className="admin-kinds">
            Jobs by kind:{" "}
            {Object.entries(jobsByKind)
              .sort((a, b) => b[1] - a[1])
              .map(([kind, count]) => `${kind} ${formatNumber(count)}`)
              .join(" · ")}
          </p>
        )}
      </Card>

      <Card
        id="users"
        title="Users"
        note={`Ranked by jobs, then approved lines, in the last ${overview.days} days. Approvals are credited to the project owner.`}
      >
        <Table<UserRow>
          size="small"
          rowKey="id"
          pagination={false}
          dataSource={users.top_users}
          columns={[
            {
              title: "Email",
              dataIndex: "email",
              render: (email: string, row) => (
                <span>
                  {email}
                  {row.is_admin && <span className="admin-tag">admin</span>}
                </span>
              ),
            },
            { title: "Username", dataIndex: "username" },
            {
              title: "Jobs",
              dataIndex: "jobs",
              align: "right",
              sorter: (a, b) => a.jobs - b.jobs,
            },
            {
              title: "Lines approved",
              dataIndex: "lines_approved",
              align: "right",
              sorter: (a, b) => a.lines_approved - b.lines_approved,
            },
            {
              title: "Last seen (UTC)",
              dataIndex: "last_seen",
              render: (value: string | null) => formatDateTime(value),
            },
            {
              title: "Joined",
              dataIndex: "created_at",
              render: (value: string) => value.slice(0, 10),
            },
          ]}
        />
      </Card>

      <Card
        id="models"
        title="Models"
        note="Success rate counts finished jobs only; median duration is created to completed, for jobs that finished."
      >
        <Table<ModelRow>
          size="small"
          rowKey={(row) => `${row.name}:${row.task}`}
          pagination={false}
          dataSource={models}
          columns={[
            { title: "Model", dataIndex: "name" },
            { title: "Task", dataIndex: "task" },
            {
              title: "Jobs",
              dataIndex: "jobs",
              align: "right",
              render: (value: number) => (
                <span className="admin-inline-bar">
                  <span
                    className="admin-inline-bar-fill"
                    style={{
                      width: `${(value / Math.max(1, ...models.map((m) => m.jobs))) * 100}%`,
                    }}
                  />
                  <span className="admin-inline-bar-value">
                    {formatNumber(value)}
                  </span>
                </span>
              ),
            },
            { title: "Done", dataIndex: "done", align: "right" },
            { title: "Failed", dataIndex: "failed", align: "right" },
            {
              title: "Success",
              dataIndex: "success_rate",
              align: "right",
              render: (value: number | null) =>
                value == null ? "n/a" : `${(value * 100).toFixed(1)}%`,
            },
            {
              title: "Median",
              dataIndex: "median_seconds",
              align: "right",
              render: (value: number | null) =>
                value == null ? "n/a" : `${value.toFixed(1)} s`,
            },
          ]}
        />
      </Card>

      <Card
        id="database"
        title="Database"
        note={`Counters are cumulative since ${database.stats_reset ? formatDateTime(database.stats_reset) : "the statistics were last reset"} (UTC), for ${database.name}.`}
      >
        <div className="admin-stats admin-stats--inner">
          <Stat
            label="Transactions committed"
            value={formatNumber(database.xact_commit)}
          />
          <Stat
            label="Rolled back"
            value={formatNumber(database.xact_rollback)}
            hint={percent(database.xact_rollback, transactions)}
          />
          <Stat
            label="Connections"
            value={formatNumber(database.numbackends)}
            hint={`${formatNumber(database.active_queries)} running a query`}
          />
          <Stat label="Cache hit rate" value={cacheHit} />
          <Stat label="Size" value={formatBytes(database.size_bytes)} />
          <Stat
            label="Rows written"
            value={formatNumber(
              database.tup_inserted +
                database.tup_updated +
                database.tup_deleted,
            )}
            hint={`${formatNumber(database.tup_returned)} read`}
          />
        </div>
        {database.pool_status && (
          <p className="admin-pool">
            This API process's pool: <code>{database.pool_status}</code>
          </p>
        )}
        <h3 className="admin-subhead">Busiest tables (rows written)</h3>
        <Table<TableRow>
          size="small"
          rowKey="name"
          pagination={false}
          dataSource={database.busiest_tables}
          columns={tableColumns()}
        />
        <h3 className="admin-subhead">Most sequential scans</h3>
        <Table<TableRow>
          size="small"
          rowKey="name"
          pagination={false}
          dataSource={database.seq_scan_tables}
          columns={tableColumns()}
        />
        <h3 className="admin-subhead">Largest tables</h3>
        <Table<TableRow>
          size="small"
          rowKey="name"
          pagination={false}
          dataSource={database.largest_tables}
          columns={tableColumns()}
        />
      </Card>

      <Card
        id="requests"
        title="Requests"
        note={`${formatNumber(requests.total_requests)} requests since this API process started at ${formatDateTime(requests.started_at)} (UTC). In memory, one process, health probes excluded.`}
      >
        <Table<RouteRow>
          size="small"
          rowKey={(row) => `${row.method} ${row.route} ${row.status_class}`}
          pagination={{ pageSize: 15, hideOnSinglePage: true }}
          dataSource={requests.routes}
          columns={[
            { title: "Method", dataIndex: "method" },
            { title: "Route", dataIndex: "route" },
            { title: "Status", dataIndex: "status_class" },
            {
              title: "Requests",
              dataIndex: "count",
              align: "right",
              render: (value: number) => formatNumber(value),
            },
            {
              title: "Avg ms",
              dataIndex: "avg_ms",
              align: "right",
            },
            {
              title: "Max ms",
              dataIndex: "max_ms",
              align: "right",
            },
          ]}
        />
      </Card>

      <Card
        id="failures"
        title="Client failures"
        note="Failure beacons sent by browsers to this API process since it started."
      >
        <DayBars
          title="Failures per day"
          days={days}
          values={failuresPerDay}
          series={[
            { key: "failures", label: "Failures", color: "--admin-failed" },
          ]}
        />
        <Table<FailureRow>
          size="small"
          rowKey={(row) => `${row.at}:${row.message}`}
          pagination={false}
          dataSource={failures.recent}
          locale={{ emptyText: "No failures reported." }}
          columns={[
            {
              title: "When (UTC)",
              dataIndex: "at",
              render: (value: string) => formatDateTime(value),
            },
            { title: "Page", dataIndex: "path" },
            { title: "Status", dataIndex: "status" },
            { title: "Source", dataIndex: "source" },
            { title: "Message", dataIndex: "message" },
          ]}
        />
      </Card>
    </div>
  );
}

/**
 * `/admin`: platform usage and database load, for `users.is_admin` accounts.
 *
 * Anyone else, including a signed-in account the API answers 403 for, gets the
 * not-found page, so the route does not advertise that it exists.
 */
export function AdminDashboardPage() {
  const router = useRouter();
  const { isAdmin } = useAuthSession();
  const [days, setDays] = useState<number>(30);

  const signedIn = hasAccessToken();
  useEffect(() => {
    if (!signedIn) navigateToLogin(router);
  }, [signedIn, router]);

  const [forbidden, setForbidden] = useState(false);
  const { data, loading, error } = useServerQuery<AdminOverview>({
    key: signedIn && isAdmin === true ? ["admin-overview", days] : null,
    tags: [],
    read: () => api.adminOverview(days),
    onError: (err) => {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return null;
      }
      if (err instanceof ApiError && err.status === 403) {
        setForbidden(true);
        return null;
      }
      return userFacingMessage(err, "Failed to load the admin overview");
    },
  });

  if (isAdmin === false || forbidden) {
    notFound();
    return null;
  }

  return (
    <AppPageShell
      currentLabel="Admin"
      title="Admin"
      subtitle="How the platform is used and how hard the database is working"
      headerActions={
        <Segmented
          aria-label="Window"
          value={days}
          onChange={(value) => setDays(Number(value))}
          options={DAY_CHOICES.map((choice) => ({
            label: `${choice} days`,
            value: choice,
          }))}
        />
      }
    >
      {error && (
        <div className="notice-banner" role="alert">
          <strong>Overview unavailable</strong>
          {error}
        </div>
      )}
      {data ? (
        <AdminOverviewView overview={data} />
      ) : (
        loading &&
        !error && <ContentRegionLoading label="Loading the overview" />
      )}
    </AppPageShell>
  );
}

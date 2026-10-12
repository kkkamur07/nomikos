# Admin dashboard

`/admin` shows how the platform is used and how hard the database is working. Only accounts
with `users.is_admin = true` can open it: the API answers `GET /admin/overview` with 401 when
signed out and 403 for any other account, and the frontend shows its not-found page. The design
is recorded in [ADR 0010](../adr/0010-admin-dashboard-and-request-metrics.md).

## Promote an admin

There is no API or UI for this, on purpose. Run SQL against the database.

1. Once per database, add the column (production never runs alembic itself):

   ```sh
   psql "$DATABASE_URL" -f scripts/platform/sql/2026-10-11_user_is_admin.sql
   ```

   Locally, `alembic upgrade head` does the same (revision `008_user_is_admin`).

2. Promote the account:

   ```sql
   UPDATE users SET is_admin = true WHERE lower(email) = lower('someone@example.com');
   ```

3. The account sees an **Admin** link in the top bar after its next page load. Demote with the
   same statement and `false`; it takes effect on the next request.

## What each number means

All windows are the last 7, 30 or 90 days (the selector at the top), bucketed by UTC day.
The Postgres cards (headline, activity, users, models, database) are cached for 60 seconds per
window in each API process, and each of their statements stops after 5 seconds; the request,
client-failure and pool numbers are always live.

### Headline

- **Users**: every account. **Active, 7 / 30 days**: distinct accounts that used a browser
  session (signed in, or refreshed an existing session) in that period. **New**: accounts created in the window.
- **Jobs**: jobs created in the window, with the failed ones underneath.
- **Client failures**: failure beacons received by this API process since it started.

### Activity per day

- **Jobs by status**: jobs created per day, stacked by their current status.
- **Corrections and approved lines**: a correction is one annotation-history snapshot (a page
  save). Approved lines are the rise in lines with approved text since the page's previous
  snapshot. History keeps only the last few snapshots per page, and a page's oldest kept
  snapshot has nothing to compare with, so its approvals are not counted: older days
  under-count, never over-count.
- **Pages uploaded** and **Lines created**: rows created in `document_parts` and `lines`,
  including lines made by segmentation.
- **Jobs by kind**: the same jobs totalled by type.

### Users

The top 20 accounts by jobs, then by approved lines, in the window. Approved lines are credited
to the owner of the project the page belongs to (snapshots record no user). **Last seen** is the
latest browser-session creation or refresh.

### Models

Jobs in the window grouped by inference model (jobs without one, such as test jobs, show as
`(no model)`). **Success** is done / (done + failed); pending and running jobs are not counted
as failures. **Median** is completion time minus creation time, for jobs that finished, so it
includes time spent waiting in the queue.

### Database

From Postgres's own statistics for the current database. These counters are cumulative since
the statistics were last reset (shown in the card note), not limited to the window.

- **Transactions committed / rolled back**: `xact_commit` and `xact_rollback`. Read-only
  requests that end in a rollback count here too, so a high rollback share is not by itself an
  error rate.
- **Connections**: backends connected to this database (`numbackends`), and how many are
  running a query right now.
- **Cache hit rate**: `blks_hit / (blks_hit + blks_read)`.
- **Rows written / read**: `tup_inserted + tup_updated + tup_deleted`, and `tup_returned`.
- **Pool**: this API process's SQLAlchemy pool (size, idle, overflow, checked out).
- **Busiest tables**: the ten tables with the most rows inserted, updated and deleted. These are
  where the write cost goes.
- **Most sequential scans**: tables read without an index most often; a large table near the top
  is a missing-index candidate.
- **Largest tables**: total size including indexes and TOAST.

### Requests

Per route template, method and status class: request count, average and maximum latency in ms.
Held in this API process's memory since it started (health probes and docs excluded), so it
resets on every restart and each replica reports only its own traffic. The numbers are also
per worker: with more than one uvicorn worker (`WEB_CONCURRENCY` above 1), each refresh shows
whichever worker answered. Nothing is written to the
database per request.

### Client failures

The browser failure beacons (`POST /client-failures`) this API process received since it
started: a per-day count and the 20 most recent, newest first. The full record stays in the API
logs as before.

# 0008. Admin dashboard and request metrics

- Status: Accepted
- Date: 2026-10-11

## Context

There was no way to see, from inside the platform, how it is used or how hard the database is
working. Questions like "who was active this week", "which models get run and how often do they
fail", or "which tables take the writes" needed a SQL console on the production host. The
database is metered, so the last question matters for cost, not only for curiosity.

The API had no notion of an operator (every account was a researcher), no request logging
beyond the security-header and CORS middleware, and the frontend failure beacon
(`POST /client-failures`) only wrote a log line.

## Decision

1. **One boolean, `users.is_admin`** (migration `007_user_is_admin`, default `false`). No roles
   table: there is exactly one privilege, read-only and all-or-nothing. No route writes the
   column; promotion is an `UPDATE` by whoever operates the database, so an account cannot grant
   itself the dashboard. `GET /me` exposes the flag so the client can show the link, and
   `require_admin` checks the database row on every admin request (401 when signed out, 403 for
   a signed-in non-admin). The frontend renders the not-found page for non-admins so the route
   does not advertise itself.
2. **One route, `GET /admin/overview?days=N`**, returning every card in one response. Each card
   is one aggregate SQL statement (sub-results come back as JSON sub-selects of that statement),
   so the cost of opening the dashboard is a fixed handful of round trips whatever the data
   size. Database load is read from `pg_stat_database` and `pg_stat_user_tables`, which the app
   role can already read, plus the SQLAlchemy pool status.
3. **Per-route request counters live in process memory only.** A pure ASGI middleware counts
   requests per (method, route template, status class) with average and maximum latency, and
   the client-failure route also appends to an in-memory ring buffer. No table, no write per
   request: writing a row per request to measure database load would add the load being
   measured. Health probes and docs paths are not counted. Route templates, never raw paths, so
   the counter set stays the size of the API.

## Consequences

- Request and client-failure numbers cover one API process since it started (`started_at` is
  shown) and reset on restart or redeploy. With several replicas, each answers with its own.
  If cross-process totals are ever needed, the counters can be scraped or shipped elsewhere;
  that is a separate decision.
- "Lines approved" has no exact source: lines carry no approval timestamp and annotation
  history snapshots record no user and keep only the last few per page. The dashboard counts
  the rise in `paired_line_count` between consecutive snapshots of a page and credits it to the
  project owner, so older days under-count. A real approval timestamp would fix this and is not
  part of this change.
- `pg_stat_*` counters are cumulative since the last statistics reset, not windowed; the
  dashboard says so next to them.
- Production never runs alembic automatically, so the column ships as
  `scripts/platform/sql/2026-10-11_user_is_admin.sql`, applied before the API build that reads
  it.

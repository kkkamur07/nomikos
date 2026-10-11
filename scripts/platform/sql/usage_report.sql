-- Read-only production usage report. SELECTs only: no temp tables, no writes.
-- Run through ~/nomikos-usage/usage_report.sh [--days N]; the window defaults to 30 days.
-- Approved text = a non-blank ground_truth line_transcriptions row (as history_service counts it).
-- Logins = auth_sessions.created_at. Durations = completed_at - started_at (done jobs only).
-- Per-user content is credited to the project owner, as the admin dashboard does.
\if :{?days}
\else
\set days 30
\endif
\set QUIET on
\pset border 0
\pset null -
\pset footer off
\set QUIET off

\echo == Accounts (window :days d)
SELECT
  (SELECT count(*) FROM users) AS users,
  (SELECT count(*) FROM users WHERE is_admin) AS admins,
  (SELECT count(*) FROM users WHERE created_at >= now() - make_interval(days => :days)) AS new,
  (SELECT count(DISTINCT user_id) FROM auth_sessions WHERE created_at >= now() - interval '7 days') AS login_7d,
  (SELECT count(DISTINCT user_id) FROM auth_sessions WHERE created_at >= now() - interval '30 days') AS login_30d;

\echo == Per user (newest first, max 50)
WITH owned AS (
  SELECT p.owner_id AS uid, count(DISTINCT p.id) AS prj, count(DISTINCT d.id) AS docs,
         count(DISTINCT dp.id) AS pages
  FROM projects p
  LEFT JOIN documents d ON d.project_id = p.id
  LEFT JOIN document_parts dp ON dp.document_id = d.id
  WHERE p.owner_id IS NOT NULL
  GROUP BY p.owner_id
), appr AS (
  SELECT p.owner_id AS uid, count(*) AS lines_ok
  FROM line_transcriptions lt
  JOIN transcriptions t ON t.id = lt.transcription_id AND t.kind = 'ground_truth'
  JOIN lines l ON l.id = lt.line_id
  JOIN document_parts dp ON dp.id = l.part_id
  JOIN documents d ON d.id = dp.document_id
  JOIN projects p ON p.id = d.project_id
  WHERE btrim(lt.text) <> '' AND p.owner_id IS NOT NULL
  GROUP BY p.owner_id
), jb AS (
  SELECT user_id AS uid, count(*) AS jobs
  FROM jobs
  WHERE created_at >= now() - make_interval(days => :days) AND user_id IS NOT NULL
  GROUP BY user_id
), seen AS (
  SELECT user_id AS uid, max(created_at)::date AS last_login
  FROM auth_sessions GROUP BY user_id
)
SELECT u.email, u.created_at::date AS joined, coalesce(o.prj, 0) AS prj,
       coalesce(o.docs, 0) AS docs, coalesce(o.pages, 0) AS pages,
       coalesce(a.lines_ok, 0) AS ok_lines, coalesce(j.jobs, 0) AS jobs, s.last_login
FROM users u
LEFT JOIN owned o ON o.uid = u.id
LEFT JOIN appr a ON a.uid = u.id
LEFT JOIN jb j ON j.uid = u.id
LEFT JOIN seen s ON s.uid = u.id
ORDER BY u.created_at DESC
LIMIT 50;

\echo == Content
SELECT
  (SELECT count(*) FROM projects) AS projects,
  (SELECT count(*) FROM documents) AS docs,
  (SELECT count(*) FROM document_parts) AS pages,
  (SELECT count(*) FROM lines) AS lines,
  (SELECT count(*) FROM line_transcriptions lt
     JOIN transcriptions t ON t.id = lt.transcription_id AND t.kind = 'ground_truth'
     WHERE btrim(lt.text) <> '') AS approved,
  (SELECT count(*) FROM documents WHERE public_share_token IS NOT NULL) AS published_docs;

\echo == Jobs by kind and status (window)
SELECT type::text AS kind, status::text AS status, count(*) AS n
FROM jobs WHERE created_at >= now() - make_interval(days => :days)
GROUP BY 1, 2 ORDER BY 1, 2;

\echo == Jobs by model (window; secs = done jobs, completed - started)
SELECT coalesce(m.name, '(no model)') AS model, count(*) AS jobs,
       count(*) FILTER (WHERE j.status = 'done') AS done,
       count(*) FILTER (WHERE j.status = 'failed') AS failed,
       round((percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM j.completed_at - j.started_at))
         FILTER (WHERE j.status = 'done' AND j.started_at IS NOT NULL AND j.completed_at IS NOT NULL))::numeric, 1) AS p50_s,
       round((percentile_cont(0.95) WITHIN GROUP (ORDER BY extract(epoch FROM j.completed_at - j.started_at))
         FILTER (WHERE j.status = 'done' AND j.started_at IS NOT NULL AND j.completed_at IS NOT NULL))::numeric, 1) AS p95_s
FROM jobs j LEFT JOIN inference_models m ON m.id = j.model_id
WHERE j.created_at >= now() - make_interval(days => :days)
GROUP BY 1 ORDER BY jobs DESC, 1;

\echo == Job duration overall (window, done jobs)
SELECT count(*) AS n,
       round((percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM completed_at - started_at)))::numeric, 1) AS p50_s,
       round((percentile_cont(0.95) WITHIN GROUP (ORDER BY extract(epoch FROM completed_at - started_at)))::numeric, 1) AS p95_s
FROM jobs
WHERE created_at >= now() - make_interval(days => :days)
  AND status = 'done' AND started_at IS NOT NULL AND completed_at IS NOT NULL;

\echo == Top 5 failure messages (window)
SELECT count(*) AS n, left(regexp_replace(coalesce(error, '(none)'), '\s+', ' ', 'g'), 80) AS error
FROM jobs
WHERE created_at >= now() - make_interval(days => :days) AND status = 'failed'
GROUP BY 2 ORDER BY n DESC, 2 LIMIT 5;

\echo == Jobs per day (last 14 days)
SELECT d::date AS day,
       count(j.id) FILTER (WHERE j.type = 'segment') AS segment,
       count(j.id) FILTER (WHERE j.type = 'transcribe') AS transcribe
FROM generate_series((now() AT TIME ZONE 'UTC')::date - 13, (now() AT TIME ZONE 'UTC')::date, interval '1 day') d
LEFT JOIN jobs j ON (j.created_at AT TIME ZONE 'UTC')::date = d::date
GROUP BY d ORDER BY d DESC;

\echo == Database size
SELECT pg_size_pretty(pg_database_size(current_database())) AS db_size;
SELECT relname AS "table", pg_size_pretty(pg_total_relation_size(relid)) AS size, n_live_tup AS rows
FROM pg_stat_user_tables ORDER BY pg_total_relation_size(relid) DESC, relname LIMIT 8;

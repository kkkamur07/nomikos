# Production deployment

Host Nomikos at **nomikos.app** with three public surfaces and Supabase as the shared database + storage backend.

Architecture overview: [technical architecture](../architecture.md).
Serverless constraints and pitfalls: [learnings - Vercel](../guides/learnings.md#serverless-api-vercel).

| Domain | Vercel project | Root directory | Role |
| -------- | ---------------- | ---------------- | ------ |
| [nomikos.app](https://nomikos.app) | `nomikos-landing` | `landing/` | Static marketing site |
| [app.nomikos.app](https://app.nomikos.app) | `nomikos-app` | `nomikos/frontend/` | Next.js App Router client |
| [api.nomikos.app](https://api.nomikos.app) | `nomikos-api` | `infrastructure/platform/` | FastAPI platform API |

Inference has no public surface of its own. Models run in an **inference
agent** - the **published package** started by a researcher, or by an operator
on a persistent host - which reaches `api.nomikos.app` outbound and listens on
nothing (ADR 0002).

Supabase setup: [`supabase.md`](supabase.md).
Local development: [`../guides/local-development.md`](../guides/local-development.md).

---

## Architecture

```text
Browser
  ├─ nomikos.app          → Vercel (static landing)
  ├─ app.nomikos.app      → Vercel (Next.js App Router)
  │     └─ REST + JWT      → api.nomikos.app (Vercel serverless FastAPI)
  │
  └─ api.nomikos.app      → Supabase Postgres + Storage

Inference (no inbound address, no port):
  nomikos agent    → claims pages from api.nomikos.app, runs models, calls back
                      (researcher's laptop, device token)

Background (persistent compute, not serverless):
  platform-worker   → runs the job types the platform executes itself
  nomikos agent    → same package, service credential, for cloud work
```

### Why local inference is the default

The **inference agent** runs both models on ONNX Runtime. Hub weights are
resolved lazily into the runtime cache; Calamari loads `best.onnx` and BLLA
loads `blla.onnx` from the registry-pinned repository. Vercel serverless
functions have strict size limits and short execution timeouts, so a hosted
function cannot be where a model runs. The hosted platform holds the queue and
persists only the result.

Nothing is deployed to make local inference work. A researcher installs the
package, pairs the machine, and runs the agent in a terminal; it connects out
to `api.nomikos.app` to claim a page and to report it, which is why no
inbound path, port, or certificate has to exist on a laptop.

Cloud inference remains optional. When it is enabled, a hosted worker runs the
same agent a laptop does and claims work with a **service credential**
(ADR 0003). There is no separate inference service to deploy.

---

## 1. Supabase (once per environment)

1. Create a Supabase project (database name `postgres`).
2. Create a **private** Storage bucket `document-media`.
3. Provision database service roles and run migrations:

```bash
cp nomikos/backend/core/.env.supabase.example nomikos/backend/core/.env.supabase
# Store credentials only in the provider's secret manager.
# Follow docs/deployment/database-roles.md before this first migration.
./scripts/platform/migrate_supabase.sh
```

1. Seed production data if needed (admin user, model catalog) - do **not** run dev seed scripts in production.

---

## 2. Vercel projects

Connect the same GitHub repo to **three** Vercel projects. Set the **Root Directory** per project (Project Settings → General).

### Landing (`nomikos-landing`)

| Setting | Value |
| --------- | ------- |
| Root Directory | `landing` |
| Framework | Other (static) |
| Build Command | *(empty)* |
| Output Directory | `.` |
| Domain | `nomikos.app`, `www.nomikos.app` → redirect to apex |

Config: [`landing/vercel.json`](../../landing/vercel.json).

### App (`nomikos-app`)

| Setting | Value |
| --------- | ------- |
| Root Directory | `nomikos/frontend` |
| Framework | Next.js |
| Build Command | `npm run build` |
| Output Directory | *(default)* |
| Domain | `app.nomikos.app` |

Environment variables (Production):

```bash
NEXT_PUBLIC_API_BASE_URL=https://api.nomikos.app
NEXT_PUBLIC_CSRF_COOKIE_NAME=greekocr-csrf
NEXT_PUBLIC_ENABLE_TEST_JOBS=false
```

The editor needs no address for the agent. It learns whether a researcher's
machine can take work from **capacity** on the account's execution-target
response, which is the same fact submission uses to fix an **execution
target** - not from anything the browser reaches.

Template: [`nomikos/frontend/.env.production.example`](../../nomikos/frontend/.env.production.example).

Config: [`nomikos/frontend/vercel.json`](../../nomikos/frontend/vercel.json).

### Platform API (`nomikos-api`)

| Setting | Value |
| --------- | ------- |
| Root Directory | `infrastructure/platform` |
| Install Command | *(empty / default)* |
| Build Command | `bash build.sh` |
| Output Directory | `.` |
| Domain | `api.nomikos.app` |
| Function region | `fra1` (Frankfurt, Europe) |

Environment variables: copy from [`nomikos/backend/core/.env.production.example`](../../nomikos/backend/core/.env.production.example).

**Critical serverless settings:**

| Variable | Production value | Why |
| ---------- | ------------------ | ----- |
| `JOB_WORKER_ENABLED` | `false` | Worker runs on persistent host |
| `JOB_SSE_NOTIFICATIONS_ENABLED` | `false` | NOTIFY listener needs long-lived process |
| `BEHIND_PROXY` | `false` (current Vercel deployment) | Forwarded headers are not trusted without a fixed proxy allowlist |
| `FORWARDED_ALLOW_IPS` | Unset (current Vercel deployment) | Set explicit IP/CIDRs before enabling `BEHIND_PROXY`; never `*` |
| `TRUST_PEER_IP` | `false` (current Vercel deployment) | The peer is the platform proxy, so IP-keyed throttles are skipped rather than made global - [`docs/security/rate-limiting.md`](../security/rate-limiting.md) |
| `CORS_ORIGINS` | `https://app.nomikos.app` | Browser origin |
| `STORAGE_BACKEND` | `supabase` | No local filesystem on Vercel |

Job progress in the browser falls back to **HTTP polling** when SSE is unavailable (already implemented in the frontend).

`FORWARDED_ALLOW_IPS` accepts only explicit proxy IPs or CIDRs. Do not trust
forwarded headers on Vercel unless the request reaches the function from a
stable, allowlisted proxy address; otherwise set `BEHIND_PROXY=false`.

With `BEHIND_PROXY=false` the direct connection peer is Vercel's proxy, not the
browser, so **per-IP rate limiting is inoperative on this deployment**. Set
`TRUST_PEER_IP=false` so IP-keyed buckets are skipped rather than collapsed into
one global bucket that would 429 real users on each other's traffic.
`/auth/login` and `/auth/register` stay capped per targeted account, which does
not depend on the network path. Full reasoning and the remaining gap:
[`docs/security/rate-limiting.md`](../security/rate-limiting.md).

Config: [`infrastructure/platform/vercel.json`](../../infrastructure/platform/vercel.json).

The `fra1` setting applies to the serverless platform API only. The landing
page and frontend SPA remain globally edge-served by Vercel. Validate API p95
latency against the Supabase project region after deployment; rollback by
removing `regions` from [`infrastructure/platform/vercel.json`](../../infrastructure/platform/vercel.json).

Vercel-specific Python runtime, bundle-size, and dependency notes:
[`vercel-platform-api.md`](vercel-platform-api.md).

### Transactional email (Resend)

Registration sends a welcome mail and each login sends a sign-in notice, both
after the response via FastAPI `BackgroundTasks`. Without a key the platform
sends nothing and logs instead, so this section is optional until real mail
is wanted.

1. In [resend.com](https://resend.com), verify the sending domain used in `EMAIL_FROM`.
2. Create an API key (`re_...`) and a Coolify API token (Coolify: Keys & Tokens,
   abilities read, write and deploy).
3. Put the secrets in `~/.nomikos-resend.env` (chmod 600, plain `KEY=VALUE`
   lines, never committed). Placeholder example:

   ```text
   RESEND_API_KEY=re_xxxxxxxxxxxx
   COOLIFY_URL=https://coolify.example.com:8000
   COOLIFY_TOKEN=1|xxxxxxxxxxxx
   ```

   `COOLIFY_URL` is the Coolify dashboard origin. Optional overrides in the
   same file: `EMAIL_FROM`, `EMAIL_REPLY_TO`, `APP_PUBLIC_URL`,
   `EMAIL_SIGNIN_NOTICES`, `APP_NAME`. Use `NOMIKOS_SECRETS_FILE` for another path.
4. From your own terminal run `scripts/platform/apply_email_env.sh --check`
   (validates the file, contacts nothing), `--dry-run` (masked plan, contacts
   nothing), then without flags to apply. `bash tests/platform/test_apply_email_env.sh`
   runs all three modes against a stub HTTP server, never against Coolify.

Hosts: the API is the Coolify application on `root@65.21.249.167`. The ssh alias
`nomikos` (95.217.208.209) is the inference box and needs no email variables.
The script finds the API application in Coolify (name contains `api`, override
with `APP_NAME`), sets the variables below through the Coolify REST API, and
redeploys, which is required because Coolify injects variables at deploy time.
**The redeploy rebuilds from the head of the tracked branch**, so every commit
pushed there since the last deploy ships too: the script prints the branch and
commit Coolify reports and waits 5 seconds for ctrl+c before it redeploys. It
never uses ssh, and the key travels to curl on stdin, never on the command line. **Unverified against production:** nothing on the Coolify server
has been inspected, so confirm the application and variables in Coolify after the
first run. Manual fallback: set the variables by hand in Coolify, Projects,
the API application, Environment Variables, then Redeploy.

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `RESEND_API_KEY` | *(unset)* | Enables sending; when unset, mail is logged, not sent |
| `EMAIL_FROM` | `Nomikos <no-reply@nomikos.app>` | Sender identity, must use a verified domain |
| `EMAIL_REPLY_TO` | *(unset)* | A mailbox someone reads. Named in both mails when set; when unset, the welcome mail invites no reply, the sign-in notice says to contact the project owner, and the env script does not write it |
| `APP_PUBLIC_URL` | `https://app.nomikos.app` | The app origin (not the `nomikos.app` landing site): the "Open Nomikos" button and the hosted logo `/email/nomikos-logo.png` |
| `EMAIL_SIGNIN_NOTICES` | `true` | Set `false` to stop sign-in notices (welcome mail still sends) |

Preview the templates without a key: `python scripts/platform/preview_emails.py <out_dir>`
renders `welcome.html`, `welcome.txt`, `signin.html`, `signin.txt`.

The welcome mail carries no username (registration is unverified, so it must not
relay caller text); the sign-in notice greets by username, which registration
limits to one line without control characters. Logs name the user id, never the
address: `email_sent id=<resend id> user_id=...` on each send, and
`email_send_failed ... error=<exception class>` on a failure.

---

## 3. Optional cloud inference + platform worker (persistent host)

Skip the agent half of this section for the default deployment, where the only
machines running models are researchers' own.

Minimum services:

| Service | Command | Port |
|---------|---------|------|
| `platform-worker` | `python -m backend.jobs.worker_main` | - |
| inference agent | `nomikos run`, one per host that should run models | - |

Cloud inference is currently disabled. Enabling it means standing up a hosted
inference agent, not a service: it installs the same
`nomikos-inference` package a laptop does, presents a **service credential**
in `NOMIKOS_SERVICE_TOKEN` instead of a device token, claims pages from
`api.nomikos.app`, and reports results to
`https://api.nomikos.app/internal/inference/job-complete`, exactly as a paired
laptop does. Set `CLOUD_INFERENCE_ENABLED=true` on the API so it fails closed
without `INFERENCE_WEBHOOK_SECRET`. Full runbook (install, credential, systemd,
ops): [`cloud-inference-worker.md`](cloud-inference-worker.md).

Use the distinct API and platform-worker database principals from
[`database-roles.md`](database-roles.md); do not give these containers the
Supabase operator/migration URI. The agent needs no database access at all.

---

## 4. DNS checklist

| Record | Type | Target |
| -------- | ------ | -------- |
| `nomikos.app` | A / CNAME | Vercel landing project |
| `www` | CNAME | Vercel (redirect to apex) |
| `app` | CNAME | Vercel app project |
| `api` | CNAME | Vercel API project |

A hosted inference agent needs no record: it calls `api.nomikos.app` and is
never called.

---

## 5. Local inference (researcher machines)

There is nothing to ship. The **published package** goes to PyPI and
researchers install it themselves:

```bash
uv tool install nomikos-inference
nomikos pair
nomikos run
```

Releasing is `.github/workflows/release.yml`: one wheel, one runner, Trusted
Publishing, no signing secret. The per-OS DMG, zip, and tarball builds it
replaced are gone along with their Developer ID and Authenticode pipelines; see
[`nomikos_inference/README.md`](../../nomikos_inference/README.md#releasing-and-what-that-changed-about-security-patching).

Patching a CVE in the shipped closure is a dependency bump plus a raise of
`INFERENCE_AGENT_MIN_VERSION`, the platform's version floor. Agents below
the floor are refused at the claim endpoint and told to upgrade, so a fix lands
without anyone reinstalling anything, which is the property four-platform frozen
installers could not offer at any price.

Nothing has to be opened, forwarded, or certified on a researcher's machine.
The agent holds a device token scoped to one account, presents it outbound, and
accepts no connection, so a laptop behind any network is a supported
deployment.

---

## 6. Pre-launch checklist

Record the required CI, secret-rotation, migration, smoke-test, metric, and
rollback evidence in the [production release record](release-evidence.md).

### Secrets and history response

Production credentials are configured only in the provider secret store for the
service that consumes them. Committed `*.env.example` files are documentation;
filled `.env*` files remain local and ignored.

If a secret scanner or Git-history review identifies a possible exposure:

1. Inventory affected paths and consumers without copying the value into an
   issue, log, or chat.
2. Create a scoped replacement in the relevant provider secret store.
3. Deploy and verify every consumer with the replacement.
4. Revoke the old credential.
5. Record the rotation date, owner, affected service, and verification result;
   coordinate history remediation separately when the value was tracked.

### Code and security gates

- [ ] Frontend typecheck, lint, build, and tests pass in CI
- [ ] First-party Python tests and Ruff checks pass in CI
- [ ] OpenAPI/generated-client drift check passes
- [ ] Dependency, secret, and container vulnerability scans pass
- [ ] No production credentials exist in the working tree, Git history, or build artifacts
- [ ] `JWT_SECRET` and `INFERENCE_WEBHOOK_SECRET` are unique per environment
- [ ] Upload, inference, callback, authorization, and cross-user isolation tests pass
- [ ] Model checkpoints use a safe loading path, pinned revision, and verified manifest/hash

### Infrastructure and application checks

- [ ] Supabase migrations applied (`alembic upgrade head`)
- [ ] Service-role bootstrap completed and each runtime has only its own DB URL
- [ ] `CORS_ORIGINS` includes only production app origins (`https://app.nomikos.app`) - never the marketing apex, since CORS runs with credentials
- [ ] `TRUST_PEER_IP=false` on Vercel, so auth throttling falls back to the per-account budget
- [ ] `/docs`, `/redoc`, and `/openapi.json` return 404 in production
- [ ] `AUTH_CSRF_COOKIE_DOMAIN=.nomikos.app` so `app.nomikos.app` can read the CSRF cookie for refresh
- [ ] Production verification uses `app.nomikos.app` / `api.nomikos.app` only - never `*.vercel.app` deployment or Preview URLs
- [ ] `ENABLE_TEST_JOB_ROUTES=false`
- [ ] Platform worker running and healthy; a hosted inference agent too if cloud inference is enabled
- [ ] Vercel bundle contains no model weights, local media, `.env` files, or training artifacts
- [ ] Docker images build, import, health-check, and run as non-root where applicable
- [ ] Release checksums, SBOMs, vulnerability scans, and provenance attestations are available
- [ ] Raw/encoded upload limits, decoded pixel limits, job limits, timeouts, and rate limits are enabled
- [ ] Upload a test page image → appears in Supabase Storage as WebP
- [ ] Login/register on `app.nomikos.app`
- [ ] Cloud segment/transcribe job completes end-to-end
- [ ] The page editor shows the three agent commands (install, `nomikos pair`, `nomikos run`), copyable and with no download link
- [ ] `INFERENCE_AGENT_MIN_VERSION` is at or below the published version, so a current agent is not refused at the claim endpoint

### Accessibility and user-flow checks

- [ ] Critical flows work with keyboard navigation only
- [ ] Modal focus is trapped and restored correctly
- [ ] Form errors are descriptive and associated with their fields
- [ ] Public canvas controls have accessible names and keyboard behavior
- [ ] No critical axe or Lighthouse accessibility findings remain

### Deployment and observation

- [ ] Record API p50/p95/p99 latency, error rate, request volume, and `/health` response before deployment
- [ ] Deploy a preview/staging build and run the smoke checks before production
- [ ] Deploy `api.nomikos.app` with function region `fra1`
- [ ] Confirm Vercel logs show the expected region and contain no secrets or submitted payloads
- [ ] During the first hour, verify health, login, upload, job submission, polling, storage, and export
- [ ] Compare European p95 latency and error rate with the pre-deployment baseline
- [ ] Keep the prior deployment available and record the rollback operator

Advance only when error rate stays within 10% of baseline and p95 latency stays
within 20% of baseline. Hold and investigate if either metric is further above
baseline. Roll back for an error rate above 2× baseline, p95 latency above 50%
baseline, a security issue, data-integrity problem, cross-user exposure, or a
broken critical flow.

### Rollback

1. Restore the last known-good Vercel deployment.
2. If the issue is specific to regional placement, remove `regions` from
   [`infrastructure/platform/vercel.json`](../../infrastructure/platform/vercel.json) and
   redeploy.
3. Re-run `/health` and the critical-flow smoke checks.
4. Compare error rate, latency, job completion, and storage writes with the
   baseline.
5. Record the incident, decision, timestamps, and any required follow-up in
   the relevant issue or [`docs/guides/learnings.md`](../guides/learnings.md).

---

## Troubleshooting

| Symptom | Likely cause |
| --------- | ---------------- |
| CORS error from app | `CORS_ORIGINS` missing `https://app.nomikos.app` |
| 401 on all API calls | Wrong `JWT_SECRET` or clock skew |
| Jobs stuck in `pending` | Platform worker not running (`JOB_WORKER_ENABLED` must be `false` on API, worker elsewhere) |
| Jobs stuck in `waiting` | No inference agent claiming for that **execution target** |
| Agent exits with 426 on its first claim | `INFERENCE_AGENT_MIN_VERSION` is above the installed version; upgrade the agent or lower the **version floor** |
| `DuplicatePreparedStatementError` | Using pooler URL without `statement_cache_size=0` - already handled in `infrastructure/db.py` |
| Media 404 | `STORAGE_BACKEND=supabase` but bucket/key wrong |
| SSE never connects | Expected on Vercel; polling fallback should still complete jobs |

## Usage report

`scripts/platform/sql/usage_report.sql` is one read-only report (SELECTs only): accounts, per-user
activity, content totals, jobs by kind, status and model, job durations, failures, jobs per day and
database size. Run it from your own terminal with `scripts/platform/usage_report.sh [--days N]`;
it reads the SQL file next to it (`USAGE_SQL` overrides), asks for the SSH password once and saves
the output to `~/nomikos-usage/report-<date>.txt`. Use `--local <dbname>` to try it against the
local Postgres first. Read-only is enforced by the database: psql runs with
`PGOPTIONS='-c default_transaction_read_only=on'`, so a write in the SQL fails; the keyword check
in the script is only an early refusal. `bash tests/platform/test_usage_report.sh` checks this
with stub `ssh` and `docker`.

## Release 2026-10-11, one script

`scripts/platform/release_2026_10_11.sh` applies the PR #165 release end to end. It runs as root
on the Coolify host (`root@65.21.249.167`), uses no ssh, and must start BEFORE the merge: production
never runs alembic, so the schema has to be in place before Coolify deploys the new API.

From your laptop, then on the host:

```bash
# 1. secrets file (mode 600): RESEND_API_KEY, COOLIFY_URL, COOLIFY_TOKEN, ADMIN_EMAIL
#    (optional: EMAIL_FROM, EMAIL_REPLY_TO, APP_PUBLIC_URL, EMAIL_SIGNIN_NOTICES, APP_NAME)
chmod 600 ~/.nomikos-release.env && scp -p ~/.nomikos-release.env root@65.21.249.167:/root/.nomikos-release.env
# 2. on the host: the release branch
ssh root@65.21.249.167
git clone --depth 1 --branch pf/integration https://github.com/kkkamur07/nomikos.git /root/nomikos-release
# 3. look first, then run (it asks once, then waits for you to merge PR #165)
bash /root/nomikos-release/scripts/platform/release_2026_10_11.sh --dry-run
bash /root/nomikos-release/scripts/platform/release_2026_10_11.sh
```

What it does, phase by phase (each phase is re-runnable and says "already applied" when there is
nothing left to do; `--from-phase N` resumes after a failure, `--help` lists the exit codes):

1. Preflight: checks the secrets file, finds the Postgres container, the production database (the one
   whose catalog has `artifact_ref` `registry://greek-calamari-v1...`) and the API container, reads
   the Coolify application, prints the plan and asks `Apply? [y/N]` (`--yes` skips the question).
2. SQL: alembic 007 check (applies 007 by hand at 006), the catalog files, `users.is_admin` plus the
   promotion of `ADMIN_EMAIL`, `blocks.label`, the `feedback` table, then the full-page block cleanup.
   Stops unless `alembic_version` reads `010_feedback` afterwards. The catalog rename refuses while a
   job on `syriac-calamari-v2` is queued or running (exit 5): wait or cancel, then run it again.
3. Env: writes the email variables on the Coolify application through `apply_email_env.sh --no-deploy`.
4. Merge gate: tells you to merge PR #165 and polls the GitHub API (or press Enter once merged).
5. Deploy wait: waits for a new healthy API container that carries `010_feedback.py`; after 20 minutes
   it offers to trigger the deploy through the Coolify API.
6. Renumber: `renumber_lines.py` inside the new API container, dry run, then `--apply`.
7. Verify: alembic version, catalog names, admin flag, feedback table, `RESEND_API_KEY` present in the
   container, `/health` inside and on `https://api.nomikos.app`; then it shreds the secrets file
   (`--keep-secrets` keeps it).

Everything it prints also goes to `/root/nomikos-release-2026-10-11.log`, with the key and token
masked. `bash tests/platform/test_release_2026_10_11.sh` runs it against a scratch database, a stub
Coolify and GitHub, and a fake docker.

Manual fallback, the same steps by hand (as the PR #165 body lists them), against the production
database with `psql -v ON_ERROR_STOP=1`, before the merge:

0. `alembic_version` must read `007_line_character_confidences` (at `006_public_sharing`, add `line_transcriptions.character_confidences JSONB` and stamp 007 first)
   and the catalog must already hold the 2026-09-20 rows (`catalog/README.md`, section Order on production)
1. `scripts/platform/catalog/2026-10-11_model_names_v2.sql` (once; it refuses a second run)
2. `scripts/platform/sql/2026-10-11_user_is_admin.sql`, then
   `UPDATE users SET is_admin = true WHERE lower(email) = lower('<you>');`
3. `scripts/platform/sql/2026-10-11_block_label.sql`, `scripts/platform/sql/2026-10-11_feedback.sql`,
   then `scripts/platform/sql/2026-10-11_drop_full_page_blocks.sql`
4. Env variables: `scripts/platform/apply_email_env.sh` (add `--no-deploy` before the merge)
5. After the deploy: `renumber_lines.py` in the API container, dry run, then `--apply`

`scripts/platform/apply_email_env.sh` (HTTPS to Coolify) and `scripts/platform/apply_catalog_sql.sh`
(one ssh connection) remain the laptop-side alternatives for the env and catalog steps.

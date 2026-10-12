#!/usr/bin/env bash
# Release 2026-10-11 (PR #165): one script, run as root ON the Coolify host
# (root@65.21.249.167), that applies the whole release in order:
#
#   1. preflight   finds Postgres, the production database and the API container,
#                  reads the Coolify application, prints the plan, asks once
#   2. sql         alembic 007 check, catalog rename, is_admin (+ promote ADMIN_EMAIL),
#                  blocks.label, feedback table, drop full-page blocks
#   3. env         writes the email variables on the Coolify application, no deploy
#   4. merge gate  waits until PR #165 is merged (GitHub API, or Enter)
#   5. deploy      waits for the new API container (healthy, carries 010_feedback.py)
#   6. renumber    renumber_lines.py inside the new API container, dry run then --apply
#   7. verify      prints a summary, then removes the secrets file
#
# Production never runs alembic (the API image CMD is uvicorn only), so the schema
# must be in place BEFORE the merge deploys the new API. Every phase is re-runnable:
# a second run finds nothing to do and says "already applied".
#
# Secrets: ${NOMIKOS_SECRETS_FILE:-/root/.nomikos-release.env}, mode 600, KEY=VALUE
# lines, parsed line by line, never sourced:
#   RESEND_API_KEY=re_...      required
#   COOLIFY_URL=https://...    required (Coolify dashboard origin)
#   COOLIFY_TOKEN=...          required (read, write, deploy abilities)
#   ADMIN_EMAIL=you@...        required (account promoted to admin)
#   EMAIL_FROM, EMAIL_REPLY_TO, APP_PUBLIC_URL, EMAIL_SIGNIN_NOTICES, APP_NAME  optional
# The key and token are never printed; the log is masked as well.
#
# Env overrides: PG_CONTAINER, PG_DATABASE, PG_USER, API_CONTAINER, API_MATCH,
# DOCKER_BIN, GITHUB_API_URL, PUBLIC_HEALTH_URL, RELEASE_LOG, MERGE_POLL_SECONDS,
# DEPLOY_POLL_SECONDS, MERGE_TIMEOUT_SECONDS, DEPLOY_TIMEOUT_SECONDS.
# RELEASE_ALLOW_NONROOT=1 skips the root check (tests only).
#
# tests/platform/test_release_2026_10_11.sh runs it against a scratch database,
# a stub Coolify and GitHub, and a fake docker for the API container.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_FILE="${NOMIKOS_SECRETS_FILE:-/root/.nomikos-release.env}"
LOG_FILE="${RELEASE_LOG:-/root/nomikos-release-2026-10-11.log}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
GITHUB_API_URL="${GITHUB_API_URL:-https://api.github.com/repos/kkkamur07/nomikos/pulls/165}"
PUBLIC_HEALTH_URL="${PUBLIC_HEALTH_URL:-https://api.nomikos.app/health}"
# 60 s: unauthenticated GitHub API calls are limited to 60 per hour per IP, so a
# faster poll hits 403 within the hour and falls back to "press Enter".
MERGE_POLL_SECONDS="${MERGE_POLL_SECONDS:-60}"
MERGE_TIMEOUT_SECONDS="${MERGE_TIMEOUT_SECONDS:-3600}"
DEPLOY_POLL_SECONDS="${DEPLOY_POLL_SECONDS:-15}"
DEPLOY_TIMEOUT_SECONDS="${DEPLOY_TIMEOUT_SECONDS:-1200}"
NEW_CODE_MARKER="/app/infrastructure/alembic/versions/010_feedback.py"
EXPECTED_NAMES="armenian-v1 coptic-v1 greek-v1 seg-v1 seg-v2 syriac-v2"

usage() {
  cat <<'EOF'
Usage: bash scripts/platform/release_2026_10_11.sh [flags]

Run as root on the Coolify host. Applies release 2026-10-11 (PR #165): SQL,
Coolify env vars, waits for the merge and the deploy, renumbers lines, verifies.

Flags:
  --dry-run          preflight and read-only checks only; GET requests only, writes nothing
  --yes              do not ask "Apply? [y/N]" (and trigger the deploy on a deploy timeout)
  --skip-merge-wait  do not wait for PR #165 to be merged (phase 4)
  --keep-secrets     keep the secrets file at the end (default: shred it)
  --from-phase N     resume at phase N (2 sql, 3 env, 4 merge gate, 5 deploy,
                     6 renumber, 7 verify); phase 1 always runs
  --help             this text

Exit codes:
  0   success (or dry run done, or nothing to do)
  1   usage error
  2   preflight failed (not root, no docker/curl/python3, secrets file, Coolify, API container)
  3   production database not found, or more than one candidate
  4   a SQL step failed, or alembic_version is not what the release expects
  5   catalog rename refused: jobs on syriac-calamari-v2 are queued or running
  6   env phase failed (Coolify)
  7   merge gate timed out or was aborted
  8   deploy wait timed out
  9   renumber failed
  10  verification failed
  11  ADMIN_EMAIL matches no user
  12  aborted at the prompt
EOF
}

ORIG_ARGS=("$@")
DRY_RUN=0 YES=0 SKIP_MERGE_WAIT=0 KEEP_SECRETS=0 FROM_PHASE=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes) YES=1 ;;
    --skip-merge-wait) SKIP_MERGE_WAIT=1 ;;
    --keep-secrets) KEEP_SECRETS=1 ;;
    --from-phase)
      [ "$#" -ge 2 ] || { echo "--from-phase needs a number" >&2; exit 1; }
      FROM_PHASE="$2"; shift ;;
    --from-phase=*) FROM_PHASE="${1#*=}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown flag: $1 (see --help)" >&2; exit 1 ;;
  esac
  shift
done
case "$FROM_PHASE" in [1-7]) ;; *) echo "--from-phase must be 1 to 7" >&2; exit 1 ;; esac

command -v python3 >/dev/null || { echo "error: python3 is required" >&2; exit 2; }

# Log everything to LOG_FILE as well. The script runs itself once more behind a
# filter that masks the key and token (read from the secrets file, never argv).
if [ -z "${_RELEASE_INNER:-}" ]; then
  ( umask 077; touch "$LOG_FILE" ) || { echo "error: cannot write the log $LOG_FILE" >&2; exit 2; }
  set +e
  _RELEASE_INNER=1 bash "${BASH_SOURCE[0]}" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"} 2>&1 |
    python3 -u -c '
import sys
secrets = []
try:
    for line in open(sys.argv[1]):
        line = line.rstrip("\r\n")
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'"'"'":
            v = v[1:-1]
        if k.strip() in ("RESEND_API_KEY", "COOLIFY_TOKEN") and len(v) >= 4:
            secrets.append(v)
except OSError:
    pass
for line in iter(sys.stdin.readline, ""):
    for s in secrets:
        line = line.replace(s, "****")
    sys.stdout.write(line)
    sys.stdout.flush()
' "$SECRETS_FILE" | tee -a "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  exit "$rc"
fi

die() { local code=$1; shift; echo "error: $*" >&2; exit "$code"; }
heading() { echo; echo "======== Phase $1: $2"; }
dk() { "$DOCKER_BIN" "$@"; }

check_safe_name() {
  case "$2" in
    ""|*[!A-Za-z0-9_.-]*) die 2 "refusing unsafe $1 name: $2" ;;
  esac
}

echo "release 2026-10-11 started $(date -u '+%Y-%m-%dT%H:%M:%SZ') (log: $LOG_FILE)"
[ "$DRY_RUN" = 0 ] || echo "DRY RUN: nothing is written; Coolify and GitHub get GET requests only."

# ---------------------------------------------------------------- phase 1
heading 1 "preflight"

if [ "$(id -u)" != 0 ] && [ "${RELEASE_ALLOW_NONROOT:-}" != 1 ]; then
  die 2 "run this as root on the Coolify host"
fi
command -v "$DOCKER_BIN" >/dev/null || die 2 "docker not found ($DOCKER_BIN)"
dk ps -q >/dev/null || die 2 "docker does not answer"
command -v curl >/dev/null || die 2 "curl is required"

[ -f "$SECRETS_FILE" ] || die 2 "secrets file not found: $SECRETS_FILE"
PERM=$(stat -c %a "$SECRETS_FILE" 2>/dev/null || stat -f %Lp "$SECRETS_FILE")
case "$PERM" in ?00) ;; *) die 2 "$SECRETS_FILE is mode $PERM; run: chmod 600 $SECRETS_FILE" ;; esac

RESEND_API_KEY="" COOLIFY_TOKEN="" COOLIFY_URL="" ADMIN_EMAIL=""
EMAIL_FROM="Nomikos <no-reply@nomikos.app>"
EMAIL_REPLY_TO=""
APP_PUBLIC_URL="https://app.nomikos.app"
EMAIL_SIGNIN_NOTICES="true"
APP_NAME="api"
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%$'\r'}"
  case "$line" in ''|'#'*) continue ;; esac
  key="${line%%=*}"
  val="${line#*=}"
  key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
  case "$val" in
    \"*\") val="${val#\"}"; val="${val%\"}" ;;
    \'*\') val="${val#\'}"; val="${val%\'}" ;;
  esac
  case "$key" in
    RESEND_API_KEY|COOLIFY_TOKEN|COOLIFY_URL|ADMIN_EMAIL|EMAIL_FROM|EMAIL_REPLY_TO|APP_PUBLIC_URL|EMAIL_SIGNIN_NOTICES|APP_NAME)
      printf -v "$key" '%s' "$val" ;;
  esac
done < "$SECRETS_FILE"
[ -n "$RESEND_API_KEY" ] || die 2 "RESEND_API_KEY is missing from $SECRETS_FILE"
[ -n "$COOLIFY_TOKEN" ] || die 2 "COOLIFY_TOKEN is missing from $SECRETS_FILE"
[ -n "$COOLIFY_URL" ] || die 2 "COOLIFY_URL is missing from $SECRETS_FILE"
[ -n "$ADMIN_EMAIL" ] || die 2 "ADMIN_EMAIL is missing from $SECRETS_FILE"
case "$RESEND_API_KEY" in re_*) ;; *) die 2 "RESEND_API_KEY must start with re_" ;; esac
case "$ADMIN_EMAIL" in *[[:space:]]*|*\'*|*\\*) die 2 "ADMIN_EMAIL looks wrong" ;; *@*) ;; *) die 2 "ADMIN_EMAIL looks wrong" ;; esac
COOLIFY_URL="${COOLIFY_URL%/}"
ENV_KEYS="RESEND_API_KEY EMAIL_FROM APP_PUBLIC_URL EMAIL_SIGNIN_NOTICES"
[ -z "$EMAIL_REPLY_TO" ] || ENV_KEYS="$ENV_KEYS EMAIL_REPLY_TO"
echo "secrets: $SECRETS_FILE (mode $PERM), key re_****, token ****, admin $ADMIN_EMAIL"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
RESP="$WORK/resp"

# coolify METHOD PATH: body to $RESP, HTTP status in $CODE. Call it directly, never
# inside $(...). The token travels in a curl config on a file descriptor, so `ps`
# never shows it.
coolify() {
  CODE=$(curl -sS -m 30 --config <(printf 'header = "Authorization: Bearer %s"\n' "$COOLIFY_TOKEN") \
    -X "$1" -H 'Accept: application/json' -o "$RESP" -w '%{http_code}' "$COOLIFY_URL$2") || CODE=000
}

coolify GET /api/v1/applications
case "$CODE" in 2*) ;; *) die 2 "Coolify: listing applications failed (HTTP $CODE); check COOLIFY_URL and the token" ;; esac
MATCHES=$(APP_NAME="$APP_NAME" python3 -c '
import json, os, sys
needle = os.environ["APP_NAME"].lower()
for a in json.load(sys.stdin):
    if needle in (a.get("name") or "").lower():
        print("\t".join(str(a.get(f) or "") for f in ("uuid", "name", "git_branch", "git_commit_sha")))
' < "$RESP")
COUNT=$(printf '%s\n' "$MATCHES" | grep -c . || true)
[ "$COUNT" = 1 ] || die 2 "Coolify: expected one application whose name contains \"$APP_NAME\", found $COUNT; set APP_NAME in the secrets file"
IFS=$'\t' read -r APP_UUID APP_LABEL GIT_BRANCH GIT_COMMIT <<<"$MATCHES" || true
check_safe_name "Coolify application uuid" "$APP_UUID"

# Postgres: the container, the superuser, and the database that holds the
# production catalog. Identified by artifact_ref, which the rename leaves alone.
DOCKER_PS_FORMAT='{{.ID}}	{{.Names}}	{{.Image}}'
if [ -n "${PG_CONTAINER:-}" ]; then
  PG_CANDIDATES="$PG_CONTAINER"
else
  PG_CANDIDATES=$(dk ps --no-trunc --format "$DOCKER_PS_FORMAT" | awk -F'\t' 'tolower($3) ~ /postgres|pgvector|timescale/ {print $2}')
fi
[ -n "$PG_CANDIDATES" ] || die 3 "no Postgres container on this host"
FOUND=""
for c in $PG_CANDIDATES; do
  check_safe_name container "$c"
  u="${PG_USER:-$(dk exec "$c" printenv POSTGRES_USER 2>/dev/null || echo postgres)}"
  check_safe_name user "$u"
  if [ -n "${PG_DATABASE:-}" ]; then
    dbs="$PG_DATABASE"
  else
    dbs=$(dk exec "$c" psql -X -U "$u" -d postgres -Atc "select datname from pg_database where not datistemplate" </dev/null 2>/dev/null || true)
  fi
  for d in $dbs; do
    check_safe_name database "$d"
    n=$(dk exec "$c" psql -X -U "$u" -d "$d" -Atc "select count(*) from inference_models where artifact_ref like 'registry://greek-calamari-v1%'" </dev/null 2>/dev/null || true)
    if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then FOUND="$FOUND$c|$u|$d"$'\n'; fi
  done
done
NFOUND=$(printf '%s' "$FOUND" | grep -c . || true)
[ "$NFOUND" -ge 1 ] || die 3 "no database holds the production catalog (artifact_ref registry://greek-calamari-v1...)"
if [ "$NFOUND" -gt 1 ]; then
  printf '%s' "$FOUND" >&2
  die 3 "more than one database holds the production catalog; set PG_CONTAINER and PG_DATABASE"
fi
IFS='|' read -r PG_C PG_U PG_D <<<"$(printf '%s' "$FOUND" | head -n 1)"

# pgq SQL: one value per row, unaligned. pgv SQL: same, with :'email' bound to ADMIN_EMAIL.
pgq() { dk exec "$PG_C" psql -X -q -v ON_ERROR_STOP=1 -U "$PG_U" -d "$PG_D" -At -c "$1" </dev/null; }
pgv() { printf '%s\n' "$1" | dk exec -i "$PG_C" psql -X -q -v ON_ERROR_STOP=1 -v email="$ADMIN_EMAIL" -U "$PG_U" -d "$PG_D" -At; }
# pgfile FILE [psql args]: the file travels on stdin, nothing is written on the host.
pgfile() { local f=$1; shift; dk exec -i "$PG_C" psql -X -v ON_ERROR_STOP=1 -U "$PG_U" -d "$PG_D" "$@" < "$f"; }

alembic_version() { pgq "select version_num from alembic_version" | head -n 1; }
catalog_names() { pgq "select coalesce(string_agg(name, ' ' order by name), '') from inference_models"; }
has_name() { case " $(catalog_names) " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
has_column() { [ "$(pgq "select count(*) from information_schema.columns where table_schema = 'public' and table_name = '$1' and column_name = '$2'")" = 1 ]; }
has_feedback() { [ "$(pgq "select to_regclass('public.feedback') is not null")" = t ]; }
admin_state() {
  if ! has_column users is_admin; then
    n=$(pgv "select count(*) from users where lower(email) = lower(:'email');")
    [ "$n" = 0 ] && echo "no such user" || echo "user exists, no is_admin column yet"
  else
    s=$(pgv "select string_agg(is_admin::text, ',') from users where lower(email) = lower(:'email');")
    case "$s" in "") echo "no such user" ;; *true*) echo "admin" ;; *) echo "not admin" ;; esac
  fi
}

# API container: name or image carries the Coolify application uuid (or API_MATCH).
API_MATCH="${API_MATCH:-$APP_UUID}"
api_candidates() { # newest first: "created id name"
  local pat=$1
  dk ps --no-trunc --format "$DOCKER_PS_FORMAT" |
    awk -F'\t' -v m="$pat" -v pg="$PG_C" '$2 != pg && (index($2, m) || index($3, m)) {print $2}' |
    while read -r name; do
      echo "$(dk inspect -f '{{.Created}}|{{.Id}}' "$name" | tr '|' ' ') $name"
    done | sort -r
}
api_list() {
  local out
  out=$(api_candidates "$API_MATCH")
  if [ -z "$out" ] && [ "$API_MATCH" = "$APP_UUID" ]; then out=$(api_candidates api); fi
  printf '%s\n' "$out" | grep . || true
}
has_new_code() { dk exec "$1" test -f "$NEW_CODE_MARKER" >/dev/null 2>&1; }
api_health() { # healthy | starting | unhealthy | none-ok | none-down
  local s
  s=$(dk inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$1" 2>/dev/null || echo gone)
  if [ "$s" = none ]; then
    if dk exec "$1" python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status == 200 else 1)" >/dev/null 2>&1; then
      s=none-ok
    else
      s=none-down
    fi
  fi
  echo "$s"
}

if [ -n "${API_CONTAINER:-}" ]; then
  check_safe_name container "$API_CONTAINER"
  info=$(dk inspect -f '{{.Created}}|{{.Id}}' "$API_CONTAINER") || die 2 "API_CONTAINER $API_CONTAINER not found"
  read -r API_CREATED API_ID <<<"$(printf '%s' "$info" | tr '|' ' ')"
  API_C="$API_CONTAINER"
else
  LIST=$(api_list)
  [ -n "$LIST" ] || die 2 "no API container found (name or image containing $API_MATCH or \"api\"); set API_CONTAINER"
  if [ "$(printf '%s\n' "$LIST" | grep -c .)" -gt 1 ]; then
    echo "several API containers, using the newest:"; printf '%s\n' "$LIST" | sed 's/^/  /'
  fi
  read -r API_CREATED API_ID API_C <<<"$(printf '%s\n' "$LIST" | head -n 1)"
fi
check_safe_name container "$API_C"
API_HAS_NEW=0
has_new_code "$API_C" && API_HAS_NEW=1

# Read the current state for the plan.
VER=$(alembic_version)
NAMES=$(catalog_names)
ADMIN=$(admin_state)
ACTIVE_SYRIAC=$(pgq "select count(*) from jobs where status in ('pending', 'waiting', 'running') and model_id = (select id from inference_models where name = 'syriac-calamari-v2')")

plan_step() { printf '  %-44s %s\n' "$1" "$2"; }
case "$VER" in
  006_public_sharing) S0="will add character_confidences and stamp 007" ;;
  007_line_character_confidences) S0="ok (at 007)" ;;
  008_user_is_admin|009_block_label|010_feedback) S0="already applied (at $VER)" ;;
  *) die 4 "alembic_version is '$VER'; this release expects 006_public_sharing or 007_line_character_confidences" ;;
esac
if has_name seg-v1; then S1="already applied (superseded by step 2)"
elif has_name kraken && has_name ppocr && has_name syriac-ppocr-v1; then S1="already applied"
else S1="will run"; fi
if has_name seg-v1; then S2="already applied"
elif [ "$ACTIVE_SYRIAC" != 0 ]; then S2="will REFUSE: $ACTIVE_SYRIAC queued or running job(s) on syriac-calamari-v2"
else S2="will run"; fi
case "$VER" in 006_*|007_*) S3="will run" ;; *) has_column users is_admin && S3="already applied" || S3="will run" ;; esac
case "$ADMIN" in admin) S3b="already applied" ;; "no such user") S3b="FAILS: no user with this email" ;; *) S3b="will run" ;; esac
case "$VER" in 006_*|007_*|008_*) S4="will run" ;; *) has_column blocks label && S4="already applied" || S4="will run" ;; esac
case "$VER" in 006_*|007_*|008_*|009_*) S5="will run" ;; *) has_feedback && S5="already applied" || S5="will run" ;; esac
if has_column blocks label; then
  # The file with COMMIT turned into ROLLBACK: it counts and writes nothing.
  sed 's/^COMMIT;$/ROLLBACK;/' "$HERE/sql/2026-10-11_drop_full_page_blocks.sql" > "$WORK/fpb_rollback.sql"
  FPB=$(pgfile "$WORK/fpb_rollback.sql" -q -At -F ' ' | head -n 1)
  case "$FPB" in "0 0") S6="already applied (no full-page blocks)" ;; *) S6="will remove ${FPB%% *} block(s), unblock ${FPB##* } line(s)" ;; esac
else
  S6="will run after step 4 (counts then)"
fi

# Env state: Coolify's current values against the ones this release writes.
env_matches() {
  coolify GET "/api/v1/applications/$APP_UUID/envs"
  case "$CODE" in 2*) ;; *) return 1 ;; esac
  KEYS="$ENV_KEYS" RESEND_API_KEY="$RESEND_API_KEY" EMAIL_FROM="$EMAIL_FROM" EMAIL_REPLY_TO="$EMAIL_REPLY_TO" \
    APP_PUBLIC_URL="$APP_PUBLIC_URL" EMAIL_SIGNIN_NOTICES="$EMAIL_SIGNIN_NOTICES" python3 -c '
import json, os, sys
try:
    data = json.load(open(sys.argv[1]))
except ValueError:
    sys.exit(1)
if isinstance(data, dict):
    data = data.get("data") or []
have = {e.get("key"): e.get("value") for e in data if isinstance(e, dict) and not e.get("is_preview")}
sys.exit(0 if all(have.get(k) == os.environ[k] for k in os.environ["KEYS"].split()) else 1)
' "$RESP"
}
if env_matches; then ENV_STATE="already applied"; else ENV_STATE="will write"; fi

GH_STATE=unknown
gh_merged() { # prints true | false | ratelimited | unavailable
  local code
  code=$(curl -sS -m 15 -H 'Accept: application/vnd.github+json' -o "$WORK/pr.json" -w '%{http_code}' "$GITHUB_API_URL" 2>/dev/null) || code=000
  case "$code" in
    200) python3 -c 'import json, sys; print("true" if json.load(open(sys.argv[1])).get("merged") else "false")' "$WORK/pr.json" 2>/dev/null || echo unavailable ;;
    403|429) echo ratelimited ;;
    *) echo unavailable ;;
  esac
}
GH_STATE=$(gh_merged)

echo
echo "Plan"
plan_step "Postgres" "container $PG_C, user $PG_U, database $PG_D"
plan_step "alembic_version" "$VER"
plan_step "inference_models" "$NAMES"
plan_step "ADMIN_EMAIL $ADMIN_EMAIL" "$ADMIN"
plan_step "API container" "$API_C (created $API_CREATED, new code: $([ "$API_HAS_NEW" = 1 ] && echo yes || echo no))"
plan_step "Coolify application" "$APP_LABEL ($APP_UUID), branch ${GIT_BRANCH:-?}, commit ${GIT_COMMIT:-?}"
plan_step "PR #165 merged" "$GH_STATE"
echo
echo "SQL steps (phase 2)"
plan_step "0 alembic 007 check" "$S0"
plan_step "1 catalog 2026-09-20 rows" "$S1"
plan_step "2 catalog rename (model_names_v2)" "$S2"
plan_step "3 users.is_admin (stamps 008)" "$S3"
plan_step "3b promote ADMIN_EMAIL" "$S3b"
plan_step "4 blocks.label (stamps 009)" "$S4"
plan_step "5 feedback table (stamps 010)" "$S5"
plan_step "6 drop full-page blocks" "$S6"
echo
echo "Env keys on $APP_LABEL (phase 3, no deploy): $ENV_STATE"
for k in $ENV_KEYS; do
  case "$k" in RESEND_API_KEY) plan_step "  $k" "re_****" ;; *) plan_step "  $k" "${!k}" ;; esac
done
[ -n "$EMAIL_REPLY_TO" ] || plan_step "  EMAIL_REPLY_TO" "(unset, not written)"
echo
echo "Phases to run: $FROM_PHASE to 7$([ "$SKIP_MERGE_WAIT" = 1 ] && echo ", merge wait skipped")"

case "$S3b" in FAILS*) die 11 "ADMIN_EMAIL $ADMIN_EMAIL matches no user; fix the secrets file before anything is applied" ;; esac

remove_secrets() {
  if [ "$KEEP_SECRETS" = 1 ]; then echo "secrets file kept (--keep-secrets): $SECRETS_FILE"; return; fi
  if command -v shred >/dev/null; then shred -u "$SECRETS_FILE"; else rm -f "$SECRETS_FILE"; fi
  echo "secrets file removed: $SECRETS_FILE"
}

if [ "$DRY_RUN" = 1 ]; then
  heading 7 "verify (read-only, dry run)"
  echo "  alembic_version   $VER"
  echo "  inference_models  $NAMES"
  echo "  admin             $ADMIN"
  echo "  feedback table    $(has_feedback && echo yes || echo no)"
  echo "  API /health       $(api_health "$API_C")"
  echo "  public health     $(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$PUBLIC_HEALTH_URL" 2>/dev/null || echo unreachable)"
  echo
  echo "Dry run done. Nothing was written."
  exit 0
fi

if [ "$YES" != 1 ]; then
  echo "Apply? [y/N]"
  read -r answer || answer=""
  case "$answer" in y|Y|yes|YES) ;; *) echo "aborted, nothing applied"; exit 12 ;; esac
fi

# ---------------------------------------------------------------- phase 2
if [ "$FROM_PHASE" -le 2 ]; then
  heading 2 "sql"

  VER=$(alembic_version)
  case "$VER" in
    006_public_sharing)
      echo "step 0: at 006, applying migration 007 by hand"
      printf '%s\n' "BEGIN;" \
        "ALTER TABLE line_transcriptions ADD COLUMN IF NOT EXISTS character_confidences JSONB;" \
        "UPDATE alembic_version SET version_num = '007_line_character_confidences' WHERE version_num = '006_public_sharing';" \
        "COMMIT;" > "$WORK/s0.sql"
      pgfile "$WORK/s0.sql" || die 4 "step 0 failed"
      [ "$(alembic_version)" = 007_line_character_confidences ] || die 4 "step 0: alembic_version is $(alembic_version), expected 007"
      ;;
    007_line_character_confidences) echo "step 0: ok (at 007)" ;;
    008_user_is_admin|009_block_label|010_feedback) echo "step 0: already applied (at $VER)" ;;
    *) die 4 "alembic_version is '$VER'; expected 006_public_sharing or 007_line_character_confidences" ;;
  esac

  F1="$HERE/catalog/2026-09-20-ppocr-rows-and-segmenter-display-names.sql"
  if has_name seg-v1; then echo "step 1: already applied (superseded by step 2)"
  elif has_name kraken && has_name ppocr && has_name syriac-ppocr-v1; then echo "step 1: already applied"
  else
    echo "step 1: $F1"
    pgfile "$F1" || die 4 "step 1 failed"
  fi

  F2="$HERE/catalog/2026-10-11_model_names_v2.sql"
  if has_name seg-v1; then echo "step 2: already applied"
  else
    echo "step 2: $F2"
    if ! pgfile "$F2" 2>"$WORK/s2.err"; then
      cat "$WORK/s2.err"
      if grep -q "queued or running" "$WORK/s2.err"; then
        die 5 "step 2 refused: jobs on syriac-calamari-v2 are queued or running. Wait for them to finish or cancel them, then re-run this script (nothing after step 1 was applied)."
      fi
      die 4 "step 2 failed (nothing in it was applied)"
    fi
    cat "$WORK/s2.err"
  fi

  F3="$HERE/sql/2026-10-11_user_is_admin.sql"
  if [ "$(alembic_version)" != 007_line_character_confidences ] && has_column users is_admin; then
    echo "step 3: already applied"
  else
    echo "step 3: $F3"
    pgfile "$F3" || die 4 "step 3 failed"
  fi
  case "$(admin_state)" in
    admin) echo "step 3b: already applied ($ADMIN_EMAIL is admin)" ;;
    "no such user") die 11 "ADMIN_EMAIL $ADMIN_EMAIL matches no user" ;;
    *)
      n=$(pgv "with u as (update users set is_admin = true where lower(email) = lower(:'email') returning 1) select count(*) from u;")
      [ "${n:-0}" -ge 1 ] || die 11 "promoting $ADMIN_EMAIL matched 0 rows"
      echo "step 3b: promoted $ADMIN_EMAIL to admin ($n row)"
      ;;
  esac

  F4="$HERE/sql/2026-10-11_block_label.sql"
  case "$(alembic_version)" in
    007_*|008_*) run4=1 ;;
    *) if has_column blocks label; then run4=0; else run4=1; fi ;;
  esac
  if [ "$run4" = 0 ]; then echo "step 4: already applied"; else echo "step 4: $F4"; pgfile "$F4" || die 4 "step 4 failed"; fi

  F5="$HERE/sql/2026-10-11_feedback.sql"
  case "$(alembic_version)" in
    007_*|008_*|009_*) run5=1 ;;
    *) if has_feedback; then run5=0; else run5=1; fi ;;
  esac
  if [ "$run5" = 0 ]; then echo "step 5: already applied"; else echo "step 5: $F5"; pgfile "$F5" || die 4 "step 5 failed"; fi

  VER=$(alembic_version)
  [ "$VER" = 010_feedback ] || die 4 "after steps 3 to 5 alembic_version is '$VER', expected 010_feedback"
  echo "alembic_version: $VER"

  F6="$HERE/sql/2026-10-11_drop_full_page_blocks.sql"
  out=$(pgfile "$F6" -q -At -F ' ') || die 4 "step 6 failed"
  out=$(printf '%s\n' "$out" | head -n 1)
  if [ "$out" = "0 0" ]; then
    echo "step 6: already applied (no full-page blocks)"
  else
    echo "step 6: removed ${out%% *} full-page block(s), unblocked ${out##* } line(s)"
  fi
  echo "inference_models: $(catalog_names)"
fi

# ---------------------------------------------------------------- phase 3
if [ "$FROM_PHASE" -le 3 ]; then
  heading 3 "env (Coolify, no deploy)"
  if env_matches; then
    echo "env: already applied ($ENV_KEYS)"
  else
    NOMIKOS_SECRETS_FILE="$SECRETS_FILE" bash "$HERE/apply_email_env.sh" --no-deploy || die 6 "setting the env variables failed"
    echo "env: written ($ENV_KEYS); they reach the container with the merge deploy"
  fi
fi

# ---------------------------------------------------------------- phase 4
if [ "$FROM_PHASE" -le 4 ]; then
  heading 4 "merge gate"
  echo "SQL and env are in place. Merge PR #165 on GitHub now."
  if [ "$SKIP_MERGE_WAIT" = 1 ]; then
    echo "merge wait skipped (--skip-merge-wait)"
  else
    start=$(date +%s)
    while :; do
      state=$(gh_merged)
      if [ "$state" = true ]; then echo "PR #165 is merged."; break; fi
      if [ "$state" = ratelimited ] || [ "$state" = unavailable ]; then
        [ "$state" = unavailable ] || echo "GitHub rate limit reached."
        echo "GitHub API not readable ($GITHUB_API_URL). Press Enter once PR #165 is merged."
        read -r _ || die 7 "no input; re-run with --from-phase 5 after the merge"
        echo "continuing as merged"
        break
      fi
      [ $(( $(date +%s) - start )) -lt "$MERGE_TIMEOUT_SECONDS" ] ||
        die 7 "PR #165 not merged after $MERGE_TIMEOUT_SECONDS s; re-run with --from-phase 4"
      echo "not merged yet; checking again in $MERGE_POLL_SECONDS s (or press Enter once merged)"
      if [ -t 0 ]; then
        if read -r -t "$MERGE_POLL_SECONDS" _; then echo "Enter pressed: continuing as merged"; break; fi
      else
        sleep "$MERGE_POLL_SECONDS"
      fi
    done
  fi
fi

# ---------------------------------------------------------------- phase 5
if [ "$FROM_PHASE" -le 5 ]; then
  heading 5 "deploy wait"
  # wait_for_new SECONDS: sets API_C to a healthy API container running the new code.
  wait_for_new() {
    local limit=$1 start created id name h
    start=$(date +%s)
    while :; do
      while read -r created id name; do
        [ -n "$name" ] || continue
        # A container is new when it differs from the one preflight saw and was
        # created later (seconds precision), or preflight's already had the new code.
        if [ "$id" = "$API_ID" ] && [ "$API_HAS_NEW" != 1 ]; then continue; fi
        if [ "$id" != "$API_ID" ] && [ "${created:0:19}" \< "${API_CREATED:0:19}" ]; then continue; fi
        has_new_code "$name" || { echo "  $name: old code"; continue; }
        h=$(api_health "$name")
        case "$h" in
          healthy|none-ok) check_safe_name container "$name"; API_C="$name"; echo "  $name: healthy, new code"; return 0 ;;
          *) echo "  $name: new code, health $h" ;;
        esac
      done <<<"$(api_list)"
      [ $(( $(date +%s) - start )) -lt "$limit" ] || return 1
      echo "waiting for the new API container; next check in $DEPLOY_POLL_SECONDS s"
      sleep "$DEPLOY_POLL_SECONDS"
    done
  }
  if [ "$API_HAS_NEW" = 1 ] && case "$(api_health "$API_C")" in healthy|none-ok) true ;; *) false ;; esac; then
    echo "deploy: already applied ($API_C runs the new code)"
  elif wait_for_new "$DEPLOY_TIMEOUT_SECONDS"; then
    echo "deploy: done ($API_C)"
  else
    echo "no new healthy API container after $DEPLOY_TIMEOUT_SECONDS s."
    trigger=0
    if [ "$YES" = 1 ]; then trigger=1
    else
      echo "Trigger the deploy through the Coolify API now? [y/N]"
      read -r answer || answer=""
      case "$answer" in y|Y|yes|YES) trigger=1 ;; esac
    fi
    [ "$trigger" = 1 ] || die 8 "deploy wait timed out; re-run with --from-phase 5"
    coolify GET "/api/v1/deploy?uuid=$APP_UUID&force=true"
    case "$CODE" in 2*) echo "deploy triggered" ;; *) die 8 "triggering the deploy failed (HTTP $CODE)" ;; esac
    wait_for_new "$DEPLOY_TIMEOUT_SECONDS" || die 8 "deploy wait timed out twice; re-run with --from-phase 5"
    echo "deploy: done ($API_C)"
  fi
fi

# ---------------------------------------------------------------- phase 6
if [ "$FROM_PHASE" -le 6 ]; then
  heading 6 "renumber lines"
  SURL=$(dk exec "$API_C" printenv SYNC_DATABASE_URL 2>/dev/null || true)
  if [ -z "$SURL" ]; then
    SURL=$(dk exec "$API_C" printenv DATABASE_URL 2>/dev/null || true)
    SURL="${SURL/+asyncpg/}"
  fi
  [ -n "$SURL" ] || die 9 "neither SYNC_DATABASE_URL nor DATABASE_URL is set in $API_C"
  where=$(printf '%s' "$SURL" | python3 -c '
import sys, urllib.parse as u
p = u.urlsplit(sys.stdin.read().strip())
print("%s:%s/%s" % (p.hostname, p.port or 5432, p.path.lstrip("/")))')
  echo "database: $where (from the API container env)"
  # The URL reaches docker through its environment, never argv.
  renumber() { SYNC_DATABASE_URL="$SURL" "$DOCKER_BIN" exec -e SYNC_DATABASE_URL "$API_C" python /app/scripts/platform/renumber_lines.py "$@"; }
  out=$(renumber 2>&1) || { printf '%s\n' "$out"; die 9 "renumber dry run failed"; }
  printf '%s\n' "$out" | sed 's/^/  /'
  if printf '%s\n' "$out" | grep -q "would renumber 0 line"; then
    echo "renumber: already applied (0 lines to change)"
  else
    out=$(renumber --apply 2>&1) || { printf '%s\n' "$out"; die 9 "renumber --apply failed"; }
    printf '%s\n' "$out" | sed 's/^/  /'
    echo "renumber: applied"
  fi
fi

# ---------------------------------------------------------------- phase 7
heading 7 "verify"
FAILED=0
verdict() { if [ "$2" = ok ]; then printf '  ok    %-22s %s\n' "$1" "$3"; else printf '  FAIL  %-22s %s\n' "$1" "$3"; FAILED=1; fi; }
VER=$(alembic_version)
if [ "$VER" = 010_feedback ]; then verdict alembic_version ok "$VER"; else verdict alembic_version fail "$VER"; fi
NAMES=$(catalog_names)
missing=""
for n in $EXPECTED_NAMES; do case " $NAMES " in *" $n "*) ;; *) missing="$missing $n" ;; esac; done
if [ -z "$missing" ]; then verdict inference_models ok "$NAMES"; else verdict inference_models fail "$NAMES (missing:$missing)"; fi
ADMIN=$(admin_state)
if [ "$ADMIN" = admin ]; then verdict "admin $ADMIN_EMAIL" ok "is_admin true"; else verdict "admin $ADMIN_EMAIL" fail "$ADMIN"; fi
if has_feedback; then verdict "feedback table" ok exists; else verdict "feedback table" fail missing; fi
klen=$( { dk exec "$API_C" printenv RESEND_API_KEY 2>/dev/null || true; } | wc -c | tr -d ' ')
if [ "$klen" -gt 1 ]; then verdict "RESEND_API_KEY in API" ok "set (value not shown)"; else verdict "RESEND_API_KEY in API" fail "not set in $API_C"; fi
h=$(api_health "$API_C")
case "$h" in healthy|none-ok) verdict "API /health" ok "$API_C $h" ;; *) verdict "API /health" fail "$API_C $h" ;; esac
pub=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$PUBLIC_HEALTH_URL" 2>/dev/null || true)
if [ "$pub" = 200 ]; then printf '  ok    %-22s %s\n' "public health" "$PUBLIC_HEALTH_URL 200"
else printf '  WARN  %-22s %s\n' "public health" "$PUBLIC_HEALTH_URL answered ${pub:-nothing} (warning only)"; fi

if [ "$FAILED" = 1 ]; then
  echo "verification failed; secrets file kept for a re-run: $SECRETS_FILE"
  exit 10
fi
remove_secrets
echo
echo "Release 2026-10-11 done. Log: $LOG_FILE"

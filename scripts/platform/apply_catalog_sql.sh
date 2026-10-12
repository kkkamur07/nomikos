#!/usr/bin/env bash
# Apply a catalog SQL file to the production Postgres (Coolify container).
#
#   bash scripts/platform/apply_catalog_sql.sh <sql-file>          # look only
#   bash scripts/platform/apply_catalog_sql.sh <sql-file> --apply  # run the SQL
#   bash scripts/platform/apply_catalog_sql.sh --rows-only         # show rows only
#
# The script opens ONE ssh connection (the password is typed once by the
# human), finds the Coolify Postgres container and the production database by
# itself, shows the current inference_models rows, and, only with --apply,
# runs the SQL file over stdin inside the container, so nothing is written
# on the server.
#
# Env overrides:
#   SSH_TARGET   host to connect to (default root@65.21.249.167)
#   SSH_BIN      ssh binary, used for every ssh call (default ssh)
#   PG_CONTAINER Postgres container name (default: discovered by image name)
#   PG_DATABASE  production database name (default: detected by content)
#   MUST_HAVE    registry ids that identify production: each one must appear
#                as artifact_ref 'registry://<id>...' in inference_models
#                (default "greek-calamari-v1 armenian-calamari-v1"). artifact_ref
#                survives the 2026-10-11 display-name rename; names do not.
set -euo pipefail

SSH_TARGET="${SSH_TARGET:-root@65.21.249.167}"
SSH_BIN="${SSH_BIN:-ssh}"
MUST_HAVE="${MUST_HAVE:-greek-calamari-v1 armenian-calamari-v1}"
SHOW="select name, provider, task, artifact_ref, default_params from inference_models order by task, name"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/platform/apply_catalog_sql.sh <sql-file> [--apply]
  bash scripts/platform/apply_catalog_sql.sh --rows-only
  bash scripts/platform/apply_catalog_sql.sh --help

Without --apply this only looks: it finds the production database, shows the
current inference_models rows, and prints the SQL it would send, indented.
With --apply it sends the file with psql -v ON_ERROR_STOP=1 over stdin inside
the container, then prints the rows again. --rows-only shows the current rows
and exits. Flags may come before or after the file.

Container, database, and user names that come back from the server or from
env overrides are accepted only when they match ^[A-Za-z0-9_.-]+$.
Anything else is refused before use.

Exit codes:
  0  success (look-only, rows-only, or applied)
  1  usage error (unknown flag, missing file)
  2  no Postgres container on this host
  3  no database holds the known registry ids (artifact_ref registry://<id>...)
  4  more than one database holds the known registry ids
  5  refusing an unsafe container, database, or user name
EOF
}

APPLY=0
ROWS_ONLY=0
SQL_FILE=""
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    case "$arg" in
      --apply) APPLY=1 ;;
      --rows-only) ROWS_ONLY=1 ;;
      --help|-h) usage; exit 0 ;;
      -*) echo "Unknown flag: $arg" >&2; exit 1 ;;
      *)
        if [ -n "$SQL_FILE" ]; then
          echo "Only one SQL file is allowed, already have: $SQL_FILE" >&2
          exit 1
        fi
        SQL_FILE="$arg"
        ;;
    esac
  done
fi

if [ "$ROWS_ONLY" -eq 0 ]; then
  [ -n "$SQL_FILE" ] || { echo "No SQL file given. Usage: bash scripts/platform/apply_catalog_sql.sh <sql-file> [--apply]" >&2; exit 1; }
  { [ -f "$SQL_FILE" ] && [ -r "$SQL_FILE" ]; } || { echo "SQL file not found or unreadable: $SQL_FILE" >&2; exit 1; }
fi

# One shared connection: the password is asked once, every later step reuses it.
CTL="/tmp/nmk-catalog-$$"
SSH_OPTS=(-o ConnectTimeout=15 -o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=120)
cleanup() { "$SSH_BIN" -o ControlPath="$CTL" -O exit "$SSH_TARGET" >/dev/null 2>&1 || true; }
trap cleanup EXIT
run() { "$SSH_BIN" "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"; }

echo "== opening one connection to $SSH_TARGET (enter the password once)"
run true
echo "== host: $SSH_TARGET ($(run hostname))"

# Remote identifiers come from the server or env overrides and are placed in
# remote shell commands, so each one is validated before use. Exit code 5
# means an unsafe container, database, or user name was refused.
check_safe_name() {
  kind="$1"
  value="$2"
  case "$value" in
    ""|*[!A-Za-z0-9_.-]*)
      echo "refusing unsafe $kind name: $value" >&2
      exit 5
      ;;
  esac
}

# 1. Find Postgres containers (Coolify names them with a random suffix).
if [ -z "${PG_CONTAINER:-}" ]; then
  # bash 3.2 on macOS has no mapfile; container names never contain spaces.
  # shellcheck disable=SC2207
  CANDIDATES=($(run "docker ps --format '{{.Names}} {{.Image}}'" | awk 'tolower($2) ~ /postgres|pgvector|timescale/ {print $1}'))
else
  check_safe_name "container" "$PG_CONTAINER"
  CANDIDATES=("$PG_CONTAINER")
fi
if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "No Postgres container on this host. The API's database is somewhere else."
  exit 2
fi
for candidate in "${CANDIDATES[@]}"; do
  check_safe_name "container" "$candidate"
done

# Rows that identify production (present since 2026-09-07): the production
# database must already hold a row whose artifact_ref is registry://<id>...
# for every registry id in MUST_HAVE (the same test as artifact_ref like
# 'registry://<id>%'). Display names change; artifact_ref does not.
holds_must_have() {
  refs="$1"
  [ -n "$refs" ] || return 1
  for m in $MUST_HAVE; do
    found=0
    for r in $refs; do
      case "$r" in "registry://$m"*) found=1; break ;; esac
    done
    [ "$found" -eq 1 ] || return 1
  done
  return 0
}

FOUND=""
if [ -n "${PG_DATABASE:-}" ]; then
  check_safe_name "database" "$PG_DATABASE"
  C="${CANDIDATES[0]}"
  U="$(run "docker exec \"$C\" printenv POSTGRES_USER" 2>/dev/null || echo postgres)"
  check_safe_name "user" "$U"
  REFS="$(run "docker exec \"$C\" psql -U \"$U\" -d \"$PG_DATABASE\" -Atc \"select artifact_ref from inference_models order by artifact_ref\"" 2>/dev/null | tr '\n' ' ' | sed 's/ $//' || true)"
  echo "== candidate: container=$C user=$U db=$PG_DATABASE"
  echo "   inference_models artifact_refs: $REFS"
  holds_must_have "$REFS" || { echo "Database $PG_DATABASE does not hold the known registry ids ($MUST_HAVE). Not changing anything."; exit 3; }
  FOUND="$C|$U|$PG_DATABASE"
else
  # 2. Pick the database that really is production: it must already hold the
  # known registry ids. More than one match is a refusal, not a guess.
  MATCHES=""
  for c in "${CANDIDATES[@]}"; do
    U="$(run "docker exec \"$c\" printenv POSTGRES_USER" 2>/dev/null || echo postgres)"
    check_safe_name "user" "$U"
    DBS="$(run "docker exec \"$c\" psql -U \"$U\" -d postgres -Atc \"select datname from pg_database where not datistemplate\"" 2>/dev/null || true)"
    for d in $DBS; do
      check_safe_name "database" "$d"
      REFS="$(run "docker exec \"$c\" psql -U \"$U\" -d \"$d\" -Atc \"select artifact_ref from inference_models order by artifact_ref\"" 2>/dev/null | tr '\n' ' ' | sed 's/ $//' || true)"
      [ -n "$REFS" ] || continue
      echo "== candidate: container=$c user=$U db=$d"
      echo "   inference_models artifact_refs: $REFS"
      if holds_must_have "$REFS"; then
        MATCHES="${MATCHES}${c}|${U}|${d}
"
      fi
    done
  done
  if [ -z "$MATCHES" ]; then
    echo "No database here holds the known registry ids ($MUST_HAVE). Not changing anything."
    exit 3
  fi
  COUNT="$(printf '%s' "$MATCHES" | grep -c .)"
  if [ "$COUNT" -gt 1 ]; then
    echo "More than one database holds the known registry ids ($MUST_HAVE). Refusing to guess:"
    printf '%s' "$MATCHES"
    exit 4
  fi
  FOUND="$(printf '%s' "$MATCHES" | head -n 1)"
fi
IFS='|' read -r C U D <<<"$FOUND"
check_safe_name "container" "$C"
check_safe_name "user" "$U"
check_safe_name "database" "$D"
echo "== production database: container=$C user=$U db=$D"
echo "== rows now:"
run "docker exec \"$C\" psql -U \"$U\" -d \"$D\" -c \"$SHOW\""

if [ "$ROWS_ONLY" -eq 1 ]; then
  exit 0
fi

if [ "$APPLY" -eq 0 ]; then
  echo "Look-only run. This is what --apply would send:"
  sed 's/^/   /' "$SQL_FILE"
  exit 0
fi

# 3. Apply. The SQL travels over stdin, so nothing is written on the server.
echo "== applying $SQL_FILE"
run "docker exec -i \"$C\" psql -v ON_ERROR_STOP=1 -U \"$U\" -d \"$D\"" < "$SQL_FILE"
echo "== rows after:"
run "docker exec \"$C\" psql -U \"$U\" -d \"$D\" -c \"$SHOW\""

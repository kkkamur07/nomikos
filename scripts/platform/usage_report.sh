#!/usr/bin/env bash
# Read-only usage report. Usage: usage_report.sh [--days N] | --local <dbname> [--days N]
#
# Runs scripts/platform/sql/usage_report.sql (next to this script; USAGE_SQL
# overrides) against production over ssh, or against a local database in the
# nomikos-db-1 container with --local. Output goes to the terminal and to
# ~/nomikos-usage/report-<date>.txt.
#
# Read-only is enforced by the database: psql runs with
# PGOPTIONS='-c default_transaction_read_only=on', so any write in the SQL
# fails. The keyword check below is only an early, friendlier refusal.
# tests/platform/test_usage_report.sh checks both with stub ssh and docker.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SQL=${USAGE_SQL:-$HERE/sql/usage_report.sql}
HOST=root@65.21.249.167
CTR=6gxpltrpm5vxa9e8hui5mcgn
SOCK=/tmp/nomikos-usage-ssh.sock
OUT_DIR=$HOME/nomikos-usage
OUT=$OUT_DIR/report-$(date +%Y-%m-%d).txt
DAYS=30
LOCAL=""
READ_ONLY='-c default_transaction_read_only=on'

while [ $# -gt 0 ]; do
  case $1 in
    --days) DAYS=${2:?--days needs a number}; shift 2 ;;
    --local) LOCAL=${2:?--local needs a database name}; shift 2 ;;
    *) echo "usage: $0 [--days N] [--local <dbname>]" >&2; exit 2 ;;
  esac
done
case $DAYS in ''|*[!0-9]*) echo "--days must be a whole number" >&2; exit 2 ;; esac
[ -f "$SQL" ] || { echo "SQL file not found: $SQL" >&2; exit 2; }

PAT='\b(insert|update|delete|alter|drop|create|truncate)\b'
if grep -Eiq "$PAT" "$SQL"; then
  echo "refusing to run: $SQL contains a write keyword:" >&2
  grep -Ein "$PAT" "$SQL" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
if [ -n "$LOCAL" ]; then
  RUN=(docker exec -i -e "PGOPTIONS=$READ_ONLY" nomikos-db-1 psql -U postgres -d "$LOCAL")
else
  ssh -M -S "$SOCK" -fN -o ControlPersist=10m "$HOST"
  trap 'ssh -S "$SOCK" -O exit "$HOST" >/dev/null 2>&1 || true' EXIT
  # ssh joins its arguments into one remote shell command, so the option is quoted here.
  RUN=(ssh -S "$SOCK" "$HOST" "docker exec -i -e 'PGOPTIONS=$READ_ONLY' $CTR psql -U nomikos_app -d nomikos")
fi

if ! "${RUN[@]}" -v days="$DAYS" -v ON_ERROR_STOP=1 -P pager=off < "$SQL" | tee "$OUT"; then
  echo "psql failed; $OUT holds a partial report" >&2
  exit 1
fi
echo "saved to $OUT" >&2

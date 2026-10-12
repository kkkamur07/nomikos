#!/usr/bin/env bash
# Stub-based tests for scripts/platform/apply_catalog_sql.sh.
# A fake ssh answers the docker/psql probes with canned output, so no
# server is needed. Plain bash, no framework. Prints PASS or FAIL per case.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/platform/apply_catalog_sql.sh"
TMP="$(mktemp -d /tmp/catalog-apply-test.XXXXXX)"
cleanup_tmp() { rm -rf "$TMP"; }
trap cleanup_tmp EXIT

PASS=0
FAIL=0
pass() { PASS=$((PASS + 1)); echo "PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }

mkdir -p "$TMP/bin"
cat > "$TMP/bin/ssh" <<'STUB_EOF'
#!/usr/bin/env bash
# Fake ssh for the catalog apply tests. Logs its arguments, captures stdin
# per call, and answers the docker/psql probes with canned output.
set -euo pipefail
LOG="${STUB_LOG:?missing STUB_LOG}"
DIR="${STUB_DIR:?missing STUB_DIR}"
MODE="${STUB_MODE:-normal}"
COUNT_FILE="$DIR/count"
n=0
if [ -f "$COUNT_FILE" ]; then n=$(cat "$COUNT_FILE"); fi
n=$((n + 1))
printf '%s' "$n" > "$COUNT_FILE"
last=""
for a in "$@"; do last="$a"; done
printf 'CALL %s: %s\n' "$n" "$*" >> "$LOG"
cat > "$DIR/stdin.$n"
printf 'ENDCALL %s\n' "$n" >> "$LOG"
case "$*" in
  *"-O exit"*) exit 0 ;;
esac
case "$last" in
  true) exit 0 ;;
  hostname) echo "fakehost"; exit 0 ;;
  *"ON_ERROR_STOP"*) echo "INSERT 0 1"; exit 0 ;;
  *"docker ps"*)
    if [ "$MODE" = "no-container" ]; then
      echo "web-ab12cd app:latest"
    else
      echo "coolify-pg-ab12cd postgres:16-pgvector"
    fi
    exit 0
    ;;
  *"printenv POSTGRES_USER"*) echo "postgres"; exit 0 ;;
  *"select datname"*)
    if [ "$MODE" = "two-dbs" ]; then
      printf 'prod_one\nprod_two\n'
    elif [ "$MODE" = "evil-db" ]; then
      printf 'nomikos;touch /tmp/pwned\n'
    else
      echo "prod_main"
    fi
    exit 0
    ;;
  *"select name from inference_models"*)
    printf 'armenian-calamari-v1\ngreek-calamari-v1\nkraken\n'
    exit 0
    ;;
  *"select name, provider, task"*)
    echo "kraken | kraken | segment | registry://blla-segment?tag=stable"
    exit 0
    ;;
  *) exit 0 ;;
esac
STUB_EOF
chmod +x "$TMP/bin/ssh"

LOG="$TMP/ssh.log"
reset_stub() {
  rm -f "$LOG" "$TMP"/stdin.*
  printf '0' > "$TMP/count"
}

# reset_stub MODE args... : run the script with the stub ssh, save output.
# Sets globals CODE, OUT, ERR. Caller reads CODE itself.
run_script() {
  mode="$1"
  shift
  reset_stub
  set +e
  STUB_MODE="$mode" STUB_LOG="$LOG" STUB_DIR="$TMP" SSH_BIN="$TMP/bin/ssh" \
    bash "$SCRIPT" "$@" < /dev/null >"$TMP/out.txt" 2>"$TMP/err.txt"
  CODE=$?
  set -e
  OUT="$TMP/out.txt"
  ERR="$TMP/err.txt"
}

{
  echo "-- test change TEST_MARKER_7f3a9c"
  echo "select 1;"
} > "$TMP/change.sql"

# (a) look-only: exit 0, shows the SQL, never pipes it to psql.
run_script normal "$TMP/change.sql"
if [ "$CODE" -ne 0 ]; then
  fail "(a) look-only exits 0 (got $CODE)"
elif grep -q "ON_ERROR_STOP" "$LOG"; then
  fail "(a) look-only never calls psql with the SQL file"
elif grep -q "TEST_MARKER_7f3a9c" "$LOG"; then
  fail "(a) look-only never pipes SQL content through ssh"
elif ! grep -q "TEST_MARKER_7f3a9c" "$OUT"; then
  fail "(a) look-only prints the SQL it would send"
else
  piped=0
  for f in "$TMP"/stdin.*; do
    if [ -f "$f" ] && grep -q "TEST_MARKER_7f3a9c" "$f"; then piped=1; fi
  done
  if [ "$piped" -ne 0 ]; then
    fail "(a) look-only leaves every ssh stdin free of SQL"
  else
    pass "(a) look-only exits 0 and never pipes the SQL to psql"
  fi
fi

# (b) --apply after the file: pipes exactly the SQL file content.
run_script normal "$TMP/change.sql" --apply
if [ "$CODE" -ne 0 ]; then
  fail "(b) --apply exits 0 (got $CODE)"
else
  applyn=$(grep "ON_ERROR_STOP" "$LOG" | head -n 1 | sed 's/^CALL \([0-9]*\):.*/\1/')
  if [ -z "$applyn" ]; then
    fail "(b) --apply sends the SQL with ON_ERROR_STOP=1"
  elif ! diff -q "$TMP/change.sql" "$TMP/stdin.$applyn" >/dev/null; then
    fail "(b) --apply pipes exactly the SQL file content"
  elif ! grep -q "rows after" "$OUT"; then
    fail "(b) --apply prints the rows again"
  else
    pass "(b) --apply pipes exactly the SQL file content"
  fi
fi

# (b2) --apply before the file works too.
run_script normal --apply "$TMP/change.sql"
if [ "$CODE" -ne 0 ]; then
  fail "(b2) flag before file exits 0 (got $CODE)"
elif ! grep -q "ON_ERROR_STOP" "$LOG"; then
  fail "(b2) flag before file still applies"
else
  pass "(b2) flag before file still applies"
fi

# (c) missing file: exit 1 with zero ssh calls.
run_script normal "$TMP/does-not-exist.sql"
if [ "$CODE" -ne 1 ]; then
  fail "(c) missing file exits 1 (got $CODE)"
elif [ -e "$LOG" ]; then
  fail "(c) missing file makes zero ssh calls"
elif [ ! -s "$ERR" ]; then
  fail "(c) missing file prints a message on stderr"
else
  pass "(c) missing file exits 1 with zero ssh calls"
fi

# (c2) unknown flag: exit 1 with zero ssh calls.
run_script normal --bogus "$TMP/change.sql"
if [ "$CODE" -ne 1 ]; then
  fail "(c2) unknown flag exits 1 (got $CODE)"
elif [ -e "$LOG" ]; then
  fail "(c2) unknown flag makes zero ssh calls"
else
  pass "(c2) unknown flag exits 1 with zero ssh calls"
fi

# (d) two qualifying databases: exit 4 and list them.
run_script two-dbs "$TMP/change.sql"
if [ "$CODE" -ne 4 ]; then
  fail "(d) two qualifying databases exit 4 (got $CODE)"
elif ! grep -q "prod_one" "$OUT" || ! grep -q "prod_two" "$OUT"; then
  fail "(d) two qualifying databases are both listed"
else
  pass "(d) two qualifying databases exit 4 and are both listed"
fi

# (e) no Postgres container: exit 2.
run_script no-container "$TMP/change.sql"
if [ "$CODE" -ne 2 ]; then
  fail "(e) no container exits 2 (got $CODE)"
else
  pass "(e) no container exits 2"
fi

# (f) --rows-only with no SQL file: exit 0, shows rows, sends nothing.
run_script normal --rows-only
if [ "$CODE" -ne 0 ]; then
  fail "(f) --rows-only exits 0 (got $CODE)"
elif ! grep -q "rows now" "$OUT"; then
  fail "(f) --rows-only shows the current rows"
elif grep -q "ON_ERROR_STOP" "$LOG"; then
  fail "(f) --rows-only never applies SQL"
else
  pass "(f) --rows-only exits 0 with no SQL file"
fi

# (g) --help: usage on stdout, exit 0, no ssh.
run_script normal --help
if [ "$CODE" -ne 0 ]; then
  fail "(g) --help exits 0 (got $CODE)"
elif ! grep -q -i "usage" "$OUT"; then
  fail "(g) --help prints usage"
elif [ -e "$LOG" ]; then
  fail "(g) --help makes zero ssh calls"
else
  pass "(g) --help prints usage with zero ssh calls"
fi

# (h) evil discovered database name: exit 5, never placed in a remote command.
run_script evil-db "$TMP/change.sql"
if [ "$CODE" -ne 5 ]; then
  fail "(h) evil database name exits 5 (got $CODE)"
elif ! grep -q "refusing unsafe database name" "$ERR"; then
  fail "(h) evil database name prints a refusal on stderr"
elif grep -q "nomikos;touch" "$LOG"; then
  fail "(h) evil database name never reaches a remote command"
else
  pass "(h) evil database name exits 5 and never reaches a remote command"
fi

# (i) unsafe PG_CONTAINER override: exit 5 before any ssh call uses it.
export PG_CONTAINER='bad;touch /tmp/pwned'
run_script normal "$TMP/change.sql"
unset PG_CONTAINER
if [ "$CODE" -ne 5 ]; then
  fail "(i) unsafe PG_CONTAINER exits 5 (got $CODE)"
elif ! grep -q "refusing unsafe container name" "$ERR"; then
  fail "(i) unsafe PG_CONTAINER prints a refusal on stderr"
elif grep -q "bad;touch" "$LOG"; then
  fail "(i) unsafe PG_CONTAINER never reaches a remote command"
else
  pass "(i) unsafe PG_CONTAINER exits 5 and never reaches a remote command"
fi

echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ]

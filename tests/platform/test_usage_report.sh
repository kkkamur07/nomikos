#!/usr/bin/env bash
# Test for scripts/platform/usage_report.sh with stub `ssh` and `docker` on PATH.
# Nothing is contacted: the stubs record their arguments, environment and stdin.
# Asserts the SQL next to the script is used and psql gets a read-only
# PGOPTIONS, locally and over ssh (as the remote shell would split it).
#
# Usage: bash tests/platform/test_usage_report.sh
# USAGE_REPORT=<path> runs it against another copy of the script.

set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=${USAGE_REPORT:-$ROOT/scripts/platform/usage_report.sh}
REPO_SQL=$ROOT/scripts/platform/sql/usage_report.sql
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

FAILS=0
check() { # check DESCRIPTION COMMAND...
  local what=$1; shift
  if "$@"; then echo "ok   $what"; else echo "FAIL $what"; FAILS=$((FAILS + 1)); fi
}

mkdir -p "$WORK/bin" "$WORK/home"
for tool in ssh docker; do
  cat > "$WORK/bin/$tool" <<EOF
#!/usr/bin/env bash
n=\$(ls "$WORK" | grep -c "^$tool\\..*\\.args\$" || true)
log="$WORK/$tool.\$n"
printf '%s\n' "\$@" > "\$log.args"
printf '%s' "\${PGOPTIONS-}" > "\$log.env"
case " \$* " in *" -fN "*|*" -O exit "*) exit 0 ;; esac
cat > "\$log.stdin"
echo "stub report"
EOF
  chmod +x "$WORK/bin/$tool"
done

run() { # run NAME ARGS...: from an unrelated directory, with stubs first on PATH
  local name=$1; shift
  (cd "$WORK" && HOME="$WORK/home" PATH="$WORK/bin:$PATH" bash "$SCRIPT" "$@" \
    >"$WORK/$name.out" 2>"$WORK/$name.err")
  echo $? > "$WORK/$name.rc"
}

echo "== --local"
run local --local scratch_db --days 7
check "--local exits 0" [ "$(cat "$WORK/local.rc")" = 0 ]
check "--local feeds the SQL next to the script" cmp -s "$WORK/docker.0.stdin" "$REPO_SQL"
check "--local passes a read-only PGOPTIONS to the container" \
  grep -qx "PGOPTIONS=-c default_transaction_read_only=on" "$WORK/docker.0.args"
check "--local passes --days" grep -qx "days=7" "$WORK/docker.0.args"
check "--local saves the report" grep -q "stub report" "$WORK/home/nomikos-usage/report-$(date +%Y-%m-%d).txt"

echo "== production (stub ssh)"
rm -f "$WORK"/docker.*
run prod --days 30
check "prod exits 0" [ "$(cat "$WORK/prod.rc")" = 0 ]
check "prod feeds the SQL next to the script" cmp -s "$WORK/ssh.1.stdin" "$REPO_SQL"
if ! python3 - "$WORK/ssh.1.args" <<'PY'
import shlex, sys

args = open(sys.argv[1]).read().splitlines()
# ssh drops its own options and joins the rest into one remote shell command.
remote = " ".join(args[args.index("root@65.21.249.167") + 1:])
words = shlex.split(remote)
ok = "PGOPTIONS=-c default_transaction_read_only=on" in words and words[0:2] == ["docker", "exec"]
print(("ok   " if ok else "FAIL ") + f"remote command sets a read-only PGOPTIONS ({remote!r})")
sys.exit(0 if ok else 1)
PY
then
  FAILS=$((FAILS + 1))
fi

echo "== write keyword"
printf 'select 1;\ndelete from users;\n' > "$WORK/bad.sql"
(cd "$WORK" && HOME="$WORK/home" PATH="$WORK/bin:$PATH" USAGE_SQL="$WORK/bad.sql" \
  bash "$SCRIPT" --local scratch_db >/dev/null 2>&1)
check "refuses SQL with a write keyword" [ $? = 1 ]

echo
if [ "$FAILS" = 0 ]; then echo "PASS: usage_report.sh"; else echo "FAILED: $FAILS check(s)"; exit 1; fi

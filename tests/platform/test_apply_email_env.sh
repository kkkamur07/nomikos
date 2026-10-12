#!/usr/bin/env bash
# Smoke test for scripts/platform/apply_email_env.sh against a stub Coolify API.
# Runs --check, --dry-run and a full run with fake secrets; the stub on
# 127.0.0.1 is the only network target. Asserts the three requests of a full
# run and that no secret reaches stdout or stderr in any mode.
#
# Usage: bash tests/platform/test_apply_email_env.sh
# APPLY_EMAIL_ENV=<path> runs it against another copy of the script.

set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=${APPLY_EMAIL_ENV:-$ROOT/scripts/platform/apply_email_env.sh}
WORK=$(mktemp -d)
STUB_PID=""
cleanup() {
  [ -z "$STUB_PID" ] || { kill "$STUB_PID" 2>/dev/null; wait "$STUB_PID" 2>/dev/null; }
  rm -rf "$WORK"
}
trap cleanup EXIT

FAILS=0
pass() { echo "ok   $*"; }
fail() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
check() { # check DESCRIPTION COMMAND...
  local what=$1; shift
  if "$@"; then pass "$what"; else fail "$what"; fi
}

FAKE_KEY="re_fakeKEY_9f8e7d6c5b4a"
FAKE_TOKEN="7|fakeTOKEN_0a1b2c3d4e5f"

cat > "$WORK/stub.py" <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

log_path, port_path = sys.argv[1], sys.argv[2]
APPS = [
    {"uuid": "app-uuid-1", "name": "nomikos-api", "git_branch": "main",
     "git_commit_sha": "abc1234def"},
    {"uuid": "app-uuid-2", "name": "nomikos-landing"},
]


class Handler(BaseHTTPRequestHandler):
    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode() if length else ""
        with open(log_path, "a") as log:
            log.write(json.dumps({"method": self.command, "path": self.path,
                                  "auth": self.headers.get("Authorization"),
                                  "body": body}) + "\n")
        if self.command == "GET" and self.path == "/api/v1/applications":
            payload = APPS
        elif self.command == "PATCH" and self.path.endswith("/envs/bulk"):
            payload = {"message": "ok"}
        elif self.command == "GET" and self.path.startswith("/api/v1/deploy"):
            payload = {"deployments": [{"message": "queued"}]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_PATCH = do_POST = _handle

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
with open(port_path, "w") as f:
    f.write(str(server.server_address[1]))
server.serve_forever()
PY

: > "$WORK/requests.log"
python3 "$WORK/stub.py" "$WORK/requests.log" "$WORK/port" &
STUB_PID=$!
for _ in $(seq 50); do [ -s "$WORK/port" ] && break; sleep 0.1; done
[ -s "$WORK/port" ] || { echo "FAIL stub server did not start"; exit 1; }
PORT=$(cat "$WORK/port")

SECRETS="$WORK/secrets.env"
cat > "$SECRETS" <<EOF
# fake values for the test
RESEND_API_KEY=$FAKE_KEY
COOLIFY_URL=http://127.0.0.1:$PORT/
COOLIFY_TOKEN="$FAKE_TOKEN"
EOF
chmod 600 "$SECRETS"

run() { # run NAME [ARGS]: stdout and stderr to $WORK/NAME.out and .err, exit code to $WORK/NAME.rc
  local name=$1; shift
  NOMIKOS_SECRETS_FILE="$SECRETS" bash "$SCRIPT" "$@" >"$WORK/$name.out" 2>"$WORK/$name.err"
  echo $? > "$WORK/$name.rc"
}
no_secret() { ! grep -qF -e "$FAKE_KEY" -e "fakeTOKEN" "$WORK/$1.out" "$WORK/$1.err"; }
nrequests() { grep -c . "$WORK/requests.log" || true; }

echo "== --check"
run check --check
check "--check exits 0" [ "$(cat "$WORK/check.rc")" = 0 ]
check "--check masks the key" grep -q "RESEND_API_KEY=re_\*\*\*\*" "$WORK/check.out"
check "--check prints no secret" no_secret check
check "--check sends no request" [ "$(nrequests)" = 0 ]

echo "== --dry-run"
run dry --dry-run
check "--dry-run exits 0" [ "$(cat "$WORK/dry.rc")" = 0 ]
check "--dry-run plans the app URL" grep -q "APP_PUBLIC_URL=https://app.nomikos.app" "$WORK/dry.out"
check "--dry-run leaves EMAIL_REPLY_TO out" bash -c "! grep -q 'EMAIL_REPLY_TO=' '$WORK/dry.out'"
check "--dry-run warns the redeploy uses the branch head" grep -q "head of the tracked branch" "$WORK/dry.out"
check "--dry-run prints no secret" no_secret dry
check "--dry-run sends no request" [ "$(nrequests)" = 0 ]

echo "== full run"
run full
echo "   exit $(cat "$WORK/full.rc"); stderr: $(tr '\n' ' ' < "$WORK/full.err")"
check "full run exits 0" [ "$(cat "$WORK/full.rc")" = 0 ]
check "full run prints no secret" no_secret full
check "full run names branch and commit before redeploying" \
  grep -q "branch: main" "$WORK/full.out"
check "full run prints the commit" grep -q "commit: abc1234def" "$WORK/full.out"
sed 's/^/   | /' "$WORK/full.out"

if ! python3 - "$WORK/requests.log" "$FAKE_KEY" "$FAKE_TOKEN" <<'PY'
import json, sys

log, key, token = sys.argv[1], sys.argv[2], sys.argv[3]
reqs = [json.loads(line) for line in open(log) if line.strip()]
seen = [(r["method"], r["path"]) for r in reqs]
expected = [
    ("GET", "/api/v1/applications"),
    ("PATCH", "/api/v1/applications/app-uuid-1/envs/bulk"),
    ("GET", "/api/v1/deploy?uuid=app-uuid-1&force=true"),
]
ok = True


def check(what, cond):
    global ok
    print(("ok   " if cond else "FAIL ") + what)
    ok = ok and cond


check(f"exactly the three requests in order (saw {seen})", seen == expected)
check("every request carries the bearer token",
      bool(reqs) and all(r["auth"] == f"Bearer {token}" for r in reqs))
bulk = next((r for r in reqs if r["path"].endswith("/envs/bulk")), None)
values = {d["key"]: d["value"] for d in json.loads(bulk["body"])["data"]} if bulk else {}
check("bulk body carries the key from the secrets file", values.get("RESEND_API_KEY") == key)
check("bulk body sets APP_PUBLIC_URL to the app", values.get("APP_PUBLIC_URL") == "https://app.nomikos.app")
check("bulk body leaves EMAIL_REPLY_TO unset", bulk is not None and "EMAIL_REPLY_TO" not in values)
check("bulk body sets exactly four keys", sorted(values) == sorted(
    ["RESEND_API_KEY", "EMAIL_FROM", "APP_PUBLIC_URL", "EMAIL_SIGNIN_NOTICES"]))
sys.exit(0 if ok else 1)
PY
then
  FAILS=$((FAILS + 1))
fi

echo "== --dry-run --no-deploy"
run drynd --dry-run --no-deploy
check "--dry-run --no-deploy exits 0" [ "$(cat "$WORK/drynd.rc")" = 0 ]
check "--dry-run --no-deploy plans no redeploy" bash -c "! grep -q 'api/v1/deploy' '$WORK/drynd.out'"

echo "== full run with --no-deploy"
: > "$WORK/requests.log"
run nodeploy --no-deploy
check "--no-deploy exits 0" [ "$(cat "$WORK/nodeploy.rc")" = 0 ]
check "--no-deploy prints no secret" no_secret nodeploy
check "--no-deploy sends GET applications then PATCH envs/bulk only" [ "$(python3 -c '
import json, sys
print(" ".join(json.loads(l)["method"] + ":" + json.loads(l)["path"] for l in open(sys.argv[1]) if l.strip()))
' "$WORK/requests.log")" = "GET:/api/v1/applications PATCH:/api/v1/applications/app-uuid-1/envs/bulk" ]
check "--no-deploy rejects an unknown flag" bash -c "NOMIKOS_SECRETS_FILE='$SECRETS' bash '$SCRIPT' --bogus >/dev/null 2>&1; [ \$? = 2 ]"

echo "== GNU stat on PATH (stat -f is file system status there)"
mkdir "$WORK/fakebin"
cat > "$WORK/fakebin/stat" <<'SH'
#!/bin/sh
if [ "$1" = "-f" ]; then
  printf '  File: "x"\n    ID: 0 Namelen: 255 Type: ext2/ext3\nBlock size: 4096\n'
  exit 0
fi
if [ "$1" = "-c" ] && [ "$2" = "%a" ]; then echo 600; exit 0; fi
exit 1
SH
chmod +x "$WORK/fakebin/stat"
PATH="$WORK/fakebin:$PATH" run gnustat --check
check "GNU-style stat: mode check passes" [ "$(cat "$WORK/gnustat.rc")" = 0 ]
check "GNU-style stat: no mode error" bash -c "! grep -q 'is mode' '$WORK/gnustat.err'"

echo
if [ "$FAILS" = 0 ]; then echo "PASS: apply_email_env.sh"; else echo "FAILED: $FAILS group(s)"; exit 1; fi

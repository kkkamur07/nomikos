#!/usr/bin/env bash
# End-to-end test for scripts/platform/release_2026_10_11.sh, no production access.
#
# - Postgres: a scratch database kalamos_pf3_rel in the local nomikos-db-1 container,
#   migrated to 006 with alembic, then 007 by hand (production's state), seeded with
#   the seven pre-rename catalog rows, one user, and a part with a full-page block.
# - Coolify, GitHub and the public /health: one python3 stub server on 127.0.0.1.
# - docker: a wrapper that forwards everything about nomikos-db-1 to the real docker
#   and fakes ps, inspect and exec for the API containers. Its exec of
#   renumber_lines.py runs the real script from this checkout against the scratch DB.
#
# Usage (from the repo root): bash tests/platform/test_release_2026_10_11.sh
# The scratch database is dropped at the end, also on failure.

set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT="$ROOT/scripts/platform/release_2026_10_11.sh"
MAIN_VENV="${MAIN_VENV:-/Users/krishuagarwal/Desktop/Programming/python/greekOCR/.venv}"
DB=kalamos_pf3_rel
export PGPASSWORD=dev
PSQL=(psql -h localhost -p 5433 -U postgres -w -X -q)
WORK=$(mktemp -d)
STUB_PID=""

cleanup() {
  [ -z "$STUB_PID" ] || { kill "$STUB_PID" 2>/dev/null; wait "$STUB_PID" 2>/dev/null; }
  "${PSQL[@]}" -d postgres -c "drop database if exists $DB" >/dev/null 2>&1
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
q() { "${PSQL[@]}" -d "$DB" -At -c "$1"; }

FAKE_KEY="re_fakeKEY_pf3rel_77aa"
FAKE_TOKEN="9|fakeTOKEN_pf3rel_88bb"
ADMIN_EMAIL="Admin@Example.test"

echo "== static checks"
check "bash -n" bash -n "$SCRIPT"
if command -v shellcheck >/dev/null; then
  check "shellcheck" shellcheck "$SCRIPT"
fi

echo "== scratch database"
"${PSQL[@]}" -d postgres -c "drop database if exists $DB" >/dev/null
"${PSQL[@]}" -d postgres -c "create database $DB" >/dev/null || { echo "FAIL cannot create $DB"; exit 1; }
(cd "$ROOT/nomikos" && MIGRATOR_DATABASE_URL="postgresql://postgres:dev@localhost:5433/$DB" PYTHONPATH=. \
  "$MAIN_VENV/bin/alembic" -c infrastructure/alembic.ini upgrade 006_public_sharing >"$WORK/alembic.log" 2>&1) ||
  { cat "$WORK/alembic.log"; echo "FAIL alembic upgrade 006"; exit 1; }
"${PSQL[@]}" -d "$DB" -v ON_ERROR_STOP=1 >/dev/null <<'SQL' || { echo "FAIL seeding"; exit 1; }
-- production's state: migration 007 applied by hand
ALTER TABLE line_transcriptions ADD COLUMN character_confidences JSONB;
UPDATE alembic_version SET version_num = '007_line_character_confidences';
INSERT INTO inference_models (id, name, provider, task, artifact_ref) VALUES
  (gen_random_uuid(), 'kraken', 'kraken', 'segment', 'registry://blla-segment?tag=stable'),
  (gen_random_uuid(), 'ppocr', 'ppocr', 'segment', 'registry://ppocr-segment?tag=stable'),
  (gen_random_uuid(), 'syriac-ppocr-v1', 'ppocr', 'transcribe', 'registry://syriac-ppocr-v1?tag=stable'),
  (gen_random_uuid(), 'greek-calamari-v1', 'calamari', 'transcribe', 'registry://greek-calamari-v1?tag=stable'),
  (gen_random_uuid(), 'armenian-calamari-v1', 'calamari', 'transcribe', 'registry://armenian-calamari-v1?tag=stable'),
  (gen_random_uuid(), 'syriac-calamari-v2', 'calamari', 'transcribe', 'registry://syriac-calamari-v2?tag=stable'),
  (gen_random_uuid(), 'coptic-calamari-v1', 'calamari', 'transcribe', 'registry://coptic-calamari-v1?tag=stable');
INSERT INTO users (id, email, username, hashed_password)
  VALUES ('00000000-0000-0000-0000-0000000000a1', 'admin@example.test', 'admin', 'x');
INSERT INTO projects (id, name, slug, owner_id)
  VALUES ('00000000-0000-0000-0000-0000000000b1', 'p', 'p', '00000000-0000-0000-0000-0000000000a1');
INSERT INTO documents (id, project_id, name, workflow)
  VALUES ('00000000-0000-0000-0000-0000000000c1', '00000000-0000-0000-0000-0000000000b1', 'd', 'draft');
INSERT INTO document_parts (id, document_id, "order", image_key, width, height) VALUES
  ('00000000-0000-0000-0000-0000000000d1', '00000000-0000-0000-0000-0000000000c1', 0, 'k1', 1000, 1500),
  ('00000000-0000-0000-0000-0000000000d2', '00000000-0000-0000-0000-0000000000c1', 1, 'k2', 1000, 1500);
-- part 1: one segmenter block covering the whole page, two lines with a gap in the order
INSERT INTO blocks (id, part_id, box, "order") VALUES
  ('00000000-0000-0000-0000-0000000000e1', '00000000-0000-0000-0000-0000000000d1',
   '{"points": [[0, 0], [1000, 0], [1000, 1500], [0, 1500]]}', 0);
INSERT INTO lines (id, part_id, block_id, baseline, "order") VALUES
  ('00000000-0000-0000-0000-0000000000f1', '00000000-0000-0000-0000-0000000000d1',
   '00000000-0000-0000-0000-0000000000e1', '[[10, 100], [900, 100]]', 3),
  ('00000000-0000-0000-0000-0000000000f2', '00000000-0000-0000-0000-0000000000d1',
   '00000000-0000-0000-0000-0000000000e1', '[[10, 200], [900, 200]]', 7);
-- part 2: a real region (small box), must survive
INSERT INTO blocks (id, part_id, box, "order") VALUES
  ('00000000-0000-0000-0000-0000000000e2', '00000000-0000-0000-0000-0000000000d2',
   '{"points": [[100, 100], [500, 100], [500, 400], [100, 400]]}', 0);
INSERT INTO lines (id, part_id, block_id, baseline, "order") VALUES
  ('00000000-0000-0000-0000-0000000000f3', '00000000-0000-0000-0000-0000000000d2',
   '00000000-0000-0000-0000-0000000000e2', '[[110, 200], [490, 200]]', 0);
SQL
pass "scratch database at 007 with the seven catalog rows"

echo "== stub server and fake docker"
mkdir -p "$WORK/bin" "$WORK/state"
cat > "$WORK/stub.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

log_path, port_path, state = sys.argv[1], sys.argv[2], sys.argv[3]
ENVS = []
APPS = [
    {"uuid": "appuuid1", "name": "nomikos-api", "git_branch": "main", "git_commit_sha": "abc1234"},
    {"uuid": "appuuid2", "name": "nomikos-landing"},
]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode() if length else ""
        with open(log_path, "a") as log:
            log.write(json.dumps({"method": self.command, "path": self.path,
                                  "auth": self.headers.get("Authorization"), "body": body}) + "\n")
        p = self.path
        if self.command == "GET" and p == "/api/v1/applications":
            return self._send(200, APPS)
        if self.command == "GET" and p == "/api/v1/applications/appuuid1/envs":
            return self._send(200, ENVS)
        if self.command == "PATCH" and p == "/api/v1/applications/appuuid1/envs/bulk":
            for item in json.loads(body)["data"]:
                ENVS[:] = [e for e in ENVS if e["key"] != item["key"]]
                ENVS.append({"key": item["key"], "value": item["value"], "is_preview": False})
            open(os.path.join(state, "patched"), "w").close()
            return self._send(200, {"message": "ok"})
        if self.command == "GET" and p.startswith("/api/v1/deploy"):
            return self._send(200, {"deployments": [{"message": "queued"}]})
        if self.command == "GET" and p == "/repos/kkkamur07/nomikos/pulls/165":
            # The merge deploys: once the env is written, the next PR read "merges" it.
            if os.path.exists(os.path.join(state, "patched")):
                open(os.path.join(state, "deployed"), "w").close()
            return self._send(200, {"number": 165, "merged": True})
        if self.command == "GET" and p == "/health":
            return self._send(200, {"status": "ok"})
        self._send(404, {"message": "not found"})

    do_GET = do_PATCH = do_POST = _handle

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
with open(port_path, "w") as f:
    f.write(str(server.server_address[1]))
server.serve_forever()
PY
: > "$WORK/requests.log"
python3 "$WORK/stub.py" "$WORK/requests.log" "$WORK/port" "$WORK/state" &
STUB_PID=$!
for _ in $(seq 50); do [ -s "$WORK/port" ] && break; sleep 0.1; done
[ -s "$WORK/port" ] || { echo "FAIL stub server did not start"; exit 1; }
PORT=$(cat "$WORK/port")

cat > "$WORK/bin/docker" <<'SH'
#!/usr/bin/env bash
# Fake docker: the API containers are invented, everything else goes to the real docker.
set -u
printf '%s\n' "$*" >> "$FAKE_STATE/../docker.log"
is_api() { case "$1" in appuuid1-old|appuuid1-new) return 0 ;; *) return 1 ;; esac; }
deployed() { [ -f "$FAKE_STATE/deployed" ]; }
case "${1:-}" in
  ps)
    printf 'pgid\tnomikos-db-1\tpostgres:16-alpine\n'
    printf 'oldid\tappuuid1-old\tappuuid1:oldsha\n'
    deployed && printf 'newid\tappuuid1-new\tappuuid1:newsha\n'
    exit 0 ;;
  inspect)
    name="${*: -1}"
    if is_api "$name"; then
      case "$*" in
        *Created*) [ "$name" = appuuid1-old ] && echo "2026-10-01T10:00:00.5Z|oldid" || echo "2026-10-11T22:00:00.25Z|newid" ;;
        *Health*) echo healthy ;;
      esac
      exit 0
    fi ;;
  exec)
    orig=("$@")
    shift
    while [ "$#" -gt 0 ]; do
      case "$1" in -i|-t) shift ;; -e) shift 2 ;; *) break ;; esac
    done
    name="$1"; shift
    if is_api "$name"; then
      case "$1 ${2:-}" in
        "test -f") [ "$name" = appuuid1-new ]; exit $? ;;
        "printenv SYNC_DATABASE_URL") exit 1 ;;
        "printenv DATABASE_URL") echo "postgresql+asyncpg://postgres:dev@localhost:5433/kalamos_pf3_rel"; exit 0 ;;
        "printenv RESEND_API_KEY") [ "$name" = appuuid1-new ] && echo "re_in_container"; exit $? ;;
        "python /app/scripts/platform/renumber_lines.py")
          shift 2
          echo "renumber url=${SYNC_DATABASE_URL:-unset}" >> "$FAKE_STATE/renumber.log"
          exec "$FAKE_VENV/bin/python" "$FAKE_REPO/scripts/platform/renumber_lines.py" "$@" ;;
        "python -c") exit 0 ;;
      esac
      echo "fake docker: unhandled exec $*" >&2; exit 99
    fi
    exec "$REAL_DOCKER" "${orig[@]}" ;;
esac
exec "$REAL_DOCKER" "$@"
SH
chmod +x "$WORK/bin/docker"

SECRETS="$WORK/release.env"
cat > "$SECRETS" <<EOF
# fake values for the test
RESEND_API_KEY=$FAKE_KEY
COOLIFY_URL=http://127.0.0.1:$PORT/
COOLIFY_TOKEN="$FAKE_TOKEN"
ADMIN_EMAIL=$ADMIN_EMAIL
EOF
chmod 600 "$SECRETS"

run() { # run NAME [ARGS]: output to $WORK/NAME.out, exit code to $WORK/NAME.rc
  local name=$1; shift
  RELEASE_ALLOW_NONROOT=1 NOMIKOS_SECRETS_FILE="${SECRETS_OVERRIDE:-$SECRETS}" RELEASE_LOG="$WORK/release.log" \
    DOCKER_BIN="$WORK/bin/docker" REAL_DOCKER="$(command -v docker)" FAKE_STATE="$WORK/state" \
    FAKE_VENV="$MAIN_VENV" FAKE_REPO="$ROOT" \
    GITHUB_API_URL="http://127.0.0.1:$PORT/repos/kkkamur07/nomikos/pulls/165" \
    PUBLIC_HEALTH_URL="http://127.0.0.1:$PORT/health" \
    PG_CONTAINER=nomikos-db-1 PG_DATABASE="$DB" PG_USER=postgres \
    MERGE_POLL_SECONDS=1 DEPLOY_POLL_SECONDS=1 DEPLOY_TIMEOUT_SECONDS=20 \
    bash "$SCRIPT" "$@" >"$WORK/$name.out" 2>&1 </dev/null
  echo $? > "$WORK/$name.rc"
}
rc() { cat "$WORK/$1.rc"; }
no_secret() { ! grep -qF -e "$FAKE_KEY" -e "fakeTOKEN" "$WORK/$1.out" "$WORK/release.log"; }
snapshot() {
  q "select (select version_num from alembic_version),
            (select string_agg(name, ',' order by name) from inference_models),
            (select count(*) from blocks),
            (select string_agg(\"order\"::text || ':' || coalesce(block_id::text, '-'), ',' order by id) from lines),
            (select count(*) from information_schema.columns where table_name = 'users' and column_name = 'is_admin'),
            to_regclass('public.feedback')"
}
methods() { python3 -c '
import json, sys
print(" ".join(json.loads(l)["method"] + ":" + json.loads(l)["path"] for l in open(sys.argv[1]) if l.strip()))
' "$WORK/requests.log"; }
show() { sed 's/^/   | /' "$WORK/$1.out"; }

echo "== --dry-run"
BEFORE=$(snapshot)
run dry --dry-run
check "--dry-run exits 0 (rc $(rc dry))" [ "$(rc dry)" = 0 ]
check "--dry-run changes nothing in the database" [ "$(snapshot)" = "$BEFORE" ]
check "--dry-run sends GET requests only" bash -c "! grep -q '\"method\": \"PATCH\"\\|\"method\": \"POST\"\\|/deploy' '$WORK/requests.log'"
check "--dry-run plans the rename" grep -q "2 catalog rename (model_names_v2) *will run" "$WORK/dry.out"
check "--dry-run sees step 1 as applied" grep -q "1 catalog 2026-09-20 rows *already applied" "$WORK/dry.out"
check "--dry-run plans the env write" grep -q "Env keys on nomikos-api (phase 3, no deploy): will write" "$WORK/dry.out"
check "--dry-run masks the key" grep -q "RESEND_API_KEY *re_\*\*\*\*" "$WORK/dry.out"
check "--dry-run prints no secret" no_secret dry
check "--dry-run keeps the secrets file" [ -f "$SECRETS" ]
[ "$FAILS" = 0 ] || show dry

echo "== full run (--yes --keep-secrets)"
run full --yes --keep-secrets
check "full run exits 0 (rc $(rc full))" [ "$(rc full)" = 0 ]
show full
check "alembic_version is 010_feedback" [ "$(q 'select version_num from alembic_version')" = 010_feedback ]
check "catalog renamed" [ "$(q "select string_agg(name, ' ' order by name) from inference_models")" = "armenian-v1 coptic-v1 greek-v1 seg-v1 seg-v2 syriac-v2" ]
check "ADMIN_EMAIL promoted" [ "$(q "select is_admin from users where email = 'admin@example.test'")" = t ]
check "full-page block gone" [ "$(q "select count(*) from blocks where id = '00000000-0000-0000-0000-0000000000e1'")" = 0 ]
check "real region kept" [ "$(q "select count(*) from blocks where id = '00000000-0000-0000-0000-0000000000e2'")" = 1 ]
check "lines unblocked and renumbered 0,1" [ "$(q "select string_agg(\"order\"::text || ':' || coalesce(block_id::text, '-'), ',' order by \"order\") from lines where part_id = '00000000-0000-0000-0000-0000000000d1'")" = "0:-,1:-" ]
check "feedback table exists" [ "$(q "select to_regclass('public.feedback') is not null")" = t ]
check "requests: env PATCHed, no deploy call" [ "$(methods | tr ' ' '\n' | grep -c 'PATCH:/api/v1/applications/appuuid1/envs/bulk')" = 1 ]
check "requests: no deploy call" bash -c "! grep -q '/api/v1/deploy' '$WORK/requests.log'"
check "renumber ran with the derived sync URL" grep -q "url=postgresql://postgres:dev@localhost:5433/$DB" "$WORK/state/renumber.log"
check "database URL never on a docker command line" bash -c "! grep -q 'postgresql' '$WORK/docker.log'"
check "database URL never printed" bash -c "! grep -q 'postgres:dev@' '$WORK/full.out' '$WORK/release.log'"
check "full run prints no secret" no_secret full
check "log file written" grep -q "Release 2026-10-11 done" "$WORK/release.log"
check "secrets file kept (--keep-secrets)" [ -f "$SECRETS" ]

echo "== second full run (everything already applied)"
run again --yes --keep-secrets
check "second run exits 0 (rc $(rc again))" [ "$(rc again)" = 0 ]
for s in "step 0: already applied" "step 1: already applied" "step 2: already applied" "step 3: already applied" \
         "step 3b: already applied" "step 4: already applied" "step 5: already applied" "step 6: already applied" \
         "env: already applied" "deploy: already applied" "renumber: already applied"; do
  check "second run: $s" grep -q "$s" "$WORK/again.out"
done
check "second run sends no second PATCH" [ "$(methods | tr ' ' '\n' | grep -c 'PATCH:')" = 1 ]
check "second run prints no secret" no_secret again
[ "$(rc again)" = 0 ] || show again

echo "== --from-phase 6"
run resume --yes --keep-secrets --from-phase 6
check "--from-phase 6 exits 0 (rc $(rc resume))" [ "$(rc resume)" = 0 ]
check "--from-phase 6 runs phase 6" grep -q "Phase 6: renumber lines" "$WORK/resume.out"
check "--from-phase 6 skips phase 2" bash -c "! grep -q 'Phase 2:' '$WORK/resume.out'"
[ "$(rc resume)" = 0 ] || show resume

echo "== secrets file removed without --keep-secrets"
cp -p "$SECRETS" "$WORK/release-copy.env"
SECRETS_OVERRIDE="$WORK/release-copy.env" run shred --yes --from-phase 7
check "--from-phase 7 exits 0 (rc $(rc shred))" [ "$(rc shred)" = 0 ]
check "secrets file removed" [ ! -e "$WORK/release-copy.env" ]

echo "== refusals"
chmod 644 "$SECRETS"
run badmode --dry-run
check "mode 644 secrets file refused with exit 2" [ "$(rc badmode)" = 2 ]
chmod 600 "$SECRETS"
run badflag --bogus
check "unknown flag exits 1" [ "$(rc badflag)" = 1 ]

echo
if [ "$FAILS" = 0 ]; then echo "PASS: release_2026_10_11.sh"; else echo "FAILED: $FAILS check(s)"; exit 1; fi

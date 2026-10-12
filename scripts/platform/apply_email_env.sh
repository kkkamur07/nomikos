#!/usr/bin/env bash
# Apply the transactional email environment variables to the production API
# through the Coolify REST API, then redeploy the API so they take effect.
#
# STATUS: UNVERIFIED AGAINST PRODUCTION.
# Nothing on the Coolify server has been inspected. The API is assumed to be a
# Coolify application on root@65.21.249.167. The ssh alias `nomikos`
# (95.217.208.209) is the INFERENCE box (systemd nomikos-worker units, no
# docker) and needs no email variables; this script never touches it, and
# never uses ssh at all (HTTPS to Coolify only).
# tests/platform/test_apply_email_env.sh runs it against a stub HTTP server.
#
# Secrets live OUTSIDE this script, in a file only you can read:
#   ${NOMIKOS_SECRETS_FILE:-$HOME/.nomikos-resend.env}   (chmod 600)
# Plain KEY=VALUE lines, `#` comments allowed, optional surrounding quotes.
# The file is parsed line by line, never sourced. Example (placeholders):
#
#   RESEND_API_KEY=re_xxxxxxxxxxxx
#   COOLIFY_URL=https://coolify.example.com:8000
#   COOLIFY_TOKEN=1|xxxxxxxxxxxx
#
# COOLIFY_URL is the Coolify dashboard origin. COOLIFY_TOKEN is created in the
# Coolify UI under Keys & Tokens with the read, write and deploy abilities.
# Optional overrides in the same file: EMAIL_FROM, EMAIL_REPLY_TO,
# APP_PUBLIC_URL, EMAIL_SIGNIN_NOTICES, APP_NAME. EMAIL_REPLY_TO is unset by
# default and then not written at all, so the mails show no reply address;
# set it only to a mailbox someone reads.
#
# What it does:
#   1. Refuses to run unless RESEND_API_KEY, COOLIFY_TOKEN and COOLIFY_URL are
#      present, the key starts with "re_", and the file is not group or world
#      readable. The key and token are never printed (masked re_**** / tok_****).
#   2. GET  /api/v1/applications          find the API application by name
#                                         (contains APP_NAME, default "api").
#   3. PATCH /api/v1/applications/{uuid}/envs/bulk
#                                         set the variables (upsert). If
#                                         the bulk endpoint is missing (older
#                                         Coolify), falls back to one PATCH
#                                         /envs per key, then POST /envs.
#   4. GET  /api/v1/deploy?uuid={uuid}&force=true
#                                         redeploy. A restart IS required:
#                                         Coolify injects variables at deploy
#                                         time, so the running container only
#                                         sees them after a redeploy.
#                                         The redeploy REBUILDS FROM THE HEAD
#                                         OF THE TRACKED BRANCH, so any commit
#                                         pushed there since the last deploy
#                                         ships too. The script prints the
#                                         branch and commit Coolify reports
#                                         and waits 5 seconds for ctrl+c.
#
# Variables set: RESEND_API_KEY, EMAIL_FROM, APP_PUBLIC_URL,
# EMAIL_SIGNIN_NOTICES, and EMAIL_REPLY_TO only when given. Read by
# nomikos/backend/core/settings/email.py.
#
# Usage: apply_email_env.sh [--check | --dry-run] [--no-deploy]
#   --check      validate the secrets file only (presence, mode, key prefix) and
#                list which keys are present, masked; contacts nothing.
#   --dry-run    print the plan with masked values; contacts nothing.
#   --no-deploy  set the variables but skip step 4 (no redeploy). For a release
#                where the merge to the tracked branch deploys anyway
#                (release_2026_10_11.sh calls it this way).
#
# Fallback (token route not possible): open the Coolify dashboard, then
#   Projects > (production project) > (API application) > Environment Variables,
# set the variables above by hand, then press Redeploy.

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

MODE=run
DEPLOY=1
for arg in "$@"; do
  case "$arg" in
    --check) MODE=check ;;
    --dry-run) MODE=dry ;;
    --no-deploy) DEPLOY=0 ;;
    *) echo "usage: $0 [--check | --dry-run] [--no-deploy]" >&2; exit 2 ;;
  esac
done

SECRETS_FILE="${NOMIKOS_SECRETS_FILE:-$HOME/.nomikos-resend.env}"

RESEND_API_KEY="" COOLIFY_TOKEN="" COOLIFY_URL=""
EMAIL_FROM="Nomikos <no-reply@nomikos.app>"
EMAIL_REPLY_TO=""   # unset: not written, the mails invite no reply
APP_PUBLIC_URL="https://app.nomikos.app"   # the app, not the landing site
EMAIL_SIGNIN_NOTICES="true"
APP_NAME="api"   # substring of the Coolify application name

[ -f "$SECRETS_FILE" ] || die "secrets file not found: $SECRETS_FILE (see the header of this script)"
PERM=$(stat -c %a "$SECRETS_FILE" 2>/dev/null || stat -f %Lp "$SECRETS_FILE")
case "$PERM" in
  ?00) ;;
  *) die "$SECRETS_FILE is mode $PERM; run: chmod 600 $SECRETS_FILE" ;;
esac

PRESENT=""
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
    RESEND_API_KEY|COOLIFY_TOKEN|COOLIFY_URL|EMAIL_FROM|EMAIL_REPLY_TO|APP_PUBLIC_URL|EMAIL_SIGNIN_NOTICES|APP_NAME)
      printf -v "$key" '%s' "$val"
      PRESENT="$PRESENT $key" ;;
  esac
done < "$SECRETS_FILE"

[ -n "$RESEND_API_KEY" ] || die "RESEND_API_KEY is missing from $SECRETS_FILE"
[ -n "$COOLIFY_TOKEN" ] || die "COOLIFY_TOKEN is missing from $SECRETS_FILE"
[ -n "$COOLIFY_URL" ] || die "COOLIFY_URL is missing from $SECRETS_FILE"
case "$RESEND_API_KEY" in re_*) ;; *) die "RESEND_API_KEY must start with re_" ;; esac
COOLIFY_URL="${COOLIFY_URL%/}"

KEYS="RESEND_API_KEY EMAIL_FROM APP_PUBLIC_URL EMAIL_SIGNIN_NOTICES"
[ -z "$EMAIL_REPLY_TO" ] || KEYS="$KEYS EMAIL_REPLY_TO"
NKEYS=$(echo "$KEYS" | wc -w | tr -d " ")

MASKED="re_****"
TOKMASK="tok_****"

if [ "$MODE" = check ]; then
  echo "ok: $SECRETS_FILE (mode $PERM)"
  echo "keys present (values masked):"
  for k in $PRESENT; do
    case $k in
      RESEND_API_KEY) echo "  $k=$MASKED" ;;
      COOLIFY_TOKEN) echo "  $k=$TOKMASK" ;;
      *) echo "  $k=set" ;;
    esac
  done
  exit 0
fi

if [ "$MODE" = dry ]; then
  echo "dry run: nothing is contacted."
  echo "Coolify:   $COOLIFY_URL (token: $TOKMASK)"
  echo "Would, in order:"
  echo "  1. GET   $COOLIFY_URL/api/v1/applications   (pick the one whose name contains \"$APP_NAME\")"
  echo "  2. PATCH $COOLIFY_URL/api/v1/applications/<uuid>/envs/bulk with:"
  for k in $KEYS; do
    case $k in
      RESEND_API_KEY) echo "       $k=$MASKED" ;;
      *) echo "       $k=${!k}" ;;
    esac
  done
  [ -n "$EMAIL_REPLY_TO" ] || echo "       (EMAIL_REPLY_TO unset: not written, the mails show no reply address)"
  if [ "$DEPLOY" = 1 ]; then
    echo "  3. GET   $COOLIFY_URL/api/v1/deploy?uuid=<uuid>&force=true"
    echo "     (redeploy: rebuilds from the head of the tracked branch, after a 5 second ctrl+c window)"
  else
    echo "  (--no-deploy: no redeploy; the variables reach the container on the next deploy)"
  fi
  exit 0
fi

command -v curl >/dev/null || die "curl is required"
command -v python3 >/dev/null || die "python3 is required"

RESP=$(mktemp)
trap 'rm -f "$RESP"' EXIT

# api METHOD PATH [JSON_BODY]: response body to $RESP, HTTP status in $CODE.
# Call it directly, never inside $(...): a subshell would lose CODE.
# The token goes through a curl config on a file descriptor and the body
# through stdin, so neither the token nor RESEND_API_KEY appears in `ps`.
api() {
  local method=$1 path=$2 body=${3:-}
  local data=()
  [ -z "$body" ] || data=(--data-binary @-)
  CODE=$(printf '%s' "$body" |
    curl -sS --config <(printf 'header = "Authorization: Bearer %s"\n' "$COOLIFY_TOKEN") \
      -X "$method" -H 'Accept: application/json' -H 'Content-Type: application/json' \
      "${data[@]+"${data[@]}"}" -o "$RESP" -w '%{http_code}' "$COOLIFY_URL$path") ||
    die "could not reach $COOLIFY_URL"
}

echo "Finding the API application (name contains \"$APP_NAME\")..."
api GET /api/v1/applications
case $CODE in 2*) ;; *) die "listing applications failed (HTTP $CODE); check the token abilities and COOLIFY_URL" ;; esac

MATCHES=$(APP_NAME="$APP_NAME" python3 -c '
import json, os, sys
needle = os.environ["APP_NAME"].lower()
for a in json.load(sys.stdin):
    if needle in (a.get("name") or "").lower():
        fields = [a["uuid"], a.get("name"), a.get("git_branch"), a.get("git_commit_sha")]
        print("\t".join(str(f or "") for f in fields))
' < "$RESP")
COUNT=$(printf '%s\n' "$MATCHES" | grep -c . || true)
if [ "$COUNT" -ne 1 ]; then
  echo "expected exactly one application matching \"$APP_NAME\", found $COUNT:" >&2
  [ -z "$MATCHES" ] || printf '%s\n' "$MATCHES" | cut -f1,2 >&2
  die "re-run with APP_NAME=<more specific substring>"
fi
IFS=$'\t' read -r UUID APP_LABEL GIT_BRANCH GIT_COMMIT <<<"$MATCHES" || true
echo "Using application: $APP_LABEL ($UUID)"

BODY=$(KEYS="$KEYS" RESEND_API_KEY="$RESEND_API_KEY" EMAIL_FROM="$EMAIL_FROM" \
  EMAIL_REPLY_TO="$EMAIL_REPLY_TO" APP_PUBLIC_URL="$APP_PUBLIC_URL" \
  EMAIL_SIGNIN_NOTICES="$EMAIL_SIGNIN_NOTICES" python3 -c '
import json, os
keys = os.environ["KEYS"].split()
print(json.dumps({"data": [{"key": k, "value": os.environ[k], "is_preview": False} for k in keys]}))
')

api PATCH "/api/v1/applications/$UUID/envs/bulk" "$BODY"
if [ "${CODE:0:1}" = 2 ]; then
  echo "Set $NKEYS variables via the bulk endpoint."
else
  echo "Bulk endpoint returned HTTP $CODE; setting variables one by one."
  for k in $KEYS; do
    ONE=$(K=$k V="${!k}" python3 -c '
import json, os
print(json.dumps({"key": os.environ["K"], "value": os.environ["V"], "is_preview": False}))')
    api PATCH "/api/v1/applications/$UUID/envs" "$ONE"
    if [ "${CODE:0:1}" != 2 ]; then
      api POST "/api/v1/applications/$UUID/envs" "$ONE"
      [ "${CODE:0:1}" = 2 ] || die "could not set $k (HTTP $CODE). Use the manual fallback in this script's header."
    fi
    echo "  set $k"
  done
fi

if [ "$DEPLOY" = 0 ]; then
  echo "--no-deploy: not redeploying. The variables reach the container on the next deploy"
  echo "of ${APP_LABEL} (branch ${GIT_BRANCH:-unknown})."
  exit 0
fi

echo
echo "Next: redeploy (required for the variables to reach the container)."
echo "This REBUILDS FROM THE HEAD OF THE TRACKED BRANCH, it is not just a restart:"
echo "  branch: ${GIT_BRANCH:-(not reported by Coolify)}"
echo "  commit: ${GIT_COMMIT:-(not reported by Coolify)}"
echo "Every commit on that branch since the last deploy ships with it."
echo "Press ctrl+c within 5 seconds to abort (the variables stay set; redeploy later in the UI)."
sleep 5
echo "Redeploying..."
api GET "/api/v1/deploy?uuid=$UUID&force=true"
[ "${CODE:0:1}" = 2 ] || die "redeploy failed (HTTP $CODE). Press Redeploy in the Coolify UI: Projects > API application."

cat <<EOF

Done. Verify once the deployment finishes (Coolify > Deployments):
  1. Register a throwaway user on $APP_PUBLIC_URL; a welcome mail
     should arrive within a minute (sign-in notices send on each login).
  2. Check the API log in Coolify (application > Logs) for an
     "email_sent id=... user_id=..." line, or open the Resend dashboard > Emails.
  3. Names only on the host: docker exec <api> env | grep -cE '^(EMAIL_|RESEND_|APP_PUBLIC_URL=)'
     should print $NKEYS.
EOF

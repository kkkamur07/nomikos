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
# APP_PUBLIC_URL, EMAIL_SIGNIN_NOTICES, APP_NAME.
#
# What it does:
#   1. Refuses to run unless RESEND_API_KEY, COOLIFY_TOKEN and COOLIFY_URL are
#      present, the key starts with "re_", and the file is not group or world
#      readable. The key and token are never printed (masked re_**** / tok_****).
#   2. GET  /api/v1/applications          find the API application by name
#                                         (contains APP_NAME, default "api").
#   3. PATCH /api/v1/applications/{uuid}/envs/bulk
#                                         set the five variables (upsert). If
#                                         the bulk endpoint is missing (older
#                                         Coolify), falls back to one PATCH
#                                         /envs per key, then POST /envs.
#   4. GET  /api/v1/deploy?uuid={uuid}&force=true
#                                         redeploy. A restart IS required:
#                                         Coolify injects variables at deploy
#                                         time, so the running container only
#                                         sees them after a redeploy.
#
# Variables set: RESEND_API_KEY, EMAIL_FROM, EMAIL_REPLY_TO, APP_PUBLIC_URL,
# EMAIL_SIGNIN_NOTICES. Read by nomikos/backend/core/settings/email.py.
#
# Usage: apply_email_env.sh [--check | --dry-run]
#   --check    validate the secrets file only (presence, mode, key prefix) and
#              list which keys are present, masked; contacts nothing.
#   --dry-run  print the plan with masked values; contacts nothing.
#
# Fallback (token route not possible): open the Coolify dashboard, then
#   Projects > (production project) > (API application) > Environment Variables,
# set the five variables above by hand, then press Redeploy.

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

MODE=run
case "${1:-}" in
  "") ;;
  --check) MODE=check ;;
  --dry-run) MODE=dry ;;
  *) echo "usage: $0 [--check | --dry-run]" >&2; exit 2 ;;
esac

SECRETS_FILE="${NOMIKOS_SECRETS_FILE:-$HOME/.nomikos-resend.env}"

RESEND_API_KEY="" COOLIFY_TOKEN="" COOLIFY_URL=""
EMAIL_FROM="Nomikos <no-reply@nomikos.app>"
EMAIL_REPLY_TO="no-reply@nomikos.app"
APP_PUBLIC_URL="https://nomikos.app"
EMAIL_SIGNIN_NOTICES="true"
APP_NAME="api"   # substring of the Coolify application name

[ -f "$SECRETS_FILE" ] || die "secrets file not found: $SECRETS_FILE (see the header of this script)"
PERM=$(stat -f %Lp "$SECRETS_FILE" 2>/dev/null || stat -c %a "$SECRETS_FILE")
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
  cat <<EOF
dry run: nothing is contacted.
Coolify:   $COOLIFY_URL (token: $TOKMASK)
Would, in order:
  1. GET   $COOLIFY_URL/api/v1/applications   (pick the one whose name contains "$APP_NAME")
  2. PATCH $COOLIFY_URL/api/v1/applications/<uuid>/envs/bulk with:
       RESEND_API_KEY=$MASKED
       EMAIL_FROM=$EMAIL_FROM
       EMAIL_REPLY_TO=$EMAIL_REPLY_TO
       APP_PUBLIC_URL=$APP_PUBLIC_URL
       EMAIL_SIGNIN_NOTICES=$EMAIL_SIGNIN_NOTICES
  3. GET   $COOLIFY_URL/api/v1/deploy?uuid=<uuid>&force=true   (redeploy)
EOF
  exit 0
fi

command -v curl >/dev/null || die "curl is required"
command -v python3 >/dev/null || die "python3 is required"

# The token goes through a curl config on stdin so it never appears in `ps`.
api() { # api METHOD PATH [JSON_BODY]; prints body, sets HTTP code in $CODE
  local method=$1 path=$2 body=${3:-} out
  local data=()
  [ -n "$body" ] && data=(--data-binary "$body")
  out=$(printf 'header = "Authorization: Bearer %s"\n' "$COOLIFY_TOKEN" |
    curl -sS --config - -X "$method" -H 'Accept: application/json' \
      -H 'Content-Type: application/json' "${data[@]+"${data[@]}"}" \
      -w '\n%{http_code}' "$COOLIFY_URL$path") || die "could not reach $COOLIFY_URL"
  CODE=${out##*$'\n'}
  printf '%s' "${out%$'\n'*}"
}

echo "Finding the API application (name contains \"$APP_NAME\")..."
LIST=$(api GET /api/v1/applications)
case $CODE in 2*) ;; *) die "listing applications failed (HTTP $CODE); check the token abilities and COOLIFY_URL" ;; esac

MATCHES=$(printf '%s' "$LIST" | APP_NAME="$APP_NAME" python3 -c '
import json, os, sys
needle = os.environ["APP_NAME"].lower()
for a in json.load(sys.stdin):
    if needle in (a.get("name") or "").lower():
        print(a["uuid"], a.get("name"), sep="\t")
')
COUNT=$(printf '%s\n' "$MATCHES" | grep -c . || true)
if [ "$COUNT" -ne 1 ]; then
  echo "expected exactly one application matching \"$APP_NAME\", found $COUNT:" >&2
  [ -n "$MATCHES" ] && printf '%s\n' "$MATCHES" >&2
  die "re-run with APP_NAME=<more specific substring>"
fi
UUID=${MATCHES%%$'\t'*}
echo "Using application: ${MATCHES#*$'\t'} ($UUID)"

BODY=$(RESEND_API_KEY="$RESEND_API_KEY" EMAIL_FROM="$EMAIL_FROM" EMAIL_REPLY_TO="$EMAIL_REPLY_TO" \
  APP_PUBLIC_URL="$APP_PUBLIC_URL" EMAIL_SIGNIN_NOTICES="$EMAIL_SIGNIN_NOTICES" python3 -c '
import json, os
keys = ["RESEND_API_KEY", "EMAIL_FROM", "EMAIL_REPLY_TO", "APP_PUBLIC_URL", "EMAIL_SIGNIN_NOTICES"]
print(json.dumps({"data": [{"key": k, "value": os.environ[k], "is_preview": False} for k in keys]}))
')

api PATCH "/api/v1/applications/$UUID/envs/bulk" "$BODY" >/dev/null
if [ "${CODE:0:1}" = 2 ]; then
  echo "Set 5 variables via the bulk endpoint."
else
  echo "Bulk endpoint returned HTTP $CODE; setting variables one by one."
  for k in RESEND_API_KEY EMAIL_FROM EMAIL_REPLY_TO APP_PUBLIC_URL EMAIL_SIGNIN_NOTICES; do
    ONE=$(K=$k V="${!k}" python3 -c '
import json, os
print(json.dumps({"key": os.environ["K"], "value": os.environ["V"], "is_preview": False}))')
    api PATCH "/api/v1/applications/$UUID/envs" "$ONE" >/dev/null
    if [ "${CODE:0:1}" != 2 ]; then
      api POST "/api/v1/applications/$UUID/envs" "$ONE" >/dev/null
      [ "${CODE:0:1}" = 2 ] || die "could not set $k (HTTP $CODE). Use the manual fallback in this script's header."
    fi
    echo "  set $k"
  done
fi

echo "Redeploying (required for the variables to reach the container)..."
api GET "/api/v1/deploy?uuid=$UUID&force=true" >/dev/null
[ "${CODE:0:1}" = 2 ] || die "redeploy failed (HTTP $CODE). Press Redeploy in the Coolify UI: Projects > API application."

cat <<EOF

Done. Verify once the deployment finishes (Coolify > Deployments):
  1. Register a throwaway user on https://app.nomikos.app; a welcome mail
     should arrive within a minute (sign-in notices send on each login).
  2. Check the API log in Coolify (application > Logs) for the "sent" line,
     or open the Resend dashboard > Emails.
  3. Names only on the host: docker exec <api> env | grep -c '^EMAIL_\|^RESEND_'
     should print 5.
EOF

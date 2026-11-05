"""Make untrusted or secret-bearing text safe for one log line or one client message.

Two jobs, kept apart on purpose:

* ``sanitize_log_field`` stops log forging: control characters (C0, DEL, CR, LF,
  ESC) become spaces and the value is capped, so one field can never start a
  second, forged log line or carry terminal escapes.
* ``redact_secrets`` masks values that must never leave the process or sit in a
  log in clear: URL userinfo and hosts, query strings, JWTs, AWS key ids and ``X-Amz-*``
  values, socket address tuples, ``host=`` values, emails, UUIDs, Windows and
  ``~`` paths, and the quoted value Postgres echoes back for a bad input.

Regexes are a backstop, not a guarantee: production clients never see free text
at all (see ``backend.core.app.error_summary``); these rules cover the log line
and non-production responses.
"""

from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@'\"]+@")
#: ``scheme://[<redacted>@]host[:port]``, after userinfo is masked.
_URL_HOST = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)((?:<redacted>@)?)[^/\s?#'\"<>@]+")
_QUERY_STRING = re.compile(r"\?[^\s'\"<>]+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?")
_AWS_KEY_ID = re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{12,}\b")
_AMZ_PARAM = re.compile(r"(?i)\b(X-Amz-[A-Za-z-]+)=[^&\s'\"]+")
_ADDRESS_TUPLE = re.compile(r"\(\s*'[^']*'\s*,\s*\d+(?:\s*,[^)]*)?\)")
_HOST_PARAM = re.compile(r"(?i)\bhost=(['\"])[^'\"]*\1")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_UUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_WINDOWS_PATH = re.compile(r"(?<![\w/])[A-Za-z]:\\[^\s'\"]*")
_HOME_PATH = re.compile(r"(?<![\w/])~[\w.-]*/[^\s'\"]*")
#: Postgres and asyncpg quote the offending bound value after these phrases.
_VALUE_ECHO = re.compile(r"(?i)invalid input|for type|value for enum")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def sanitize_log_field(value: str, *, max_len: int) -> str:
    """``value`` on one line: every control character a space, then capped."""
    return _CONTROL_CHARS.sub(" ", value).strip()[:max_len]


def _mask_quoted(match: re.Match[str]) -> str:
    quote = match.group(0)[0]
    return f"{quote}<value>{quote}"


def redact_secrets(text: str) -> str:
    """``text`` with credentials, tokens, bound values, hosts and paths masked."""
    text = _USERINFO.sub(r"\1<redacted>@", text)
    text = _URL_HOST.sub(r"\1\2<host>", text)
    text = _QUERY_STRING.sub("?<redacted>", text)
    text = _AMZ_PARAM.sub(r"\1=<redacted>", text)
    text = _JWT.sub("<jwt>", text)
    text = _AWS_KEY_ID.sub("<aws-key>", text)
    text = _ADDRESS_TUPLE.sub("(<address>)", text)
    text = _HOST_PARAM.sub("host=<redacted>", text)
    echo = _VALUE_ECHO.search(text)
    if echo is not None:
        start = echo.start()
        text = text[:start] + _QUOTED.sub(_mask_quoted, text[start:])
    text = _EMAIL.sub("<email>", text)
    text = _UUID.sub("<uuid>", text)
    text = _WINDOWS_PATH.sub("<path>", text)
    return _HOME_PATH.sub("<path>", text)

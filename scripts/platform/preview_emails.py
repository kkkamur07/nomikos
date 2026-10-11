#!/usr/bin/env python3
"""Render the welcome and sign-in notice emails to HTML + text for review.

Usage: python scripts/platform/preview_emails.py <out_dir>
Writes welcome.html, welcome.txt, signin.html, signin.txt with sample values.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import ensure_nomikos_on_path


def main() -> int:
    ensure_nomikos_on_path()
    from backend.notifications.templates import signin_notice_email, welcome_email

    if len(sys.argv) != 2:
        print("usage: preview_emails.py <out_dir>", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    welcome = welcome_email("Ada", "https://nomikos.app")
    welcome.to = "ada@example.com"
    signin = signin_notice_email(
        "Ada", datetime(2026, 10, 11, 9, 30, tzinfo=UTC), "https://nomikos.app"
    )
    signin.to = "ada@example.com"

    (out_dir / "welcome.html").write_text(welcome.html, encoding="utf-8")
    (out_dir / "welcome.txt").write_text(welcome.text, encoding="utf-8")
    (out_dir / "signin.html").write_text(signin.html, encoding="utf-8")
    (out_dir / "signin.txt").write_text(signin.text, encoding="utf-8")
    print(f"Wrote 4 files to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

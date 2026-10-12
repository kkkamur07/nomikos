"""Renumber every page's lines to a dense 0..n-1 reading order (one-off data fix, ADR 0008).

Before ADR 0008 a drawn line was stored at ``max(order) + 1`` and a deleted one left its
number behind, so existing pages carry gaps, duplicates and lines numbered after lines
they sit above. This keeps each page's current sequence, ``(order, created_at, id)``,
regrouped region-first as the API numbers it (ADR 0011: the lines of each block in block
order, then the lines in no block), and otherwise only closes the gaps and breaks the
ties; it does not move a line within its region. A line drawn
under the old rule stays last on its page: delete it and draw it again to place it.

Dry run by default: it prints every part whose numbering would change and writes
nothing. ``--apply`` writes, one transaction per part, under the same row lock the API
takes for line writes, so it is safe while the app is up.

Production, from a checkout on the box that can reach the database:

    SYNC_DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<db> \\
        uv run python scripts/platform/renumber_lines.py            # dry run, all parts
    SYNC_DATABASE_URL=... uv run python scripts/platform/renumber_lines.py --apply
    SYNC_DATABASE_URL=... uv run python scripts/platform/renumber_lines.py --document <id>
    SYNC_DATABASE_URL=... uv run python scripts/platform/renumber_lines.py --part <id> --apply

Use a direct (session) connection, not a transaction pooler port.
"""

from __future__ import annotations

import argparse
import os
import sys
from uuid import UUID

from _bootstrap import ensure_nomikos_on_path

ensure_nomikos_on_path()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import Connection  # noqa: E402

from backend.document.application.reading_order import (  # noqa: E402
    LineShape,
    block_ranks,
    renumber,
)


def _part_ids(conn: Connection, *, document_id: UUID | None, part_id: UUID | None) -> list[UUID]:
    if part_id is not None:
        return [part_id]
    if document_id is not None:
        rows = conn.execute(
            text('select id from document_parts where document_id = :d order by "order", id'),
            {"d": document_id},
        )
    else:
        rows = conn.execute(text('select id from document_parts order by document_id, "order", id'))
    return [row[0] for row in rows]


def _changes(conn: Connection, part_id: UUID) -> list[tuple[UUID, int, int]]:
    """``(line_id, old_order, new_order)`` for every line whose order would change."""
    rows = conn.execute(
        text(
            'select id, "order", block_id from lines where part_id = :p '
            'order by "order", created_at, id'
        ),
        {"p": part_id},
    ).all()
    lines = [LineShape(id=row[0], order=row[1], block_id=row[2]) for row in rows]
    # The same ranks the API uses, from blocks listed as the repository lists them.
    blocks = conn.execute(
        text('select id from blocks where part_id = :p order by "order", created_at'),
        {"p": part_id},
    ).all()
    orders = renumber(lines, block_ranks(LineShape(id=row[0]) for row in blocks))
    return [
        (line.id, line.order, orders[line.id]) for line in lines if line.order != orders[line.id]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--document", type=UUID, help="only this document's parts")
    scope.add_argument("--part", type=UUID, help="only this part")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="report only (default)")
    mode.add_argument("--apply", action="store_true", help="write the new orders")
    args = parser.parse_args(argv)

    url = os.environ.get("SYNC_DATABASE_URL")
    if not url:
        print("SYNC_DATABASE_URL is not set", file=sys.stderr)
        return 2

    engine = create_engine(url)
    changed_parts = 0
    changed_lines = 0
    with engine.connect() as conn:
        part_ids = _part_ids(conn, document_id=args.document, part_id=args.part)
    for part_id in part_ids:
        with engine.begin() as conn:
            if args.apply:
                conn.execute(
                    text("select id from document_parts where id = :p for update"), {"p": part_id}
                )
            changes = _changes(conn, part_id)
            if not changes:
                continue
            changed_parts += 1
            changed_lines += len(changes)
            moves = ", ".join(f"{old}->{new}" for _, old, new in changes[:8])
            more = f" (+{len(changes) - 8} more)" if len(changes) > 8 else ""
            print(f"part {part_id}: {len(changes)} line(s) renumbered: {moves}{more}")
            if args.apply:
                conn.execute(
                    text('update lines set "order" = :o where id = :i'),
                    [{"o": new, "i": line_id} for line_id, _, new in changes],
                )

    verb = "renumbered" if args.apply else "would renumber"
    print(f"{verb} {changed_lines} line(s) on {changed_parts} of {len(part_ids)} part(s)")
    if not args.apply and changed_parts:
        print("dry run: nothing written; pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

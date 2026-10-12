"""renumber_lines.py numbers a page region-first, as the API does (ADR 0008, 0011)."""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "platform"))

import renumber_lines  # noqa: E402


def test_a_page_not_numbered_region_first_is_regrouped():
    # A PP-OCR page segmented before regions came first: the left column's lines are
    # interleaved with the right column's. The API would regroup it on its next write.
    engine = create_engine("sqlite://")
    part = str(uuid.uuid4())
    left, right = str(uuid.uuid4()), str(uuid.uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(text('create table blocks (id, part_id, "order", created_at)'))
        conn.execute(text('create table lines (id, part_id, "order", created_at, block_id)'))
        conn.execute(
            text("insert into blocks values (:i, :p, :o, :c)"),
            [
                {"i": left, "p": part, "o": 0, "c": now},
                {"i": right, "p": part, "o": 1, "c": now + timedelta(seconds=1)},
            ],
        )
        conn.execute(
            text("insert into lines values (:i, :p, :o, :c, :b)"),
            [
                {"i": f"l{n}", "p": part, "o": n, "c": now, "b": (left, right)[n % 2]}
                for n in range(4)
            ],
        )
        changes = renumber_lines._changes(conn, part)

    assert sorted((line_id, new) for line_id, _, new in changes) == [("l1", 2), ("l2", 1)]

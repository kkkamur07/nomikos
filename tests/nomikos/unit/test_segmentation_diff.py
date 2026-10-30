"""The four training classes, derived from a run record and the live page.

Each test names the editing act it stands for, because that is what the classes
are for: "the model drew a line nobody had to touch", "a human redrew it", "a
human deleted a false positive", "a human drew a line the model missed". A class
that is right for the wrong reason would still make the retraining set wrong.
"""

from __future__ import annotations

import uuid

from backend.document.domain.segmentation_diff import (
    LiveLine,
    RunLine,
    diff_segmentation_run,
    run_lines_from_payload,
)

RUN_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_RUN_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _proposed(external_id: str, order: int, box: list[list[float]]) -> RunLine:
    return RunLine(external_id=external_id, order=order, points=box)


def _live(
    external_id: str | None,
    order: int,
    box: list[list[float]],
    *,
    run_id: uuid.UUID | None = RUN_ID,
    line_id: uuid.UUID | None = None,
) -> LiveLine:
    return LiveLine(
        id=line_id or uuid.uuid4(),
        order=order,
        points=box,
        segmentation_run_id=run_id if external_id else None,
        run_external_id=external_id,
    )


def test_a_line_nobody_touched_is_unchanged():
    proposal = _proposed("l1", 0, _box(10, 10, 90, 30))
    live = _live("l1", 0, _box(10, 10, 90, 30))

    diff = diff_segmentation_run([proposal], [live], run_id=RUN_ID)

    assert [item.external_id for item in diff.unchanged] == ["l1"]
    assert diff.unchanged[0].line_id == live.id
    assert not diff.corrected and not diff.deleted and not diff.added


def test_an_integer_polygon_read_back_as_floats_is_still_unchanged():
    # The two sides take different JSONB round trips. 10 against 10.0 is not an edit.
    proposal = _proposed("l1", 0, [[10, 10], [90, 10], [90, 30], [10, 30]])
    live = _live("l1", 0, [[10.0, 10.0], [90.0, 10.0], [90.0, 30.0], [10.0, 30.0]])

    diff = diff_segmentation_run([proposal], [live], run_id=RUN_ID)

    assert len(diff.unchanged) == 1
    assert not diff.corrected


def test_a_redrawn_line_is_corrected_and_carries_both_geometries():
    proposal = _proposed("l1", 0, _box(10, 10, 90, 30))
    live = _live("l1", 0, _box(12, 11, 94, 31))

    diff = diff_segmentation_run([proposal], [live], run_id=RUN_ID)

    assert len(diff.corrected) == 1
    corrected = diff.corrected[0]
    assert corrected.model_points == _box(10, 10, 90, 30)
    assert corrected.human_points == (_box(12, 11, 94, 31),)
    assert corrected.line_ids == (live.id,)
    assert not diff.unchanged and not diff.deleted and not diff.added


def test_a_deleted_false_positive_keeps_the_models_geometry():
    # The model drew a smudge in the margin; a human deleted it. The live page has
    # no trace of the line, so the run record is the only place its polygon exists.
    kept = _proposed("l1", 0, _box(10, 10, 90, 30))
    smudge = _proposed("l2", 1, _box(10, 40, 20, 46))

    diff = diff_segmentation_run(
        [kept, smudge], [_live("l1", 0, _box(10, 10, 90, 30))], run_id=RUN_ID
    )

    assert [item.external_id for item in diff.deleted] == ["l2"]
    assert diff.deleted[0].model_points == _box(10, 40, 20, 46)
    assert diff.deleted[0].order == 1


def test_a_line_the_model_missed_is_added():
    proposal = _proposed("l1", 0, _box(10, 10, 90, 30))
    drawn_by_hand = _live(None, 1, _box(10, 40, 90, 60))

    diff = diff_segmentation_run(
        [proposal], [_live("l1", 0, _box(10, 10, 90, 30)), drawn_by_hand], run_id=RUN_ID
    )

    assert [item.line_id for item in diff.added] == [drawn_by_hand.id]
    assert diff.added[0].points == _box(10, 40, 90, 60)


def test_lines_linked_to_a_run_that_stores_no_geometry_are_added_not_lost():
    """A back-filled run has ``lines = []`` but its page's lines still link to it."""
    first = _live("l1", 0, _box(10, 10, 90, 30))
    second = _live("l2", 1, _box(10, 40, 90, 60))

    diff = diff_segmentation_run([], [second, first], run_id=RUN_ID)

    assert [item.line_id for item in diff.added] == [first.id, second.id]
    assert diff.unchanged == diff.corrected == diff.deleted == ()


def test_a_split_is_one_correction_carrying_both_pieces():
    # Segment health cut a two-column segment in half. Both pieces inherit the
    # origin, so the pair reads as one line the model got wrong, not as one
    # correction plus a phantom human line.
    proposal = _proposed("l1", 0, _box(10, 10, 900, 30))
    left = _live("l1", 0, _box(10, 10, 440, 30))
    right = _live("l1", 1, _box(460, 10, 900, 30))

    diff = diff_segmentation_run([proposal], [left, right], run_id=RUN_ID)

    assert len(diff.corrected) == 1
    assert diff.corrected[0].line_ids == (left.id, right.id)
    assert diff.corrected[0].human_points == (_box(10, 10, 440, 30), _box(460, 10, 900, 30))
    assert not diff.added and not diff.deleted


def test_a_merge_reads_as_a_grown_primary_plus_a_deleted_fragment():
    # apply_merge keeps the primary's row and deletes the fragment's. The fragment
    # is therefore a deletion carrying its original geometry, and the primary is a
    # correction whose polygon grew to cover both.
    primary = _proposed("l1", 0, _box(10, 10, 400, 30))
    fragment = _proposed("l2", 1, _box(410, 10, 460, 30))
    merged = _live("l1", 0, _box(10, 10, 460, 30))

    diff = diff_segmentation_run([primary, fragment], [merged], run_id=RUN_ID)

    assert [item.external_id for item in diff.deleted] == ["l2"]
    assert diff.deleted[0].model_points == _box(410, 10, 460, 30)
    assert [item.external_id for item in diff.corrected] == ["l1"]
    assert diff.corrected[0].human_points == (_box(10, 10, 460, 30),)


def test_a_line_kept_from_an_older_run_is_not_matched_by_external_id_alone():
    # External ids are unique inside a run, not across runs. A line surviving from
    # last week's run happens to be called "l1" too; pairing it with this week's
    # "l1" would invent a correction and hide a real deletion.
    proposal = _proposed("l1", 0, _box(10, 10, 90, 30))
    stale = _live("l1", 0, _box(10, 10, 90, 30), run_id=OTHER_RUN_ID)

    diff = diff_segmentation_run([proposal], [stale], run_id=RUN_ID)

    assert [item.external_id for item in diff.deleted] == ["l1"]
    assert [item.line_id for item in diff.added] == [stale.id]
    assert not diff.unchanged and not diff.corrected


def test_an_empty_page_reports_every_proposal_as_deleted():
    proposals = [_proposed("l1", 0, _box(10, 10, 90, 30)), _proposed("l2", 1, _box(10, 40, 90, 60))]

    diff = diff_segmentation_run(proposals, [], run_id=RUN_ID)

    assert [item.external_id for item in diff.deleted] == ["l1", "l2"]
    assert not diff.unchanged and not diff.corrected and not diff.added


def test_classes_come_back_in_page_order():
    proposals = [
        _proposed("l3", 2, _box(10, 70, 90, 90)),
        _proposed("l1", 0, _box(10, 10, 90, 30)),
        _proposed("l2", 1, _box(10, 40, 90, 60)),
    ]

    diff = diff_segmentation_run(proposals, [], run_id=RUN_ID)

    assert [item.external_id for item in diff.deleted] == ["l1", "l2", "l3"]


def test_the_stored_payload_reads_back_into_run_lines():
    payload = [
        {
            "external_id": "l1",
            "order": 0,
            "points": [[1, 1], [2, 1], [2, 2], [1, 2]],
            "baseline": {"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
            "kraken_ceiling": None,
        }
    ]

    lines = run_lines_from_payload(payload)

    assert lines == [RunLine(external_id="l1", order=0, points=[[1, 1], [2, 1], [2, 2], [1, 2]])]


def test_an_entry_with_no_external_id_is_skipped_rather_than_fatal():
    # One unusable entry must not take the whole page's provenance down with it.
    payload = [{"order": 0, "points": [[1, 1]]}, {"external_id": "l2", "order": 1, "points": []}]

    assert [line.external_id for line in run_lines_from_payload(payload)] == ["l2"]
    assert run_lines_from_payload(None) == []

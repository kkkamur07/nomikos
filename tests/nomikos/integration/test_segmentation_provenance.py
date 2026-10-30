"""Model output versus human correction, end to end against Postgres.

The pure diff is covered by ``tests/nomikos/unit/test_segmentation_diff.py``. What
that cannot check is the half that lives in the database: that the run is written
at all, that the origin link on a line survives each of the five ways a human can
change a page, and that the two things a user can delete - job history and the
model transcription layer - no longer take the provenance with them.

Every test here names the editing act it stands for. A mutation path that quietly
dropped the link would still pass a test that only asserted the geometry.
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.nomikos.integration.helpers import (  # noqa: F401
    documents_url,
    make_part,
    return_pooled_connections_before_leaving,
)

SEGMENT_MODEL_REF = "registry://blla-segment?tag=stable"


PAGE_WIDTH = 2479.0
PAGE_HEIGHT = 3508.0
LEFT_COLUMN = (300.0, 1100.0)
RIGHT_COLUMN = (1380.0, 2180.0)


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _canonical(lines: list[tuple[str, int, list[list[float]]]]):
    from backend.ml.domain.segment import (
        CanonicalLine,
        CanonicalSegmentResult,
    )

    return CanonicalSegmentResult(
        blocks=[],
        lines=[
            CanonicalLine(
                external_id=external_id,
                order=order,
                # The bottom edge, in the ``{"points": [...]}`` shape the editor and
                # segment health both read. A GeoJSON baseline parses to nothing here
                # and every finder would silently skip the line.
                baseline={"points": [points[3], points[2]]},
                points=points,
                source_metadata={"adapter": "blla"},
            )
            for external_id, order, points in lines
        ],
    )


def _segment_page(
    part_id: str,
    lines: list[tuple[str, int, list[list[float]]]],
    *,
    with_model: bool = True,
    params: dict | None = None,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Run one segmenter pass over a page exactly as the job callback does.

    Returns ``(job_id, segmentation_run_id)``. The job and the registry row are
    seeded here rather than mocked, because "the identity survives deleting the
    job" is only a real statement about real rows.
    """
    from backend.document.application.segment_merge_service import SegmentMergeService
    from backend.document.infrastructure.orm_models import DocumentPart
    from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
    from backend.ml.application.model_identity import identity_for_job
    from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
    from infrastructure.db import sync_system_session

    job_id = uuid.uuid4()
    with sync_system_session() as session:
        # The uploaded fixture is a 1x1 PNG, and segment health's thresholds are all
        # relative to the page, so a 1x1 part makes every finder report nothing.
        # Give the page a manuscript's dimensions, which is also what the run
        # records as the size its coordinates are in.
        part = session.get(DocumentPart, uuid.UUID(part_id))
        part.width = int(PAGE_WIDTH)
        part.height = int(PAGE_HEIGHT)
        model_id = None
        if with_model:
            model_id = uuid.uuid4()
            session.add(
                InferenceModel(
                    id=model_id,
                    name=f"blla-segment-{uuid.uuid4().hex[:6]}",
                    provider="kraken",
                    task=InferenceTask.segment,
                    artifact_ref=SEGMENT_MODEL_REF,
                    default_params={"heatmap_threshold": 0.17},
                )
            )
            session.flush()
        job = Job(
            id=job_id,
            type=JobType.segment,
            status=JobStatus.done,
            payload={"ml_params": params if params is not None else {"heatmap_threshold": 0.3}},
            model_id=model_id,
            document_part_id=uuid.UUID(part_id),
        )
        session.add(job)
        session.flush()
        summary = SegmentMergeService().apply_sync(
            session,
            part_id=uuid.UUID(part_id),
            canonical_segment=_canonical(lines),
            job_id=job_id,
            commit=False,
            model_identity=identity_for_job(job),
        )
        session.commit()
        return job_id, summary.segmentation_run_id


def _stored_lines(part_id: str) -> list:
    from backend.document.infrastructure.orm_models import Line
    from infrastructure.db import sync_system_session

    with sync_system_session() as session:
        return list(
            session.execute(
                select(
                    Line.id,
                    Line.order,
                    Line.points,
                    Line.source,
                    Line.segmentation_run_id,
                    Line.run_external_id,
                    Line.updated_at,
                    Line.updated_by,
                )
                .where(Line.part_id == uuid.UUID(part_id))
                .order_by(Line.order)
            ).all()
        )


def _origins(part_id: str) -> dict[str, uuid.UUID | None]:
    """``external_id -> segmentation_run_id`` for every live line that has one."""
    return {
        row.run_external_id: row.segmentation_run_id
        for row in _stored_lines(part_id)
        if row.run_external_id is not None
    }


def _stored_run(run_id: uuid.UUID):
    from backend.document.infrastructure.orm_models import SegmentationRun
    from infrastructure.db import sync_system_session

    with sync_system_session() as session:
        return session.execute(
            select(SegmentationRun).where(SegmentationRun.id == run_id)
        ).scalar_one_or_none()


def _diff(client: TestClient, headers: dict[str, str], project: dict, document_id, part_id) -> dict:
    response = client.get(
        f"{documents_url(project['id'])}/{document_id}/parts/{part_id}/segmentation-runs/diff",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _segmented_page(
    client: TestClient,
    headers: dict[str, str],
    project: dict,
    *,
    name: str,
    lines: list[tuple[str, int, list[list[float]]]] | None = None,
) -> tuple[str, str, uuid.UUID]:
    document_id, part_id = make_part(client, headers, project, name=name)
    _job_id, run_id = _segment_page(
        part_id,
        lines
        or [
            ("l1", 0, _box(10, 10, 90, 30)),
            ("l2", 1, _box(10, 40, 90, 60)),
        ],
    )
    return document_id, part_id, run_id


def _row(x0: float, x1: float, y: float, height: float = 100.0) -> list[list[float]]:
    return _box(x0, y - height / 2, x1, y + height / 2)


def _two_column_page() -> list[tuple[str, int, list[list[float]]]]:
    """Twenty rows of two columns, the shape segment health's thresholds assume.

    Borrowed from ``tests/nomikos/unit/test_segment_health_service.py``: the
    finders are calibrated on real page statistics, so a four-line toy page trips
    none of them and a test built on one asserts nothing.
    """
    lines: list[tuple[str, int, list[list[float]]]] = []
    order = 0
    for row in range(20):
        y = 400.0 + row * 120.0
        lines.append((f"l{order}", order, _row(*LEFT_COLUMN, y)))
        order += 1
        lines.append((f"l{order}", order, _row(*RIGHT_COLUMN, y)))
        order += 1
    return lines


# --- 1. the run itself -------------------------------------------------------


def test_a_segment_run_records_the_output_and_the_model_that_produced_it(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id = make_part(client, owner_headers, owner_project, name="Recorded run")
    _job_id, run_id = _segment_page(part_id, [("l1", 0, _box(10, 10, 90, 30))])

    run = _stored_run(run_id)

    assert run is not None
    assert run.line_count == 1
    assert [line["external_id"] for line in run.lines] == ["l1"]
    assert run.lines[0]["points"] == _box(10, 10, 90, 30)
    assert run.registry_model_id == "blla-segment"
    assert run.registry_tag == "stable"
    assert run.model_artifact_ref == SEGMENT_MODEL_REF
    assert run.model_provider == "kraken"
    # The effective params the job ran with, not the registry defaults.
    assert run.params == {"heatmap_threshold": 0.3}


def test_every_line_the_merge_inserts_points_back_at_the_run(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Linked lines"
    )

    assert _origins(part_id) == {"l1": run_id, "l2": run_id}


def test_a_fresh_page_reports_every_proposed_line_as_unchanged(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Untouched page"
    )

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)

    assert {item["external_id"] for item in diff["unchanged"]} == {"l1", "l2"}
    assert diff["corrected"] == [] and diff["deleted"] == [] and diff["added"] == []
    assert diff["run"]["id"] == str(run_id)
    assert diff["run"]["registry_model_id"] == "blla-segment"


def test_a_second_run_is_recorded_beside_the_first_and_is_the_default(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, first_run = _segmented_page(
        client, owner_headers, owner_project, name="Two runs"
    )
    _job_id, second_run = _segment_page(part_id, [("r2l1", 0, _box(11, 11, 91, 31))])

    listed = client.get(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/segmentation-runs",
        headers=owner_headers,
    )
    assert listed.status_code == 200, listed.text
    assert [run["id"] for run in listed.json()] == [str(second_run), str(first_run)]
    # Nothing rewrote the first run: it still says what it said.
    assert [line["external_id"] for line in _stored_run(first_run).lines] == ["l1", "l2"]
    assert _diff(client, owner_headers, owner_project, document_id, part_id)["run"]["id"] == str(
        second_run
    )


# --- 2. every mutation path keeps the origin link ---------------------------


def test_patching_one_lines_geometry_keeps_its_origin(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Patched line"
    )
    line_id = str(_stored_lines(part_id)[0].id)

    patched = client.patch(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines/{line_id}",
        headers=owner_headers,
        json={"points": _box(12, 11, 94, 31)},
    )
    assert patched.status_code == 200, patched.text
    # The line now claims a human drew it, which is true, and still remembers which
    # proposal it started as, which is the whole point.
    assert patched.json()["source"] == "manual"
    assert _origins(part_id) == {"l1": run_id, "l2": run_id}

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    corrected = {item["external_id"]: item for item in diff["corrected"]}
    assert set(corrected) == {"l1"}
    assert corrected["l1"]["model_points"] == _box(10, 10, 90, 30)
    assert corrected["l1"]["human_points"] == [_box(12, 11, 94, 31)]
    assert {item["external_id"] for item in diff["unchanged"]} == {"l2"}


def test_a_bulk_save_of_the_whole_page_keeps_every_origin(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    # The editor sends every line on every save. Before the origin lived in its own
    # columns, a client that sent its own ``source_metadata`` could rewrite the
    # provenance of the whole page in one request.
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Bulk saved"
    )
    lines = _stored_lines(part_id)

    replaced = client.put(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "id": str(lines[0].id),
                    "order": 0,
                    "points": _box(12, 11, 94, 31),
                    "source": "manual",
                    "source_metadata": {"client": "wrote this"},
                },
                {"id": str(lines[1].id), "order": 1, "points": _box(10, 40, 90, 60)},
            ]
        },
    )
    assert replaced.status_code == 200, replaced.text

    assert _origins(part_id) == {"l1": run_id, "l2": run_id}
    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert [item["external_id"] for item in diff["corrected"]] == ["l1"]
    assert [item["external_id"] for item in diff["unchanged"]] == ["l2"]


def test_a_new_line_in_a_bulk_save_has_no_origin_and_reads_as_added(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Bulk added"
    )
    lines = _stored_lines(part_id)

    replaced = client.put(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {"id": str(lines[0].id), "order": 0, "points": _box(10, 10, 90, 30)},
                {"id": str(lines[1].id), "order": 1, "points": _box(10, 40, 90, 60)},
                {"order": 2, "points": _box(10, 70, 90, 90), "source": "manual"},
            ]
        },
    )
    assert replaced.status_code == 200, replaced.text

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert len(diff["added"]) == 1
    assert diff["added"][0]["points"] == _box(10, 70, 90, 90)
    assert len(diff["unchanged"]) == 2


def test_drawing_a_line_by_hand_reads_as_a_line_the_model_missed(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Hand drawn"
    )
    created = client.post(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={"order": 2, "kind": "polygon", "points": _box(10, 70, 90, 90)},
    )
    assert created.status_code == 201, created.text

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)

    assert [item["line_id"] for item in diff["added"]] == [created.json()["id"]]


def test_splitting_a_spanning_segment_gives_both_pieces_the_same_origin(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    # One proposed segment swallowed both columns. Segment health cuts it at the
    # gutter; both halves are still that one proposal, so the diff must report one
    # correction carrying two live ids rather than a correction plus a phantom line.
    lines = _two_column_page()
    lines.append(("crosser", len(lines), _row(LEFT_COLUMN[0], RIGHT_COLUMN[1], 500.0)))
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Split page", lines=lines
    )
    crosser_id = str(
        next(row.id for row in _stored_lines(part_id) if row.run_external_id == "crosser")
    )

    base = f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/segment-health"
    report = client.get(base, headers=owner_headers)
    assert report.status_code == 200, report.text
    assert [item["line_id"] for item in report.json()["spanning"]] == [crosser_id]

    applied = client.post(f"{base}/splits", headers=owner_headers, json={"line_id": crosser_id})
    assert applied.status_code == 200, applied.text

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    corrected = {item["external_id"]: item for item in diff["corrected"]}
    assert "crosser" in corrected
    assert len(corrected["crosser"]["line_ids"]) == 2
    assert len(corrected["crosser"]["human_points"]) == 2
    # No piece of a split is ever a line the model missed, and nothing was lost.
    assert diff["added"] == []
    assert diff["deleted"] == []


def test_trimming_two_overlapping_lines_keeps_both_origins(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    # Two proposals drawn over each other. A trim rewrites both polygons and must
    # leave both still attributed to the run that proposed them.
    lines = _two_column_page()
    # The pair sits below the twenty rows, sharing 30% of one mask's height: over
    # the 20% that makes an overlap a defect, under the 50% that calls it a
    # duplicate and refuses a trim. The geometry is the calibrated unit fixture.
    lines.append(("upper", len(lines), _row(*LEFT_COLUMN, 2860.0, height=120.0)))
    lines.append(("lower", len(lines) + 1, _row(*LEFT_COLUMN, 2944.0, height=120.0)))
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Trimmed page", lines=lines
    )
    base = f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/segment-health"
    report = client.get(base, headers=owner_headers)
    assert report.status_code == 200, report.text
    overlaps = [item for item in report.json()["overlaps"] if not item["duplicate"]]
    assert overlaps, report.text

    applied = client.post(
        f"{base}/trims",
        headers=owner_headers,
        json={"upper_id": overlaps[0]["upper_id"], "lower_id": overlaps[0]["lower_id"]},
    )
    assert applied.status_code == 200, applied.text

    # Every proposal still has its live line, and every live line its origin.
    assert len(_origins(part_id)) == len(lines)
    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert diff["added"] == [] and diff["deleted"] == []
    # A trim clips the pair's shared band off whichever outline it has to move, so
    # only one of the two is rewritten here. What matters for provenance is that the
    # rewrite reads as a correction of the model's line rather than a new line, and
    # that the untouched half is still attributed to the same run.
    corrected = {item["external_id"] for item in diff["corrected"]}
    unchanged = {item["external_id"] for item in diff["unchanged"]}
    assert "lower" in corrected
    assert "upper" in corrected | unchanged
    assert _origins(part_id)["upper"] == run_id and _origins(part_id)["lower"] == run_id


def test_deleting_one_line_reports_the_models_geometry_as_deleted(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Deleted line"
    )
    line_id = str(_stored_lines(part_id)[1].id)

    deleted = client.delete(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines/{line_id}",
        headers=owner_headers,
    )
    assert deleted.status_code == 204, deleted.text

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert [item["external_id"] for item in diff["deleted"]] == ["l2"]
    assert diff["deleted"][0]["model_points"] == _box(10, 40, 90, 60)
    assert [item["external_id"] for item in diff["unchanged"]] == ["l1"]


def test_a_bulk_save_that_drops_a_line_reports_it_as_deleted(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Bulk dropped"
    )
    lines = _stored_lines(part_id)

    replaced = client.put(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={"lines": [{"id": str(lines[0].id), "order": 0, "points": _box(10, 10, 90, 30)}]},
    )
    assert replaced.status_code == 200, replaced.text

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert [item["external_id"] for item in diff["deleted"]] == ["l2"]
    assert diff["deleted"][0]["model_points"] == _box(10, 40, 90, 60)


def test_a_segment_health_delete_reports_the_suspects_geometry(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    # A margin speck the segmenter drew and a reviewer removed: the false positive
    # the training set most needs to know about, and the one the old schema erased.
    speck = _row(2300.0, 2340.0, 1000.0, height=20.0)
    lines = _two_column_page()
    lines.append(("speck", len(lines), speck))
    lines.append(("below", len(lines) + 1, _row(*LEFT_COLUMN, 3300.0)))
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Suspect page", lines=lines
    )
    speck_id = str(next(row.id for row in _stored_lines(part_id) if row.run_external_id == "speck"))

    base = f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/segment-health"
    report = client.get(base, headers=owner_headers)
    assert report.status_code == 200, report.text
    assert speck_id in {item["line_id"] for item in report.json()["suspects"]}

    applied = client.post(f"{base}/deletions", headers=owner_headers, json={"line_id": speck_id})
    assert applied.status_code == 200, applied.text

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert [item["external_id"] for item in diff["deleted"]] == ["speck"]
    assert diff["deleted"][0]["model_points"] == speck


def test_a_merge_leaves_the_absorbed_fragment_visible_as_deleted(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    # A merge keeps the primary's row and deletes the fragment's. The fragment is
    # not a false positive, but from the page it is just as gone, so the run record
    # has to be what still holds its polygon.
    fragment_points = _row(960.0, 1060.0, 2900.0)
    lines = _two_column_page()
    lines.append(("primary", len(lines), _row(300.0, 900.0, 2900.0)))
    lines.append(("fragment", len(lines) + 1, fragment_points))
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Merge page", lines=lines
    )
    by_external = {row.run_external_id: str(row.id) for row in _stored_lines(part_id)}

    base = f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/segment-health"
    report = client.get(base, headers=owner_headers)
    assert report.status_code == 200, report.text
    offered = [
        item
        for item in report.json()["fragments"]
        if item["fragment_id"] == by_external["fragment"]
        and item["primary_id"] == by_external["primary"]
    ]
    assert offered, report.text

    applied = client.post(
        f"{base}/merges",
        headers=owner_headers,
        json={"primary_id": by_external["primary"], "fragment_id": by_external["fragment"]},
    )
    assert applied.status_code == 200, applied.text

    diff = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert [item["external_id"] for item in diff["deleted"]] == ["fragment"]
    assert diff["deleted"][0]["model_points"] == fragment_points
    # The line that absorbed it is a correction whose polygon grew, not a new line.
    assert "primary" in {item["external_id"] for item in diff["corrected"]}
    assert diff["added"] == []


def test_a_re_segment_leaves_the_first_runs_record_and_its_pruned_lines_readable(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    # Re-segmentation deletes the machine lines nobody touched and inserts fresh
    # ones. Against the first run that is a page with nothing left, which is
    # exactly what happened, and the first run still says what it proposed.
    document_id, part_id, first_run = _segmented_page(
        client, owner_headers, owner_project, name="Re-segmented"
    )
    _job_id, second_run = _segment_page(
        part_id, [("n1", 0, _box(11, 11, 91, 31)), ("n2", 1, _box(11, 41, 91, 61))]
    )

    against_first = client.get(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}"
        f"/segmentation-runs/diff?run_id={first_run}",
        headers=owner_headers,
    )
    assert against_first.status_code == 200, against_first.text
    body = against_first.json()

    assert {item["external_id"] for item in body["deleted"]} == {"l1", "l2"}
    assert body["deleted"][0]["model_points"] == _box(10, 10, 90, 30)
    assert {item["line_id"] for item in body["added"]} == {
        str(row.id) for row in _stored_lines(part_id)
    }
    # And the second run is clean against the page it produced.
    assert len(_diff(client, owner_headers, owner_project, document_id, part_id)["unchanged"]) == 2


# --- 4. what a user can delete no longer takes the provenance ---------------


def test_clearing_job_history_leaves_the_run_and_its_model_identity(
    client: TestClient,
    owner_user: dict[str, str],
    owner_headers: dict[str, str],
    owner_project: dict,
) -> None:
    from backend.jobs.infrastructure.orm_models import Job, JobStatus
    from infrastructure.db import sync_system_session
    from tests.nomikos.integration.helpers import user_id_for_email

    document_id, part_id = make_part(client, owner_headers, owner_project, name="Cleared history")
    job_id, run_id = _segment_page(part_id, [("l1", 0, _box(10, 10, 90, 30))])
    # The clear route is scoped to a project through the job's document, and to
    # the caller's own jobs; attach the job to this document and to the owner so
    # the real endpoint is what removes it.
    with sync_system_session() as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        job.document_id = uuid.UUID(document_id)
        job.user_id = user_id_for_email(owner_user["email"])
        job.status = JobStatus.done
        session.commit()

    cleared = client.delete(
        f"/jobs/history?project_id={owner_project['id']}", headers=owner_headers
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["deleted"] >= 1
    with sync_system_session() as session:
        assert session.execute(select(Job).where(Job.id == job_id)).scalar_one_or_none() is None

    run = _stored_run(run_id)
    assert run is not None
    # The FK is nulled, which is what SET NULL means, and the copies are why that
    # is survivable: the run still names the model, the tag and the params.
    assert run.created_by_job_id is None
    assert run.registry_model_id == "blla-segment"
    assert run.registry_tag == "stable"
    assert run.params == {"heatmap_threshold": 0.3}
    assert [line["external_id"] for line in run.lines] == ["l1"]
    # And the page's lines still point at it.
    assert _origins(part_id) == {"l1": run_id}


def test_a_model_transcription_layer_keeps_its_identity_when_the_job_is_deleted(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    from backend.document.infrastructure.orm_models import Transcription
    from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
    from infrastructure.db import sync_system_session

    layer_id, job_id = _transcribe_page(
        client, owner_headers, owner_project, name="Transcribed page"
    )[1:3]

    with sync_system_session() as session:
        layer = session.execute(
            select(Transcription).where(Transcription.id == layer_id)
        ).scalar_one()
        assert layer.model_name is not None
        assert layer.registry_model_id == "greek-calamari"
        assert layer.registry_tag == "v1"
        assert layer.model_params == {"batch_size": 4}
        session.execute(select(Job).where(Job.id == job_id)).scalar_one()

    with sync_system_session() as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        assert job.type == JobType.transcribe and job.status == JobStatus.done
        session.delete(job)
        session.commit()

    with sync_system_session() as session:
        layer = session.execute(
            select(Transcription).where(Transcription.id == layer_id)
        ).scalar_one()
        assert layer.created_by_job_id is None
        assert layer.registry_model_id == "greek-calamari"
        assert layer.registry_tag == "v1"
        assert layer.model_artifact_ref == "registry://greek-calamari?tag=v1"


def _transcribe_page(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict, *, name: str
) -> tuple[tuple[str, str], uuid.UUID, uuid.UUID, uuid.UUID]:
    """A page with one line and one model transcription layer over it.

    Returns ``((document_id, part_id), transcription_id, job_id, line_id)``.
    """
    from backend.document.application.transcribe_merge_service import TranscribeMergeService
    from backend.document.infrastructure.orm_models import Line, LineGeometryKind, LineSource
    from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
    from backend.ml.application.model_identity import identity_for_job
    from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
    from infrastructure.db import sync_system_session
    from nomikos_inference.contracts.transcribe import TranscribeRunResponse

    document_id, part_id = make_part(client, owner_headers, owner_project, name=name)
    line_id = uuid.uuid4()
    job_id = uuid.uuid4()
    model_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Line(
                id=line_id,
                part_id=uuid.UUID(part_id),
                baseline={"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                kind=LineGeometryKind.polygon,
                points=_box(10, 10, 90, 30),
                source=LineSource.kraken,
                order=0,
            )
        )
        session.add(
            InferenceModel(
                id=model_id,
                name=f"greek-calamari-{uuid.uuid4().hex[:6]}",
                provider="calamari",
                task=InferenceTask.transcribe,
                artifact_ref="registry://greek-calamari?tag=v1",
                default_params={},
            )
        )
        session.flush()
        job = Job(
            id=job_id,
            type=JobType.transcribe,
            status=JobStatus.done,
            payload={"ml_params": {"batch_size": 4}},
            model_id=model_id,
            document_id=uuid.UUID(document_id),
            document_part_id=uuid.UUID(part_id),
        )
        session.add(job)
        session.flush()
        line = session.get(Line, line_id)
        result = TranscribeMergeService().apply_sync(
            session,
            document_id=uuid.UUID(document_id),
            part_id=uuid.UUID(part_id),
            job_id=job_id,
            lines_with_output=[
                (
                    line,
                    TranscribeRunResponse(
                        text="Αβγ",
                        confidence=0.9,
                        character_confidences=[
                            {"char": "Α", "confidence": 0.9},
                            {"char": "β", "confidence": 0.9},
                            {"char": "γ", "confidence": 0.9},
                        ],
                    ),
                )
            ],
            commit=False,
            model_identity=identity_for_job(job),
        )
        session.commit()
    return (document_id, part_id), uuid.UUID(result["transcription_id"]), job_id, line_id


def test_accepting_model_text_records_which_layer_it_came_from(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    from backend.document.infrastructure.orm_models import (
        LineTranscription,
        Transcription,
        TranscriptionKind,
    )
    from infrastructure.db import sync_system_session

    (document_id, part_id), layer_id, _job_id, line_id = _transcribe_page(
        client, owner_headers, owner_project, name="Accepted text"
    )

    copied = client.post(
        f"{documents_url(owner_project['id'])}/{document_id}/transcriptions/{layer_id}"
        "/copy-to-ground-truth",
        headers=owner_headers,
        json={},
    )
    assert copied.status_code == 200, copied.text

    def ground_truth_row():
        with sync_system_session() as session:
            return session.execute(
                select(LineTranscription)
                .join(Transcription, Transcription.id == LineTranscription.transcription_id)
                .where(
                    LineTranscription.line_id == line_id,
                    Transcription.kind == TranscriptionKind.ground_truth,
                )
            ).scalar_one()

    row = ground_truth_row()
    assert row.text == "Αβγ"
    assert row.seeded_from_transcription_id == layer_id
    ground_truth_id = row.transcription_id

    # A later human correction of the same row keeps the note: this is still text
    # that started as the model's, which is what makes it worth measuring.
    patched = client.patch(
        f"{documents_url(owner_project['id'])}/{document_id}/transcriptions/{ground_truth_id}"
        f"/lines/{line_id}",
        headers=owner_headers,
        json={"text": "Αβγδ"},
    )
    assert patched.status_code == 200, patched.text
    row = ground_truth_row()
    assert row.text == "Αβγδ"
    assert row.seeded_from_transcription_id == layer_id


def test_removing_the_ground_truth_removes_the_accepted_from_note_with_it(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    from backend.document.infrastructure.orm_models import (
        LineTranscription,
        Transcription,
        TranscriptionKind,
    )
    from infrastructure.db import sync_system_session

    (document_id, part_id), layer_id, _job_id, line_id = _transcribe_page(
        client, owner_headers, owner_project, name="Unpaired text"
    )
    copied = client.post(
        f"{documents_url(owner_project['id'])}/{document_id}/transcriptions/{layer_id}"
        "/copy-to-ground-truth",
        headers=owner_headers,
        json={},
    )
    assert copied.status_code == 200, copied.text

    # There is no unpair route; the way a ground-truth row is removed is an explicit
    # null ``approved_text`` on the bulk line save, which is the path
    # ``GroundTruthText.write`` documents as "removes the row rather than storing an
    # empty string". The note has nothing left to describe and goes with it.
    base = f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}"
    cleared = client.put(
        f"{base}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "id": str(line_id),
                    "order": 0,
                    "points": _box(10, 10, 90, 30),
                    "approved_text": None,
                }
            ]
        },
    )
    assert cleared.status_code == 200, cleared.text

    with sync_system_session() as session:
        remaining = session.execute(
            select(LineTranscription)
            .join(Transcription, Transcription.id == LineTranscription.transcription_id)
            .where(
                LineTranscription.line_id == line_id,
                Transcription.kind == TranscriptionKind.ground_truth,
            )
        ).scalar_one_or_none()
    assert remaining is None


# --- 5. reviewer and time ---------------------------------------------------


def test_marking_a_page_reviewed_records_who_and_when_and_unmarking_clears_both(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict, owner_user: dict
) -> None:
    from tests.nomikos.integration.helpers import user_id_for_email

    document_id, part_id = make_part(client, owner_headers, owner_project, name="Reviewed page")
    base = f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}"

    document = client.get(
        f"{documents_url(owner_project['id'])}/{document_id}", headers=owner_headers
    )
    assert document.status_code == 200, document.text
    fresh = next(part for part in document.json()["parts"] if part["id"] == part_id)
    assert fresh["reviewed"] is False
    assert fresh["reviewed_at"] is None
    assert fresh["reviewed_by"] is None

    marked = client.patch(base, headers=owner_headers, json={"reviewed": True})
    assert marked.status_code == 200, marked.text
    body = marked.json()
    assert body["reviewed"] is True
    assert body["reviewed_at"] is not None
    assert body["reviewed_by"] == str(user_id_for_email(owner_user["email"]))

    unmarked = client.patch(base, headers=owner_headers, json={"reviewed": False})
    assert unmarked.status_code == 200, unmarked.text
    assert unmarked.json()["reviewed_at"] is None
    assert unmarked.json()["reviewed_by"] is None


# --- 6. line edit stamps ----------------------------------------------------


def test_editing_a_line_stamps_when_and_who(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict, owner_user: dict
) -> None:
    from tests.nomikos.integration.helpers import user_id_for_email

    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Stamped line"
    )
    before = _stored_lines(part_id)
    assert before[0].updated_at is None and before[0].updated_by is None

    line_id = str(before[0].id)
    patched = client.patch(
        f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}/lines/{line_id}",
        headers=owner_headers,
        json={"points": _box(12, 11, 94, 31)},
    )
    assert patched.status_code == 200, patched.text

    after = {row.id: row for row in _stored_lines(part_id)}
    edited = after[before[0].id]
    assert edited.updated_at is not None
    assert edited.updated_by == user_id_for_email(owner_user["email"])
    # The line nobody touched is still unstamped.
    assert after[before[1].id].updated_at is None


# --- 7. back-fill -----------------------------------------------------------


def test_the_backfill_rebuilds_a_run_and_its_lines_read_as_added(
    client: TestClient,
    owner_headers: dict[str, str],
    owner_project: dict,
    monkeypatch,
) -> None:
    """A page segmented before runs were recorded: the job and the lines' breadcrumbs
    survive, the run row does not. The back-fill must rebuild the run, relink the
    lines, and the diff must then list every one of them as added."""
    import os
    import sys
    from pathlib import Path

    from sqlalchemy import delete, update

    from backend.document.infrastructure.orm_models import Line, SegmentationRun
    from infrastructure.db import sync_system_session

    document_id, part_id = make_part(client, owner_headers, owner_project, name="Back-filled")
    job_id, run_id = _segment_page(
        part_id, [("l1", 0, _box(10, 10, 90, 30)), ("l2", 1, _box(10, 40, 90, 60))]
    )
    with sync_system_session() as session:
        session.execute(
            update(Line)
            .where(Line.part_id == uuid.UUID(part_id))
            .values(segmentation_run_id=None, run_external_id=None)
        )
        session.execute(delete(SegmentationRun).where(SegmentationRun.id == run_id))
        session.commit()

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "platform"))
    import backfill_segmentation_runs

    monkeypatch.setenv("BACKFILL_DATABASE_URL", os.environ["SYNC_DATABASE_URL"])
    assert backfill_segmentation_runs.run(apply=True) == 0

    with sync_system_session() as session:
        rebuilt = session.execute(
            select(SegmentationRun).where(SegmentationRun.created_by_job_id == job_id)
        ).scalar_one()
        assert rebuilt.lines == []
        assert rebuilt.registry_model_id == "blla-segment"
    assert _origins(part_id) == {"l1": rebuilt.id, "l2": rebuilt.id}

    body = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert len(body["added"]) == 2
    assert body["unchanged"] == body["corrected"] == body["deleted"] == []


def test_restoring_a_snapshot_brings_a_deleted_proposal_back_unchanged(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    document_id, part_id, run_id = _segmented_page(
        client, owner_headers, owner_project, name="Restored proposal"
    )
    part_url = f"{documents_url(owner_project['id'])}/{document_id}/parts/{part_id}"
    snapshot = client.post(f"{part_url}/history", headers=owner_headers)
    assert snapshot.status_code in (200, 201), snapshot.text

    doomed = next(row for row in _stored_lines(part_id) if row.run_external_id == "l2")
    deleted = client.delete(f"{part_url}/lines/{doomed.id}", headers=owner_headers)
    assert deleted.status_code == 204, deleted.text
    assert _origins(part_id) == {"l1": run_id}

    restored = client.post(
        f"{part_url}/history/{snapshot.json()['id']}/restore", headers=owner_headers
    )
    assert restored.status_code == 200, restored.text

    assert _origins(part_id) == {"l1": run_id, "l2": run_id}
    body = _diff(client, owner_headers, owner_project, document_id, part_id)
    assert sorted(item["external_id"] for item in body["unchanged"]) == ["l1", "l2"]
    assert body["added"] == body["deleted"] == []

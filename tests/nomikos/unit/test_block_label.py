"""Region labels (ADR 0011): the request schemas, and grouping a page by region."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError as PydanticValidationError

from backend.document.api.schemas import BlockCreateRequest, BlockPatchRequest, BlockResponse
from backend.document.application.patch_fields import BLOCK_PATCH_FIELDS
from backend.document.application.regions import (
    RegionHeading,
    contains,
    line_in_box,
    lines_by_region,
    region_text,
)


def test_create_takes_an_optional_label_and_assigns_contained_lines_by_default():
    body = BlockCreateRequest.model_validate({"order": 0, "box": {"points": []}})
    assert body.label is None
    assert body.assign_contained is True

    named = BlockCreateRequest.model_validate(
        {"order": 0, "box": {"points": []}, "label": "  4v ", "assign_contained": False}
    )
    assert named.label == "4v"
    assert named.assign_contained is False


def test_a_label_longer_than_the_column_is_refused():
    with pytest.raises(PydanticValidationError):
        BlockCreateRequest.model_validate({"order": 0, "box": {}, "label": "x" * 65})


def test_patch_label_null_or_empty_clears_it_and_omitted_leaves_it():
    assert BlockPatchRequest.model_validate({"label": None}).model_dump(exclude_unset=True) == {
        "label": None
    }
    assert BlockPatchRequest.model_validate({"label": "  "}).model_dump(exclude_unset=True) == {
        "label": None
    }
    assert BlockPatchRequest.model_validate({"order": 2}).model_dump(exclude_unset=True) == {
        "order": 2
    }
    assert "label" in BLOCK_PATCH_FIELDS


def test_block_response_carries_the_label():
    assert "label" in BlockResponse.model_fields


# --- geometry ---


def test_contains_is_even_odd_and_needs_a_polygon():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert contains(square, 5, 5)
    assert not contains(square, 15, 5)
    assert not contains([[0, 0], [10, 10]], 5, 5)


def test_a_line_is_in_a_box_when_its_centre_is():
    box = {"points": [[0, 0], [100, 0], [100, 100], [0, 100]]}
    assert line_in_box([[10, 10], [90, 10], [90, 20], [10, 20]], box)
    # Mostly outside: the centre is at x 110.
    assert not line_in_box([[80, 10], [140, 10], [140, 20], [80, 20]], box)
    assert not line_in_box([], box)
    assert not line_in_box([[10, 10]], None)


# --- grouping ---


def _block(order: int, label: str | None = None):
    return SimpleNamespace(id=uuid.uuid4(), order=order, label=label)


def _line(text: str, block=None):
    return SimpleNamespace(text=text, block_id=block.id if block is not None else None)


def test_a_page_without_regions_is_one_group_in_its_own_order():
    lines = [_line("a"), _line("b")]
    assert lines_by_region([], lines) == [(None, lines)]


def test_regions_group_in_block_order_with_unknown_and_loose_lines_last():
    first, second = _block(0), _block(1)
    stray = _line("stray")
    stray.block_id = uuid.uuid4()  # a block not on this page
    lines = [_line("s1", second), _line("loose"), _line("f1", first), stray, _line("s2", second)]
    groups = lines_by_region([first, second], lines)
    assert [(block, [line.text for line in members]) for block, members in groups] == [
        (first, ["f1"]),
        (second, ["s1", "s2"]),
        (None, ["loose", "stray"]),
    ]


def test_region_text_heads_labelled_regions_that_have_text():
    labelled, unlabelled, empty = _block(0, "4v"), _block(1), _block(2, "6r")
    lines = [
        _line("a", labelled),
        _line("b", unlabelled),
        _line("  ", empty),
        _line("c"),
    ]
    assert region_text([labelled, unlabelled, empty], lines, lambda line: line.text) == [
        RegionHeading("4v"),
        "a",
        "b",
        "c",
    ]


# --- document PDF ---


def test_the_document_pdf_draws_each_labelled_region_heading():
    import io

    from pypdf import PdfReader

    from backend.annotation.application.transcription_pdf_service import _PdfLine
    from backend.document.application.document_export_pdf import (
        DocumentTranscriptionPdfService,
        _pdf_regions,
        _PdfPage,
    )

    blocks = [
        SimpleNamespace(label="4v", box={"points": [[50, 80], [650, 80], [650, 400]]}),
        SimpleNamespace(label=None, box={"points": [[750, 80], [1400, 80], [1400, 400]]}),
        SimpleNamespace(label="5r", box={}),
    ]
    regions = _pdf_regions(blocks)
    assert [(region.label, region.x, region.y) for region in regions] == [("4v", 50, 80)]

    line = _PdfLine(text="verso", points=[[50, 100], [650, 100], [650, 140], [50, 140]])
    content = DocumentTranscriptionPdfService()._render_document_pdf(
        [_PdfPage(width=1500, height=1000, lines=[line], regions=regions)]
    )
    text = PdfReader(io.BytesIO(content)).pages[0].extract_text()
    assert "4v" in text
    assert "verso" in text

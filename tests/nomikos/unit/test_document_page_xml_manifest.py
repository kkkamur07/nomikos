"""The manifest a document PAGE XML archive carries, without a database.

The archive is what a training set is built from, so these tests pin the two
things a person rebuilding that set later has to be able to read back: which
pages count as corrected, and how many of the baselines the platform invented.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from backend.annotation.application.page_xml_export_service import PageLineCounts, PageXmlBundle
from backend.document.application.document_export_page_xml import (
    CORRECTED_RULE,
    MANIFEST_VERSION,
    _manifest,
    _manifest_page,
)
from backend.document.application.document_export_selection import DocumentExportSelection


def _part(*, order, reviewed):
    return SimpleNamespace(id=uuid4(), order=order, reviewed=reviewed)


def _bundle(stem, counts):
    return PageXmlBundle(
        stem=stem,
        xml=b"<PcGts/>",
        image_filename=f"{stem}.webp",
        image=b"RIFF",
        line_counts=counts,
    )


def _selection(parts):
    document = SimpleNamespace(id=uuid4(), name="Σιναϊτικός κώδικας")
    return DocumentExportSelection(
        document=document,
        parts=list(parts),
        page_numbers={part.id: number for number, part in enumerate(parts, start=1)},
    )


def test_a_reviewed_page_with_ground_truth_is_corrected() -> None:
    part = _part(order=0, reviewed=True)
    selection = _selection([part])
    counts = PageLineCounts(total=3, kraken=2, manual=1, derived_baseline=1, ground_truth=3)

    page = _manifest_page(selection, part, _bundle("0001", counts))

    assert page["page_number"] == 1
    assert page["xml"] == "0001.xml"
    assert page["image"] == "0001.webp"
    assert page["reviewed"] is True
    assert page["corrected"] is True
    assert page["lines"]["derived_baseline"] == 1
    assert page["lines"]["ground_truth"] == 3


def test_a_reviewed_page_with_no_ground_truth_is_not_corrected() -> None:
    # Somebody ticked "reviewed" on a page whose lines carry only model output.
    part = _part(order=0, reviewed=True)

    page = _manifest_page(_selection([part]), part, _bundle("0001", PageLineCounts(total=4)))

    assert page["reviewed"] is True
    assert page["corrected"] is False


def test_an_unreviewed_page_with_ground_truth_is_not_corrected() -> None:
    # Transcribed but never signed off: conservative, so it does not count.
    part = _part(order=0, reviewed=False)
    counts = PageLineCounts(total=4, ground_truth=4)

    page = _manifest_page(_selection([part]), part, _bundle("0001", counts))

    assert page["corrected"] is False


def test_the_manifest_states_the_rule_the_archive_was_built_under() -> None:
    parts = [_part(order=0, reviewed=True), _part(order=1, reviewed=False)]
    selection = _selection(parts)
    pages = [
        _manifest_page(
            selection, parts[0], _bundle("0001", PageLineCounts(total=2, ground_truth=2))
        ),
        _manifest_page(selection, parts[1], _bundle("0002", PageLineCounts(total=2))),
    ]

    manifest = _manifest(selection, pages, reviewed_only=False)

    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["document"]["name"] == "Σιναϊτικός κώδικας"
    assert manifest["reviewed_only"] is False
    assert manifest["corrected_rule"] == CORRECTED_RULE
    assert manifest["page_count"] == 2
    assert manifest["corrected_page_count"] == 1
    assert manifest["derived_baseline"]["height_fraction"] == 0.68
    assert "baseline:derived" in manifest["derived_baseline"]["marked_in_xml_as"]


def test_an_empty_selection_still_produces_a_readable_manifest() -> None:
    manifest = _manifest(_selection([]), [], reviewed_only=True)

    assert manifest["page_count"] == 0
    assert manifest["corrected_page_count"] == 0
    assert manifest["reviewed_only"] is True

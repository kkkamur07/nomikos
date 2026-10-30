"""Unit test for the PAGE XML builder in scripts/hf/export_release_corpus.py.

No network: a fake two-column part (left column x 0 to 100, right column
x 200 to 300, same y values) with transcription order left column first
then right column must come out in text order, not geometric y order. The
ReadingOrder group must hold a RegionRefIndexed pointing at the TextRegion,
never a TextLineRefIndexed, and each TextLine carries a Transkribus-style
readingOrder custom attribute. Unpaired lines sort after all paired lines
and carry no TextEquiv.
"""

from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path
from xml.etree.ElementTree import fromstring

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hf" / "export_release_corpus.py"
NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def _load():
    spec = importlib.util.spec_from_file_location("export_release_corpus", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _column(x0: int, x1: int, index: int, line_id: str, text: str | None) -> dict:
    y = 100 + index * 60
    return {
        "id": line_id,
        "points": [[x0, y], [x1, y], [x1, y + 40], [x0, y + 40]],
        "baseline": [[x0, y + 30], [x1, y + 30]],
        "text": text,
        "text_order": None,
    }


def test_page_xml_builder_follows_text_order_and_valid_reading_order():
    builder = _load()
    left = [_column(0, 100, i, f"left-{i}", f"left text {i}") for i in range(2)]
    right = [_column(200, 300, i, f"right-{i}", f"right text {i}") for i in range(2)]
    for position, line in enumerate(left + right):
        line["text_order"] = position
    stray = _column(0, 100, 5, "stray", None)
    lines = right + [stray] + left  # scrambled input order on purpose
    xml = builder.build_page_xml("part-1", "0001_page.png", 800, 600, lines)
    root = fromstring(xml)  # noqa: S314 - XML produced by this test, not untrusted input
    assert root.tag == f"{{{NAMESPACE}}}PcGts"

    ns = {"p": NAMESPACE}
    assert root.findall(".//p:TextLineRefIndexed", ns) == []
    text_lines = root.findall(".//p:TextLine", ns)
    assert len(text_lines) == 5
    ids = [element.get("id") for element in text_lines]
    # Text order (left column first, then right), unpaired last.
    assert ids == [
        "line_left-0",
        "line_left-1",
        "line_right-0",
        "line_right-1",
        "line_stray",
    ]

    regions = root.findall(".//p:TextRegion", ns)
    assert len(regions) == 1
    refs = root.findall(".//p:RegionRefIndexed", ns)
    assert len(refs) == 1
    assert refs[0].get("index") == "0"
    assert refs[0].get("regionRef") == regions[0].get("id")

    for index, element in enumerate(text_lines[:4]):
        assert element.get("custom") == f"readingOrder {{index:{index};}}"
    unicodes = {
        element.get("id"): element.find("p:TextEquiv/p:Unicode", ns) for element in text_lines
    }
    assert unicodes["line_left-0"].text == unicodedata.normalize("NFC", "left text 0")
    assert unicodes["line_right-1"].text == unicodedata.normalize("NFC", "right text 1")

    stray_element = next(element for element in text_lines if element.get("id") == "line_stray")
    assert stray_element.find("p:TextEquiv", ns) is None
    assert stray_element.get("custom") == ("readingOrder {index:4;} structure {type:unpaired;}")

    page = root.find("p:Page", ns)
    assert page is not None
    assert page.get("imageFilename") == "0001_page.png"
    assert page.get("imageWidth") == "800"
    assert page.get("imageHeight") == "600"
    metadata = root.find("p:Metadata", ns)
    assert metadata is not None
    assert metadata.find("p:Creator", ns).text == "NOMIKOS"
    assert metadata.find("p:Created", ns) is not None
    assert metadata.find("p:LastChange", ns) is not None

    for element in text_lines:
        coords = element.find("p:Coords", ns).get("points")
        assert all(part.split(",")[0].lstrip("-").isdigit() for part in coords.split())


def _gt(text: str) -> list[dict]:
    return [
        {"transcription_kind": "model", "text": "model guess"},
        {"transcription_kind": "ground_truth", "text": text},
    ]


def test_collect_part_lines_takes_ground_truth_text_and_candidate_order():
    builder = _load()
    box = [[0, 0], [10, 0], [10, 10], [0, 10]]
    layout = {
        "lines": [
            {"id": "corrected", "points": box, "line_transcriptions": _gt("fixed text")},
            {"id": "typed", "points": box, "line_transcriptions": _gt("typed text")},
            {"id": "bare", "points": box, "line_transcriptions": []},
        ]
    }
    pairing = {
        "text_lines": [
            {"order": 0, "text": "other line", "paired_line_id": None},
            {"order": 1, "text": "old candidate", "paired_line_id": "corrected"},
        ]
    }

    lines, text_count, paired_count = builder.collect_part_lines(layout, pairing)

    by_id = {line["id"]: line for line in lines}
    assert by_id["corrected"]["text"] == "fixed text"
    assert by_id["corrected"]["text_order"] == 1
    assert by_id["typed"]["text"] == "typed text"
    assert by_id["typed"]["text_order"] is None
    assert by_id["bare"]["text"] is None
    assert (text_count, paired_count) == (2, 2)

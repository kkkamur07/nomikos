"""Unit test for the PAGE XML builder in scripts/hf/export_release_corpus.py.

No network: a fake part with two paired lines and one unpaired line must
give valid XML in the PAGE 2019-07-15 namespace, top to bottom reading
order, round-tripped text, and no TextEquiv on the unpaired line.
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


def test_page_xml_builder_orders_text_and_marks_unpaired():
    builder = _load()
    lines = [
        {
            "id": "lower",
            "points": [[10, 300], [200, 300], [200, 340], [10, 340]],
            "baseline": [[10, 330], [200, 330]],
            "text": "δεύτερη γραμμή",
        },
        {
            "id": "upper",
            "points": [[10, 100], [200, 100], [200, 140], [10, 140]],
            "baseline": [[10, 130], [200, 130]],
            "text": "πρώτη γραμμή",
        },
        {
            "id": "stray",
            "points": [[10, 200], [200, 200], [200, 240], [10, 240]],
            "baseline": [[10, 230], [200, 230]],
            "text": None,
        },
    ]
    xml = builder.build_page_xml("part-1", "0001_page.png", 800, 600, lines)
    root = fromstring(xml)  # noqa: S314 - XML produced by this test, not untrusted input
    assert root.tag == f"{{{NAMESPACE}}}PcGts"

    ns = {"p": NAMESPACE}
    text_lines = root.findall(".//p:TextLine", ns)
    assert len(text_lines) == 3
    ids = [element.get("id") for element in text_lines]
    # Top to bottom by mean baseline y: upper (130), stray (230), lower (330).
    assert ids == ["line_upper", "line_stray", "line_lower"]

    refs = root.findall(".//p:TextLineRefIndexed", ns)
    assert [ref.get("textLineId") for ref in refs] == ids

    unicodes = {
        element.get("id"): element.find("p:TextEquiv/p:Unicode", ns) for element in text_lines
    }
    assert unicodes["line_upper"] is not None
    assert unicodes["line_upper"].text == unicodedata.normalize("NFC", "πρώτη γραμμή")
    assert unicodes["line_lower"] is not None
    assert unicodes["line_lower"].text == unicodedata.normalize("NFC", "δεύτερη γραμμή")

    stray = next(element for element in text_lines if element.get("id") == "line_stray")
    assert stray.find("p:TextEquiv", ns) is None
    assert stray.get("custom") == "unpaired"

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

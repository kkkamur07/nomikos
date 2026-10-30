"""Unit test for scripts/hf/build_release_datasets.py. No network, no real data."""

from __future__ import annotations

import csv
import importlib.util
import json
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement
from xml.etree.ElementTree import parse as xml_parse
from xml.etree.ElementTree import tostring as xml_tostring

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hf" / "build_release_datasets.py"
NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def _load():
    spec = importlib.util.spec_from_file_location("build_release_datasets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tag(name: str) -> str:
    return f"{{{NAMESPACE}}}{name}"


def _fake_page_xml(image_name: str, width: int, height: int, lines: list[dict]) -> bytes:
    root = Element(_tag("PcGts"))
    metadata = SubElement(root, _tag("Metadata"))
    SubElement(metadata, _tag("Creator")).text = "NOMIKOS"
    page = SubElement(
        root,
        _tag("Page"),
        {
            "imageFilename": image_name,
            "imageWidth": str(width),
            "imageHeight": str(height),
        },
    )
    region = SubElement(page, _tag("TextRegion"), {"id": "region_p", "type": "paragraph"})
    for position, line in enumerate(lines):
        text_line = SubElement(
            region,
            _tag("TextLine"),
            {"id": f"line_{position}", "custom": f"readingOrder {{index:{position};}}"},
        )
        SubElement(
            text_line,
            _tag("Coords"),
            {"points": " ".join(f"{x},{y}" for x, y in line["poly"])},
        )
        if line["text"] is not None:
            equiv = SubElement(text_line, _tag("TextEquiv"))
            SubElement(equiv, _tag("Unicode")).text = line["text"]
    return xml_tostring(root, encoding="utf-8", xml_declaration=True)


@pytest.fixture()
def fake_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    builder = _load()
    raw = tmp_path / "raw"
    monkeypatch.setitem(
        builder.MANUSCRIPTS,
        "ms-ship",
        {
            "project_slug": "proj",
            "doc_slug": "doc-ship",
            "shelfmark": "Shelf Ship",
            "source_url": "https://example.invalid/ship",
        },
    )
    monkeypatch.setitem(
        builder.MANUSCRIPTS,
        "ms-noimg",
        {
            "project_slug": "proj",
            "doc_slug": "doc-noimg",
            "shelfmark": "Shelf Noimg",
            "source_url": "https://example.invalid/noimg",
        },
    )
    monkeypatch.setitem(builder.SHIP_IMAGES, "ms-ship", True)
    monkeypatch.setitem(builder.SHIP_IMAGES, "ms-noimg", False)
    monkeypatch.setitem(
        builder.REPOS,
        "test-repo",
        {"language": "grc", "manuscripts": ["ms-ship", "ms-noimg"]},
    )
    recorded: dict[str, dict] = {}

    def stub_write_parquet(path: Path, columns: dict, schema_kind: str) -> None:
        recorded[schema_kind] = columns
        path.with_suffix(".json").write_text(json.dumps(columns, default=str))

    monkeypatch.setattr(builder, "write_parquet", stub_write_parquet)

    def write_page(doc: str, order: int, lines: list[dict], size=(60, 80)):
        base = raw / "proj" / doc
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "page").mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", size, "white")
        image.save(base / "images" / f"{order:04d}_part.png")
        (base / "page" / f"{order:04d}_part.xml").write_bytes(
            _fake_page_xml(f"{order:04d}_part.png", size[0], size[1], lines)
        )

    write_page(
        "doc-ship",
        0,
        [
            {"poly": [[10, 10], [30, 10], [30, 20], [10, 20]], "text": "alpha"},
            {"poly": [[10, 30], [30, 30], [30, 40], [10, 40]], "text": "beta"},
            {"poly": [[10, 50], [30, 50], [30, 60], [10, 60]], "text": None},
        ],
    )
    write_page("doc-ship", 1, [{"poly": [[5, 5], [9, 5], [9, 9], [5, 9]], "text": None}])
    write_page(
        "doc-noimg",
        0,
        [{"poly": [[1, 1], [11, 1], [11, 11], [1, 11]], "text": "gamma"}],
    )
    out = tmp_path / "release"
    counts = builder.build_repo(raw, out, tmp_path / "cards", "test-repo", {})
    return builder, out / "test-repo", counts, recorded, raw


def test_release_layout_manifest_and_xml(fake_tree):
    _, repo_dir, counts, recorded, _ = fake_tree
    assert counts == {"pages": 2, "lines": 3, "images_shipped": 1}
    # The zero-paired page is excluded everywhere.
    assert not (repo_dir / "page-xml" / "ms-ship" / "0001.xml").exists()
    assert (repo_dir / "images" / "ms-ship" / "0000.png").exists()
    assert not (repo_dir / "images" / "ms-noimg").exists()
    # One row per paired line, in reading order.
    assert len(recorded["lines"]["line_id"]) == 3
    assert recorded["lines"]["text"] == ["alpha", "beta", "gamma"]
    assert recorded["lines"]["line_index"] == [0, 1, 0]
    assert recorded["lines"]["image"][2] is None
    assert recorded["lines"]["image_shipped"] == [True, True, False]
    assert recorded["pages"]["page_id"] == ["ms-ship/0000", "ms-noimg/0000"]
    assert recorded["pages"]["objects"][0]["paired"] == [True, True, False]
    assert recorded["pages"]["image"][1] is None
    with (repo_dir / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["manuscript"], row["page_index"]) for row in rows] == [
        ("ms-ship", "0"),
        ("ms-noimg", "0"),
    ]
    assert [row["n_paired"] for row in rows] == ["2", "1"]
    assert [row["image_sha256"] != "" for row in rows] == [True, False]
    root = xml_parse(str(repo_dir / "page-xml" / "ms-ship" / "0000.xml")).getroot()  # noqa: S314 - XML this test wrote, not untrusted input
    page = root.find(_tag("Page"))
    image_ref = page.get("imageFilename")
    assert image_ref == "../../images/ms-ship/0000.png"
    xml_dir = repo_dir / "page-xml" / "ms-ship"
    assert (xml_dir / image_ref).resolve() == (repo_dir / "images" / "ms-ship" / "0000.png")
    assert page.get("imageWidth") == "60"
    unshipped = xml_parse(str(repo_dir / "page-xml" / "ms-noimg" / "0000.xml")).getroot()  # noqa: S314 - XML this test wrote, not untrusted input
    assert unshipped.find(_tag("Page")).get("imageFilename") == "0000.png"


def test_rebuild_drops_withdrawn_page_files(fake_tree, tmp_path: Path):
    builder, repo_dir, _, _, raw = fake_tree
    assert (repo_dir / "page-xml" / "ms-ship" / "0000.xml").exists()
    # Page 0000 loses its last paired line: its files must go on the next build.
    (raw / "proj" / "doc-ship" / "page" / "0000_part.xml").write_bytes(
        _fake_page_xml(
            "0000_part.png",
            60,
            80,
            [{"poly": [[10, 10], [30, 10], [30, 20], [10, 20]], "text": None}],
        )
    )
    counts = builder.build_repo(raw, repo_dir.parent, tmp_path / "cards", "test-repo", {})
    assert counts == {"pages": 1, "lines": 1, "images_shipped": 0}
    assert not (repo_dir / "page-xml" / "ms-ship" / "0000.xml").exists()
    assert not (repo_dir / "images" / "ms-ship" / "0000.png").exists()
    assert (repo_dir / "page-xml" / "ms-noimg" / "0000.xml").exists()
    assert sorted(path.name for path in repo_dir.parent.iterdir()) == ["test-repo"]


def test_ship_images_switch():
    builder = _load()
    assert builder.SHIP_IMAGES == {
        "bnf-grec-1360": True,
        "bnf-armenien-172": True,
        "ucla-armenian-ms-72": True,
        "alqosh-65": True,
        "bav-borg-copt-109": False,
        "naples-ib-10": False,
    }


def _local_opening_xml() -> bytes:
    root = Element(_tag("PcGts"))
    metadata = SubElement(root, _tag("Metadata"))
    SubElement(metadata, _tag("Creator")).text = "NOMIKOS"
    page = SubElement(
        root,
        _tag("Page"),
        {"imageFilename": "opening.png", "imageWidth": "1000", "imageHeight": "800"},
    )
    region = SubElement(page, _tag("TextRegion"), {"id": "region_p", "type": "paragraph"})
    lines = [
        ("right", [[800, 100], [950, 100], [950, 130], [800, 130]], "right text"),
        ("left", [[100, 200], [300, 200], [300, 230], [100, 230]], "left text"),
        ("stray", [[800, 400], [950, 400], [950, 430], [800, 430]], None),
    ]
    for position, (name, poly, text) in enumerate(lines):
        text_line = SubElement(
            region,
            _tag("TextLine"),
            {"id": f"line_{name}", "custom": f"readingOrder {{index:{position};}}"},
        )
        SubElement(
            text_line,
            _tag("Coords"),
            {"points": " ".join(f"{x},{y}" for x, y in poly)},
        )
        SubElement(
            text_line,
            _tag("Baseline"),
            {"points": f"{poly[0][0]},{poly[0][1] + 20} {poly[1][0]},{poly[1][1] + 20}"},
        )
        if text is not None:
            equiv = SubElement(text_line, _tag("TextEquiv"))
            SubElement(equiv, _tag("Unicode")).text = text
    return xml_tostring(root, encoding="utf-8", xml_declaration=True)


def test_local_page_xml_reading_order_and_folio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip(
        "backend.annotation.application.page_xml_geometry",
        reason="derive_baseline lands with the fix/training-data-export PR",
    )
    builder = _load()
    raw = tmp_path / "raw"
    monkeypatch.setitem(
        builder.MANUSCRIPTS,
        "bnf-grec-1360",
        {
            "project_slug": "proj",
            "doc_slug": "doc-grk",
            "shelfmark": "Shelf Grec",
            "source_url": "https://example.invalid/grec",
        },
    )
    monkeypatch.setitem(builder.LOCAL_PAGE_XML, "bnf-grec-1360", {2: "opening.xml"})
    recorded: dict[str, dict] = {}

    def stub_write_parquet(path: Path, columns: dict, schema_kind: str) -> None:
        recorded[schema_kind] = columns
        path.with_suffix(".json").write_text(json.dumps(columns, default=str))

    monkeypatch.setattr(builder, "write_parquet", stub_write_parquet)

    def write_page(order: int, lines: list[dict]):
        base = raw / "proj" / "doc-grk"
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "page").mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (1000, 800), "white")
        image.save(base / "images" / f"{order:04d}_part.png")
        (base / "page" / f"{order:04d}_part.xml").write_bytes(
            _fake_page_xml(f"{order:04d}_part.png", 1000, 800, lines)
        )

    lone = {"poly": [[10, 10], [30, 10], [30, 20], [10, 20]], "text": None}
    # Order 0 carries no paired text and no local entry, so it drops out.
    write_page(0, [dict(lone)])
    write_page(
        1,
        [
            {
                "poly": [[10, 10], [30, 10], [30, 20], [10, 20]],
                "text": "production text",
            }
        ],
    )
    # Order 2 carries no production text; the local opening rescues it.
    write_page(2, [dict(lone)])
    local_base = tmp_path / "local"
    (local_base / "xml").mkdir(parents=True)
    (local_base / "xml" / "opening.xml").write_bytes(_local_opening_xml())

    out = tmp_path / "release"
    counts = builder.build_repo(
        raw, out, tmp_path / "cards", "greek-manuscript-htr", {}, local_base
    )
    assert counts == {"pages": 2, "lines": 3, "images_shipped": 2}
    repo_dir = out / "greek-manuscript-htr"
    with (repo_dir / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Folios come from the raw part order, not the filtered position.
    assert [(row["page_index"], row["folio"]) for row in rows] == [
        ("1", "1v-2r"),
        ("2", "2v-3r"),
    ]
    assert [row["text_source"] for row in rows] == ["production", "local-page-xml"]
    # Left folio block sorts before the right block; empty text is unpaired.
    assert recorded["lines"]["text"] == ["production text", "left text", "right text"]
    assert recorded["lines"]["folio"] == ["1v-2r", "2v-3r", "2v-3r"]
    assert recorded["lines"]["text_source"] == [
        "production",
        "local-page-xml",
        "local-page-xml",
    ]
    assert recorded["pages"]["folio"] == ["1v-2r", "2v-3r"]
    assert recorded["pages"]["text_source"] == ["production", "local-page-xml"]
    assert recorded["pages"]["objects"][1]["paired"] == [True, True, False]
    # The stored Baseline is ignored and one is derived inside the polygon.
    assert recorded["lines"]["baseline"][1] == [[102, 220], [298, 220]]
    root = xml_parse(str(repo_dir / "page-xml" / "bnf-grec-1360" / "0002.xml")).getroot()  # noqa: S314 - XML this test wrote, not untrusted input
    texts = [
        element.find(f"{_tag('TextEquiv')}/{_tag('Unicode')}")
        for element in root.findall(f".//{_tag('TextLine')}")
    ]
    assert [(element.text if element is not None else None) for element in texts] == [
        "left text",
        "right text",
        None,
    ]


def test_folio_labels():
    builder = _load()
    assert [builder.folio_for_page("bav-borg-copt-109", order) for order in range(4)] == [
        "1r",
        "1v",
        "2r",
        "2v",
    ]
    assert builder.folio_for_page("bav-borg-copt-109", 17) == "9v"
    assert builder.folio_for_page("naples-ib-10", 0) == "c 10"
    assert builder.folio_for_page("naples-ib-10", 17) == "c 27"
    assert builder.folio_for_page("naples-ib-10", 18) == "c 27v"
    assert builder.folio_for_page("bnf-grec-1360", 0) == "1r"
    assert builder.folio_for_page("bnf-grec-1360", 1) == "1v-2r"
    assert builder.folio_for_page("bnf-grec-1360", 4) == "4v-5r"
    assert builder.folio_for_page("alqosh-65", 0) == ""


def test_crop_size(fake_tree):
    builder, repo_dir, _, _, _ = fake_tree
    blob = (repo_dir / "images" / "ms-ship" / "0000.png").read_bytes()
    crop = builder.crop_line(blob, [10, 10, 30, 20], (60, 80))
    with Image.open(BytesIO(crop)) as handle:
        # Tight box 20x10 plus 4 px padding on every side.
        assert handle.size == (28, 18)


def test_parquet_writer_round_trip(tmp_path: Path):
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    builder = _load()
    lines_path = tmp_path / "lines.parquet"
    builder.write_parquet(
        lines_path,
        {
            "line_id": ["m/0000/000"],
            "manuscript": ["m"],
            "shelfmark": ["Shelf"],
            "page_index": [0],
            "line_index": [0],
            "text": ["alpha"],
            "image": [{"bytes": b"png", "path": "m/0000/000.png"}],
            "polygon": [[[10, 10], [30, 20]]],
            "baseline": [[[10, 18], [30, 18]]],
            "bbox": [[10, 10, 30, 20]],
            "page_width": [60],
            "page_height": [80],
            "image_shipped": [True],
            "source_url": ["https://example.invalid"],
            "folio": ["1r"],
            "text_source": ["production"],
        },
        "lines",
    )
    table = pq.read_table(lines_path)
    assert table.num_rows == 1
    assert table.column("text").to_pylist() == ["alpha"]
    assert table.schema.field("image").type == pyarrow.struct(
        [pyarrow.field("bytes", pyarrow.binary()), pyarrow.field("path", pyarrow.string())]
    )

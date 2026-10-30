#!/usr/bin/env python3
"""Build the four Hugging Face release folders from the raw export.

Reads `data/hf-release/raw/` (page images, PAGE XML, layout/pairing JSON)
plus `data/hf-release/inventory.json`, and writes one folder per language
under `data/hf-release/release/`, each with native PAGE XML, a Parquet
`lines` config, a Parquet `pages` config, page images only where the holding
library allows redistribution, a manifest.csv, and the dataset card copied
from `docs/datasets/release-cards/<repo>.md`.

No network: everything comes from the local staging tree. Nothing is
uploaded here.

Parquet is written with `datasets` when it is installed, else with
`pyarrow` using the Hub image struct `{bytes: binary, path: string}`.
Neither package is a project dependency; both are optional imports.

Example::

    python scripts/hf/build_release_datasets.py
    python scripts/hf/build_release_datasets.py --repo greek-manuscript-htr
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import fromstring as xml_fromstring
from xml.etree.ElementTree import tostring as xml_tostring

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "data" / "hf-release" / "raw"
DEFAULT_OUT = REPO_ROOT / "data" / "hf-release" / "release"
DEFAULT_CARDS = REPO_ROOT / "docs" / "datasets" / "release-cards"

PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

# Per-manuscript image switch. Flipped by the project owner's release decision.
SHIP_IMAGES = {
    "bnf-grec-1360": True,
    "bnf-armenien-172": True,
    "ucla-armenian-ms-72": True,
    "alqosh-65": True,
    "bav-borg-copt-109": False,
    "naples-ib-10": False,
}

# Earlier NOMIKOS PAGE XML export of the same transcription, per raw part
# order, for pages whose production part carries no paired text. The file
# names resolve under the --grec1360-local directory (its xml/ subfolder).
LOCAL_PAGE_XML = {
    "bnf-grec-1360": {0: "page1.xml", 1: "page2.xml", 2: "page3.xml"},
}

# Centre line between the two folios of a Grec 1360 opening, in pixels.
GREC_FOLIO_SPLIT_X = 725

MANUSCRIPTS = {
    "bnf-grec-1360": {
        "project_slug": "byzantine-greek-manuscripts",
        "doc_slug": "grec1360",
        "shelfmark": "Paris, Bibliothèque nationale de France, Grec 1360",
        "source_url": "https://gallica.bnf.fr/",
    },
    "bnf-armenien-172": {
        "project_slug": "armenian",
        "doc_slug": "ms-p-172",
        "shelfmark": ("Paris, Bibliothèque nationale de France, Arménien 172, ff. 215v-223v"),
        "source_url": "https://gallica.bnf.fr/",
    },
    "ucla-armenian-ms-72": {
        "project_slug": "armenian",
        "doc_slug": "ms-ucla-ms",
        "shelfmark": ("Los Angeles, UCLA Library Special Collections, Armenian MS 72, ff. 64v-73v"),
        "source_url": "https://digital.library.ucla.edu/catalog/ark:/21198/zz0026w379",
    },
    "alqosh-65": {
        "project_slug": "eastsyriac",
        "doc_slug": "alqosh65",
        "shelfmark": "Alqosh, Chaldean collection (Notre-Dame des Semences), MS 65",
        "source_url": "https://www.vhmml.org/",
    },
    "bav-borg-copt-109": {
        "project_slug": "coptic",
        "doc_slug": "vatican-copt-borg-109",
        "shelfmark": (
            "Vatican City, Biblioteca Apostolica Vaticana, Borg. copt. 109, cass. XXIX, fasc. 159"
        ),
        "source_url": "https://digi.vatlib.it/view/MSS_Borg.copt.109",
    },
    "naples-ib-10": {
        "project_slug": "coptic",
        "doc_slug": "napoli-ib1010-27",
        "shelfmark": ('Naples, Biblioteca Nazionale "Vittorio Emanuele III", IB.10.10-24'),
        "source_url": "",
    },
}

REPOS = {
    "greek-manuscript-htr": {
        "language": "grc",
        "manuscripts": ["bnf-grec-1360"],
    },
    "armenian-manuscript-htr": {
        "language": "xcl",
        "manuscripts": ["bnf-armenien-172", "ucla-armenian-ms-72"],
    },
    "syriac-manuscript-htr": {
        "language": "syc",
        "manuscripts": ["alqosh-65"],
    },
    "coptic-manuscript-htr": {
        "language": "cop",
        "manuscripts": ["bav-borg-copt-109", "naples-ib-10"],
    },
}

CROP_PADDING = 4


class BuildError(RuntimeError):
    """The release cannot be built from the staging tree as it stands."""


def _ns(tag: str) -> str:
    return f"{{{PAGE_XML_NAMESPACE}}}{tag}"


def _parse_points(text: str) -> list[list[int]]:
    points = []
    for chunk in text.split():
        x, y = chunk.split(",")
        points.append([int(round(float(x))), int(round(float(y)))])
    return points


def parse_page_xml_from_bytes(data: bytes) -> dict:
    """Read PAGE XML bytes into plain dicts (document order kept)."""
    from xml.etree.ElementTree import fromstring

    root = fromstring(data)  # noqa: S314 - XML this script wrote, not untrusted input
    page = root.find(_ns("Page"))
    if page is None:
        raise BuildError("page XML: no Page element")
    lines = []
    for element in root.findall(f".//{_ns('TextLine')}"):
        coords = element.find(_ns("Coords"))
        baseline = element.find(_ns("Baseline"))
        unicode = element.find(f"{_ns('TextEquiv')}/{_ns('Unicode')}")
        lines.append(
            {
                "id": element.get("id", ""),
                "polygon": _parse_points(coords.get("points", "")),
                "baseline": (
                    _parse_points(baseline.get("points", "")) if baseline is not None else None
                ),
                "text": unicode.text if unicode is not None else None,
            }
        )
    return {
        "image_filename": page.get("imageFilename", ""),
        "width": int(page.get("imageWidth", "0")),
        "height": int(page.get("imageHeight", "0")),
        "lines": lines,
    }


def parse_page_xml(path: Path) -> dict:
    """Read one exported PAGE XML into plain dicts (document order kept)."""
    return parse_page_xml_from_bytes(path.read_bytes())


_export_builder = None


def _production_xml_builder():
    """Load build_page_xml from the sibling export script (never edited here)."""
    global _export_builder
    if _export_builder is None:
        import importlib.util

        sibling = Path(__file__).resolve().parent / "export_release_corpus.py"
        spec = importlib.util.spec_from_file_location("export_release_corpus", sibling)
        if spec is None or spec.loader is None:
            raise BuildError(f"cannot load {sibling}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _export_builder = module.build_page_xml
    return _export_builder


def folio_for_page(manuscript: str, order: int) -> str:
    """Folio or image label for a raw part order. Empty when not known."""
    if manuscript == "bav-borg-copt-109":
        if 0 <= order <= 17:
            return f"{order // 2 + 1}{'r' if order % 2 == 0 else 'v'}"
        return ""
    if manuscript == "naples-ib-10":
        if order == 18:
            return "c 27v"
        if 0 <= order <= 17:
            return f"c {10 + order}"
        return ""
    if manuscript == "bnf-grec-1360":
        # Raw order 0 carries text on the right-hand page only (every line
        # x-centre at or above 725), so it is fol. 1r. Order N >= 1 is the
        # opening fol. Nv (left) and fol. N+1r (right): order 4 holds 26
        # paired lines left of 725 (fol. 4v) and 6 right of it (fol. 5r).
        if order == 0:
            return "1r"
        return f"{order}v-{order + 1}r"
    return ""


def sort_grec_opening(line_dicts: list[dict]) -> list[dict]:
    """Order a Grec 1360 opening: left folio top to bottom, then right folio.

    Lines whose polygon x-centre is below 725 come first, top to bottom by
    mean baseline y (polygon mean y when no baseline), then the lines at or
    above 725, top to bottom.
    """

    def key(line: dict) -> tuple[int, float]:
        polygon = line.get("polygon") or line.get("points") or []
        centre = sum(point[0] for point in polygon) / len(polygon) if polygon else 0.0
        ref = line.get("baseline") or polygon
        mean = sum(point[1] for point in ref) / len(ref) if ref else 0.0
        return (0 if centre < GREC_FOLIO_SPLIT_X else 1, mean)

    return sorted(line_dicts, key=key)


def _derive_baseline():
    """``derive_baseline`` from the backend, imported from this checkout."""
    nomikos = str(REPO_ROOT / "nomikos")
    if nomikos not in sys.path:
        sys.path.insert(0, nomikos)
    from backend.annotation.application.page_xml_geometry import derive_baseline

    return derive_baseline


def load_local_grec_page(local_base: Path, manuscript: str, order: int) -> dict | None:
    """Read one local PAGE XML page for a raw part order, or None.

    Returns width, height and lines in build_page_xml input shape, with
    text_order following the opening reading order. Empty text is unpaired.
    Baselines are derived from the polygons.
    """
    wanted = LOCAL_PAGE_XML.get(manuscript, {}).get(order)
    if wanted is None:
        return None
    path = local_base / "xml" / wanted
    if not path.exists():
        raise BuildError(f"{manuscript} order {order}: no local PAGE XML at {path}")
    page = parse_page_xml(path)
    derive_baseline = _derive_baseline()
    entries = []
    for raw_line in page["lines"]:
        text = raw_line["text"]
        if text is not None and not text.strip():
            text = None
        points = [[float(x), float(y)] for x, y in raw_line["polygon"]]
        entries.append(
            {
                "id": raw_line["id"],
                "points": points,
                # The old export wrote the polygon into Baseline, so the stored
                # baseline is ignored and one is derived from the polygon.
                "baseline": derive_baseline(points),
                "text": text,
                "text_order": None,
            }
        )
    for position, line in enumerate(sort_grec_opening(entries)):
        if line["text"] is not None:
            line["text_order"] = position
    return {"width": page["width"], "height": page["height"], "entries": entries}


def tight_bbox(polygon: list[list[int]]) -> list[int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def crop_line(image_bytes: bytes, bbox: list[int], page_size: tuple[int, int]) -> bytes:
    """Cut the padded, page-clipped line box and return it as PNG bytes."""
    from PIL import Image

    width, height = page_size
    x0 = max(0, bbox[0] - CROP_PADDING)
    y0 = max(0, bbox[1] - CROP_PADDING)
    x1 = min(width, bbox[2] + CROP_PADDING)
    y1 = min(height, bbox[3] + CROP_PADDING)
    with Image.open(BytesIO(image_bytes)) as handle:
        crop = handle.convert("RGB").crop((x0, y0, x1, y1))
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        return buffer.getvalue()


def load_part_ids(inventory_path: Path) -> dict[tuple[str, str, int], str]:
    """Map (project_slug, doc_slug, part_order) to platform part id."""
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, str, int], str] = {}
    for project in inventory["projects"]:
        for document in project["documents"]:
            for part in document["parts"]:
                mapping[(project["slug"], document["slug"], int(part["order"] or 0))] = str(
                    part["id"]
                )
    return mapping


def collect_manuscript_pages(
    raw: Path,
    manuscript: str,
    part_ids: dict[tuple[str, str, int], str],
    grec_local: Path | None = None,
) -> list[dict]:
    """Gather the included pages (at least one paired line) of one manuscript.

    Page NNNN is raw part order N: folio labels and file numbers both come
    from that order, never from the position in this filtered list. Greek
    orders listed in LOCAL_PAGE_XML take lines and text from the local PAGE
    XML export and the page image from the raw production export.
    """
    from PIL import Image

    spec = MANUSCRIPTS[manuscript]
    ship = SHIP_IMAGES[manuscript]
    base = raw / spec["project_slug"] / spec["doc_slug"]
    xml_files = sorted((base / "page").glob("*.xml"))
    if not xml_files:
        raise BuildError(f"{base}: no exported pages")
    pages = []
    for xml_path in xml_files:
        order = int(xml_path.stem.split("_", 1)[0])
        matches = sorted((base / "images").glob(f"{order:04d}_*"))
        if not matches:
            raise BuildError(f"{base}: no image for page order {order}")
        ext = matches[0].suffix.lstrip(".") or "png"
        image_bytes = matches[0].read_bytes() if ship else None
        local = (
            load_local_grec_page(grec_local, manuscript, order) if grec_local is not None else None
        )
        if local is not None:
            with Image.open(matches[0]) as handle:
                raw_size = handle.size
            if (local["width"], local["height"]) != (raw_size[0], raw_size[1]):
                raise BuildError(
                    f"{manuscript} order {order}: local PAGE XML size "
                    f"{local['width']}x{local['height']} != raw image {raw_size[0]}x{raw_size[1]}"
                )
            _build_xml = _production_xml_builder()
            entries = [
                {**line, "id": line["id"].removeprefix("line_")} for line in local["entries"]
            ]
            part_id = part_ids.get((spec["project_slug"], spec["doc_slug"], order), "")
            xml_bytes = _build_xml(
                part_id or f"{manuscript}-{order}",
                f"{order:04d}.{ext}",
                local["width"],
                local["height"],
                entries,
            )
            page = parse_page_xml_from_bytes(xml_bytes)
            page["xml_bytes"] = xml_bytes
            text_source = "local-page-xml"
        else:
            page = parse_page_xml(xml_path)
            text_source = "production"
        paired = [line for line in page["lines"] if line["text"] is not None]
        if not paired:
            continue
        pages.append(
            {
                "manuscript": manuscript,
                "page_index": order,
                "part_id": part_ids.get((spec["project_slug"], spec["doc_slug"], order), ""),
                "width": page["width"],
                "height": page["height"],
                "lines": page["lines"],
                "ext": ext,
                "image_bytes": image_bytes,
                "folio": folio_for_page(manuscript, order),
                "text_source": text_source,
                "local_xml": local is not None,
                "xml_bytes": page.get("xml_bytes"),
            }
        )
    pages.sort(key=lambda item: item["page_index"])
    return pages


def hub_image(blob: bytes | None, name: str) -> dict | None:
    if blob is None:
        return None
    return {"bytes": blob, "path": name}


def write_parquet(path: Path, columns: dict, schema_kind: str) -> None:
    """Write one Parquet file with `datasets` if present, else `pyarrow`."""
    try:
        from datasets import Dataset
        from datasets import Image as HFImage

        dataset = Dataset.from_dict(columns)
        for name in ("image",):
            if name in columns:
                dataset = dataset.cast_column(name, HFImage())
        dataset.to_parquet(str(path))
        return
    except ImportError:
        pass
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise BuildError(
            "writing Parquet needs the `datasets` or `pyarrow` package, and neither is installed"
        ) from exc
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    if schema_kind == "lines":
        schema = pa.schema(
            [
                pa.field("line_id", pa.string()),
                pa.field("manuscript", pa.string()),
                pa.field("shelfmark", pa.string()),
                pa.field("page_index", pa.int32()),
                pa.field("line_index", pa.int32()),
                pa.field("text", pa.string()),
                pa.field("image", image_type, nullable=True),
                pa.field("polygon", pa.list_(pa.list_(pa.int32()))),
                pa.field(
                    "baseline",
                    pa.list_(pa.field("point", pa.list_(pa.int32()), nullable=True)),
                    nullable=True,
                ),
                pa.field("bbox", pa.list_(pa.int32())),
                pa.field("page_width", pa.int32()),
                pa.field("page_height", pa.int32()),
                pa.field("image_shipped", pa.bool_()),
                pa.field("source_url", pa.string()),
                pa.field("folio", pa.string()),
                pa.field("text_source", pa.string()),
            ]
        )
    else:
        objects_type = pa.struct(
            [
                pa.field("id", pa.list_(pa.string())),
                pa.field("bbox", pa.list_(pa.list_(pa.int32()))),
                pa.field("polygon", pa.list_(pa.list_(pa.list_(pa.int32())))),
                pa.field(
                    "baseline",
                    pa.list_(pa.field("line", pa.list_(pa.list_(pa.int32())), nullable=True)),
                ),
                pa.field("text", pa.list_(pa.string())),
                pa.field("paired", pa.list_(pa.bool_())),
            ]
        )
        schema = pa.schema(
            [
                pa.field("page_id", pa.string()),
                pa.field("manuscript", pa.string()),
                pa.field("shelfmark", pa.string()),
                pa.field("page_index", pa.int32()),
                pa.field("image", image_type, nullable=True),
                pa.field("width", pa.int32()),
                pa.field("height", pa.int32()),
                pa.field("image_shipped", pa.bool_()),
                pa.field("source_url", pa.string()),
                pa.field("folio", pa.string()),
                pa.field("text_source", pa.string()),
                pa.field("objects", objects_type),
            ]
        )
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, str(path))


def rewrite_page_xml_text(data: bytes, image_name: str) -> bytes:
    """Return the page XML with imageFilename set to the release image path."""
    root = xml_fromstring(data)  # noqa: S314 - XML this script wrote, not untrusted input
    page = root.find(_ns("Page"))
    if page is None:
        raise BuildError("page XML: no Page element")
    page.set("imageFilename", image_name)
    return xml_tostring(root, encoding="utf-8", xml_declaration=True)


def release_image_path(manuscript: str, image_name: str, ship: bool) -> str:
    """imageFilename as seen from page-xml/<manuscript>/, where images/ is two levels up.

    Unshipped images have no release folder, so the bare file name is kept.
    """
    return f"../../images/{manuscript}/{image_name}" if ship else image_name


def build_repo(
    raw: Path,
    out: Path,
    cards: Path,
    repo: str,
    part_ids: dict[tuple[str, str, int], str],
    grec_local: Path | None = None,
) -> dict:
    """Build one release folder. Returns counts for the summary.

    The folder is built fresh beside the target and swapped in only after the
    build succeeds, so files of withdrawn pages (or images no longer shipped)
    never survive from an earlier build.
    """
    out.mkdir(parents=True, exist_ok=True)
    final_dir = out / repo
    staging = out / f".{repo}.building"
    retired = out / f".{repo}.old"
    for leftover in (staging, retired):
        if leftover.exists():
            shutil.rmtree(leftover)
    try:
        counts = _write_repo(raw, staging, cards, repo, part_ids, grec_local)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if final_dir.exists():
        final_dir.rename(retired)
    staging.rename(final_dir)
    shutil.rmtree(retired, ignore_errors=True)
    return counts


def _write_repo(
    raw: Path,
    repo_dir: Path,
    cards: Path,
    repo: str,
    part_ids: dict[tuple[str, str, int], str],
    grec_local: Path | None,
) -> dict:
    """Write one release folder into an empty ``repo_dir``."""
    spec = REPOS[repo]
    (repo_dir / "data" / "lines").mkdir(parents=True, exist_ok=True)
    (repo_dir / "data" / "pages").mkdir(parents=True, exist_ok=True)
    line_rows: dict[str, list] = {name: [] for name in _LINES_COLUMNS}
    page_rows: dict[str, list] = {name: [] for name in _PAGES_COLUMNS}
    manifest_rows = []
    for manuscript in spec["manuscripts"]:
        mspec = MANUSCRIPTS[manuscript]
        ship = SHIP_IMAGES[manuscript]
        pages = collect_manuscript_pages(raw, manuscript, part_ids, grec_local)
        if ship:
            (repo_dir / "images" / manuscript).mkdir(parents=True, exist_ok=True)
        (repo_dir / "page-xml" / manuscript).mkdir(parents=True, exist_ok=True)
        for page in pages:
            index = page["page_index"]
            tag = f"{index:04d}"
            image_name = f"{tag}.{page['ext']}"
            xml_image = release_image_path(manuscript, image_name, ship)
            if ship and page["image_bytes"] is not None:
                (repo_dir / "images" / manuscript / image_name).write_bytes(page["image_bytes"])
                digest = hashlib.sha256(page["image_bytes"]).hexdigest()
            else:
                digest = ""
            if page.get("local_xml"):
                if page.get("xml_bytes") is None:
                    raise BuildError(f"{manuscript} page {tag}: missing local page XML")
                (repo_dir / "page-xml" / manuscript / f"{tag}.xml").write_bytes(
                    rewrite_page_xml_text(page["xml_bytes"], xml_image)
                )
            else:
                raw_xml = sorted(
                    (raw / mspec["project_slug"] / mspec["doc_slug"] / "page").glob(f"{tag}_*.xml")
                )
                if not raw_xml:
                    raise BuildError(f"{manuscript} page {tag}: no exported XML")
                (repo_dir / "page-xml" / manuscript / f"{tag}.xml").write_bytes(
                    rewrite_page_xml_text(raw_xml[0].read_bytes(), xml_image)
                )
            page_id = f"{manuscript}/{tag}"
            object_ids: list[str] = []
            object_bboxes: list[list[int]] = []
            object_polygons: list[list[list[int]]] = []
            object_baselines: list[list[list[int]] | None] = []
            object_texts: list[str] = []
            object_paired: list[bool] = []
            line_index = 0
            for position, line in enumerate(page["lines"]):
                box = tight_bbox(line["polygon"])
                object_ids.append(f"{page_id}/{position:03d}")
                object_bboxes.append(box)
                object_polygons.append(line["polygon"])
                object_baselines.append(line["baseline"])
                object_texts.append(line["text"] or "")
                object_paired.append(line["text"] is not None)
                if line["text"] is None:
                    continue
                crop = (
                    crop_line(page["image_bytes"], box, (page["width"], page["height"]))
                    if ship and page["image_bytes"] is not None
                    else None
                )
                line_rows["line_id"].append(f"{page_id}/{line_index:03d}")
                line_rows["manuscript"].append(manuscript)
                line_rows["shelfmark"].append(mspec["shelfmark"])
                line_rows["page_index"].append(index)
                line_rows["line_index"].append(line_index)
                line_rows["text"].append(line["text"])
                line_rows["image"].append(
                    hub_image(crop, f"{page_id}/{line_index:03d}.png") if crop is not None else None
                )
                line_rows["polygon"].append(line["polygon"])
                line_rows["baseline"].append(line["baseline"])
                line_rows["bbox"].append(box)
                line_rows["page_width"].append(page["width"])
                line_rows["page_height"].append(page["height"])
                line_rows["image_shipped"].append(ship)
                line_rows["source_url"].append(mspec["source_url"])
                line_rows["folio"].append(page["folio"])
                line_rows["text_source"].append(page["text_source"])
                line_index += 1
            page_rows["page_id"].append(page_id)
            page_rows["manuscript"].append(manuscript)
            page_rows["shelfmark"].append(mspec["shelfmark"])
            page_rows["page_index"].append(index)
            page_rows["image"].append(
                hub_image(page["image_bytes"], f"{manuscript}/{image_name}")
                if ship and page["image_bytes"] is not None
                else None
            )
            page_rows["width"].append(page["width"])
            page_rows["height"].append(page["height"])
            page_rows["image_shipped"].append(ship)
            page_rows["source_url"].append(mspec["source_url"])
            page_rows["folio"].append(page["folio"])
            page_rows["text_source"].append(page["text_source"])
            page_rows["objects"].append(
                {
                    "id": object_ids,
                    "bbox": object_bboxes,
                    "polygon": object_polygons,
                    "baseline": object_baselines,
                    "text": object_texts,
                    "paired": object_paired,
                }
            )
            manifest_rows.append(
                {
                    "manuscript": manuscript,
                    "page_index": index,
                    "platform_part_id": page["part_id"],
                    "width": page["width"],
                    "height": page["height"],
                    "n_lines": len(page["lines"]),
                    "n_paired": line_index,
                    "image_shipped": ship,
                    "image_sha256": digest,
                    "source_url": mspec["source_url"],
                    "folio": page["folio"],
                    "text_source": page["text_source"],
                }
            )
    write_parquet(repo_dir / "data" / "lines" / "train-00000-of-00001.parquet", line_rows, "lines")
    write_parquet(repo_dir / "data" / "pages" / "train-00000-of-00001.parquet", page_rows, "pages")
    with (repo_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    card = cards / f"{repo}.md"
    if card.exists():
        shutil.copyfile(card, repo_dir / "README.md")
    else:
        print(f"warning: no card at {card}, skipping README copy", file=sys.stderr)
    licence = cards / f"{repo}.LICENSE.md"
    if licence.exists():
        shutil.copyfile(licence, repo_dir / "LICENSE.md")
    else:
        print(f"warning: no licence at {licence}, skipping LICENSE copy", file=sys.stderr)
    return {
        "pages": len(manifest_rows),
        "lines": len(line_rows["line_id"]),
        "images_shipped": sum(1 for row in manifest_rows if row["image_shipped"]),
    }


_LINES_COLUMNS = (
    "line_id",
    "manuscript",
    "shelfmark",
    "page_index",
    "line_index",
    "text",
    "image",
    "polygon",
    "baseline",
    "bbox",
    "page_width",
    "page_height",
    "image_shipped",
    "source_url",
    "folio",
    "text_source",
)

_PAGES_COLUMNS = (
    "page_id",
    "manuscript",
    "shelfmark",
    "page_index",
    "image",
    "width",
    "height",
    "image_shipped",
    "source_url",
    "folio",
    "text_source",
    "objects",
)

_MANIFEST_COLUMNS = (
    "manuscript",
    "page_index",
    "platform_part_id",
    "width",
    "height",
    "n_lines",
    "n_paired",
    "image_shipped",
    "image_sha256",
    "source_url",
    "folio",
    "text_source",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the four Hugging Face release folders from the raw export.",
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cards",
        type=Path,
        default=DEFAULT_CARDS,
        help="Committed dataset cards to copy as README.md",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="inventory.json path (default: <raw>/../inventory.json)",
    )
    parser.add_argument(
        "--grec1360-local",
        type=Path,
        default=REPO_ROOT / "data" / "dataset" / "grec1360",
        help="local Grec 1360 PAGE XML tree (its xml/ subfolder)",
    )
    parser.add_argument("--repo", default=None, help="Build one repo only")
    return parser.parse_args(argv)


def write_summary(out: Path, results: dict[str, dict]) -> None:
    """Write the release SUMMARY.md from per-repo counts."""
    lines = ["# Release summary", ""]
    for repo in sorted(results):
        counts = results[repo]
        lines.append(
            f"- {repo}: pages={counts['pages']} lines={counts['lines']} "
            f"images_shipped={counts['images_shipped']}"
        )
    lines.append("")
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repos = [args.repo] if args.repo else sorted(REPOS)
        for repo in repos:
            if repo not in REPOS:
                raise BuildError(f"unknown repo {repo}")
        inventory_path = args.inventory or (args.raw.parent / "inventory.json")
        if not inventory_path.exists():
            raise BuildError(f"no inventory at {inventory_path}")
        part_ids = load_part_ids(inventory_path)
        results = {}
        for repo in repos:
            counts = build_repo(args.raw, args.out, args.cards, repo, part_ids, args.grec1360_local)
            results[repo] = counts
            print(f"{repo}: pages={counts['pages']} lines={counts['lines']}", flush=True)
        if args.repo is None:
            write_summary(args.out, results)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

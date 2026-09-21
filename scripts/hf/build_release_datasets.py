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
from xml.etree.ElementTree import parse as xml_parse
from xml.etree.ElementTree import tostring as xml_tostring

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "data" / "hf-release" / "raw"
DEFAULT_OUT = REPO_ROOT / "data" / "hf-release" / "release"
DEFAULT_CARDS = REPO_ROOT / "docs" / "datasets" / "release-cards"

PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

# Per-manuscript image switch. A later permission flips one value.
SHIP_IMAGES = {
    "bnf-grec-1360": True,
    "bnf-armenien-172": True,
    "ucla-armenian-ms-72": False,
    "alqosh-65": False,
    "bav-borg-copt-109": False,
    "naples-ib-10": False,
}

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


def parse_page_xml(path: Path) -> dict:
    """Read one exported PAGE XML into plain dicts (document order kept)."""
    root = xml_parse(str(path)).getroot()  # noqa: S314 - XML this script wrote, not untrusted input
    page = root.find(_ns("Page"))
    if page is None:
        raise BuildError(f"{path}: no Page element")
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
    raw: Path, manuscript: str, part_ids: dict[tuple[str, str, int], str]
) -> list[dict]:
    """Gather the included pages (at least one paired line) of one manuscript."""
    spec = MANUSCRIPTS[manuscript]
    ship = SHIP_IMAGES[manuscript]
    base = raw / spec["project_slug"] / spec["doc_slug"]
    xml_files = sorted((base / "page").glob("*.xml"))
    if not xml_files:
        raise BuildError(f"{base}: no exported pages")
    pages = []
    for xml_path in xml_files:
        order = int(xml_path.stem.split("_", 1)[0])
        page = parse_page_xml(xml_path)
        paired = [line for line in page["lines"] if line["text"] is not None]
        if not paired:
            continue
        matches = sorted((base / "images").glob(f"{order:04d}_*"))
        if not matches:
            raise BuildError(f"{base}: no image for page order {order}")
        ext = matches[0].suffix.lstrip(".") or "png"
        image_bytes = matches[0].read_bytes() if ship else None
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
                pa.field("objects", objects_type),
            ]
        )
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, str(path))


def rewrite_page_xml_text(xml_path: Path, image_name: str) -> bytes:
    """Return the page XML with imageFilename set to the release image name."""
    root = xml_parse(str(xml_path)).getroot()  # noqa: S314 - XML this script wrote, not untrusted input
    page = root.find(_ns("Page"))
    if page is None:
        raise BuildError(f"{xml_path}: no Page element")
    page.set("imageFilename", image_name)
    return xml_tostring(root, encoding="utf-8", xml_declaration=True)


def build_repo(
    raw: Path,
    out: Path,
    cards: Path,
    repo: str,
    part_ids: dict[tuple[str, str, int], str],
) -> dict:
    """Build one release folder. Returns counts for the summary."""
    spec = REPOS[repo]
    repo_dir = out / repo
    (repo_dir / "data" / "lines").mkdir(parents=True, exist_ok=True)
    (repo_dir / "data" / "pages").mkdir(parents=True, exist_ok=True)
    line_rows: dict[str, list] = {name: [] for name in _LINES_COLUMNS}
    page_rows: dict[str, list] = {name: [] for name in _PAGES_COLUMNS}
    manifest_rows = []
    for manuscript in spec["manuscripts"]:
        mspec = MANUSCRIPTS[manuscript]
        ship = SHIP_IMAGES[manuscript]
        pages = collect_manuscript_pages(raw, manuscript, part_ids)
        if ship:
            (repo_dir / "images" / manuscript).mkdir(parents=True, exist_ok=True)
        (repo_dir / "page-xml" / manuscript).mkdir(parents=True, exist_ok=True)
        for page in pages:
            index = page["page_index"]
            tag = f"{index:04d}"
            image_name = f"{tag}.{page['ext']}"
            if ship and page["image_bytes"] is not None:
                (repo_dir / "images" / manuscript / image_name).write_bytes(page["image_bytes"])
                digest = hashlib.sha256(page["image_bytes"]).hexdigest()
            else:
                digest = ""
            raw_xml = sorted(
                (raw / mspec["project_slug"] / mspec["doc_slug"] / "page").glob(f"{tag}_*.xml")
            )
            if not raw_xml:
                raise BuildError(f"{manuscript} page {tag}: no exported XML")
            (repo_dir / "page-xml" / manuscript / f"{tag}.xml").write_bytes(
                rewrite_page_xml_text(raw_xml[0], image_name)
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
    parser.add_argument("--repo", default=None, help="Build one repo only")
    return parser.parse_args(argv)


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
        for repo in repos:
            counts = build_repo(args.raw, args.out, args.cards, repo, part_ids)
            print(f"{repo}: pages={counts['pages']} lines={counts['lines']}", flush=True)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

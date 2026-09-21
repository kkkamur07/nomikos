#!/usr/bin/env python3
"""Inventory and PAGE XML export of the docs seed account corpora.

Read-only against production (``https://api.nomikos.app``): the only
non-GET request this script ever sends is the login POST. Two subcommands:

``inventory`` lists every project, document and part the account can see
and writes ``<out>/inventory.json`` plus a readable ``<out>/inventory.md``.

``export`` saves every page image unchanged plus a PAGE XML file built
locally from the layout and pairing GET responses (the server also has a
per-part ``page-xml`` GET route, but its element layout does not match the
release spec, so the script builds the XML itself in the same style).

Example::

    NOMIKOS_EMAIL=dev@example.com NOMIKOS_PASSWORD=... \\
        python scripts/hf/export_release_corpus.py inventory --out data/hf-release
    NOMIKOS_EMAIL=dev@example.com NOMIKOS_PASSWORD=... \\
        python scripts/hf/export_release_corpus.py export --out data/hf-release
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from xml.etree.ElementTree import Element, QName, SubElement, register_namespace, tostring

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "hf-release"

API_BASE_URL = "https://api.nomikos.app"
LOGIN_PATH = "/auth/login"

PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
PAGE_XML_SCHEMA = (
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
)
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"

register_namespace("", PAGE_XML_NAMESPACE)
register_namespace("xsi", XSI_NAMESPACE)

MIN_REQUEST_INTERVAL = 0.5
MAX_RETRIES = 5


class ApiError(RuntimeError):
    """An API call failed with a status the retry policy does not cover."""


class ApiClient:
    """Thin GET-only client (login POST excepted) with a gentle retry policy."""

    def __init__(self, base_url: str = API_BASE_URL, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._last_at = 0.0

    def _guard(self, method: str, path: str) -> None:
        if method.upper() != "GET" and path != LOGIN_PATH:
            raise ApiError(f"refusing non-GET {method} {path}: production is read-only")

    def _pace(self) -> None:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_at)
        if wait > 0:
            time.sleep(wait)
        self._last_at = time.monotonic()

    def request(self, method: str, path: str, **kwargs) -> object:
        """Send one request, retrying 429 and 5xx with backoff. Returns parsed JSON or bytes."""
        import httpx

        self._guard(method, path)
        url = self.base_url + path
        headers = dict(kwargs.pop("headers", {}))
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        backoff = 1.0
        for attempt in range(MAX_RETRIES + 1):
            self._pace()
            try:
                response = httpx.request(method, url, headers=headers, timeout=60.0, **kwargs)
            except httpx.HTTPError as exc:
                if attempt >= MAX_RETRIES:
                    raise ApiError(f"{method} {path} failed: {exc}") from exc
                time.sleep(backoff)
                backoff *= 2
                continue
            if response.status_code == 401:
                raise ApiError(
                    "got 401 from the API: the NOMIKOS_EMAIL/NOMIKOS_PASSWORD "
                    "credentials are missing or wrong"
                )
            if response.status_code in (429,) or 500 <= response.status_code < 600:
                if attempt >= MAX_RETRIES:
                    raise ApiError(f"{method} {path} failed with {response.status_code}")
                retry_after = response.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else backoff)
                backoff *= 2
                continue
            if response.status_code >= 400:
                raise ApiError(f"{method} {path} failed with {response.status_code}")
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type or not content_type:
                try:
                    return response.json()
                except ValueError:
                    return response.content
            return response.content, response.headers.get("content-type", "")
        raise ApiError(f"{method} {path} failed after retries")

    def get_json(self, path: str, params: dict | None = None) -> object:
        return self.request("GET", path, params=params)

    def get_bytes(self, path: str, params: dict | None = None) -> tuple[bytes, str]:
        result = self.request("GET", path, params=params)
        if isinstance(result, tuple):
            return result
        if isinstance(result, bytes):
            return result, ""
        raise ApiError(f"GET {path} did not return bytes")

    def login(self, email: str, password: str) -> str:
        """The single allowed POST: exchange seed credentials for a bearer token."""
        import httpx

        self._pace()
        url = self.base_url + LOGIN_PATH
        try:
            response = httpx.post(url, json={"email": email, "password": password}, timeout=60.0)
        except httpx.HTTPError as exc:
            raise ApiError(f"login failed: {exc}") from exc
        if response.status_code == 401:
            raise ApiError(
                "login got 401: the NOMIKOS_EMAIL/NOMIKOS_PASSWORD credentials are wrong"
            )
        if response.status_code >= 400:
            raise ApiError(f"login failed with {response.status_code}")
        token = response.json().get("access_token")
        if not token:
            raise ApiError("login response carried no access token")
        self.token = token
        return token


def credentials_from_env() -> tuple[str, str]:
    email = os.environ.get("NOMIKOS_EMAIL", "")
    password = os.environ.get("NOMIKOS_PASSWORD", "")
    if not email or not password:
        raise ApiError("set NOMIKOS_EMAIL and NOMIKOS_PASSWORD in the environment")
    return email, password


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.strip().lower(), flags=re.UNICODE)
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug or "unnamed"


def paged_get(client: ApiClient, path: str, params: dict | None = None) -> list:
    items: list = []
    cursor: str | None = None
    while True:
        query = dict(params or {})
        query["limit"] = 200
        if cursor:
            query["cursor"] = cursor
        payload = client.get_json(path, params=query)
        if isinstance(payload, dict):
            items.extend(payload.get("items", []))
            cursor = payload.get("next_cursor")
        elif isinstance(payload, list):
            items.extend(payload)
            cursor = None
        else:
            break
        if not cursor:
            break
    return items


def geometry_points(geometry: object) -> list[list[float]]:
    if not isinstance(geometry, dict):
        return []
    points = geometry.get("points")
    if not isinstance(points, list):
        return []
    clean: list[list[float]] = []
    for point in points:
        if (
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(value, (int, float)) for value in point)
        ):
            clean.append([float(point[0]), float(point[1])])
    return clean


def mean_y(points: list[list[float]]) -> float:
    if not points:
        return 0.0
    return sum(point[1] for point in points) / len(points)


def part_image_stem(image_url: str, part_id: str) -> tuple[str, str]:
    """Split an image URL or key into (stem, ext). Falls back to the part id."""
    base = urlparse(image_url).path.rsplit("/", 1)[-1] if image_url else ""
    base = base.split("?")[0].strip()
    if not base or "." not in base:
        return (part_id, "")
    stem, ext = base.rsplit(".", 1)
    stem = stem.strip() or part_id
    return stem, ext.lower()


def ext_for_content_type(content_type: str, fallback: str) -> str:
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/tiff": "tif",
    }
    for key, ext in mapping.items():
        if key in content_type:
            return ext
    return fallback


def _page_tag(name: str) -> str:
    return str(QName(PAGE_XML_NAMESPACE, name))


def _xsi_tag(name: str) -> str:
    return str(QName(XSI_NAMESPACE, name))


def _format_coord(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _points_str(points: list[list[float]]) -> str:
    return " ".join(f"{_format_coord(x)},{_format_coord(y)}" for x, y in points)


def collect_part_lines(layout: dict, pairing: dict) -> tuple[list[dict], int, int]:
    """Join layout segments with paired transcription text.

    Returns (lines, text_line_count, paired_line_count) where each line holds
    id, points, baseline points and optional text. Lines that no text line
    pairs to are unpaired. Text lines paired to an unknown segment id are
    ignored for the XML but still counted as text lines.
    """
    text_lines = pairing.get("text_lines", []) if isinstance(pairing, dict) else []
    text_by_segment: dict[str, list[str]] = {}
    for entry in text_lines:
        segment_id = entry.get("paired_line_id")
        if segment_id:
            text_by_segment.setdefault(str(segment_id), []).append(entry.get("text", ""))
    raw_lines = layout.get("lines", []) if isinstance(layout, dict) else []
    lines: list[dict] = []
    for raw in raw_lines:
        points = [[float(x), float(y)] for x, y in raw.get("points", [])]
        baseline = geometry_points(raw.get("baseline")) or []
        texts = text_by_segment.get(str(raw.get("id")), [])
        text = "\n".join(texts) if texts else None
        lines.append(
            {
                "id": str(raw.get("id")),
                "points": points,
                "baseline": baseline,
                "text": text,
            }
        )
    paired = sum(1 for line in lines if line["text"] is not None)
    return lines, len(text_lines), paired


def sort_lines_top_to_bottom(lines: list[dict]) -> list[dict]:
    """Order lines by mean baseline y (polygon mean y when no baseline)."""

    def key(line: dict) -> tuple[float, float, str]:
        baseline = line.get("baseline") or []
        points = baseline if baseline else line.get("points", [])
        xs = [point[0] for point in points] if points else [0.0]
        return (mean_y(points), min(xs), str(line.get("id")))

    return sorted(lines, key=key)


def build_page_xml(
    part_id: str,
    image_filename: str,
    width: int,
    height: int,
    lines: list[dict],
    created: str | None = None,
) -> bytes:
    """Render PAGE XML for one part. Lines are ordered top to bottom.

    Each entry of ``lines`` holds ``id``, ``points`` (polygon), ``baseline``
    (polyline, may be empty) and ``text`` (None for unpaired segments). Text
    is NFC normalised and otherwise untouched. Coordinates are integers in
    image pixels.
    """
    stamp = created or datetime.now(UTC).isoformat()
    ordered = sort_lines_top_to_bottom(lines)
    root = Element(
        _page_tag("PcGts"),
        {
            _xsi_tag("schemaLocation"): PAGE_XML_SCHEMA,
            "pcGtsId": f"pcgts_{part_id}",
        },
    )
    metadata = SubElement(root, _page_tag("Metadata"))
    SubElement(metadata, _page_tag("Creator")).text = "NOMIKOS"
    SubElement(metadata, _page_tag("Created")).text = stamp
    SubElement(metadata, _page_tag("LastChange")).text = stamp
    page = SubElement(
        root,
        _page_tag("Page"),
        {
            "imageFilename": image_filename,
            "imageWidth": str(int(width)),
            "imageHeight": str(int(height)),
        },
    )
    reading = SubElement(page, _page_tag("ReadingOrder"))
    group = SubElement(
        reading,
        _page_tag("OrderedGroup"),
        {"id": f"reading_order_{part_id}", "caption": "Lines top to bottom"},
    )
    for index, line in enumerate(ordered):
        SubElement(
            group,
            _page_tag("TextLineRefIndexed"),
            {"index": str(index), "textLineId": f"line_{line['id']}"},
        )
    region = SubElement(
        page,
        _page_tag("TextRegion"),
        {"id": f"region_{part_id}", "type": "paragraph"},
    )
    for line in ordered:
        attrs = {"id": f"line_{line['id']}"}
        if line.get("text") is None:
            attrs["custom"] = "unpaired"
        text_line = SubElement(region, _page_tag("TextLine"), attrs)
        polygon = [[round(x), round(y)] for x, y in line.get("points", [])]
        SubElement(text_line, _page_tag("Coords"), {"points": _points_str(polygon)})
        baseline = line.get("baseline") or []
        if baseline:
            snapped = [[round(x), round(y)] for x, y in baseline]
            SubElement(text_line, _page_tag("Baseline"), {"points": _points_str(snapped)})
        if line.get("text") is not None:
            equiv = SubElement(text_line, _page_tag("TextEquiv"))
            SubElement(equiv, _page_tag("Unicode")).text = unicodedata.normalize(
                "NFC", line["text"]
            )
    return tostring(root, encoding="utf-8", xml_declaration=True)


def collect_inventory(client: ApiClient, project_filter: list[str] | None = None) -> dict:
    """Walk projects, documents and parts. Returns a JSON-able inventory dict."""
    wanted = set(project_filter or [])
    projects = paged_get(client, "/projects")
    inventory: dict = {"projects": []}
    for project in projects:
        if wanted and project.get("name") not in wanted:
            continue
        entry: dict = {
            "id": project.get("id"),
            "name": project.get("name"),
            "slug": project.get("slug") or slugify(project.get("name", "")),
            "documents": [],
        }
        documents = paged_get(client, f"/projects/{project['id']}/documents")
        for document in documents:
            detail = client.get_json(f"/projects/{project['id']}/documents/{document['id']}")
            if not isinstance(detail, dict):
                raise ApiError(f"document {document['id']} did not return JSON")
            doc_entry: dict = {
                "id": document.get("id"),
                "name": document.get("name"),
                "slug": slugify(document.get("name", "")),
                "workflow": document.get("workflow"),
                "parts": [],
            }
            for part in detail.get("parts", []):
                layout = client.get_json(
                    f"/projects/{project['id']}/documents/{document['id']}"
                    f"/parts/{part['id']}/layout"
                )
                pairing = client.get_json(
                    f"/projects/{project['id']}/documents/{document['id']}"
                    f"/parts/{part['id']}/pairing"
                )
                if not isinstance(layout, dict) or not isinstance(pairing, dict):
                    raise ApiError(f"part {part['id']} did not return JSON")
                _, text_count, paired_count = collect_part_lines(layout, pairing)
                stem, ext = part_image_stem(part.get("image_url", ""), part["id"])
                doc_entry["parts"].append(
                    {
                        "id": part.get("id"),
                        "order": part.get("order"),
                        "image_file": stem + (f".{ext}" if ext else ""),
                        "width": part.get("width"),
                        "height": part.get("height"),
                        "segment_count": len(layout.get("lines", [])),
                        "text_line_count": text_count,
                        "paired_line_count": paired_count,
                        "reviewed": part.get("reviewed"),
                        "published": part.get("published"),
                    }
                )
            doc_entry["parts"].sort(key=lambda p: (p.get("order") or 0, p.get("id")))
            entry["documents"].append(doc_entry)
        inventory["projects"].append(entry)
    return inventory


def write_inventory_markdown(inventory: dict, path: Path) -> None:
    lines = ["# Corpus inventory", ""]
    for project in inventory["projects"]:
        lines.append(f"## {project['name']}")
        lines.append("")
        lines.append("| Document | Parts | Segments | Text lines | Paired |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        project_parts = 0
        project_segments = 0
        project_texts = 0
        project_paired = 0
        for document in project["documents"]:
            parts = document["parts"]
            segments = sum(p["segment_count"] for p in parts)
            texts = sum(p["text_line_count"] for p in parts)
            paired = sum(p["paired_line_count"] for p in parts)
            project_parts += len(parts)
            project_segments += segments
            project_texts += texts
            project_paired += paired
            lines.append(f"| {document['name']} | {len(parts)} | {segments} | {texts} | {paired} |")
        lines.append(
            f"| Total | {project_parts} | {project_segments} | {project_texts} | {project_paired} |"
        )
        lines.append("")
        for document in project["documents"]:
            lines.append(f"### {project['name']} / {document['name']}")
            lines.append("")
            lines.append("| Order | Part id | Image | W x H | Segments | Text | Paired | Review |")
            lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | --- |")
            for part in document["parts"]:
                width = part["width"] if part["width"] is not None else "?"
                height = part["height"] if part["height"] is not None else "?"
                lines.append(
                    f"| {part['order']} | {part['id']} | {part['image_file']} "
                    f"| {width} x {height} | {part['segment_count']} "
                    f"| {part['text_line_count']} | {part['paired_line_count']} "
                    f"| reviewed={part['reviewed']} published={part['published']} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_inventory(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    email, password = credentials_from_env()
    client = ApiClient(base_url=args.base_url)
    client.login(email, password)
    inventory = collect_inventory(client, args.project or None)
    (out / "inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_inventory_markdown(inventory, out / "inventory.md")
    total_parts = sum(
        len(document["parts"])
        for project in inventory["projects"]
        for document in project["documents"]
    )
    print(f"projects={len(inventory['projects'])} parts={total_parts} out={out}")
    return 0


def real_image_size(image: bytes) -> tuple[int, int]:
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(image)) as handle:
        return int(handle.width), int(handle.height)


def cmd_export(args: argparse.Namespace) -> int:
    out = args.out
    email, password = credentials_from_env()
    client = ApiClient(base_url=args.base_url)
    client.login(email, password)
    inventory_path = out / "inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        inventory = collect_inventory(client, args.project or None)
        out.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(
            json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_inventory_markdown(inventory, out / "inventory.md")
    wanted = set(args.project or [])
    saved_images = 0
    saved_xml = 0
    for project in inventory["projects"]:
        if wanted and project.get("name") not in wanted:
            continue
        for document in project["documents"]:
            images_dir = out / "raw" / project["slug"] / document["slug"] / "images"
            page_dir = out / "raw" / project["slug"] / document["slug"] / "page"
            images_dir.mkdir(parents=True, exist_ok=True)
            page_dir.mkdir(parents=True, exist_ok=True)
            for part in document["parts"]:
                order = part.get("order") or 0
                stem, url_ext = part_image_stem(part.get("image_file", ""), part["id"])
                safe_stem = slugify(stem) or part["id"]
                prefix = f"{int(order):04d}_{safe_stem}"
                existing_xml = sorted(page_dir.glob(f"{prefix}.xml"))
                existing_images = sorted(images_dir.glob(f"{prefix}.*"))
                if args.skip_existing and existing_xml and existing_images:
                    continue
                image, content_type = client.get_bytes(f"/media/parts/{part['id']}")
                ext = ext_for_content_type(content_type, url_ext or "png")
                image_name = f"{prefix}.{ext}"
                image_path = images_dir / image_name
                image_path.write_bytes(image)
                saved_images += 1
                try:
                    width, height = real_image_size(image)
                except Exception:
                    width = part.get("width") or 0
                    height = part.get("height") or 0
                layout = client.get_json(
                    f"/projects/{project['id']}/documents/{document['id']}"
                    f"/parts/{part['id']}/layout"
                )
                pairing = client.get_json(
                    f"/projects/{project['id']}/documents/{document['id']}"
                    f"/parts/{part['id']}/pairing"
                )
                if not isinstance(layout, dict) or not isinstance(pairing, dict):
                    raise ApiError(f"part {part['id']} did not return JSON")
                part_lines, _, _ = collect_part_lines(layout, pairing)
                xml = build_page_xml(part["id"], image_name, width, height, part_lines)
                (page_dir / f"{prefix}.xml").write_bytes(xml)
                saved_xml += 1
    write_validation(out, inventory)
    print(f"images={saved_images} xml={saved_xml} out={out}")
    return 0


def write_validation(out: Path, inventory: dict) -> None:
    """Check the exported tree against the inventory and write validation.md."""
    from xml.etree.ElementTree import parse as xml_parse

    report = ["# Export validation", ""]
    mismatches: list[str] = []
    for project in inventory["projects"]:
        report.append(f"## {project['name']}")
        report.append("")
        report.append("| Document | Images | XML | Segments | Paired (inventory) | Paired (XML) |")
        report.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for document in project["documents"]:
            base = out / "raw" / project["slug"] / document["slug"]
            images = sorted((base / "images").glob("*")) if (base / "images").exists() else []
            xml_files = sorted((base / "page").glob("*.xml")) if (base / "page").exists() else []
            inv_segments = sum(p["segment_count"] for p in document["parts"])
            inv_paired = sum(p["paired_line_count"] for p in document["parts"])
            xml_paired = 0
            xml_lines = 0
            for xml_path in xml_files:
                try:
                    root = xml_parse(str(xml_path)).getroot()  # noqa: S314 - XML this script wrote, not untrusted input
                except Exception as exc:
                    mismatches.append(f"{xml_path}: does not parse: {exc}")
                    continue
                ns = {"p": PAGE_XML_NAMESPACE}
                page = root.find("p:Page", ns)
                if page is None:
                    mismatches.append(f"{xml_path}: no Page element")
                    continue
                name = page.get("imageFilename", "")
                sibling = xml_path.parent.parent / "images" / name
                if not sibling.exists():
                    sibling = base / "images" / name
                if not sibling.exists():
                    mismatches.append(f"{xml_path}: imageFilename {name} not found")
                    continue
                try:
                    actual_w, actual_h = real_image_size(sibling.read_bytes())
                except Exception as exc:
                    mismatches.append(f"{xml_path}: cannot read image: {exc}")
                    continue
                if str(page.get("imageWidth")) != str(actual_w) or str(
                    page.get("imageHeight")
                ) != str(actual_h):
                    mismatches.append(
                        f"{xml_path}: XML size "
                        f"{page.get('imageWidth')}x{page.get('imageHeight')} "
                        f"differs from real {actual_w}x{actual_h}"
                    )
                for text_line in root.findall(".//p:TextLine", ns):
                    xml_lines += 1
                    equiv = text_line.find("p:TextEquiv", ns)
                    if equiv is not None:
                        xml_paired += 1
            if xml_paired != inv_paired:
                mismatches.append(
                    f"{project['name']}/{document['name']}: paired XML={xml_paired} "
                    f"differs from inventory={inv_paired}"
                )
            if len(images) != len(document["parts"]) or len(xml_files) != len(document["parts"]):
                mismatches.append(
                    f"{project['name']}/{document['name']}: files "
                    f"images={len(images)} xml={len(xml_files)} "
                    f"parts={len(document['parts'])}"
                )
            report.append(
                f"| {document['name']} | {len(images)} | {len(xml_files)} "
                f"| {inv_segments} | {inv_paired} | {xml_paired} |"
            )
        report.append("")
    if mismatches:
        report.append("## Mismatches")
        report.append("")
        for item in mismatches:
            report.append(f"- {item}")
    else:
        report.append("No mismatches.")
    report.append("")
    (out / "validation.md").write_text("\n".join(report), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory and PAGE XML export of the docs seed account corpora.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Staging output folder (default: data/hf-release)",
    )
    parser.add_argument(
        "--base-url",
        default=API_BASE_URL,
        help="API base URL (default: https://api.nomikos.app)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="List projects, documents and parts.")
    inv.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Staging output folder (default: data/hf-release)",
    )
    inv.add_argument("--base-url", default=API_BASE_URL, help="API base URL")
    inv.add_argument(
        "--project",
        action="append",
        default=[],
        help="Project name to include (repeatable, default: all)",
    )
    exp = sub.add_parser("export", help="Export page images plus PAGE XML.")
    exp.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Staging output folder (default: data/hf-release)",
    )
    exp.add_argument("--base-url", default=API_BASE_URL, help="API base URL")
    exp.add_argument(
        "--project",
        action="append",
        default=[],
        help="Project name to include (repeatable, default: all)",
    )
    exp.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip parts whose image and XML already exist",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "inventory":
            return cmd_inventory(args)
        return cmd_export(args)
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inventory and PAGE XML export of the docs seed account corpora.

Read-only against production (``https://api.nomikos.app``): the only
non-GET request this script ever sends is the login POST. Two subcommands:

``inventory`` lists every project, document and part the account can see
and writes ``<out>/inventory.json`` plus a readable ``<out>/inventory.md``.

``export`` saves every page image unchanged plus a PAGE XML file built
locally from the layout and pairing GET responses (the server also has a
per-part ``page-xml`` GET route, but its element layout does not match the
release spec, so the script builds the XML itself in the same style). It
also saves both GET responses unchanged under ``json/`` per part, and
``export --from-json`` rebuilds the XML from those snapshots with no
network.

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
        self._last_raw = b""

    def last_raw(self) -> bytes:
        """Raw body of the most recent response (for byte-identical snapshots)."""
        return self._last_raw

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
            self._last_raw = response.content
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
    id, points, baseline points, optional text and optional text_order (the
    position of its text in the transcription, which the scholars typed in
    reading order). Lines that no text line pairs to are unpaired
    (text_order None). Text lines paired to an unknown segment id are
    ignored for the XML but still counted as text lines.
    """
    text_lines = pairing.get("text_lines", []) if isinstance(pairing, dict) else []
    text_by_segment: dict[str, list[tuple[int, str]]] = {}
    for list_index, entry in enumerate(text_lines):
        segment_id = entry.get("paired_line_id")
        if segment_id:
            order = entry.get("order")
            position = int(order) if isinstance(order, int) else list_index
            text_by_segment.setdefault(str(segment_id), []).append(
                (position, entry.get("text", ""))
            )
    raw_lines = layout.get("lines", []) if isinstance(layout, dict) else []
    lines: list[dict] = []
    for raw in raw_lines:
        points = [[float(x), float(y)] for x, y in raw.get("points", [])]
        baseline = geometry_points(raw.get("baseline")) or []
        hits = sorted(text_by_segment.get(str(raw.get("id")), []))
        text = "\n".join(text for _, text in hits) if hits else None
        lines.append(
            {
                "id": str(raw.get("id")),
                "points": points,
                "baseline": baseline,
                "text": text,
                "text_order": hits[0][0] if hits else None,
            }
        )
    paired = sum(1 for line in lines if line["text"] is not None)
    return lines, len(text_lines), paired


def sort_lines_reading_order(lines: list[dict]) -> list[dict]:
    """Order lines as readers read them.

    Paired lines come first in transcription text order (the scholars typed
    the text in reading order, which survives multi-column pages where a pure
    geometric sort would interleave columns). Unpaired segments follow, sorted
    by mean baseline y (polygon mean y when no baseline) then min x.
    """

    def key(line: dict) -> tuple[int, int, float, float, str]:
        baseline = line.get("baseline") or []
        points = baseline if baseline else line.get("points", [])
        xs = [point[0] for point in points] if points else [0.0]
        text_order = line.get("text_order")
        if text_order is not None:
            return (0, int(text_order), 0.0, 0.0, "")
        return (1, 0, mean_y(points), min(xs), str(line.get("id")))

    return sorted(lines, key=key)


def build_page_xml(
    part_id: str,
    image_filename: str,
    width: int,
    height: int,
    lines: list[dict],
    created: str | None = None,
) -> bytes:
    """Render PAGE XML for one part. Lines follow transcription reading order.

    Each entry of ``lines`` holds ``id``, ``points`` (polygon), ``baseline``
    (polyline, may be empty), ``text`` (None for unpaired segments) and
    ``text_order`` (position in the transcription, None for unpaired). Text
    is NFC normalised and otherwise untouched. Coordinates are integers in
    image pixels. Line order is the document order of the ``TextLine``
    elements; the ``ReadingOrder`` group points at the region only.
    """
    stamp = created or datetime.now(UTC).isoformat()
    ordered = sort_lines_reading_order(lines)
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
    region_id = f"region_{part_id}"
    reading = SubElement(page, _page_tag("ReadingOrder"))
    group = SubElement(
        reading,
        _page_tag("OrderedGroup"),
        {
            "id": f"reading_order_{part_id}",
            "caption": "Lines in transcription reading order",
        },
    )
    group.append(Element(_page_tag("RegionRefIndexed"), {"index": "0", "regionRef": region_id}))
    region = SubElement(
        page,
        _page_tag("TextRegion"),
        {"id": region_id, "type": "paragraph"},
    )
    for index, line in enumerate(ordered):
        if line.get("text") is None:
            custom = f"readingOrder {{index:{index};}} structure {{type:unpaired;}}"
        else:
            custom = f"readingOrder {{index:{index};}}"
        attrs = {"id": f"line_{line['id']}", "custom": custom}
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


def _export_dirs(out: Path, project: dict, document: dict) -> tuple[Path, Path, Path]:
    base = out / "raw" / project["slug"] / document["slug"]
    return base / "images", base / "page", base / "json"


def _part_prefix(part: dict) -> str:
    order = part.get("order") or 0
    stem, _ = part_image_stem(part.get("image_file", ""), part["id"])
    safe_stem = slugify(stem) or part["id"]
    return f"{int(order):04d}_{safe_stem}"


def _normalise_timestamps(xml: bytes) -> bytes:
    text = xml.decode("utf-8")
    text = re.sub(r"<Created>.*?</Created>", "<Created>STAMP</Created>", text)
    text = re.sub(r"<LastChange>.*?</LastChange>", "<LastChange>STAMP</LastChange>", text)
    return text.encode("utf-8")


def cmd_export(args: argparse.Namespace) -> int:
    out = args.out
    from_json = args.from_json
    client: ApiClient | None = None
    if from_json:
        inventory_path = out / "inventory.json"
        if not inventory_path.exists():
            raise ApiError("no inventory.json: run a network export first")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
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
    if client is None and not from_json:
        raise ApiError("no API client: cannot reach production")
    wanted = set(args.project or [])
    saved_images = 0
    saved_xml = 0
    rebuilt = 0
    identical = 0
    for project in inventory["projects"]:
        if wanted and project.get("name") not in wanted:
            continue
        for document in project["documents"]:
            images_dir, page_dir, json_dir = _export_dirs(out, project, document)
            images_dir.mkdir(parents=True, exist_ok=True)
            page_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)
            for part in document["parts"]:
                prefix = _part_prefix(part)
                xml_path = page_dir / f"{prefix}.xml"
                if from_json:
                    if args.skip_existing and xml_path.exists():
                        continue
                    layout = json.loads(
                        (json_dir / f"{prefix}.layout.json").read_text(encoding="utf-8")
                    )
                    pairing = json.loads(
                        (json_dir / f"{prefix}.pairing.json").read_text(encoding="utf-8")
                    )
                    matches = sorted(images_dir.glob(f"{prefix}.*"))
                    if not matches:
                        raise ApiError(f"no image for {prefix}: run a network export first")
                    image = matches[0].read_bytes()
                    image_name = matches[0].name
                    try:
                        width, height = real_image_size(image)
                    except Exception:
                        width = part.get("width") or 0
                        height = part.get("height") or 0
                    part_lines, _, _ = collect_part_lines(layout, pairing)
                    fresh = build_page_xml(part["id"], image_name, width, height, part_lines)
                    rebuilt += 1
                    if not xml_path.exists():
                        xml_path.write_bytes(fresh)
                        saved_xml += 1
                    elif _normalise_timestamps(fresh) == _normalise_timestamps(
                        xml_path.read_bytes()
                    ):
                        identical += 1
                    else:
                        raise ApiError(f"{prefix}: rebuilt XML differs from stored XML")
                    continue
                if client is None:
                    raise ApiError("no API client: cannot reach production")
                existing_images = sorted(images_dir.glob(f"{prefix}.*"))
                if args.skip_existing and xml_path.exists() and existing_images:
                    continue
                image, content_type = client.get_bytes(f"/media/parts/{part['id']}")
                _, url_ext = part_image_stem(part.get("image_file", ""), part["id"])
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
                layout_raw = client.last_raw()
                pairing = client.get_json(
                    f"/projects/{project['id']}/documents/{document['id']}"
                    f"/parts/{part['id']}/pairing"
                )
                pairing_raw = client.last_raw()
                if not isinstance(layout, dict) or not isinstance(pairing, dict):
                    raise ApiError(f"part {part['id']} did not return JSON")
                (json_dir / f"{prefix}.layout.json").write_bytes(layout_raw)
                (json_dir / f"{prefix}.pairing.json").write_bytes(pairing_raw)
                part_lines, _, _ = collect_part_lines(layout, pairing)
                xml = build_page_xml(part["id"], image_name, width, height, part_lines)
                xml_path.write_bytes(xml)
                saved_xml += 1
    write_validation(out, inventory)
    if from_json:
        print(f"rebuilt={rebuilt} identical={identical} out={out}")
    else:
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
    exp.add_argument(
        "--from-json",
        action="store_true",
        help="Rebuild PAGE XML from saved layout/pairing JSON with no network",
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

"""Compare a local model run with the same run through the platform.

Findings reproduced by the local run (read from the backend, not assumed):

(a) Image bytes: the worker never receives pixels in its claim. It fetches
the single stored page object behind a signed link to ``part.image_key``
(``nomikos/backend/jobs/application/inference_dispatcher.py:180-194``
``page_image_key_for_job``; the link is signed in
``nomikos/backend/jobs/application/job_claim_service.py:186``
``claim_one_page``). An authenticated client fetches exactly those bytes
with ``GET /media/parts/{part_id}`` and no width query
(``nomikos/backend/document/api/media.py:27-46`` ``get_part_image``):
``part_image_response`` serves the stored original when the width is None
and a derived thumbnail rendering otherwise
(``nomikos/backend/document/api/media_responses.py:80-100``,
``nomikos/backend/document/application/part_service.py:448-450``
``_read_part_bytes``). This script always downloads with no width.

(b) Params: an explicit ``model_id`` contributes its catalog
``default_params``; request params are merged over them for segment
(``nomikos/backend/document/application/document_job_enqueue.py:144-175``
``enqueue_segment_part``) while transcribe carries only
``default_params`` plus ``line_ids``
(``nomikos/backend/document/application/document_job_enqueue.py:90-120``
``enqueue_transcribe_part``; request overrides come from a part, document
or project binding in
``nomikos/backend/ml/application/model_service.py:200``
``resolve_for_part``). The dispatcher passes the stored
``job.payload["ml_params"]`` to the worker verbatim for segment
(``nomikos/backend/jobs/application/inference_dispatcher.py:128``
``_build_segment_request``) and as the base params under the line list for
transcribe
(``nomikos/backend/jobs/application/inference_dispatcher.py:162``
``_build_transcribe_request``). This script reuses the finished job
payload ``ml_params`` as the local params, so the merge is reproduced
rather than reimplemented.

(c) Transcribe geometry: the dispatcher sends one entry per line with only
``line_id``, ``line_index`` and ``points``
(``nomikos/backend/jobs/application/inference_dispatcher.py:163-170``),
read in ``(Line.order, Line.created_at)`` order
(``nomikos/backend/document/application/transcribe_merge_service.py:28-35``
``load_lines``). No mask, no baseline, no rounding or int conversion: the
Postgres ``JSONB`` floats
(``nomikos/backend/document/infrastructure/orm_models.py:179``) travel
verbatim into ``TranscribeLineRegion.points``
(``nomikos_inference/contracts/transcribe.py:42-44``). This script builds
the local line list from the same ``GET .../parts/{part_id}/lines``
endpoint, sorted the same way, filtered to the job ``line_ids``.

(d) Stored segment lines carry the model polygon. ``job.result`` for a
segment job is only the merge summary (counts such as ``added_lines``),
written at ``nomikos/backend/jobs/application/job_callback_service.py:308``
from ``_apply_segment_merge``
(``nomikos/backend/jobs/application/job_callback_service.py:122-139``), so
the platform side is read back from ``GET .../parts/{part_id}/lines``
after the job is done. The callback maps the worker response onto the
canonical DTO field for field with no coordinate transform
(``nomikos/backend/ml/application/segment_mapping.py:14-40``), and the
merge stores ``points``, ``baseline``, ``mask`` and ``kind`` verbatim on
the ``Line`` row
(``nomikos/backend/document/application/segment_merge_service.py:127-143``);
only ``source_metadata`` gains ``external_id`` and ``job_id`` keys
(``nomikos/backend/document/application/segment_merge_service.py:122-126``),
which the comparator ignores. Compare stored ``points`` with local
``points`` like with like.

(e) Transcribe job results are flat summaries
(``nomikos/backend/document/application/transcribe_merge_service.py:95-118``):
``{"transcription_id": ..., "lines": [{"line_id", "text", "confidence"}]}``
with an optional ``failed_line_indexes`` list added by the callback
(``nomikos/backend/jobs/application/job_callback_service.py:229``). There
is no per-line ``output`` object and no ``line_index`` on the platform
side; the local contract shape keeps both, so each side has its own
normaliser and lines pair on ``line_id`` only.

(f) Failed lines are absent from the summary: the indexes are positions
in the dispatched region order. The dispatcher filters its
``(Line.order, Line.created_at)`` rows by the payload id set, so one
dispatched order serves both cases: the sorted page lines, filtered to
``payload.line_ids`` when the job has them
(``nomikos/backend/jobs/application/inference_dispatcher.py:156-170``).
It is trusted only when its length fits the summary plus the failures
and the surviving positions match the result ids in order. Mapped
failures are run locally and reported per line as failed on the
platform, and any run with one in scope is a MISMATCH, never IDENTICAL.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import sys
import time
from typing import Any
from urllib import parse as _urlparse
from urllib import request as _urlrequest

try:
    import httpx as _httpx
except ImportError:
    _httpx = None

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
except ImportError:
    _ShapelyPolygon = None

_HAS_SHAPELY = _ShapelyPolygon is not None

SEGMENT_REPLACE_FLAG = "i_know_segment_replaces_the_lines"
TERMINAL_STATUSES = ("done", "failed", "cancelled")
_POLY_TOLERANCE_PX = 1.0
_CONFIDENCE_EPS = 1e-4

_SECRET_ENV_NAMES = (
    "NOMIKOS_TOKEN",
    "NOMIKOS_EMAIL",
    "NOMIKOS_PASSWORD",
    "LOCUST_EMAIL",
    "LOCUST_PASSWORD",
)


def _read_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            if key:
                values[key] = value
    return values


def _credentials(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    file_values = _read_env_file(args.env_file) if args.env_file else {}
    token = os.environ.get("NOMIKOS_TOKEN", file_values.get("NOMIKOS_TOKEN", ""))
    email = os.environ.get(
        "NOMIKOS_EMAIL",
        file_values.get(
            "NOMIKOS_EMAIL", file_values.get("LOCUST_EMAIL", os.environ.get("LOCUST_EMAIL", ""))
        ),
    )
    password = os.environ.get(
        "NOMIKOS_PASSWORD",
        file_values.get(
            "NOMIKOS_PASSWORD",
            file_values.get("LOCUST_PASSWORD", os.environ.get("LOCUST_PASSWORD", "")),
        ),
    )
    return (token or None, email or None, password or None)


def _scrub(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {key: _scrub(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, secrets) for item in value]
    if isinstance(value, str):
        cleaned = value
        for secret in secrets:
            if secret:
                cleaned = cleaned.replace(secret, "[REDACTED]")
        return cleaned
    return value


def parse_artifact_ref(artifact_ref: str) -> tuple[str, str]:
    parsed = _urlparse.urlparse(artifact_ref)
    if parsed.scheme != "registry" or not parsed.netloc:
        raise ValueError(f"unsupported artifact_ref: {artifact_ref!r}")
    query = _urlparse.parse_qs(parsed.query)
    tag = (query.get("tag", ["stable"])[0] or "stable").strip() or "stable"
    return parsed.netloc, tag


def resolve_model(catalog: list[dict[str, Any]], spec: str) -> dict[str, Any]:
    wanted = spec.strip()
    for entry in catalog:
        if str(entry.get("id", "")) == wanted:
            return entry
    for entry in catalog:
        ref = str(entry.get("artifact_ref", ""))
        try:
            registry_id, _ = parse_artifact_ref(ref)
        except ValueError:
            continue
        if registry_id == wanted:
            return entry
    lowered = wanted.lower()
    for entry in catalog:
        if str(entry.get("name", "")).lower() == lowered:
            return entry
    raise ValueError(f"model {spec!r} not found in catalog of {len(catalog)} entries")


def _as_floats(points: Any) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(points, list):
        return out
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                out.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                continue
    return out


def _geom_points(geom: Any) -> list[list[float]]:
    if geom is None:
        return []
    if isinstance(geom, dict):
        for key in ("points", "poly", "polygon"):
            if key in geom:
                return _as_floats(geom[key])
        return []
    return _as_floats(geom)


def max_coord_diff(first: list[list[float]], second: list[list[float]]) -> float | None:
    if len(first) != len(second):
        return None
    if not first:
        return 0.0
    diff = 0.0
    for (x1, y1), (x2, y2) in zip(first, second, strict=True):
        diff = max(diff, abs(x1 - x2), abs(y1 - y2))
    return diff


def _polygon_area(points: list[list[float]]) -> float:
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _convex_hull(points: list[list[float]]) -> list[list[float]]:
    """Monotone chain convex hull, counter-clockwise, no duplicate endpoint."""
    unique = sorted({(point[0], point[1]) for point in points})
    if len(unique) <= 1:
        return [list(point) for point in unique]

    def cross(
        origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [[x, y] for x, y in (lower[:-1] + upper[:-1])]


def _clip_against_edge(
    polygon: list[list[float]],
    edge_start: list[float],
    edge_end: list[float],
) -> list[list[float]]:
    """Clip a polygon to the left of the directed edge (Sutherland-Hodgman)."""
    if not polygon:
        return []
    out: list[list[float]] = []
    edge_x = edge_end[0] - edge_start[0]
    edge_y = edge_end[1] - edge_start[1]

    def inside(point: list[float]) -> bool:
        return edge_x * (point[1] - edge_start[1]) - edge_y * (point[0] - edge_start[0]) >= 0

    def intersect(first: list[float], second: list[float]) -> list[float]:
        direction_x = second[0] - first[0]
        direction_y = second[1] - first[1]
        denom = edge_x * direction_y - edge_y * direction_x
        if denom == 0.0:
            return list(second)
        ratio = (edge_x * (first[1] - edge_start[1]) - edge_y * (first[0] - edge_start[0])) / denom
        return [first[0] + ratio * direction_x, first[1] + ratio * direction_y]

    count = len(polygon)
    for index in range(count):
        current = polygon[index]
        previous = polygon[index - 1]
        c_in = inside(current)
        p_in = inside(previous)
        if c_in:
            if not p_in:
                out.append(intersect(previous, current))
            out.append(current)
        elif p_in:
            out.append(intersect(previous, current))
    return out


def _convex_intersection_area(first: list[list[float]], second: list[list[float]]) -> float:
    """Intersection area of two convex polygons via Sutherland-Hodgman clipping."""
    clipped = [list(point) for point in first]
    count = len(second)
    for index in range(count):
        clipped = _clip_against_edge(clipped, second[index], second[(index + 1) % count])
        if len(clipped) < 3:
            return 0.0
    return _polygon_area(clipped)


def _shapely_areas(
    first: list[list[float]], second: list[list[float]]
) -> tuple[float, float] | None:
    """Intersection and union areas via shapely, or None when unusable."""
    if not _HAS_SHAPELY or _ShapelyPolygon is None:
        return None
    try:
        shape_first = _ShapelyPolygon(first).buffer(0)
        shape_second = _ShapelyPolygon(second).buffer(0)
        return (
            float(shape_first.intersection(shape_second).area),
            float(shape_first.union(shape_second).area),
        )
    except Exception:
        return None


def polygon_intersection_area(first: list[list[float]], second: list[list[float]]) -> float:
    if len(first) < 3 or len(second) < 3:
        return 0.0
    areas = _shapely_areas(first, second)
    if areas is not None:
        return areas[0]
    hull_first = _convex_hull(first)
    hull_second = _convex_hull(second)
    if len(hull_first) < 3 or len(hull_second) < 3:
        return 0.0
    return _convex_intersection_area(hull_first, hull_second)


def polygon_iou(first: list[list[float]], second: list[list[float]]) -> float:
    areas = _shapely_areas(first, second)
    if areas is not None:
        inter, union = areas
        if union <= 0.0:
            return 0.0
        return min(1.0, max(0.0, inter / union))
    area_first = _polygon_area(_convex_hull(first))
    area_second = _polygon_area(_convex_hull(second))
    if area_first <= 0.0 or area_second <= 0.0:
        return 0.0
    inter = polygon_intersection_area(first, second)
    union = area_first + area_second - inter
    if union <= 0.0:
        return 0.0
    return min(1.0, max(0.0, inter / union))


_SEGMENT_RANKS = {"IDENTICAL": 0, "NUMERIC": 1, "MISMATCH": 2}
_TRANSCRIBE_RANKS = {"IDENTICAL": 0, "CONFIDENCE_ONLY": 1, "MISMATCH": 2}


def _worsen(current: str, candidate: str, ranks: dict[str, int]) -> str:
    """A verdict only ever gets worse; never downgrade a worse verdict."""
    if ranks[candidate] > ranks[current]:
        return candidate
    return current


def _norm_segment_line(line: dict[str, Any]) -> dict[str, Any]:
    meta = line.get("source_metadata") or {}
    return {
        "order": line.get("order"),
        "kind": line.get("kind"),
        "points": _as_floats(line.get("points")),
        "baseline": _geom_points(line.get("baseline")),
        "mask": _geom_points(line.get("mask")),
        "role": meta.get("role"),
        "suspect": meta.get("suspect"),
        "external_id": line.get("external_id"),
    }


def compare_segment_lines(
    local_lines: list[dict[str, Any]], platform_lines: list[dict[str, Any]]
) -> dict[str, Any]:
    local = [_norm_segment_line(line) for line in local_lines]
    platform = [_norm_segment_line(line) for line in platform_lines]
    local_by_order = sorted(local, key=lambda line: (line["order"] is None, line["order"]))
    platform_by_order = sorted(platform, key=lambda line: (line["order"] is None, line["order"]))
    result: dict[str, Any] = {
        "local_line_count": len(local),
        "platform_line_count": len(platform),
        "verdict": "IDENTICAL",
        "order_pairing": [],
        "iou_pairing": [],
        "only_local": [],
        "only_platform": [],
        "order_differences": [],
        "line_differences": [],
    }
    if not local and not platform:
        result["verdict"] = "EMPTY"
        result["max_coordinate_diff"] = 0.0
        return result
    verdict = "IDENTICAL"
    if len(local) != len(platform):
        verdict = _worsen(verdict, "MISMATCH", _SEGMENT_RANKS)
    pairs = list(zip(local_by_order, platform_by_order, strict=False))
    max_diff = 0.0
    for position, (left, right) in enumerate(pairs):
        entry: dict[str, Any] = {
            "position": position,
            "local_order": left["order"],
            "platform_order": right["order"],
            "points_diff": max_coord_diff(left["points"], right["points"]),
            "baseline_diff": max_coord_diff(left["baseline"], right["baseline"]),
            "mask_diff": max_coord_diff(left["mask"], right["mask"]),
            "kind_equal": left["kind"] == right["kind"],
            "role_equal": left["role"] == right["role"],
            "suspect_equal": left["suspect"] == right["suspect"],
        }
        result["order_pairing"].append(entry)
        problems: list[str] = []
        if left["order"] != right["order"]:
            problems.append("order")
            result["order_differences"].append(entry)
        for key in ("points_diff", "baseline_diff", "mask_diff"):
            value = entry[key]
            if value is None:
                problems.append(key)
            else:
                max_diff = max(max_diff, value)
                if value > _POLY_TOLERANCE_PX:
                    problems.append(key)
        if not entry["kind_equal"]:
            problems.append("kind")
        if not entry["role_equal"]:
            problems.append("role")
        if not entry["suspect_equal"]:
            problems.append("suspect")
        if problems:
            result["line_differences"].append({"position": position, "fields": problems})
    matched_platform: set[int] = set()
    for local_index, left in enumerate(local):
        best_index = -1
        best_iou = 0.0
        for platform_index, right in enumerate(platform):
            if platform_index in matched_platform:
                continue
            score = polygon_iou(left["points"], right["points"])
            if score > best_iou:
                best_iou = score
                best_index = platform_index
        if best_index >= 0 and best_iou > 0.0:
            matched_platform.add(best_index)
            result["iou_pairing"].append(
                {"local": local_index, "platform": best_index, "iou": best_iou}
            )
            partner = platform[best_index]
            if (
                best_iou >= 0.5
                and left["order"] is not None
                and partner["order"] is not None
                and left["order"] != partner["order"]
            ):
                result["order_differences"].append(
                    {
                        "local": local_index,
                        "platform": best_index,
                        "local_order": left["order"],
                        "platform_order": partner["order"],
                        "iou": best_iou,
                    }
                )
        else:
            result["only_local"].append(local_index)
    result["only_platform"] = [
        index for index in range(len(platform)) if index not in matched_platform
    ]
    if result["only_local"] or result["only_platform"] or result["line_differences"]:
        verdict = _worsen(verdict, "MISMATCH", _SEGMENT_RANKS)
    elif max_diff > 0.0:
        verdict = _worsen(verdict, "NUMERIC", _SEGMENT_RANKS)
    result["verdict"] = verdict
    result["max_coordinate_diff"] = max_diff
    return result


def _levenshtein(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for index, char_first in enumerate(first, start=1):
        current = [index]
        for other, char_second in enumerate(second, start=1):
            cost = 0 if char_first == char_second else 1
            current.append(
                min(previous[other] + 1, current[other - 1] + 1, previous[other - 1] + cost)
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def _norm_local_transcribe_line(line: dict[str, Any]) -> dict[str, Any]:
    """Local contract shape: per-line ``output`` with text and confidence."""
    output = line.get("output") or {}
    return {
        "line_id": line.get("line_id"),
        "line_index": line.get("line_index"),
        "text": output.get("text", ""),
        "confidence": output.get("confidence"),
        "error": line.get("error"),
    }


def _norm_platform_transcribe_line(line: dict[str, Any]) -> dict[str, Any]:
    """Platform summary shape: flat ``line_id``, ``text``, ``confidence``."""
    return {
        "line_id": line.get("line_id"),
        "line_index": None,
        "text": line.get("text", ""),
        "confidence": line.get("confidence"),
        "error": None,
    }


def compare_transcribe_lines(
    local_lines: list[dict[str, Any]],
    platform_lines: list[dict[str, Any]],
    failed_line_indexes: list[int] | None = None,
    failed_line_ids: list[str] | None = None,
) -> dict[str, Any]:
    local = [_norm_local_transcribe_line(line) for line in local_lines]
    platform = [_norm_platform_transcribe_line(line) for line in platform_lines]
    failed_ids = {str(line_id) for line_id in (failed_line_ids or [])}
    result: dict[str, Any] = {
        "local_line_count": len(local),
        "platform_line_count": len(platform),
        "verdict": "IDENTICAL",
        "line_differences": [],
        "max_confidence_diff": 0.0,
        "max_cer": 0.0,
        "only_local": [],
        "only_platform": [],
        "failed_line_indexes": list(failed_line_indexes or []),
    }
    if not local and not platform:
        result["verdict"] = "EMPTY"
        return result
    verdict = "IDENTICAL"
    if len(local) != len(platform):
        verdict = _worsen(verdict, "MISMATCH", _TRANSCRIBE_RANKS)
    by_id_platform = {line["line_id"]: line for line in platform if line["line_id"] is not None}
    local_order = sorted(local, key=lambda item: (item["line_index"] is None, item["line_index"]))
    for left in local_order:
        line_id = left["line_id"]
        if line_id is None or line_id not in by_id_platform:
            result["only_local"].append(line_id)
            verdict = _worsen(verdict, "MISMATCH", _TRANSCRIBE_RANKS)
            continue
        right = by_id_platform[line_id]
        left_text = left["text"] or ""
        right_text = right["text"] or ""
        cer = character_error_rate(right_text, left_text)
        result["max_cer"] = max(result["max_cer"], cer)
        conf_diff: float | None = None
        if left["confidence"] is not None and right["confidence"] is not None:
            conf_diff = abs(float(left["confidence"]) - float(right["confidence"]))
            result["max_confidence_diff"] = max(result["max_confidence_diff"], conf_diff)
        platform_error = "failed on the platform" if line_id in failed_ids else None
        text_equal = left_text == right_text
        error_equal = (left["error"] or None) == platform_error
        entry: dict[str, Any] = {
            "line_id": line_id,
            "line_index": left["line_index"],
            "local_text": left_text,
            "platform_text": right_text,
            "text_equal": text_equal,
            "cer": cer,
            "confidence_diff": conf_diff,
            "local_error": left["error"],
            "platform_error": platform_error,
        }
        if not text_equal or not error_equal:
            result["line_differences"].append(entry)
            verdict = _worsen(verdict, "MISMATCH", _TRANSCRIBE_RANKS)
        elif conf_diff is not None and conf_diff > _CONFIDENCE_EPS:
            result["line_differences"].append(entry)
            verdict = _worsen(verdict, "CONFIDENCE_ONLY", _TRANSCRIBE_RANKS)
    local_ids = {line["line_id"] for line in local}
    platform_ids = {line["line_id"] for line in platform}
    result["only_platform"] = sorted(
        str(item) for item in (platform_ids - local_ids) if item is not None
    )
    if result["only_local"] or result["only_platform"]:
        verdict = _worsen(verdict, "MISMATCH", _TRANSCRIBE_RANKS)
    result["verdict"] = verdict
    return result


def parse_platform_transcribe(
    result: Any,
) -> tuple[str | None, list[dict[str, Any]], list[int]]:
    """Split the flat platform transcribe summary into its parts."""
    if not isinstance(result, dict):
        raise ApiError("platform transcribe result is not an object")
    lines = result.get("lines")
    if not isinstance(lines, list):
        raise ApiError("platform transcribe result has no lines list")
    failed = result.get("failed_line_indexes") or []
    failed_indexes = [int(index) for index in failed if isinstance(index, int)]
    transcription_id = result.get("transcription_id")
    return (
        str(transcription_id) if transcription_id is not None else None,
        [line for line in lines if isinstance(line, dict)],
        failed_indexes,
    )


class ApiError(Exception):
    pass


class ApiClient:
    """Small API client with an injectable transport for offline tests.

    The transport is a callable ``(method, url, headers, body)`` returning
    ``(status_code, response_body_bytes)``. When None, a default transport
    backed by httpx (or urllib when httpx is unavailable) is used.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        transport: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._transport = transport or self._default_transport

    def _default_transport(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        if _httpx is not None:
            try:
                with _httpx.Client(timeout=60.0, follow_redirects=True) as client:
                    response = client.request(method, url, headers=headers, content=body)
                    return response.status_code, response.content
            except _httpx.HTTPError as error:
                raise ApiError(f"{method} {url} failed: {error}") from error
        data = body if method in ("POST", "PUT", "PATCH") else None
        request = _urlrequest.Request(  # noqa: S310
            url, data=data, method=method, headers=headers or {}
        )
        try:
            with _urlrequest.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.status, response.read()
        except Exception as error:
            message = str(error)
            if hasattr(error, "read"):
                try:
                    payload = error.read()
                    detail = json.loads(payload.decode("utf-8"))
                    raise ApiError(f"{method} {url} failed: {detail}") from error
                except (ValueError, UnicodeDecodeError):
                    pass
            raise ApiError(f"{method} {url} failed: {message}") from error

    def _headers(self, as_json: bool) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if as_json:
            headers["Content-Type"] = "application/json"
        return headers

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        status, raw = self._transport(method, url, self._headers(payload is not None), body)
        if status >= 400:
            raise ApiError(f"{method} {path} returned {status}: {raw[:500]!r}")
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def request_bytes(self, method: str, path: str) -> bytes:
        url = self.base_url + path
        status, raw = self._transport(method, url, self._headers(False), None)
        if status >= 400:
            raise ApiError(f"{method} {path} returned {status}")
        return raw

    def login(self, email: str, password: str) -> str:
        url = self.base_url + "/auth/login"
        body = json.dumps({"email": email, "password": password}).encode("utf-8")
        status, raw = self._transport("POST", url, {"Content-Type": "application/json"}, body)
        if status >= 400:
            raise ApiError(f"POST /auth/login returned {status}")
        data = json.loads(raw.decode("utf-8"))
        return str(data["access_token"])

    def list_models(self) -> list[dict[str, Any]]:
        data = self.request_json("GET", "/inference/models")
        return list(data) if isinstance(data, list) else []

    def get_job(self, job_id: str) -> dict[str, Any]:
        data = self.request_json("GET", f"/jobs/{job_id}")
        if not isinstance(data, dict):
            raise ApiError(f"GET /jobs/{job_id} returned a non-object")
        return data

    def list_part_lines(
        self, project_id: str, document_id: str, part_id: str
    ) -> list[dict[str, Any]]:
        data = self.request_json(
            "GET", f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/lines"
        )
        return list(data) if isinstance(data, list) else []

    def page_image_bytes(self, part_id: str) -> bytes:
        return self.request_bytes("GET", f"/media/parts/{part_id}")

    def enqueue_segment(
        self, project_id: str, document_id: str, part_id: str, model_id: str
    ) -> dict[str, Any]:
        data = self.request_json(
            "POST",
            f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/segment",
            {"model_id": model_id},
        )
        if not isinstance(data, dict):
            raise ApiError("segment enqueue returned a non-object")
        return data

    def enqueue_transcribe(
        self,
        project_id: str,
        document_id: str,
        part_id: str,
        model_id: str,
        line_ids: list[str] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model_id": model_id}
        if line_ids is not None:
            payload["line_ids"] = line_ids
        data = self.request_json(
            "POST",
            f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/transcribe",
            payload,
        )
        if not isinstance(data, dict):
            raise ApiError("transcribe enqueue returned a non-object")
        return data


def local_package_versions() -> dict[str, str | None]:
    nomikos_version: str | None = None
    try:
        from importlib import metadata as _metadata

        nomikos_version = _metadata.version("nomikos-inference")
    except Exception:
        try:
            import nomikos_inference as _package

            nomikos_version = getattr(_package, "__version__", None)
        except Exception:
            nomikos_version = None
    onnx_version: str | None = None
    try:
        import onnxruntime as _ort

        onnx_version = getattr(_ort, "__version__", None)
    except Exception:
        onnx_version = None
    return {"nomikos_inference": nomikos_version, "onnxruntime": onnx_version}


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    return value


def build_transcribe_params(
    part_lines: list[dict[str, Any]],
    job_line_ids: list[str] | None,
    base_params: dict[str, Any],
    line_limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(
        part_lines,
        key=lambda line: (
            line.get("order") if isinstance(line.get("order"), int) else 0,
            str(line.get("created_at") or ""),
        ),
    )
    if job_line_ids:
        wanted = set(job_line_ids)
        ordered = [line for line in ordered if str(line.get("id")) in wanted]
    if line_limit is not None:
        ordered = ordered[:line_limit]
    regions = []
    for index, line in enumerate(ordered):
        regions.append(
            {
                "line_id": str(line.get("id")),
                "line_index": index,
                "points": line.get("points"),
            }
        )
    if not regions:
        raise ApiError(
            "no line regions to transcribe (empty line_ids intersection "
            "or --line-limit 0); the platform refuses empty transcribe jobs, "
            "so the local run would transcribe the whole page instead"
        )
    params = dict(base_params)
    params.pop("lines", None)
    params["lines"] = regions
    return params, ordered


def render_report_markdown(report: dict[str, Any]) -> str:
    header = report["header"]
    comparison = report["comparison"]
    lines = [
        "# Parity report: local vs platform",
        "",
    ]
    if comparison.get("verdict") == "NOT_COMPARABLE":
        lines.append(f"NOTICE: {comparison.get('reason', 'not comparable')}")
        lines.append("")
    lines.extend(
        [
            f"Task: {header['task']}",
            f"Model: {header['model_name']} (registry {header['registry_model_id']}, "
            f"tag {header['registry_tag']})",
            f"Job: {header['job_id']}",
            f"Verdict: {comparison['verdict']}",
            "",
            "## Header",
            "",
            f"API host: {header['api_host']}",
            f"Local nomikos_inference: {header['local_nomikos_inference_version']}",
            f"Local onnxruntime: {header['local_onnxruntime_version']}",
            "Worker version: not exposed by the API",
            f"Image sha256: {header['image_sha256']} ({header['image_bytes']} bytes)",
            f"Params: {json.dumps(header['params'], sort_keys=True)}",
        ]
    )
    if header.get("merge_summary") is not None:
        lines.append(f"Merge summary: {json.dumps(header['merge_summary'], sort_keys=True)}")
    if header.get("transcription_id") is not None:
        lines.append(f"Transcription: {header['transcription_id']}")
    if header.get("lines_requested") is not None:
        lines.append(f"Lines requested: {header['lines_requested']}")
    if header.get("lines_compared") is not None:
        lines.append(f"Lines compared: {header['lines_compared']}")
    lines.extend(["", "## Comparison", ""])
    if comparison.get("verdict") == "NOT_COMPARABLE":
        lines.append(f"No comparison: {comparison.get('reason', 'not comparable')}")
        lines.append("")
        return "\n".join(lines) + "\n"
    lines.append(f"Local lines: {comparison.get('local_line_count')}")
    lines.append(f"Platform lines: {comparison.get('platform_line_count')}")
    if header["task"] == "segment":
        lines.append(f"Max coordinate diff (px): {comparison.get('max_coordinate_diff')}")
        if comparison.get("order_differences"):
            lines.append(f"Order differences: {len(comparison['order_differences'])}")
        for item in comparison.get("line_differences", [])[:50]:
            lines.append(f"Line {item['position']}: {', '.join(item['fields'])}")
        if comparison.get("only_local"):
            lines.append(f"Only local: {comparison['only_local']}")
        if comparison.get("only_platform"):
            lines.append(f"Only platform: {comparison['only_platform']}")
    else:
        lines.append(f"Max CER: {comparison.get('max_cer')}")
        lines.append(f"Max confidence diff: {comparison.get('max_confidence_diff')}")
        if comparison.get("failed_line_indexes"):
            lines.append(f"Failed line indexes: {comparison['failed_line_indexes']}")
        for item in comparison.get("line_differences", [])[:50]:
            marker = " [platform failed]" if item.get("platform_error") else ""
            lines.append(
                f"Line {item.get('line_id')} (index {item.get('line_index')}): "
                f"local {item.get('local_text')!r} platform {item.get('platform_text')!r} "
                f"CER {item.get('cer')}{marker}"
            )
        if comparison.get("only_local"):
            lines.append(f"Only local: {comparison['only_local']}")
        if comparison.get("only_platform"):
            lines.append(f"Only platform: {comparison['only_platform']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a local nomikos_inference run with the same platform job. "
            "--project-id, --document-id and --part-id are required for both tasks."
        )
    )
    parser.add_argument("--api", required=True)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--task", choices=("segment", "transcribe"), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--i-know-segment-replaces-the-lines", action="store_true")
    parser.add_argument(
        "--allow-merged",
        action="store_true",
        help="compare segment output even when the part holds protected lines",
    )
    parser.add_argument(
        "--trust-job-model",
        action="store_true",
        help="compare against a job whose model cannot be verified from its payload",
    )
    parser.add_argument("--line-limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def poll_job(client: ApiClient, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        job = client.get_job(job_id)
        if str(job.get("status")) in TERMINAL_STATUSES:
            return job
        if time.monotonic() >= deadline:
            raise ApiError(f"timed out waiting for job {job_id}")
        time.sleep(5.0)


def _default_out_dir() -> str:
    stamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("/tmp", "nmk-parity", stamp)  # noqa: S108


def collect_secret_values(args: argparse.Namespace) -> list[str]:
    secrets: list[str] = []
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            secrets.append(value)
    if args.env_file:
        try:
            file_values = _read_env_file(args.env_file)
        except OSError:
            file_values = {}
        for name in _SECRET_ENV_NAMES:
            value = file_values.get(name)
            if value:
                secrets.append(value)
    return [secret for secret in secrets if secret]


_MODEL_IDENTIFYING_KEYS = ("registry_model_id", "registry_id", "model", "model_id")


def cross_check_model(payload: dict[str, Any], registry_model_id: str, model_uuid: str) -> bool:
    """Fail when the job payload names a different model than --model resolved.

    Returns True when the payload carries a readable identifier that agrees,
    False when it carries nothing identifying the model.
    """
    ml_params = payload.get("ml_params")
    if not isinstance(ml_params, dict):
        return False
    verified = False
    for key in (*_MODEL_IDENTIFYING_KEYS, "artifact_ref"):
        value = ml_params.get(key)
        if value is None:
            continue
        text = str(value)
        if key == "artifact_ref":
            try:
                candidate, _ = parse_artifact_ref(text)
            except ValueError:
                continue
        else:
            candidate = text
        verified = True
        if candidate not in (registry_model_id, model_uuid):
            raise ApiError(
                f"job payload {key} {text!r} disagrees with --model "
                f"(registry {registry_model_id}, id {model_uuid})"
            )
    return verified


def _parse_dt(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        parsed = _datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_datetime.UTC)
    return parsed


_MODEL_LINE_SOURCES = ("kraken", "model")


def segment_staleness_reason(
    stored_lines: list[dict[str, Any]], job_id: str, job_completed_at: Any
) -> str | None:
    """Name why stored lines no longer represent the selected segment job.

    Every model-produced stored line (``source`` kraken or model, not manual
    geometry) must name the selected job in ``source_metadata.job_id``
    (stamped by ``SegmentMergeService``). A line naming another job that was
    created after the selected job finished is newer state, so the stored
    lines are not the pure model output. Older foreign lines are pre-existing
    state covered by the protected-lines gate instead.
    """
    finished = _parse_dt(job_completed_at)
    for line in stored_lines:
        if line.get("manual_geometry"):
            continue
        if str(line.get("source") or "") not in _MODEL_LINE_SOURCES:
            continue
        meta = line.get("source_metadata")
        if isinstance(meta, dict) and str(meta.get("job_id") or "") == str(job_id):
            continue
        created = _parse_dt(line.get("created_at"))
        if created is not None and finished is not None and created <= finished:
            continue
        return f"lines no longer come from job {job_id}"
    return None


def _sort_part_lines(part_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        part_lines,
        key=lambda line: (
            line.get("order") if isinstance(line.get("order"), int) else 0,
            str(line.get("created_at") or ""),
        ),
    )


_EXIT_BY_VERDICT = {
    "IDENTICAL": 0,
    "NUMERIC": 1,
    "CONFIDENCE_ONLY": 1,
    "MISMATCH": 1,
    "EMPTY": 3,
    "NOT_COMPARABLE": 3,
}


def _transcribe_unmappable(
    args: argparse.Namespace,
    model_entry: dict[str, Any],
    registry_model_id: str,
    registry_tag: str,
    model_uuid: str,
    job_id: str,
    base_params: dict[str, Any],
    transcription_id: str | None,
    model_verification: str,
    lines_requested: int,
    lines_compared: int,
    summary: dict[str, Any],
    out_dir: str,
) -> int:
    reason = "failed line indexes cannot be mapped to line ids"
    print(reason, file=sys.stderr)
    header = _base_header(
        args,
        model_entry,
        registry_model_id,
        registry_tag,
        model_uuid,
        job_id,
        dict(base_params),
        None,
        None,
    )
    header["transcription_id"] = transcription_id
    header["model_verification"] = model_verification
    header["lines_requested"] = lines_requested
    header["lines_compared"] = lines_compared
    comparison = {"verdict": "NOT_COMPARABLE", "reason": reason}
    return _write_report(args, header, comparison, None, summary, out_dir)


def _write_private_file(path: str, content: str) -> None:
    """Write a report file readable only by its owner."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.chmod(path, 0o600)


def _write_report(
    args: argparse.Namespace,
    header: dict[str, Any],
    comparison: dict[str, Any],
    local_dump: Any,
    platform_result: Any,
    out_dir: str,
) -> int:
    secrets = collect_secret_values(args)
    report: dict[str, Any] = {
        "header": header,
        "comparison": comparison,
        "local_result": local_dump,
        "platform_result": platform_result,
    }
    report = _scrub(report, secrets)
    os.makedirs(out_dir, mode=0o700, exist_ok=True)
    os.chmod(out_dir, 0o700)
    _write_private_file(
        os.path.join(out_dir, "report.json"),
        json.dumps(report, indent=2, sort_keys=True, default=str),
    )
    _write_private_file(os.path.join(out_dir, "report.md"), render_report_markdown(report))
    verdict = comparison.get("verdict", "MISMATCH")
    print(f"Verdict: {verdict} (report in {out_dir})")
    return _EXIT_BY_VERDICT.get(verdict, 1)


def _base_header(
    args: argparse.Namespace,
    model_entry: dict[str, Any],
    registry_model_id: str,
    registry_tag: str,
    model_uuid: str,
    job_id: str,
    params: dict[str, Any],
    image_sha256: str | None,
    image_size: int | None,
) -> dict[str, Any]:
    versions = local_package_versions()
    return {
        "api_host": args.api,
        "task": args.task,
        "model_name": model_entry.get("name"),
        "registry_model_id": registry_model_id,
        "registry_tag": registry_tag,
        "model_id": model_uuid,
        "job_id": job_id,
        "local_nomikos_inference_version": versions["nomikos_inference"],
        "local_onnxruntime_version": versions["onnxruntime"],
        "params": params,
        "image_sha256": image_sha256,
        "image_bytes": image_size,
    }


def run_comparison(
    args: argparse.Namespace,
    client: ApiClient,
    out_dir: str,
) -> int:
    if args.task == "segment" and args.enqueue and not args.i_know_segment_replaces_the_lines:
        print(
            "Refusing: a segment job replaces the part lines. "
            "Re-run with --i-know-segment-replaces-the-lines to proceed.",
            file=sys.stderr,
        )
        return 2
    catalog = client.list_models()
    if args.model is None:
        raise ApiError("missing required --model")
    model_entry = resolve_model(catalog, args.model)
    registry_model_id, registry_tag = parse_artifact_ref(str(model_entry["artifact_ref"]))
    model_uuid = str(model_entry.get("id"))

    job_id = args.job_id
    if args.enqueue:
        if args.task == "segment":
            current = client.list_part_lines(args.project_id, args.document_id, args.part_id)
            print(f"Part currently has {len(current)} lines; segment will replace them.")
            created = client.enqueue_segment(
                args.project_id, args.document_id, args.part_id, model_uuid
            )
        else:
            if args.line_limit is not None and args.line_limit < 1:
                raise ApiError("--line-limit must be at least 1")
            enqueue_ids: list[str] | None = None
            if args.line_limit is not None:
                candidates = _sort_part_lines(
                    client.list_part_lines(args.project_id, args.document_id, args.part_id)
                )
                enqueue_ids = [str(line.get("id")) for line in candidates[: args.line_limit]]
                if not enqueue_ids:
                    raise ApiError("no line regions to transcribe (--line-limit 0 on no lines)")
            created = client.enqueue_transcribe(
                args.project_id, args.document_id, args.part_id, model_uuid, enqueue_ids
            )
        job_id = str(created.get("job_id") or created.get("id"))
        if not job_id or job_id == "None":
            raise ApiError(f"enqueue response has no job id: {created!r}")
    if not job_id:
        raise ApiError("missing required --job-id (or use --enqueue to create one)")
    job = poll_job(client, job_id, args.timeout)
    if str(job.get("status")) == "failed":
        print(f"Platform job {job_id} failed: {job.get('error')}", file=sys.stderr)
        return 2
    if str(job.get("status")) != "done":
        print(
            f"Platform job {job_id} ended with status {job.get('status')}",
            file=sys.stderr,
        )
        return 2
    if str(job.get("type")) != args.task:
        raise ApiError(f"job {job_id} has type {job.get('type')!r} but --task is {args.task!r}")
    if (
        str(job.get("document_id")) != args.document_id
        or str(job.get("document_part_id")) != args.part_id
    ):
        raise ApiError(
            f"job {job_id} targets document {job.get('document_id')} "
            f"part {job.get('document_part_id')} but --document-id is "
            f"{args.document_id} and --part-id is {args.part_id}"
        )
    print("Note: registry id and tag come from --model, the job does not record them.")
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    model_verified = cross_check_model(payload, registry_model_id, model_uuid)
    if model_verified:
        model_verification = "verified"
    elif args.enqueue:
        model_verification = "enqueued by this run"
    elif args.trust_job_model:
        model_verification = "model of the job not verified"
    else:
        reason = (
            "model of the job could not be verified; "
            "re-run with --trust-job-model to compare anyway"
        )
        print(reason, file=sys.stderr)
        header = _base_header(
            args,
            model_entry,
            registry_model_id,
            registry_tag,
            model_uuid,
            job_id,
            dict(payload.get("ml_params") or {}),
            None,
            None,
        )
        header["model_verification"] = "model of the job not verified"
        comparison = {"verdict": "NOT_COMPARABLE", "reason": reason}
        return _write_report(args, header, comparison, None, job.get("result"), out_dir)
    base_params = payload.get("ml_params") if isinstance(payload.get("ml_params"), dict) else {}
    params: dict[str, Any] = dict(base_params)
    part_id = args.part_id

    if args.task == "segment":
        stored_lines = _sort_part_lines(
            client.list_part_lines(args.project_id, args.document_id, part_id)
        )
        summary = job.get("result") if isinstance(job.get("result"), dict) else {}
        stale_reason = segment_staleness_reason(stored_lines, job_id, job.get("completed_at"))
        if stale_reason is not None:
            print(stale_reason, file=sys.stderr)
            header = _base_header(
                args,
                model_entry,
                registry_model_id,
                registry_tag,
                model_uuid,
                job_id,
                params,
                None,
                None,
            )
            header["merge_summary"] = summary
            header["model_verification"] = model_verification
            comparison = {
                "verdict": "NOT_COMPARABLE",
                "reason": stale_reason,
                "merge_summary": summary,
                "stored_line_count": len(stored_lines),
            }
            return _write_report(
                args, header, comparison, None, {"merge_summary": summary}, out_dir
            )
        protected = sum(
            int(summary.get(key) or 0)
            for key in (
                "preserved_manual_lines",
                "preserved_transcribed_lines",
                "skipped_covered_lines",
            )
        )
        if protected > 0 and not args.allow_merged:
            print(
                "Stored lines mix model output with protected lines; "
                "re-run with --allow-merged to compare anyway.",
                file=sys.stderr,
            )
            header = _base_header(
                args,
                model_entry,
                registry_model_id,
                registry_tag,
                model_uuid,
                job_id,
                params,
                None,
                None,
            )
            header["merge_summary"] = summary
            header["model_verification"] = model_verification
            comparison = {
                "verdict": "NOT_COMPARABLE",
                "reason": (
                    "the part held protected lines (manual, transcribed, or covered), "
                    "so the stored lines are not the pure model output. "
                    "Re-run with --allow-merged to compare anyway."
                ),
                "protected_lines": protected,
                "merge_summary": summary,
                "stored_line_count": len(stored_lines),
            }
            return _write_report(
                args, header, comparison, None, {"merge_summary": summary}, out_dir
            )
        image_bytes = client.page_image_bytes(part_id)
        local_dump = _run_local(args.task, registry_model_id, registry_tag, image_bytes, params)
        comparison = compare_segment_lines(local_dump.get("lines", []), stored_lines)
        header = _base_header(
            args,
            model_entry,
            registry_model_id,
            registry_tag,
            model_uuid,
            job_id,
            params,
            hashlib.sha256(image_bytes).hexdigest(),
            len(image_bytes),
        )
        header["merge_summary"] = summary
        header["model_verification"] = model_verification
        return _write_report(
            args,
            header,
            comparison,
            local_dump,
            {"merge_summary": summary, "lines": stored_lines},
            out_dir,
        )

    part_lines = client.list_part_lines(args.project_id, args.document_id, part_id)
    summary = job.get("result") if isinstance(job.get("result"), dict) else {}
    transcription_id, platform_lines, failed_indexes = parse_platform_transcribe(summary)
    result_ids = [
        str(line.get("line_id")) for line in platform_lines if line.get("line_id") is not None
    ]
    # One dispatched order for both cases: the dispatcher filters its
    # database-ordered rows by the payload id set, so failed indexes are
    # positions in (order, created_at) sequence, never in payload order
    # (inference_dispatcher.py:156-170, load_lines order).
    payload_ids = payload.get("line_ids")
    if isinstance(payload_ids, list) and payload_ids:
        selective_ids: list[str] | None = [str(line_id) for line_id in payload_ids]
    else:
        selective_ids = None
    candidate_ids = selective_ids if selective_ids is not None else result_ids
    page_ids = {str(line.get("id")) for line in part_lines}
    early_missing = [line_id for line_id in candidate_ids if line_id not in page_ids]
    sorted_page = _sort_part_lines(part_lines)
    failed_ids = []
    if early_missing:
        requested_ids = candidate_ids
        missing_ids = early_missing
    else:
        if selective_ids is not None:
            wanted = set(selective_ids)
            dispatched_ids = [
                str(line.get("id")) for line in sorted_page if str(line.get("id")) in wanted
            ]
        else:
            dispatched_ids = [str(line.get("id")) for line in sorted_page]
        if failed_indexes:
            failed_set = set(failed_indexes)
            kept_ids = [
                line_id for pos, line_id in enumerate(dispatched_ids) if pos not in failed_set
            ]
            trusted = (
                len(dispatched_ids) == len(result_ids) + len(failed_indexes)
                and kept_ids == result_ids
            )
            if trusted:
                failed_ids = [dispatched_ids[index] for index in sorted(failed_set)]
            else:
                return _transcribe_unmappable(
                    args,
                    model_entry,
                    registry_model_id,
                    registry_tag,
                    model_uuid,
                    job_id,
                    base_params,
                    transcription_id,
                    model_verification,
                    len(result_ids) + len(failed_indexes),
                    len(result_ids),
                    summary,
                    out_dir,
                )
        if args.line_limit is not None:
            requested_ids = dispatched_ids[: args.line_limit]
        else:
            requested_ids = dispatched_ids
        missing_ids = [line_id for line_id in requested_ids if line_id not in page_ids]
    if missing_ids:
        reason = (
            f"job line {missing_ids[0]} no longer exists on the page "
            f"({len(missing_ids)} of {len(requested_ids)} requested lines missing)"
        )
        print(reason, file=sys.stderr)
        header = _base_header(
            args,
            model_entry,
            registry_model_id,
            registry_tag,
            model_uuid,
            job_id,
            dict(base_params),
            None,
            None,
        )
        header["transcription_id"] = transcription_id
        header["model_verification"] = model_verification
        header["lines_requested"] = len(requested_ids)
        header["lines_compared"] = len(requested_ids) - len(missing_ids)
        comparison = {
            "verdict": "NOT_COMPARABLE",
            "reason": reason,
            "missing_line_ids": missing_ids,
        }
        return _write_report(args, header, comparison, None, summary, out_dir)
    params, _ = build_transcribe_params(part_lines, requested_ids or None, base_params, None)
    image_bytes = client.page_image_bytes(part_id)
    local_dump = _run_local(args.task, registry_model_id, registry_tag, image_bytes, params)
    requested_set = set(requested_ids)
    platform_subset = [line for line in platform_lines if str(line.get("line_id")) in requested_set]
    present_ids = {str(line.get("line_id")) for line in platform_subset}
    for failed_id in failed_ids:
        if failed_id in requested_set and failed_id not in present_ids:
            platform_subset.append({"line_id": failed_id, "text": "", "confidence": None})
            present_ids.add(failed_id)
    failed_in_scope = [failed_id for failed_id in failed_ids if failed_id in requested_set]
    comparison = compare_transcribe_lines(
        local_dump.get("lines", []), platform_subset, failed_indexes, failed_in_scope
    )
    header = _base_header(
        args,
        model_entry,
        registry_model_id,
        registry_tag,
        model_uuid,
        job_id,
        params,
        hashlib.sha256(image_bytes).hexdigest(),
        len(image_bytes),
    )
    header["transcription_id"] = transcription_id
    header["model_verification"] = model_verification
    header["lines_requested"] = len(requested_ids) if requested_ids else len(params["lines"])
    header["lines_compared"] = len(platform_subset)
    return _write_report(args, header, comparison, local_dump, summary, out_dir)


def _run_local(
    task: str,
    registry_model_id: str,
    registry_tag: str,
    image_bytes: bytes,
    params: dict[str, Any],
) -> Any:
    from nomikos_inference.contracts.common import InferenceTask
    from nomikos_inference.jobs.runner import run_model

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    local_response = run_model(
        task=InferenceTask(task),
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        image_bytes=image_bytes,
        params=params,
    )
    return _model_dump(local_response)


def main(argv: list[str] | None = None, transport: Any = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task == "segment" and args.enqueue and not args.i_know_segment_replaces_the_lines:
        print(
            "Refusing: a segment job replaces the part lines. "
            "Re-run with --i-know-segment-replaces-the-lines to proceed.",
            file=sys.stderr,
        )
        return 2
    token, email, password = _credentials(args)
    if token is None:
        if email is None or password is None:
            print(
                "Missing credentials: set NOMIKOS_TOKEN or "
                "NOMIKOS_EMAIL plus NOMIKOS_PASSWORD (or --env-file).",
                file=sys.stderr,
            )
            return 2
        client = ApiClient(args.api, "", transport=transport)
        token = client.login(email, password)
    client = ApiClient(args.api, token, transport=transport)
    out_dir = args.out or _default_out_dir()
    try:
        return run_comparison(args, client, out_dir)
    except ApiError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

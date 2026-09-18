"""Shared plumbing for the offline BLLA segmentation experiments.

Everything here calls the production entry point
``nomikos_inference.architectures.blla.run_blla_segment`` on real image bytes,
so variant V0 is production byte for byte. Variants differ only in *what image
bytes* are handed to it and how the returned coordinates are mapped back into
the source page.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MAIN = Path("/Users/krishuagarwal/Desktop/Programming/python/greekOCR")
BLLA_INPUT_HEIGHT = 1800


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
def blla_weights() -> tuple[Path, str]:
    """Resolve the pinned blla-segment artifact exactly as the job runner does."""
    from nomikos_inference.contracts.common import InferenceTask
    from nomikos_inference.registry.resolve import resolve_registry_entry
    from nomikos_inference.weights import resolve_weights_source

    entry = resolve_registry_entry(
        registry_model_id="blla-segment",
        registry_tag="stable",
        task=InferenceTask.segment,
        registry_path=MAIN / "nomikos_inference" / "registry.yaml",
    )
    version = entry.versions["stable"]
    path = resolve_weights_source(
        version.weights_source,
        registry_model_id="blla-segment",
        registry_tag="stable",
        hub_revision=version.hub_revision,
        artifact_sha256=version.artifact_sha256,
        architecture=entry.architecture.value,
    )
    return path, version.artifact_sha256


# --------------------------------------------------------------------------
# one segmentation call
# --------------------------------------------------------------------------
@dataclass
class RawLines:
    lines: list[dict[str, Any]]
    seconds: float


def _encode(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def segment_image(
    image: Image.Image,
    *,
    model_path: Path,
    artifact_sha256: str,
    heatmap_threshold: float | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
) -> RawLines:
    """Segment one PIL image and return lines shifted by ``offset``.

    ``offset`` is the position of this image's top-left corner inside the
    source page, so a crop or tile comes back in source pixels.
    """
    from nomikos_inference.architectures.blla import run_blla_segment

    params: dict[str, Any] = {}
    if heatmap_threshold is not None:
        params["heatmap_threshold"] = heatmap_threshold
    payload = _encode(image)
    started = time.perf_counter()
    response = run_blla_segment(
        payload,
        model_path=model_path,
        artifact_sha256=artifact_sha256,
        params=params or None,
    )
    seconds = time.perf_counter() - started
    dx, dy = offset
    out = []
    for line in response.lines:
        baseline = [[float(x) + dx, float(y) + dy] for x, y in line.baseline["points"]]
        polygon = [[float(x) + dx, float(y) + dy] for x, y in line.points]
        out.append({"baseline": baseline, "polygon": polygon})
    return RawLines(lines=out, seconds=seconds)


# --------------------------------------------------------------------------
# prediction files
# --------------------------------------------------------------------------
def write_prediction(
    out_dir: Path,
    variant: str,
    slug: str,
    image_size: tuple[int, int],
    seconds: float,
    lines: list[dict[str, Any]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "page": slug,
        "variant": variant,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "seconds": round(float(seconds), 3),
        "lines": [
            {
                "baseline": [[round(float(x), 2), round(float(y), 2)] for x, y in ln["baseline"]],
                "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in ln["polygon"]],
            }
            for ln in lines
        ],
    }
    path = out_dir / f"{slug}.json"
    path.write_text(json.dumps(payload))
    return path


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def baseline_y(baseline: list[list[float]]) -> float:
    import statistics

    return statistics.median(p[1] for p in baseline)


def baseline_x_extent(baseline: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in baseline]
    return min(xs), max(xs)


def dedupe_lines(
    lines: list[dict[str, Any]], *, pitch: float
) -> list[dict[str, Any]]:
    """Drop near-duplicate lines produced where two tiles overlap.

    Two lines are duplicates when their baselines sit within 0.3 * pitch
    vertically and their x extents overlap by at least half of the shorter one.
    The survivor is the longer baseline, which is the one whose tile saw more
    of the line.
    """
    order = sorted(
        range(len(lines)),
        key=lambda i: -(baseline_x_extent(lines[i]["baseline"])[1]
                        - baseline_x_extent(lines[i]["baseline"])[0]),
    )
    kept: list[int] = []
    for i in order:
        yi = baseline_y(lines[i]["baseline"])
        x0i, x1i = baseline_x_extent(lines[i]["baseline"])
        clash = False
        for j in kept:
            yj = baseline_y(lines[j]["baseline"])
            if abs(yi - yj) > 0.3 * pitch:
                continue
            x0j, x1j = baseline_x_extent(lines[j]["baseline"])
            inter = min(x1i, x1j) - max(x0i, x0j)
            shorter = min(x1i - x0i, x1j - x0j)
            if shorter > 0 and inter >= 0.5 * shorter:
                clash = True
                break
        if not clash:
            kept.append(i)
    return [lines[i] for i in sorted(kept, key=lambda i: baseline_y(lines[i]["baseline"]))]

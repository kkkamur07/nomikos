#!/usr/bin/env python3
"""Shared PAGE-XML line cutting for the PP-OCR recognition parity scripts.

Both ``verify_ppocr_rec_parity.py`` (tensor inputs for the ONNX-vs-kraken
gate) and ``verify_ppocr_rec_adapter.py`` (PNG crops for the serving-adapter
gate) cut the same lines with kraken's own extractor. This module owns that
walk so the two scripts cannot drift apart: the XML cleaning, the page-image
lookup, the polygon-vs-box branch and the line budget live here, once. Each
script keeps only its own ``take`` step (tensors vs PNG bytes).

This is a library, not a script: no argparse, no ``main``. kraken and PIL
stay function-local so importing this module never requires model weights.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Line-image file suffixes accepted for --lines and page directories.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")


class PageXMLLine(Protocol):
    """The kraken PAGE-XML line surface both scripts rely on."""

    boundary: Sequence[Sequence[float]] | None
    baseline: Sequence[Sequence[float]]
    text: str | None


@dataclasses.dataclass
class CutLine:
    """One kraken-cut line crop plus the facts both scripts need about it."""

    name: str
    crop: PILImage
    truth: str | None
    # valid_norm mirrors CTCRecognitionInferenceMixin._recognition_pred:
    # False on the baselines path, True on the box path. For 3-channel RGB
    # input both build identical transforms (the flag only selects the
    # center-norm branch for single-channel input); it is threaded through
    # so tensor construction matches kraken's line for line.
    valid_norm: bool


def strip_custom_attributes(xml_text: str) -> str:
    """Drop ``custom="..."`` so kraken's ``parse_page_custom`` accepts the file."""
    import re

    return re.sub(r' custom="[^"]*"', "", xml_text)


def has_real_boundary(line: PageXMLLine) -> bool:
    """A boundary that is not just the baseline polyline stored twice."""
    if line.boundary is None:
        return False
    return list(map(tuple, line.boundary)) != list(map(tuple, line.baseline))


def greedy_decode(logits, labels):
    """Collapse repeats, drop blank 0, map through the codec (display order)."""
    best = logits.argmax(1)
    parts = []
    previous = -1
    for label in best:
        label = int(label)
        if label == 0:
            previous = 0
            continue
        if label != previous:
            parts.append(labels.get(label, ""))
        previous = label
    return "".join(parts)


def find_page_image(pages_dir: Path, stem: str) -> Path | None:
    """First page image matching ``stem`` under ``pages_dir``, if any."""
    for suffix in IMAGE_SUFFIXES:
        candidate = pages_dir / (stem + suffix)
        if candidate.is_file():
            return candidate
    return None


def read_xml_page(xml_path: Path, workspace: Path):
    """Parse one PAGE-XML file after cleaning; None when it has no baselines."""
    from kraken.lib.xml import XMLPage

    cleaned = workspace / xml_path.name
    cleaned.write_text(
        strip_custom_attributes(xml_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    page = XMLPage(str(cleaned), linetype="baselines")
    if page.type != "baselines" or not page.lines:
        print(f"{xml_path.name} has no baselines, skipping page", file=sys.stderr)
        return None
    return page


def cut_page_crops(
    page, image, stem: str, *, already: int, max_lines: int | None
) -> tuple[
    list[CutLine],
    dict[str, int],
]:
    """Cut one page's sorted lines; uncuttable lines are skipped with a warning."""
    from kraken.containers import Segmentation
    from kraken.lib.segmentation import extract_polygons

    records: list[CutLine] = []
    counters = {"polygon_lines": 0, "box_lines": 0, "skipped_lines": 0}
    for number, xml_line in enumerate(page.get_sorted_lines()):
        if max_lines is not None and already + len(records) >= max_lines:
            break
        name = f"{stem}#{number}"
        try:
            if has_real_boundary(xml_line):
                single = dataclasses.replace(page.to_container(), lines=[xml_line])
                crop, _ = next(extract_polygons(image, single))
                counters["polygon_lines"] += 1
                records.append(CutLine(name=name, crop=crop, truth=xml_line.text, valid_norm=False))
            else:
                box = xml_line.to_bbox(text_direction="horizontal-lr")
                single = Segmentation(
                    type="bbox",
                    imagename=page.imagename,
                    text_direction="horizontal-lr",
                    script_detection=True,
                    lines=[box],
                    regions={},
                )
                crop, _ = next(extract_polygons(image, single))
                counters["box_lines"] += 1
                records.append(CutLine(name=name, crop=crop, truth=xml_line.text, valid_norm=True))
        except (StopIteration, ValueError) as error:
            counters["skipped_lines"] += 1
            print(f"cannot cut {name}: {error}", file=sys.stderr)
    return records, counters


def collect_page_crops(
    pages_dir: Path | None, xml_dir: Path | None, *, max_lines: int | None
) -> tuple[list[CutLine], dict[str, int]]:
    """Cut every XML line with kraken's extractor; returns crops and counters.

    The temporary directory for cleaned XML lives exactly as long as this
    call, so an exception mid-loop cannot leave cleaned copies in /tmp.
    """
    from PIL import Image

    records: list[CutLine] = []
    counters = {"pages": 0, "polygon_lines": 0, "box_lines": 0, "skipped_lines": 0}
    if xml_dir is None or pages_dir is None:
        return records, counters
    with tempfile.TemporaryDirectory(prefix="ppocr-rec-xml-") as workspace:
        scratch = Path(workspace)
        for xml_path in sorted(xml_dir.glob("*.xml")):
            if max_lines is not None and len(records) >= max_lines:
                break
            page_path = find_page_image(pages_dir, xml_path.stem)
            if page_path is None:
                print(f"no page image for {xml_path.name}, skipping page", file=sys.stderr)
                continue
            page = read_xml_page(xml_path, scratch)
            if page is None:
                continue
            counters["pages"] += 1
            with Image.open(page_path) as image:
                rgb = image.convert("RGB")
                page_records, page_counters = cut_page_crops(
                    page, rgb, xml_path.stem, already=len(records), max_lines=max_lines
                )
            records.extend(page_records)
            for key in ("polygon_lines", "box_lines", "skipped_lines"):
                counters[key] += page_counters[key]
    return records, counters


def collect_file_crops(
    line_files: Sequence[Path | str], *, already: int, max_lines: int | None
) -> list[CutLine]:
    """Cut extra line image files as plain RGB images, honouring the budget."""
    from PIL import Image

    records: list[CutLine] = []
    for extra in line_files:
        if max_lines is not None and already + len(records) >= max_lines:
            break
        with Image.open(extra) as image:
            records.append(
                CutLine(
                    name=Path(extra).name,
                    crop=image.convert("RGB"),
                    truth=None,
                    valid_norm=False,
                )
            )
    return records

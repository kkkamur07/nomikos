"""Schema-level sanity for an exported PAGE XML file, with no kraken on the path.

A trainer is the wrong place to find out that an export is malformed: by then the
file is in somebody's training folder, the error is a stack trace from a parser
nobody here wrote, and the page it came from is three commands away. This module
asks the questions that do not need a segmenter to answer, so a unit test can ask
them of every branch of the renderer, and ``check_page_xml_with_kraken.py`` is
left to answer only the one question that genuinely needs kraken: whether kraken
agrees.

Every check is a statement about the *file*, never about the manuscript. Nothing
here knows whether the transcription is correct, only whether a trainer can read
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree.ElementTree import Element, ParseError, fromstring

PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
_NS = {"p": PAGE_XML_NAMESPACE}


@dataclass
class PageXmlValidation:
    """What a file got wrong, and how much of it there was.

    ``problems`` is a list rather than a first failure: one malformed line in a
    forty-line page is worth reporting alongside the other thirty-nine being fine,
    and a caller sweeping a training folder wants the whole picture per file.
    """

    line_count: int = 0
    region_count: int = 0
    derived_baseline_count: int = 0
    #: Lines whose ``TextEquiv/Unicode`` is present but empty. Counted, not faulted:
    #: a page can be segmented and only half transcribed, and that export is still a
    #: valid file and still trains a segmenter. A recognition run is what cares, so
    #: it gets a number to filter on rather than a failed file.
    lines_without_text: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_page_xml(xml: bytes | str) -> PageXmlValidation:
    """Parse ``xml`` and report every way it would fail a segmenter trainer."""
    result = PageXmlValidation()
    try:
        # The input is this platform's own export, re-read to check it. It is not
        # untrusted third-party XML, and no entity expansion is enabled.
        root = fromstring(xml)  # noqa: S314
    except ParseError as error:
        result.problems.append(f"not well-formed XML: {error}")
        return result

    if root.tag != f"{{{PAGE_XML_NAMESPACE}}}PcGts":
        result.problems.append(
            f"root is {root.tag}, expected PcGts in the PAGE 2019-07-15 namespace"
        )
        return result

    page = root.find("p:Page", _NS)
    if page is None:
        result.problems.append("no Page element")
        return result
    if not page.get("imageFilename"):
        result.problems.append("Page has no imageFilename")

    regions = page.findall("p:TextRegion", _NS)
    result.region_count = len(regions)
    _check_reading_order(page, regions, result)

    for region in regions:
        region_id = region.get("id") or "<unnamed region>"
        _check_custom(region, region_id, result)
        region_coords = region.find("p:Coords", _NS)
        if region_coords is None:
            # kraken drops a region with no boundary, so a page that looks like two
            # columns here reads back as zero regions there.
            result.problems.append(f"{region_id}: TextRegion has no Coords")
        else:
            _check_integer_points(region_coords, region_id, result)
        for text_line in region.findall("p:TextLine", _NS):
            result.line_count += 1
            _check_text_line(text_line, result)
    return result


def _check_reading_order(page: Element, regions: list[Element], result: PageXmlValidation) -> None:
    """A ReadingOrder is optional, but one that points at nothing is worse than none."""
    reading_order = page.find("p:ReadingOrder", _NS)
    if reading_order is None:
        return
    region_ids = {region.get("id") for region in regions}
    referenced = [
        ref.get("regionRef")
        for ref in reading_order.iter(f"{{{PAGE_XML_NAMESPACE}}}RegionRefIndexed")
    ]
    for region_ref in referenced:
        if region_ref not in region_ids:
            result.problems.append(f"ReadingOrder references unknown region {region_ref!r}")
    missing = region_ids - set(referenced)
    if missing:
        result.problems.append(f"ReadingOrder omits regions {sorted(str(m) for m in missing)}")


#: The tags kraken's ``page.py`` acts on. Writing one is a claim about the line or
#: region class, or about reading order, so an exporter has to mean it.
_KRAKEN_TAGS = frozenset({"structure", "readingOrder", "split"})


def _check_custom(element: Element, label: str, result: PageXmlValidation) -> dict[str, str]:
    """``custom`` parses under the Transkribus grammar kraken implements.

    kraken's ``parse_page_custom`` splits on ``}`` and then on ``{``, so a
    brace-less string raises ``ValueError`` and takes the whole file down with it.
    That is not something it recovers from, which is why this is a problem and not
    a note.
    """
    custom = element.get("custom")
    if custom is None:
        return {}
    values: dict[str, str] = {}
    for chunk in (part for part in custom.split("}") if part.strip()):
        if "{" not in chunk:
            result.problems.append(f"{label}: custom {custom!r} is not `tag {{key:value;}}`")
            return {}
        tag, _, body = chunk.partition("{")
        if tag.strip() in _KRAKEN_TAGS:
            result.problems.append(f"{label}: custom claims the kraken tag {tag.strip()!r}")
        for pair in (item.strip() for item in body.split(";") if item.strip()):
            key, _, value = pair.partition(":")
            values[key.strip()] = value.strip()
    return values


def _check_integer_points(element: Element | None, label: str, result: PageXmlValidation) -> None:
    """PAGE's ``PointsType`` is integer pairs, and kraken calls ``int()`` on each one.

    A float coordinate is not a rounding difference to a reader, it is a
    ``ValueError`` that fails the file.
    """
    if element is None:
        return
    for pair in (element.get("points") or "").split():
        for component in pair.split(","):
            try:
                int(component)
            except ValueError:
                result.problems.append(f"{label}: non-integer coordinate {component!r}")
                return


def _check_text_line(text_line: Element, result: PageXmlValidation) -> None:
    line_id = text_line.get("id") or "<unnamed>"
    if _check_custom(text_line, line_id, result).get("baseline") == "derived":
        result.derived_baseline_count += 1

    _check_integer_points(text_line.find("p:Coords", _NS), line_id, result)
    _check_integer_points(text_line.find("p:Baseline", _NS), line_id, result)

    coords = _points(text_line.find("p:Coords", _NS))
    if len(coords) < 3:
        result.problems.append(f"{line_id}: Coords has {len(coords)} points, needs at least 3")

    baseline_element = text_line.find("p:Baseline", _NS)
    if baseline_element is None:
        result.problems.append(f"{line_id}: no Baseline")
        return
    baseline = _points(baseline_element)
    if len(baseline) < 2:
        result.problems.append(f"{line_id}: Baseline has {len(baseline)} points, needs at least 2")
        return
    if not _strictly_increasing_x(baseline):
        result.problems.append(f"{line_id}: Baseline x is not strictly increasing left to right")
    if coords and not _within_bounds(baseline, coords):
        result.problems.append(f"{line_id}: Baseline leaves the Coords bounding box")

    unicode_element = text_line.find("p:TextEquiv/p:Unicode", _NS)
    if unicode_element is None:
        result.problems.append(f"{line_id}: no TextEquiv/Unicode")
    elif not (unicode_element.text or "").strip():
        result.lines_without_text += 1


def _points(element: Element | None) -> list[tuple[float, float]]:
    if element is None:
        return []
    points: list[tuple[float, float]] = []
    for pair in (element.get("points") or "").split():
        x, _, y = pair.partition(",")
        try:
            points.append((float(x), float(y)))
        except ValueError:
            continue
    return points


def _strictly_increasing_x(points: list[tuple[float, float]]) -> bool:
    return all(later[0] > earlier[0] for earlier, later in zip(points, points[1:], strict=False))


def _within_bounds(baseline: list[tuple[float, float]], coords: list[tuple[float, float]]) -> bool:
    """The baseline inside the polygon's bounding box, to within half a pixel.

    The box rather than the polygon itself: PAGE coordinates are routinely rounded
    to integers on the way out, and a point-in-polygon test at that precision fails
    on baselines that are correct. The box catches the failure that actually
    happens, a baseline belonging to a different line or a different page.
    """
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return all(
        min(xs) - 0.5 <= x <= max(xs) + 0.5 and min(ys) - 0.5 <= y <= max(ys) + 0.5
        for x, y in baseline
    )

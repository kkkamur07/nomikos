"""The schema-level checks a PAGE export has to pass before anybody trains on it."""

from __future__ import annotations

from backend.annotation.application.page_xml_validation import (
    PAGE_XML_NAMESPACE,
    validate_page_xml,
)

_REGION_COORDS = '<Coords points="0,0 600,0 600,800 0,800"/>'

_GOOD_LINE = """
    <TextLine id="line_1" custom="nomikos {source:kraken; kind:polygon;}">
      <Coords points="10,100 110,100 110,120 10,120"/>
      <Baseline points="12,114 108,115"/>
      <TextEquiv><Unicode>ἐν ἀρχῇ</Unicode></TextEquiv>
    </TextLine>
"""


def _page(lines: str = _GOOD_LINE, *, reading_order: str = "", regions: str = "") -> str:
    body = regions or f'<TextRegion id="r1">{_REGION_COORDS}{lines}</TextRegion>'
    return (
        f'<?xml version="1.0" encoding="utf-8"?>'
        f'<PcGts xmlns="{PAGE_XML_NAMESPACE}">'
        f'<Page imageFilename="page.webp" imageWidth="600" imageHeight="800">'
        f"{reading_order}{body}"
        f"</Page></PcGts>"
    )


def test_a_well_formed_page_passes_and_is_counted() -> None:
    result = validate_page_xml(_page())

    assert result.ok
    assert result.line_count == 1
    assert result.region_count == 1
    assert result.derived_baseline_count == 0


def test_a_derived_baseline_is_counted_from_the_custom_string() -> None:
    line = _GOOD_LINE.replace("kind:polygon;", "kind:polygon; baseline:derived;")

    assert validate_page_xml(_page(line)).derived_baseline_count == 1


def test_broken_xml_is_a_problem_not_an_exception() -> None:
    result = validate_page_xml("<PcGts>")

    assert not result.ok
    assert "not well-formed" in result.problems[0]


def test_the_wrong_root_or_namespace_is_rejected() -> None:
    for xml in ("<Page/>", '<PcGts xmlns="http://example.invalid/page"/>'):
        result = validate_page_xml(xml)
        assert not result.ok
        assert "expected PcGts" in result.problems[0]


def test_a_missing_page_element_is_reported() -> None:
    result = validate_page_xml(f'<PcGts xmlns="{PAGE_XML_NAMESPACE}"/>')

    assert result.problems == ["no Page element"]


def test_a_page_without_an_image_filename_is_reported() -> None:
    result = validate_page_xml(_page().replace(' imageFilename="page.webp"', ""))

    assert "Page has no imageFilename" in result.problems


def test_a_two_point_coords_is_not_a_polygon() -> None:
    line = _GOOD_LINE.replace('points="10,100 110,100 110,120 10,120"', 'points="10,100 110,120"')

    assert any(
        "Coords has 2 points" in problem for problem in validate_page_xml(_page(line)).problems
    )


def test_a_missing_baseline_is_reported() -> None:
    line = _GOOD_LINE.replace('<Baseline points="12,114 108,115"/>', "")

    assert "line_1: no Baseline" in validate_page_xml(_page(line)).problems


def test_a_one_point_baseline_is_reported() -> None:
    line = _GOOD_LINE.replace('points="12,114 108,115"', 'points="12,114"')

    assert any(
        "Baseline has 1 points" in problem for problem in validate_page_xml(_page(line)).problems
    )


def test_a_baseline_that_doubles_back_is_reported() -> None:
    line = _GOOD_LINE.replace('points="12,114 108,115"', 'points="108,115 12,114"')

    assert any(
        "not strictly increasing" in problem for problem in validate_page_xml(_page(line)).problems
    )


def test_a_baseline_outside_the_polygon_box_is_reported() -> None:
    line = _GOOD_LINE.replace('points="12,114 108,115"', 'points="12,900 108,901"')

    assert any(
        "leaves the Coords bounding box" in problem
        for problem in validate_page_xml(_page(line)).problems
    )


def test_a_line_without_text_is_counted_not_faulted() -> None:
    """A segmented page can be only half transcribed, and that file is still valid."""
    line = _GOOD_LINE.replace("<Unicode>ἐν ἀρχῇ</Unicode>", "<Unicode></Unicode>")

    result = validate_page_xml(_page(line))

    assert result.ok
    assert result.lines_without_text == 1


def test_a_line_without_a_text_equiv_is_reported() -> None:
    line = _GOOD_LINE.replace("<TextEquiv><Unicode>ἐν ἀρχῇ</Unicode></TextEquiv>", "")

    assert "line_1: no TextEquiv/Unicode" in validate_page_xml(_page(line)).problems


def test_a_reading_order_pointing_at_a_missing_region_is_reported() -> None:
    reading_order = (
        '<ReadingOrder><OrderedGroup id="ro">'
        '<RegionRefIndexed index="0" regionRef="r1"/>'
        '<RegionRefIndexed index="1" regionRef="ghost"/>'
        "</OrderedGroup></ReadingOrder>"
    )

    result = validate_page_xml(_page(reading_order=reading_order))

    assert any("unknown region 'ghost'" in problem for problem in result.problems)


def test_a_reading_order_that_forgets_a_region_is_reported() -> None:
    regions = (
        f'<TextRegion id="r1">{_REGION_COORDS}{_GOOD_LINE}</TextRegion>'
        f'<TextRegion id="r2">{_REGION_COORDS}</TextRegion>'
    )
    reading_order = (
        '<ReadingOrder><OrderedGroup id="ro">'
        '<RegionRefIndexed index="0" regionRef="r1"/>'
        "</OrderedGroup></ReadingOrder>"
    )

    result = validate_page_xml(_page(reading_order=reading_order, regions=regions))

    assert any("omits regions ['r2']" in problem for problem in result.problems)


def test_every_bad_line_is_reported_not_only_the_first() -> None:
    bad = _GOOD_LINE.replace('id="line_1"', 'id="line_2"').replace(
        '<Baseline points="12,114 108,115"/>', ""
    )

    result = validate_page_xml(_page(_GOOD_LINE + bad + bad.replace("line_2", "line_3")))

    assert result.line_count == 3
    assert result.problems == ["line_2: no Baseline", "line_3: no Baseline"]


def test_a_custom_string_without_braces_is_reported() -> None:
    """The grammar kraken 7.1 implements, checked here so nobody needs kraken to find it.

    ``parse_page_custom`` splits on ``{`` and unpacks two values, so the old
    ``source:kraken; kind:polygon`` raised ValueError and failed the whole file.
    """
    line = _GOOD_LINE.replace(
        'custom="nomikos {source:kraken; kind:polygon;}"',
        'custom="source:kraken; kind:polygon"',
    )

    result = validate_page_xml(_page(line))

    assert any("is not `tag {key:value;}`" in problem for problem in result.problems)


def test_a_region_custom_without_braces_is_reported() -> None:
    regions = (
        f'<TextRegion id="r1" custom="type:paragraph">{_REGION_COORDS}{_GOOD_LINE}</TextRegion>'
    )

    result = validate_page_xml(_page(regions=regions))

    assert any("r1: custom" in problem for problem in result.problems)


def test_claiming_a_tag_kraken_acts_on_is_reported() -> None:
    """``structure`` and ``readingOrder`` change what kraken thinks a line is."""
    for tag in ("structure {type:default;}", "readingOrder {index:0;}"):
        line = _GOOD_LINE.replace(
            'custom="nomikos {source:kraken; kind:polygon;}"', f'custom="{tag}"'
        )

        result = validate_page_xml(_page(line))

        assert any("custom claims the kraken tag" in problem for problem in result.problems)


def test_a_fractional_coordinate_in_coords_is_reported() -> None:
    """PAGE's PointsType is integer pairs, and kraken calls ``int()`` on each one."""
    line = _GOOD_LINE.replace(
        'points="10,100 110,100 110,120 10,120"', 'points="10,100.5 110,100 110,120 10,120"'
    )

    result = validate_page_xml(_page(line))

    assert any("non-integer coordinate '100.5'" in problem for problem in result.problems)


def test_a_fractional_coordinate_in_a_baseline_is_reported() -> None:
    line = _GOOD_LINE.replace('points="12,114 108,115"', 'points="12,113.6 108,113.6"')

    result = validate_page_xml(_page(line))

    assert any("non-integer coordinate '113.6'" in problem for problem in result.problems)


def test_a_fractional_coordinate_in_a_region_boundary_is_reported() -> None:
    regions = f'<TextRegion id="r1"><Coords points="0,0 600,0 600,799.5 0,799.5"/>{_GOOD_LINE}</TextRegion>'

    result = validate_page_xml(_page(regions=regions))

    assert any("r1: non-integer coordinate '799.5'" in problem for problem in result.problems)


def test_a_region_without_coords_is_reported() -> None:
    """kraken drops a boundary-less region, and every line inside it goes with it."""
    regions = f'<TextRegion id="r1">{_GOOD_LINE}</TextRegion>'

    result = validate_page_xml(_page(regions=regions))

    assert "r1: TextRegion has no Coords" in result.problems

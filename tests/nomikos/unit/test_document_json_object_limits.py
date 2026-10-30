"""Free-form JSON objects on line and block writes are bounded.

Round-2 QA evidence: ``LineCreateRequest`` accepted a 50,000-key, 5.4 MB ``mask`` and
``baseline``. Every request schema that writes such an object now refuses one above
``MAX_JSON_OBJECT_BYTES`` or ``MAX_JSON_OBJECT_KEYS`` with a 422, while the largest
real geometry (``MAX_LINE_GEOMETRY_POINTS`` points) still fits with room to spare.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.document.api.schemas import (
    MAX_JSON_OBJECT_BYTES,
    MAX_JSON_OBJECT_KEYS,
    MAX_LINE_GEOMETRY_POINTS,
    BlockCreateRequest,
    BlockPatchRequest,
    LineCreateRequest,
    LinePatchRequest,
    LinesReplaceRequest,
)

_SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
_BIG = {f"k{i}": "x" * 100 for i in range(50_000)}
_WIDE = {f"k{i}": i for i in range(MAX_JSON_OBJECT_KEYS + 1)}
_HEAVY = {"points": "x" * MAX_JSON_OBJECT_BYTES}
_LARGEST_REAL = {"points": [[123456.789, 987654.321] for _ in range(MAX_LINE_GEOMETRY_POINTS)]}


def _line(**fields: object) -> dict:
    return {"order": 0, "points": _SQUARE, **fields}


@pytest.mark.parametrize("value", [_BIG, _WIDE, _HEAVY])
@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda **kw: LineCreateRequest.model_validate(_line(**kw)), "mask"),
        (lambda **kw: LineCreateRequest.model_validate(_line(**kw)), "baseline"),
        (lambda **kw: LinePatchRequest.model_validate(kw), "mask"),
        (lambda **kw: LinePatchRequest.model_validate(kw), "baseline"),
        (lambda **kw: LinesReplaceRequest.model_validate({"lines": [_line(**kw)]}), "mask"),
        (lambda **kw: LinesReplaceRequest.model_validate({"lines": [_line(**kw)]}), "baseline"),
        (
            lambda **kw: LinesReplaceRequest.model_validate({"lines": [_line(**kw)]}),
            "source_metadata",
        ),
        (lambda **kw: BlockCreateRequest.model_validate({"order": 0, **kw}), "box"),
        (lambda **kw: BlockPatchRequest.model_validate(kw), "box"),
    ],
)
def test_oversized_json_objects_are_refused(build, field: str, value: dict) -> None:
    with pytest.raises(ValidationError):
        build(**{field: value})


def test_the_largest_real_geometry_is_accepted_everywhere() -> None:
    assert (
        LineCreateRequest.model_validate(_line(mask=_LARGEST_REAL, baseline=_LARGEST_REAL)).mask
        == _LARGEST_REAL
    )
    assert LinePatchRequest.model_validate({"mask": _LARGEST_REAL}).mask == _LARGEST_REAL
    replace = LinesReplaceRequest.model_validate(
        {"lines": [_line(baseline=_LARGEST_REAL, source_metadata={"engine": "kraken"})]}
    )
    assert replace.lines[0].baseline == _LARGEST_REAL
    assert BlockCreateRequest.model_validate({"order": 0, "box": _LARGEST_REAL}).box


def test_nullable_objects_still_accept_null() -> None:
    assert LinePatchRequest.model_validate({"mask": None}).mask is None
    assert LineCreateRequest.model_validate(_line(mask=None)).mask is None


def test_not_null_patch_fields_still_reject_null() -> None:
    with pytest.raises(ValidationError, match="must not be null"):
        LinePatchRequest.model_validate({"baseline": None})
    with pytest.raises(ValidationError, match="must not be null"):
        BlockPatchRequest.model_validate({"box": None})

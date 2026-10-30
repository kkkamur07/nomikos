"""Segment task request/response contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from nomikos_inference.contracts.common import (
    MAX_GEOMETRY_POINTS,
    MAX_KRAKEN_CEILING_POINTS,
    MAX_SEGMENT_BLOCKS,
    MAX_SEGMENT_LINES,
)


class SegmentGeometryKind(StrEnum):
    polygon = "polygon"
    rectangle = "rectangle"


def _validate_point_pairs(value: list[list[float]]) -> list[list[float]]:
    if any(len(point) != 2 for point in value):
        raise ValueError("each point must contain x and y")
    return value


class SegmentBlock(BaseModel):
    external_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    box: dict[str, object]


class SegmentLine(BaseModel):
    external_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    block_external_id: str | None = None
    baseline: dict[str, object]
    mask: dict[str, object] | None = None
    kind: SegmentGeometryKind = SegmentGeometryKind.polygon
    points: list[list[float]] = Field(min_length=4, max_length=MAX_GEOMETRY_POINTS)
    kraken_ceiling: list[list[float]] | None = Field(
        default=None, max_length=MAX_KRAKEN_CEILING_POINTS
    )
    source_metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: list[list[float]]) -> list[list[float]]:
        return _validate_point_pairs(value)

    @field_validator("kraken_ceiling")
    @classmethod
    def validate_kraken_ceiling(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        return _validate_point_pairs(value)


class SegmentRunResponse(BaseModel):
    blocks: list[SegmentBlock] = Field(default_factory=list, max_length=MAX_SEGMENT_BLOCKS)
    lines: list[SegmentLine] = Field(default_factory=list, max_length=MAX_SEGMENT_LINES)


__all__ = [
    "SegmentBlock",
    "SegmentGeometryKind",
    "SegmentLine",
    "SegmentRunResponse",
]

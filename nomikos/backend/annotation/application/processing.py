"""Platform-owned Processing pipeline for exported line images."""

from PIL import Image, ImageDraw

SUPPORTED_STEPS = {"rectify"}


def _clamp(value: float, low: int, high: int) -> float:
    return min(max(value, low), high)


def _bbox(points: list[tuple[float, float]], width: int, height: int) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0 = int(_clamp(min(xs), 0, width - 1))
    y0 = int(_clamp(min(ys), 0, height - 1))
    x1 = int(_clamp(max(xs), x0 + 1, width))
    y1 = int(_clamp(max(ys), y0 + 1, height))
    return x0, y0, x1, y1


def _mask_bbox_rectify(page_image: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
    x0, y0, x1, y1 = _bbox(points, page_image.width, page_image.height)
    crop = page_image.crop((x0, y0, x1, y1))

    shifted = [(x - x0, y - y0) for x, y in points]
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon(shifted, fill=255)

    white = Image.new(crop.mode, crop.size, "white")
    white.paste(crop, mask=mask)
    return white


def rectify(page_image: Image.Image, segment: dict) -> Image.Image:
    raw_points = segment.get("points") or []
    points = [(float(x), float(y)) for x, y in raw_points]
    if not points:
        raise ValueError("Segment has no points")
    if len(points) < 3:
        return page_image.crop(_bbox(points, page_image.width, page_image.height))

    return _mask_bbox_rectify(page_image, points)


def apply_step(image: Image.Image, segment: dict, step: str) -> Image.Image:
    if step not in SUPPORTED_STEPS:
        raise ValueError(f"Unsupported processing step: {step}")
    return rectify(image, segment)

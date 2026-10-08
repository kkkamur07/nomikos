"""Random, print-aware line-image augmentation for the Calamari recognizer.

Pool operations are tagged light / mild / heavy. Each training copy uses one
strength from the configurable 5-copy plan (default two easy, two mild, one
hard); a ``none`` copy is returned untouched.

light: small geometric shift or global tone change (pad, translate,
    perspective, shear, tone, sharpen); script polarity stays the same.
mild: visible distortion, dirt, or stroke thickening (rotate, elastic, jpeg /
    pixelate, posterize, vignette, noise); the line is still gray and readable.
heavy: domain change, heavy print noise, blur, ruling, or stroke loss (printlike,
    otsu, sauvola, invert, thin, holes, box blur, motion blur, equalize, ruled lines).
"""

from __future__ import annotations

import random
from collections.abc import Callable

import cv2
import numpy
from torch import Tensor


SYRIAC_CONDENSED_PX_PER_CHAR = 13.0
SYRIAC_CONDENSED_WIDTH_SCALE_RANGE = (1.0, 2.0)
SYRIAC_SPACED_WIDTH_SCALE_RANGE = (0.5, 1.5)


def jitter_syriac_line_width(image: Tensor, text: str = "") -> Tensor:
    """Stretch or compress Syriac line width; keep height and all ink.

    Already-dense lines (2024 / chapter 4, under 13 px/char) are only
    stretched. Spaced 2025 lines may be compressed toward that density.
    Variant 0 must skip this; the returned tensor may change width.
    """
    width = image.shape[0]
    n_chars = max(len(text), 1)
    px_per_char = width / n_chars
    if px_per_char < SYRIAC_CONDENSED_PX_PER_CHAR:
        low, high = SYRIAC_CONDENSED_WIDTH_SCALE_RANGE
    else:
        low, high = SYRIAC_SPACED_WIDTH_SCALE_RANGE
    return scale_line_width(image, random.uniform(low, high))


def scale_line_width(image: Tensor, scale: float) -> Tensor:
    """Resize only the CTC time axis. Height stays the Calamari line height."""
    if image.ndim != 3 or image.shape[-1] != 1:
        raise ValueError("Calamari width jitter requires a (width, height, 1) image tensor.")
    if scale <= 0.0:
        raise ValueError(f"Width scale must be positive, got {scale}.")

    original_dtype = image.dtype
    pixels = image.squeeze(-1).detach().cpu().numpy().astype(numpy.float32, copy=True)
    width, height = pixels.shape
    new_width = max(1, round(width * scale))
    if new_width == width:
        return image.clone()
    resized = cv2.resize(pixels, (height, new_width), interpolation=cv2.INTER_LINEAR)
    return Tensor(resized).to(dtype=original_dtype).unsqueeze(-1)


def augment_grayscale_line(
    image: numpy.ndarray,
    strength: str,
    operation_count: int,
    *,
    ink_bright: bool = False,
) -> numpy.ndarray:
    """Apply the easy, mild, or hard pool to a ``(height, width)`` grayscale line.

    TrOCR and other image-layout callers use this. The ops themselves are still
    the width-major Calamari stack; the axis swap stays inside this function.
    Input is paper-bright by default; pass ``ink_bright=True`` for ink = 255.
    """
    if image.ndim != 2:
        raise ValueError("Line augmentation requires a 2D grayscale array.")
    if _is_identity(strength, operation_count):
        return image.copy()
    original_dtype = image.dtype
    pixels = image.astype(numpy.float32, copy=False)
    augmented = _augment_width_major(
        pixels.T, strength, operation_count, ink_bright=ink_bright
    ).T
    return augmented.astype(original_dtype, copy=False)


def augment_legacy_line_image(image: Tensor, strength: str, operation_count: int) -> Tensor:
    """Apply the easy, mild, or hard pool to an ink-bright Calamari ``(width, height, 1)`` tensor."""
    if image.ndim != 3 or image.shape[-1] != 1:
        raise ValueError("Calamari augmentation requires a (width, height, 1) image tensor.")
    if _is_identity(strength, operation_count):
        return image.clone()

    original_dtype = image.dtype
    pixels = image.squeeze(-1).detach().cpu().numpy()
    augmented = _augment_width_major(
        pixels.astype(numpy.float32, copy=False),
        strength,
        operation_count,
        ink_bright=True,
    )
    return Tensor(augmented.astype(pixels.dtype, copy=False)).to(dtype=original_dtype).unsqueeze(-1)


def _augment_width_major(
    pixels: numpy.ndarray,
    strength: str,
    operation_count: int,
    *,
    ink_bright: bool = False,
) -> numpy.ndarray:
    """Run the legacy ink-bright ops on a ``(width, height)`` array.

    Paper-bright input is inverted into ink space and back; ink-bright input
    (``ink_bright=True``) is used as is and returned ink-bright.
    """
    scale = 255.0 if float(pixels.max(initial=0.0)) > 1.0 else 1.0
    normalized = numpy.clip(pixels / scale, 0.0, 1.0)
    ink = normalized if ink_bright else 1.0 - normalized
    augmented = _apply_strength(ink, strength, operation_count)
    if ink_bright:
        return numpy.clip(augmented * scale, 0.0, scale)
    return numpy.clip((1.0 - augmented) * scale, 0.0, scale)


def _apply_strength(image: numpy.ndarray, strength: str, operation_count: int) -> numpy.ndarray:
    operations = _operations_for_strength(strength)
    count = operation_count
    if count > len(operations):
        raise ValueError(f"Calamari {strength} pool has fewer than {count} operations.")
    augmented = image
    for operation in random.sample(operations, count):
        augmented = numpy.clip(operation(augmented), 0.0, 1.0)
    return augmented


def _is_identity(strength: str, operation_count: int) -> bool:
    """True for a ``none`` copy or zero ops: the line is returned untouched."""
    _operations_for_strength(strength)  # reject unknown strengths early
    return strength == "none" or operation_count == 0


def _operations_for_strength(strength: str) -> tuple[Callable[[numpy.ndarray], numpy.ndarray], ...]:
    if strength == "none":
        return ()
    if strength == "easy":
        return (
            _line_pad,
            _translate,
            _perspective,
            _tone,
            _sharpen,
            _shear,
        )
    if strength == "mild":
        return (
            _smooth_elastic_distortion,
            _random_rotate,
            _camera_degradation,
            _posterize,
            _stroke_thicken,
            _anisotropic_scale,
            _bleed_through,
            _illumination_gradient,
            _vignette,
            _ink_fade,
            _gaussian_noise,
            _salt_pepper,
        )
    if strength == "hard":
        return (
            _printlike_degradation,
            _binarize_otsu,
            _binarize_sauvola,
            _invert,
            _stroke_thin,
            _stroke_holes,
            _box_blur,
            _motion_blur,
            _equalize_histogram,
            _ruled_lines,
        )
    raise ValueError(f"Unknown Calamari augmentation strength {strength!r}.")


def _line_pad(image: numpy.ndarray) -> numpy.ndarray:
    """light: pad along the line with empty background."""
    return _random_pad(image, (0, max(2, image.shape[1] * 2)))


def _random_pad(image: numpy.ndarray, horizontal: tuple[int, int]) -> numpy.ndarray:
    """light: pad along the line with empty background."""
    left, right = numpy.random.randint(*horizontal, size=2)
    return cv2.copyMakeBorder(
        image,
        int(left),
        int(right),
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def _random_rotate(image: numpy.ndarray, maximum_degrees: float = 5.0) -> numpy.ndarray:
    """mild: ±5° rotation on a width-major line."""
    width, height = image.shape[:2]
    angle = float(numpy.random.uniform(-maximum_degrees, maximum_degrees))
    transform = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        transform,
        (height, width),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _smooth_elastic_distortion(image: numpy.ndarray) -> numpy.ndarray:
    """mild: ±3 px smooth warp."""
    sigma = random.uniform(2.0, 5.0)
    return _distort_with_noise(image, _bounded_gaussian_noise(image.shape, sigma, 3.0))


def _printlike_degradation(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: paper texture, blotches, and uneven ink (ocrodeg-style)."""
    return _printlike_multiscale(image, blur=1.0, inverted=True)


def _camera_degradation(image: numpy.ndarray) -> numpy.ndarray:
    """mild wrapper: jpeg compression or pixelate."""
    return random.choice((_jpeg_compression, _pixelate))(image)


def _tone(image: numpy.ndarray) -> numpy.ndarray:
    """light: one linear tone map, gain 0.7–1.3 about the mean plus offset ±0.1"""
    gain = random.uniform(0.7, 1.3)
    offset = random.uniform(-0.1, 0.1)
    return (image - image.mean()) * gain + image.mean() + offset


def _jpeg_compression(image: numpy.ndarray) -> numpy.ndarray:
    """mild: JPEG quality fixed at 15."""
    encoded, buffer = cv2.imencode(
        ".jpg",
        numpy.rint(image * 255.0).astype(numpy.uint8),
        [cv2.IMWRITE_JPEG_QUALITY, 15],
    )
    if not encoded:
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return decoded.astype(numpy.float32) / 255.0


def _pixelate(image: numpy.ndarray) -> numpy.ndarray:
    """mild: downsample to 45–75% then nearest-neighbor upscale."""
    width, height = image.shape
    scale = random.uniform(0.45, 0.75)
    reduced = cv2.resize(
        image,
        (max(1, round(height * scale)), max(1, round(width * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(reduced, (height, width), interpolation=cv2.INTER_NEAREST)


def _box_blur(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: 3×3 or 5×5 box blur."""
    return cv2.blur(image, (random.choice((3, 5)),) * 2)


def _motion_blur(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: 5–9 px horizontal motion blur."""
    kernel_size = random.choice((5, 7, 9))
    kernel = numpy.zeros((kernel_size, kernel_size), dtype=numpy.float32)
    cv2.line(
        kernel,
        (0, kernel_size // 2),
        (kernel_size - 1, kernel_size // 2),
        1.0,
        1,
    )
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def _equalize_histogram(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: histogram equalization."""
    equalized = cv2.equalizeHist(numpy.rint(image * 255.0).astype(numpy.uint8))
    return equalized.astype(numpy.float32) / 255.0


def _sharpen(image: numpy.ndarray) -> numpy.ndarray:
    """light: unsharp mask."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, borderType=cv2.BORDER_REFLECT)
    return image + random.uniform(0.2, 0.8) * (image - blurred)


def _posterize(image: numpy.ndarray) -> numpy.ndarray:
    """mild: quantize to 16 gray levels."""
    levels = 16
    return numpy.floor(image * (levels - 1)) / (levels - 1)


def _stroke_thicken(image: numpy.ndarray) -> numpy.ndarray:
    """mild: thicken ink-bright strokes (3x3 dilate)."""
    return cv2.dilate(
        image,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )


def _stroke_thin(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: thin ink-bright strokes (3x3 erode)."""
    return cv2.erode(
        image,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )


def _stroke_holes(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: punch 2–4 px holes through strokes, removing 5–10% of the ink."""
    ink = image > 0.5
    ink_total = int(ink.sum())
    if ink_total == 0:
        return image
    ink_coords = numpy.argwhere(ink)
    radius = random.randint(2, 4)
    target = random.uniform(0.05, 0.10)
    mask = numpy.zeros(image.shape[:2], dtype=numpy.uint8)
    for _ in range(2000):
        x, y = ink_coords[random.randrange(len(ink_coords))]
        cv2.circle(mask, (int(y), int(x)), radius, 1, thickness=-1)
        if numpy.count_nonzero(mask[ink]) >= target * ink_total:
            break
    transformed = image.copy()
    transformed[mask.astype(bool)] = 0.0
    return transformed


def _warp_width_major(
    image: numpy.ndarray,
    matrix: numpy.ndarray,
    *,
    perspective: bool = False,
) -> numpy.ndarray:
    width, height = image.shape[:2]
    size = (height, width)
    if perspective:
        return cv2.warpPerspective(
            image,
            matrix,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return cv2.warpAffine(
        image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _shear(image: numpy.ndarray) -> numpy.ndarray:
    """light: horizontal slant, factor ±0.28."""
    width, height = image.shape[:2]
    factor = float(numpy.random.uniform(-0.28, 0.28))
    matrix = numpy.array(
        [[1.0, 0.0, 0.0], [-factor, 1.0, factor * height / 2]], dtype=numpy.float32
    )
    return _warp_width_major(image, matrix)


def _anisotropic_scale(image: numpy.ndarray) -> numpy.ndarray:
    """mild: independent width/height scale, then crop or pad."""
    width, height = image.shape[:2]
    scaled_width = max(8, int(width * random.uniform(0.82, 1.22)))
    scaled_height = max(8, int(height * random.uniform(0.82, 1.18)))
    scaled = cv2.resize(image, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)
    canvas = numpy.zeros((width, height), dtype=image.dtype)
    paste_w, paste_h = min(width, scaled.shape[0]), min(height, scaled.shape[1])
    dst_x = (width - paste_w) // 2
    dst_y = (height - paste_h) // 2
    src_x = (scaled.shape[0] - paste_w) // 2
    src_y = (scaled.shape[1] - paste_h) // 2
    canvas[dst_x : dst_x + paste_w, dst_y : dst_y + paste_h] = scaled[
        src_x : src_x + paste_w, src_y : src_y + paste_h
    ]
    return canvas


def _translate(image: numpy.ndarray) -> numpy.ndarray:
    """light: shift along the line and its height."""
    width, height = image.shape[:2]
    matrix = numpy.array(
        [
            [1.0, 0.0, random.uniform(-0.16, 0.16) * height],
            [0.0, 1.0, random.uniform(-0.07, 0.07) * width],
        ],
        dtype=numpy.float32,
    )
    return _warp_width_major(image, matrix)


def _perspective(image: numpy.ndarray) -> numpy.ndarray:
    """light: trapezoid warp; shrinks inward onto empty fill."""
    width, height = image.shape[:2]
    jitter_x = 0.14 * height
    jitter_y = 0.08 * width
    source = numpy.float32([[0, 0], [height - 1, 0], [height - 1, width - 1], [0, width - 1]])
    dest = numpy.float32(
        [
            [random.uniform(0, jitter_x), random.uniform(0, jitter_y)],
            [height - 1 - random.uniform(0, jitter_x), random.uniform(0, jitter_y)],
            [height - 1 - random.uniform(0, jitter_x), width - 1 - random.uniform(0, jitter_y)],
            [random.uniform(0, jitter_x), width - 1 - random.uniform(0, jitter_y)],
        ]
    )
    return _warp_width_major(image, cv2.getPerspectiveTransform(source, dest), perspective=True)


def _bleed_through(image: numpy.ndarray) -> numpy.ndarray:
    """mild: faded, flipped copy of the same line as verso ghosting."""
    ghost = numpy.flip(image, axis=0)
    ghost = numpy.roll(ghost, int(random.uniform(-0.05, 0.05) * image.shape[0]), axis=0)
    return numpy.maximum(image, ghost * random.uniform(0.14, 0.3))


def _paper_field(image: numpy.ndarray, field: numpy.ndarray) -> numpy.ndarray:
    paper = 1.0 - image
    return 1.0 - numpy.clip(paper * field, 0.0, 1.0)


def _illumination_gradient(image: numpy.ndarray) -> numpy.ndarray:
    """mild: linear lighting ramp about 0.65–1.3."""
    width, height = image.shape[:2]
    start, end = random.uniform(0.65, 0.85), random.uniform(1.05, 1.3)
    if random.choice((True, False)):
        ramp = numpy.linspace(start, end, width, dtype=numpy.float32)[:, None]
        field = numpy.repeat(ramp, height, axis=1)
    else:
        ramp = numpy.linspace(start, end, height, dtype=numpy.float32)[None, :]
        field = numpy.repeat(ramp, width, axis=0)
    return _paper_field(image, field)


def _vignette(image: numpy.ndarray) -> numpy.ndarray:
    """mild: darken the borders."""
    width, height = image.shape[:2]
    ys, xs = numpy.ogrid[:width, :height]
    radius = numpy.sqrt(
        ((ys - width / 2) / (width / 2 + 1e-6)) ** 2
        + ((xs - height / 2) / (height / 2 + 1e-6)) ** 2
    )
    field = 1.0 - random.uniform(0.25, 0.5) * numpy.clip(radius - 0.15, 0.0, 1.0) ** 2
    return _paper_field(image, field)


def _ruled_lines(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: one or two ruling strokes through the line."""
    result = image.copy()
    width, height = image.shape[:2]
    for _ in range(random.randint(1, 2)):
        y = int(random.uniform(0.2, 0.8) * height)
        shade = random.uniform(0.35, 0.75)
        result[:, max(0, y - 1) : min(height, y + 1)] = numpy.maximum(
            result[:, max(0, y - 1) : min(height, y + 1)],
            shade,
        )
        if random.random() < 0.4:
            wobble = numpy.sin(
                numpy.linspace(0, random.uniform(6.0, 12.0), width)
            ) * random.uniform(0.4, 1.1)
            for x, delta in enumerate(wobble):
                yy = int(numpy.clip(y + delta, 0, height - 1))
                result[x, yy] = max(result[x, yy], shade)
    return result


def _ink_fade(image: numpy.ndarray) -> numpy.ndarray:
    """mild: weaken ink-bright strokes, fade 0.25–0.55."""
    fade = random.uniform(0.25, 0.55)
    ink = (image > 0.2).astype(numpy.float32)
    return image * (1.0 - ink * fade)


def _binarize_otsu(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: global Otsu."""
    pixels = numpy.rint(image * 255.0).astype(numpy.uint8)
    _threshold, binary = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(numpy.float32) / 255.0


def _binarize_sauvola(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: local mean/std threshold."""
    pixels = image.astype(numpy.float32)
    window = random.choice((15, 21, 25))
    mean = cv2.blur(pixels, (window, window))
    sqmean = cv2.blur(pixels * pixels, (window, window))
    std = numpy.sqrt(numpy.maximum(sqmean - mean * mean, 0.0))
    threshold = mean * (1.0 + random.uniform(0.15, 0.35) * ((std / 0.5) - 1.0))
    return (pixels > threshold).astype(numpy.float32)


def _invert(image: numpy.ndarray) -> numpy.ndarray:
    """heavy: swap polarity (white-on-black after the outer invert)."""
    return 1.0 - image


def _gaussian_noise(image: numpy.ndarray) -> numpy.ndarray:
    """mild: additive grain, sigma 0.03–0.08 in ink space."""
    return image + numpy.random.normal(0.0, random.uniform(0.03, 0.08), image.shape).astype(
        numpy.float32
    )


def _salt_pepper(image: numpy.ndarray) -> numpy.ndarray:
    """mild: 0.4–1.5% black/white specks."""
    result = image.copy()
    amount = random.uniform(0.004, 0.015)
    salt = numpy.random.random(image.shape) < amount / 2
    pepper = numpy.random.random(image.shape) < amount / 2
    result[salt] = 1.0
    result[pepper] = 0.0
    return result


def _bounded_gaussian_noise(shape: tuple[int, ...], sigma: float, maxdelta: float) -> numpy.ndarray:
    width, height = shape[:2]
    deltas = numpy.random.rand(2, width, height)
    for axis, values in enumerate(deltas):
        deltas[axis] = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    deltas -= deltas.min()
    deltas /= deltas.max()
    return (2 * deltas - 1) * maxdelta


def _distort_with_noise(image: numpy.ndarray, deltas: numpy.ndarray) -> numpy.ndarray:
    if deltas.shape != (2, *image.shape[:2]):
        raise ValueError("Calamari distortion offsets must match image dimensions.")
    width, height = image.shape[:2]
    coordinates = numpy.transpose(
        numpy.array(numpy.meshgrid(range(width), range(height))),
        axes=[0, 2, 1],
    )
    deltas = deltas + coordinates
    return cv2.remap(
        image,
        deltas[1].astype(numpy.float32),
        deltas[0].astype(numpy.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _random_blotches(image: numpy.ndarray, foreground: float, background: float) -> numpy.ndarray:
    fg = _random_blobs(image.shape[:2], foreground, 10)
    bg = _random_blobs(image.shape[:2], background, 10)
    return numpy.minimum(numpy.maximum(image, fg), 1 - bg)


def _random_blobs(shape: tuple[int, int], density: float, size: int) -> numpy.ndarray:
    width, height = shape
    count = max(1, int(density * width * height))
    mask = numpy.zeros((width, height), numpy.uint8)
    for _ in range(count):
        mask[random.randint(0, width - 1), random.randint(0, height - 1)] = 1
    distance = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    mask = numpy.array(distance < size, dtype=numpy.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=size / 4, borderType=cv2.BORDER_REFLECT)
    mask -= mask.min()
    maximum = float(mask.max())
    if maximum:
        mask /= maximum
    noise = cv2.GaussianBlur(
        numpy.random.rand(width, height),
        (0, 0),
        sigmaX=size / 4,
        borderType=cv2.BORDER_REFLECT,
    )
    noise -= noise.min()
    maximum = float(noise.max())
    if maximum:
        noise /= maximum
    return numpy.array(mask * noise > 0.5, dtype=numpy.float32)


def _make_multiscale_noise_uniform(
    shape: tuple[int, int], *, scale_range: tuple[float, float] = (1.0, 100.0)
) -> numpy.ndarray:
    minimum, maximum = numpy.log10(scale_range)
    scales = numpy.random.uniform(size=4)
    scales = numpy.add.accumulate(scales)
    scales -= scales.min()
    scales /= scales.max()
    scales = 10 ** (scales * (maximum - minimum) + minimum)
    weights = 2.0 * numpy.random.uniform(size=4)
    result = _make_noise_at_scale(shape, scales[0]) * weights[0]
    for scale, weight in zip(scales, weights, strict=True):
        result += _make_noise_at_scale(shape, scale) * weight
    result -= result.min()
    result /= result.max()
    return result


def _make_noise_at_scale(shape: tuple[int, int], scale: float) -> numpy.ndarray:
    width, height = shape
    source = numpy.random.rand(int(width / scale + 1), int(height / scale + 1))
    noise = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return noise[:width, :height]


def _printlike_multiscale(
    image: numpy.ndarray,
    *,
    blur: float,
    inverted: bool,
) -> numpy.ndarray:
    selector = image if inverted else 1 - image
    selector = _random_blotches(selector, 3 * 5e-5, 5e-5)
    paper = 0.8 + 0.2 * _make_multiscale_noise_uniform(image.shape[:2])
    ink = 0.2 * _make_multiscale_noise_uniform(image.shape[:2])
    blurred = (
        cv2.GaussianBlur(selector, (0, 0), sigmaX=blur, borderType=cv2.BORDER_REFLECT) + selector
    ) / 2
    printed = blurred * ink + (1 - blurred) * paper
    return 1 - printed if inverted else printed

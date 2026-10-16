"""Serving must build the exact picture the Calamari models were trained on.

Every Calamari model in ``nomikos_inference/registry.yaml`` was cut by
``src/preprocessing_data/syriac/xml_to_data.py::crop_polygon``, written to disk
by that file's ``save_crop``, and loaded during training by
``src/models/calamari/data.py::_load_line_image``. Serving has three chances to
disagree with that, and until 2026-09-07 it took all three: it cut a bare
bounding box rather than the masked polygon, it ran the legacy TensorFlow
Calamari processors (invert, centre-line dewarp, 16 px pad) instead of the
trainer's three-step resize, and it never had a padding to cut at. None of the
three shows up as an exception; all three show up as a model that reads a
manuscript worse than its own validation numbers say it can.

So these tests do not restate the recipe in a second implementation, which could
only ever drift alongside the first. They import the **real training code** and
assert the serving output is byte-identical to it. No weights and no ``ml``
marker: this has to run in the default job, because the default job is where a
preprocessing change lands.

One crop function serves all four Calamari models. What is per model is the
padding it was exported with: Greek used 0 and the other three used 12, so
the crop parity test runs at both and the second half of this file covers
the registry fields that carry the number.
"""

from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nomikos_inference.architectures.calamari.preprocessing import (
    TRAINING_CROP_PADDING,
    crop_line,
    crop_line_on_white,
    preprocess_line_array_to_calamari_tensor,
    preprocess_line_image_bytes_to_calamari_tensor,
)
from nomikos_inference.contracts.common import LineCrop
from nomikos_inference.registry import RegistryDocument
from tests.fixtures.paths import REPO_ROOT, TRANSCRIBE_LINE

_LINE_HEIGHT = 48

# The dataset exporter is a script, not an importable package member: it lives
# outside any package and runs its work under a ``__main__`` guard. Its
# top-level imports are cv2/numpy/PIL/stdlib and its module constants are inert,
# so loading it straight from its file costs nothing and gets the genuine
# functions rather than a copy of them. Two of them matter here: ``crop_polygon``
# cut every training line, and ``save_crop`` wrote it to disk.
_CROP_POLYGON_PATH = REPO_ROOT / "src/preprocessing_data/syriac/xml_to_data.py"


def _load_training_exporter():
    spec = importlib.util.spec_from_file_location("_syriac_xml_to_data", _CROP_POLYGON_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_exporter = _load_training_exporter()
crop_polygon = _exporter.crop_polygon
parse_points = _exporter.parse_points
save_crop = _exporter.save_crop
TRAINING_PADDING_CONSTANT = _exporter.PADDING


def _load_training_line_loader():
    """Import the trainer's dataset loader with the repo root on ``sys.path``."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.models.calamari.data import _load_line_image

    return _load_line_image


# ---------------------------------------------------------------------------
# The page and the polygons
# ---------------------------------------------------------------------------


def _synthetic_page(height: int = 300, width: int = 420) -> np.ndarray:
    """A parchment-ish RGB page with dark strokes, deterministic per seed."""
    rng = np.random.default_rng(20260907)
    page = np.full((height, width, 3), (219, 201, 166), dtype=np.int16)
    page += rng.integers(-14, 15, page.shape)
    for _ in range(120):  # iron-gall ink
        x = int(rng.integers(0, width - 10))
        y = int(rng.integers(0, height - 22))
        page[y : y + int(rng.integers(6, 20)), x : x + int(rng.integers(2, 9))] = (58, 40, 28)
    return page.clip(0, 255).astype(np.uint8)


# Float coordinates with fractional parts, the way a segmenter hands them over.
# The last three sit against a page edge so the padded box has to be clamped
# rather than run off the array, which is where an off-by-one in the bounds
# would show.
_POLYGONS: dict[str, list[list[float]]] = {
    "interior_irregular": [
        [61.4, 84.9],
        [190.7, 78.2],
        [263.1, 96.6],
        [258.8, 131.3],
        [124.5, 139.9],
        [58.2, 118.7],
    ],
    "concave_zigzag": [
        [40.9, 170.2],
        [120.3, 155.8],
        [130.6, 190.4],
        [210.1, 162.7],
        [300.5, 205.9],
        [180.2, 231.4],
        [45.7, 214.8],
    ],
    "against_top_left": [
        [0.4, 0.9],
        [70.6, 3.2],
        [96.8, 26.1],
        [2.7, 31.5],
    ],
    "against_bottom_right": [
        [330.5, 262.3],
        [419.9, 258.1],
        [419.2, 299.6],
        [327.8, 296.4],
    ],
    "thin_triangle_on_edge": [
        [408.7, 40.2],
        [419.4, 44.8],
        [405.1, 61.9],
    ],
}


# The two paddings in the shipped registry. 12 is the exporter's own PADDING and
# what the Armenian and Syriac sets used; 0 is what the Greek finetuning crops
# were written with, and running Greek at 12 costs it every exact line it has
# (125/204 down to 0/204).
_PADDINGS = [0, TRAINING_CROP_PADDING]


@pytest.mark.parametrize("padding", _PADDINGS)
@pytest.mark.parametrize("name", sorted(_POLYGONS))
def test_serving_crop_is_byte_identical_to_the_training_crop(name: str, padding: int) -> None:
    """``crop_line_on_white`` must reproduce ``crop_polygon`` pixel for pixel."""
    rgb = _synthetic_page()
    page = Image.fromarray(rgb, mode="RGB")
    # ``crop_polygon`` was written against a ``cv2.imread`` array, which is BGR.
    page_bgr = rgb[:, :, ::-1].copy()
    points = _POLYGONS[name]
    # The exporter never sees floats: its ``parse_points`` rounds every PAGE-XML
    # coordinate first. Feed the training function exactly what that parser
    # would hand it, so the rounding is under test too, not just the crop.
    int_points = parse_points(" ".join(f"{x},{y}" for x, y in points))

    serving = crop_line_on_white(page, points, padding=padding)
    training, _bbox = crop_polygon(page_bgr, int_points, padding, keep_color=False)

    assert serving.dtype == np.uint8
    assert serving.shape == training.shape, name
    np.testing.assert_array_equal(
        serving, training, err_msg=f"crop skew for polygon {name} at padding {padding}"
    )


@pytest.mark.parametrize("name", sorted(_POLYGONS))
def test_padding_zero_is_the_mask_without_the_margin(name: str) -> None:
    """Greek's convention, spelled out: same mask, no widening of the box.

    Parametrizing the test above already covers padding 0 against the training
    function. This one pins what the number means, so a refactor that quietly
    turned "padding 0" into "no mask" (the reading this repo held until
    2026-09-07) would fail here rather than pass both branches.
    """
    page = Image.fromarray(_synthetic_page(), mode="RGB")
    points = _POLYGONS[name]

    unpadded = crop_line_on_white(page, points, padding=0)
    padded = crop_line_on_white(page, points, padding=TRAINING_CROP_PADDING)

    # A smaller box, or an equal one where the page edge already clamped it.
    assert unpadded.shape[0] <= padded.shape[0], name
    assert unpadded.shape[1] <= padded.shape[1], name
    # Still a mask, not a bare box: none of these polygons fills its own bounding
    # box, so some pixel inside the tight box must have been painted white.
    assert (unpadded == 255).any(), name


def test_serving_default_padding_matches_the_exporter_constant() -> None:
    """The default here is the exporter's own PADDING, and must stay that way.

    It is a default, not a rule: ``greek-calamari-v1`` overrides it with 0 in the
    registry. What this pins is that the fallback for an entry built outside the
    loader is the number the exporter actually used.
    """
    assert TRAINING_CROP_PADDING == TRAINING_PADDING_CONSTANT == 12


def test_a_masked_crop_is_not_the_bare_bounding_box() -> None:
    """Guard the guard: the parity above would also hold for two no-op crops.

    ``crop_polygon``'s whole contribution is painting the neighbouring lines'
    ascenders white. If both sides silently degraded to a plain box the test
    above would still be green, so pin that the mask actually removes ink.
    """
    rgb = _synthetic_page()
    points = _POLYGONS["concave_zigzag"]
    masked = crop_line_on_white(Image.fromarray(rgb, mode="RGB"), points)

    assert (masked == 255).any(), "no pixel was painted white, so nothing was masked"
    # The concave notch cuts into the bounding box, so some in-box pixel that is
    # ink on the page has to have been whitened.
    assert masked.min() < 128, "the crop kept no ink at all"


def test_a_polygon_entirely_off_the_page_raises() -> None:
    """Degenerate geometry is a per-line ValueError, not a silent empty array."""
    page = Image.fromarray(_synthetic_page(), mode="RGB")
    with pytest.raises(ValueError):
        crop_line_on_white(page, [[-900.0, -900.0], [-880.0, -900.0], [-880.0, -880.0]])


# ---------------------------------------------------------------------------
# The tensor the model is handed
# ---------------------------------------------------------------------------

# Heights below, above and exactly at the model's line height, plus a one-pixel
# sliver: the width formula's ``max(1, round(...))`` and PIL's resize both have
# their edge cases there.
_CROP_SHAPES = [(31, 97), (73, 214), (48, 156), (25, 1), (60, 3), (48, 1)]


def _grayscale_crop(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    crop = np.full((height, width), 214, dtype=np.int16)
    crop += rng.integers(-30, 31, crop.shape)
    crop[height // 3 : max(height // 3 + 1, 2 * height // 3), : max(1, width // 2)] = 52
    return crop.clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize(("height", "width"), _CROP_SHAPES, ids=lambda value: str(value))
def test_serving_tensor_is_byte_identical_to_the_training_loader(
    tmp_path: Path, height: int, width: int
) -> None:
    """Both serving entry points must equal ``_load_line_image`` exactly."""
    pytest.importorskip("torch")
    load_line_image = _load_training_line_loader()

    array = _grayscale_crop(height, width, seed=height * 1000 + width)
    path = tmp_path / f"line_{height}x{width}.png"
    Image.fromarray(array, mode="L").save(path, format="PNG")

    expected = load_line_image(path, _LINE_HEIGHT).numpy()[None]
    assert expected.dtype == np.uint8

    from_bytes = preprocess_line_image_bytes_to_calamari_tensor(
        path.read_bytes(), line_height=_LINE_HEIGHT
    )
    from_array = preprocess_line_array_to_calamari_tensor(array, line_height=_LINE_HEIGHT)

    for label, actual in (("bytes", from_bytes), ("array", from_array)):
        assert actual.dtype == np.uint8, label
        assert actual.shape == expected.shape, label
        np.testing.assert_array_equal(
            actual, expected, err_msg=f"train/serve tensor skew via {label} at {height}x{width}"
        )


def test_recipe_properties_on_a_real_manuscript_line() -> None:
    """Pin the observable recipe without torch, on a real line crop.

    Shape, dtype, absence of padding and orientation of the ink are the four
    properties the legacy processors broke. This test needs no training import,
    so it still fails loudly in an environment where the parity tests above skip.
    """
    with Image.open(TRANSCRIBE_LINE) as source:
        width, height = source.size

    tensor = preprocess_line_image_bytes_to_calamari_tensor(
        TRANSCRIBE_LINE.read_bytes(), line_height=_LINE_HEIGHT
    )
    expected_width = round(width * _LINE_HEIGHT / height)

    assert tensor.dtype == np.uint8
    # batch x time x height x channel, and the time axis is exactly the scaled
    # width: a padded pipeline would make it wider (the legacy one added 16 px).
    assert tensor.shape == (1, expected_width, _LINE_HEIGHT, 1)
    # Dark ink on light parchment, kept that way. Inverted input would put the
    # mean far below the midpoint.
    assert tensor.mean() > 127


# ---------------------------------------------------------------------------
# The bytes the model is handed, and the registry fields behind them
# ---------------------------------------------------------------------------


def test_crop_line_dispatches_on_the_registry_value() -> None:
    """The dispatcher passes the registry's padding through untouched."""
    page = Image.fromarray(_synthetic_page(), mode="RGB")
    points = _POLYGONS["interior_irregular"]

    for padding in _PADDINGS:
        np.testing.assert_array_equal(
            crop_line(page, points, LineCrop.polygon_white, padding=padding),
            crop_line_on_white(page, points, padding=padding),
            err_msg=f"dispatcher lost the padding at {padding}",
        )


def test_the_runner_hands_on_the_bytes_the_exporter_wrote(tmp_path: Path) -> None:
    """The model was trained on JPEG, so serving must hand it JPEG.

    ``save_crop`` wrote every training crop as a grayscale JPEG at
    ``quality=82, optimize=True`` and the trainer's loader read that file back,
    so the pixels the weights were fitted on had already been through a lossy
    encoder. A lossless PNG here would be a *cleaner* picture than any the model
    has ever seen, which is not the same thing as a correct one. Compare against
    the real ``save_crop``, not against a re-spelling of its options.
    """
    from nomikos_inference.jobs.runner import _crop_line_image

    rgb = _synthetic_page()
    page = Image.fromarray(rgb, mode="RGB")
    points = _POLYGONS["concave_zigzag"]
    crop = crop_line_on_white(page, points, padding=TRAINING_CROP_PADDING)

    training_path = tmp_path / "training_crop.jpg"
    save_crop(training_path, crop, keep_color=False)
    with Image.open(training_path) as written:
        training_pixels = np.asarray(written.convert("L"), dtype=np.uint8)

    served = _crop_line_image(page, b"unused-page-bytes", points)
    with Image.open(BytesIO(served)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.mode == "L"
        served_pixels = np.asarray(decoded, dtype=np.uint8)

    np.testing.assert_array_equal(
        served_pixels, training_pixels, err_msg="serving encodes the crop unlike save_crop"
    )
    # The round trip is lossy, which is the whole point: if these matched, the
    # test above would prove nothing about the encoder.
    assert not np.array_equal(training_pixels, crop)


def test_the_runner_passes_the_padding_it_is_given(tmp_path: Path) -> None:
    """A different padding has to reach the encoded bytes, not just the crop."""
    from nomikos_inference.jobs.runner import _crop_line_image

    page = Image.fromarray(_synthetic_page(), mode="RGB")
    points = _POLYGONS["interior_irregular"]

    sizes = []
    for padding in _PADDINGS:
        encoded = _crop_line_image(page, b"unused", points, line_crop_padding=padding)
        with Image.open(BytesIO(encoded)) as decoded:
            sizes.append(decoded.size)

    assert sizes[0] != sizes[1], "padding never reached the crop"


def _registry_document(task: str, **extra: object) -> dict:
    entry = {
        "task": task,
        "architecture": "calamari" if task == "transcribe" else "blla",
        "device": "cpu",
        "versions": {"stable": {"weights_source": "file://local/best.onnx"}},
    }
    entry.update(extra)
    return {"models": {f"{task}-model": entry}}


def test_a_transcribe_entry_without_line_crop_is_rejected() -> None:
    """No default: a guess about someone else's training data is what ADR 0007 is about."""
    with pytest.raises(ValueError, match="transcribe-model"):
        RegistryDocument.model_validate(_registry_document("transcribe", line_crop_padding=12))


def test_a_transcribe_entry_without_line_crop_padding_is_rejected() -> None:
    """The padding is the field the three models actually disagree on."""
    with pytest.raises(ValueError, match="line_crop_padding"):
        RegistryDocument.model_validate(_registry_document("transcribe", line_crop="polygon-white"))


def test_a_negative_line_crop_padding_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative line_crop_padding"):
        RegistryDocument.model_validate(
            _registry_document("transcribe", line_crop="polygon-white", line_crop_padding=-1)
        )


@pytest.mark.parametrize("padding", _PADDINGS)
def test_a_complete_transcribe_entry_loads(padding: int) -> None:
    document = RegistryDocument.model_validate(
        _registry_document("transcribe", line_crop="polygon-white", line_crop_padding=padding)
    )
    entry = document.models["transcribe-model"]

    assert entry.line_crop == LineCrop.polygon_white
    assert entry.line_crop_padding == padding


def test_a_segment_entry_needs_neither_crop_field() -> None:
    document = RegistryDocument.model_validate(_registry_document("segment"))
    entry = document.models["segment-model"]

    assert entry.line_crop is None
    assert entry.line_crop_padding is None


@pytest.mark.parametrize("field", [{"line_crop": "polygon-white"}, {"line_crop_padding": 12}])
def test_a_segment_entry_may_not_set_a_crop_field(field: dict) -> None:
    """Segmentation has no line crops, so either field there is a misunderstanding."""
    with pytest.raises(ValueError, match="segment-model"):
        RegistryDocument.model_validate(_registry_document("segment", **field))


def test_the_shipped_registry_states_a_crop_for_every_transcribe_model() -> None:
    """The real file, not a fixture: these five are what production serves.

    Greek's 0 is the one that has to survive an edit. It looks like an omission
    and it is not: it is 125 exact lines out of 204 against 0 out of 204.
    Coptic's 12 is measured: CER 0.0177 at 12 against 0.1890 at 0 on 2823 platform
    Coptic project lines through best.pt at batch size 1, minimum flat across 10
    to 12, measured 2026-09-18 on the platform's Coptic project lines; full curve
    in docs/inference/models-and-datasets.md."""
    from nomikos_inference.registry import load_registry

    models = load_registry().models
    stated = {
        model_id: (entry.line_crop, entry.line_crop_padding)
        for model_id, entry in models.items()
        if entry.task == "transcribe"
    }

    assert stated == {
        "syriac-calamari-v2": (LineCrop.polygon_white, 12),
        "greek-calamari-v1": (LineCrop.polygon_white, 0),
        "armenian-calamari-v1": (LineCrop.polygon_white, 12),
        "syriac-ppocr-v1": (LineCrop.polygon_white, 0),
        "coptic-calamari-v1": (LineCrop.polygon_white, 12),
    }


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------


def _load_training_codec():
    pytest.importorskip("torch")
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.models.calamari.codec import CharacterCodec

    return CharacterCodec


_CHARSET = ["", " ", "α", "β", "γ", ",", "."]


def _softmax_for(labels: list[int], *, seed: int) -> np.ndarray:
    """A softmax whose argmax is ``labels``, with a distinct runner-up per frame."""
    rng = np.random.default_rng(seed)
    softmax = rng.uniform(0.0, 0.2, (len(labels), len(_CHARSET)))
    for frame, label in enumerate(labels):
        softmax[frame, label] = 0.6 + rng.uniform(0.0, 0.3)
    return softmax / softmax.sum(axis=1, keepdims=True)


_LABEL_SEQUENCES = {
    "leading_space": [0, 1, 1, 2, 0, 2, 3],
    "trailing_spaces": [2, 2, 0, 3, 5, 1, 0, 1],
    "space_only": [0, 1, 0, 1, 1],
    "both_edges": [1, 2, 0, 3, 1],
    "no_edge_space": [0, 2, 3, 0, 3, 4, 6, 0],
}


@pytest.mark.parametrize("labels", list(_LABEL_SEQUENCES.values()), ids=list(_LABEL_SEQUENCES))
def test_the_decoder_returns_what_the_trainer_codec_returns(labels: list[int]) -> None:
    """Edge whitespace included.

    The trainer's ``decode_ctc`` collapses repeats, drops blanks and returns
    the rest untouched, and its loader reads ``.gt.txt`` without stripping, so
    a model that learned an edge space emits one. Serving used to trim it, and
    over three whole training documents that trim was the only thing left
    between the trainer's text and the runner's.
    """
    from nomikos_inference.architectures.calamari.adapter import _decode_greedy

    codec = _load_training_codec()(tuple(_CHARSET))
    softmax = _softmax_for(labels, seed=sum(labels))

    text, confidences = _decode_greedy(softmax, charset=_CHARSET)

    assert text == codec.decode_ctc(list(np.argmax(softmax, axis=1)))
    assert len(confidences) == len(text)

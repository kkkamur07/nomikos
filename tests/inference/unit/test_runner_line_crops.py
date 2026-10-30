"""What each transcribe architecture's batch lines look like on the way in.

Calamari was trained on grayscale JPEG q82 crops and still gets them. The
PP-OCR recognizer was trained on kraken ``extract_polygons`` crops of the RGB
page with no lossy round trip, so its lines arrive as RGB PNG: the colour of
the page reaches the model, and nothing outside the polygon does.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from nomikos_inference.contracts.common import InferenceTask, LineCrop, RegistryArchitecture
from nomikos_inference.contracts.transcribe import (
    CharacterConfidence,
    TranscribeBatchRunResponse,
    TranscribeRunResponse,
)
from nomikos_inference.jobs import runner

PAGE_COLOUR = (200, 120, 40)
LINE = [[10.0, 10.0], [190.0, 10.0], [190.0, 60.0], [10.0, 60.0]]
TRIANGLE = [[10.0, 10.0], [190.0, 10.0], [10.0, 60.0]]


def _response(text: str) -> TranscribeRunResponse:
    return TranscribeRunResponse(
        text=text,
        confidence=0.5,
        character_confidences=[CharacterConfidence(char=char, confidence=0.5) for char in text],
    )


def _page() -> bytes:
    output = BytesIO()
    Image.new("RGB", (200, 80), PAGE_COLOUR).save(output, format="PNG")
    return output.getvalue()


def _entry(architecture: RegistryArchitecture, *, line_crop, padding) -> SimpleNamespace:
    return SimpleNamespace(
        architecture=architecture,
        line_crop=line_crop,
        line_crop_padding=padding,
        versions={
            "stable": SimpleNamespace(
                weights_source="file://unused", hub_revision=None, artifact_sha256=None
            )
        },
    )


@pytest.fixture
def run_lines(monkeypatch: pytest.MonkeyPatch):
    """Run a batch through ``run_model`` and return the crops each adapter received."""
    monkeypatch.setattr(runner, "validate_image_bytes", lambda *_args: None)
    monkeypatch.setattr(runner, "validate_request_params", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "get_inference_settings",
        lambda: SimpleNamespace(inference_registry_path=Path("registry.yaml")),
    )
    monkeypatch.setattr(runner, "resolve_weights_source", lambda *_a, **_k: Path("unused.onnx"))
    received: dict[str, list[bytes]] = {"calamari": [], "ppocr_rec": []}

    def fake_many(name: str):
        def run_many(line_images: list[bytes], **_kwargs) -> list[TranscribeRunResponse]:
            received[name].extend(line_images)
            return [_response("x") for _ in line_images]

        return run_many

    monkeypatch.setattr(runner, "run_calamari_transcribe_many", fake_many("calamari"))
    monkeypatch.setattr(runner, "run_ppocr_rec_transcribe_many", fake_many("ppocr_rec"))

    def run(entry: SimpleNamespace, points: list[list[float]]):
        monkeypatch.setattr(runner, "resolve_registry_entry", lambda **_kwargs: entry)
        output = runner.run_model(
            task=InferenceTask.transcribe,
            registry_model_id="model",
            registry_tag="stable",
            image_bytes=_page(),
            params={"lines": [{"line_id": "l0", "line_index": 0, "points": points}]},
        )
        return output, received

    return run


def test_ppocr_rec_lines_reach_the_model_as_lossless_rgb(run_lines) -> None:
    entry = _entry(RegistryArchitecture.ppocr_rec, line_crop=LineCrop.polygon_white, padding=0)

    output, received = run_lines(entry, LINE)

    assert isinstance(output, TranscribeBatchRunResponse)
    assert received["calamari"] == []
    crop = Image.open(BytesIO(received["ppocr_rec"][0]))
    assert (crop.format, crop.mode, crop.size) == ("PNG", "RGB", (181, 51))
    # Lossless: every pixel inside the box is exactly the page colour.
    assert {tuple(pixel) for pixel in np.asarray(crop).reshape(-1, 3)} == {PAGE_COLOUR}


def test_ppocr_rec_crop_paints_outside_the_polygon_white(run_lines) -> None:
    entry = _entry(RegistryArchitecture.ppocr_rec, line_crop=LineCrop.polygon_white, padding=0)

    _, received = run_lines(entry, TRIANGLE)

    pixels = np.asarray(Image.open(BytesIO(received["ppocr_rec"][0])))
    assert tuple(pixels[1, 1]) == PAGE_COLOUR
    assert tuple(pixels[-1, -1]) == (255, 255, 255)


def test_calamari_lines_still_reach_the_model_as_grayscale_jpeg(run_lines) -> None:
    entry = _entry(RegistryArchitecture.calamari, line_crop=LineCrop.polygon_white, padding=0)

    _, received = run_lines(entry, LINE)

    assert received["ppocr_rec"] == []
    crop = Image.open(BytesIO(received["calamari"][0]))
    assert (crop.format, crop.mode, crop.size) == ("JPEG", "L", (181, 51))


@pytest.mark.parametrize(
    ("line_crop", "padding"),
    [(None, 0), (LineCrop.polygon_white, None)],
)
@pytest.mark.parametrize(
    "architecture", [RegistryArchitecture.calamari, RegistryArchitecture.ppocr_rec]
)
def test_a_batch_without_registry_crop_settings_is_refused(
    run_lines, architecture: RegistryArchitecture, line_crop, padding
) -> None:
    """No silent 12 px: the padding alone moves Greek from CER 0.050 to 0.304."""
    entry = _entry(architecture, line_crop=line_crop, padding=padding)

    with pytest.raises(ValueError, match="must set line_crop and line_crop_padding"):
        run_lines(entry, LINE)


def test_a_single_line_request_needs_no_crop_settings(
    run_lines, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole image is the line; there is nothing to cut."""
    monkeypatch.setattr(
        runner,
        "run_ppocr_rec_transcribe",
        lambda image_bytes, **_kwargs: _response("whole"),
    )
    monkeypatch.setattr(
        runner,
        "resolve_registry_entry",
        lambda **_kwargs: _entry(RegistryArchitecture.ppocr_rec, line_crop=None, padding=None),
    )

    output = runner.run_model(
        task=InferenceTask.transcribe,
        registry_model_id="model",
        registry_tag="stable",
        image_bytes=_page(),
        params=None,
    )

    assert isinstance(output, TranscribeRunResponse)
    assert output.text == "whole"


def test_an_unknown_segment_architecture_is_refused(
    run_lines, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "resolve_registry_entry",
        lambda **_kwargs: _entry(RegistryArchitecture.calamari, line_crop=None, padding=None),
    )

    with pytest.raises(ValueError, match="unsupported segment architecture: calamari"):
        runner.run_model(
            task=InferenceTask.segment,
            registry_model_id="model",
            registry_tag="stable",
            image_bytes=_page(),
        )

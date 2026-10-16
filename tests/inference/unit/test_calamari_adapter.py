"""Calamari adapter response helpers and artifact preflight.

Torch-free by construction, like everything under ``tests/inference``: the
published wheel cannot import Torch under ADR 0006, so a test that needs it
belongs in ``tests/export`` beside the graph it exercises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nomikos_inference.architectures.calamari.adapter import (
    CalamariUnavailableError,
    _load_session,
    run_calamari_transcribe_many,
)
from nomikos_inference.architectures.shared.transcribe import response_from_decoded

# The artifact preflight cases that used to stand here - a retired ``.pt``
# refused by suffix, a digest checked before the file is opened, a missing
# artifact reported ahead of anything else - moved to
# ``test_architecture_contract``, which runs all three over both architectures
# rather than over Calamari alone.


def test_response_from_decoded_aligns_character_confidences() -> None:
    response = response_from_decoded("ܡܪܝ", [0.8, 0.9, 0.7])
    assert response.text == "ܡܪܝ"
    assert response.confidence == pytest.approx(0.8)
    assert [entry.char for entry in response.character_confidences] == ["ܡ", "ܪ", "ܝ"]
    assert [entry.confidence for entry in response.character_confidences] == [0.8, 0.9, 0.7]


def test_response_from_decoded_fills_missing_confidences() -> None:
    response = response_from_decoded("ab", [0.5])
    assert response.text == "ab"
    assert [entry.confidence for entry in response.character_confidences] == [0.5, 0.5]


def test_load_session_rejects_a_file_that_is_not_an_onnx_graph(tmp_path: Path) -> None:
    artifact = tmp_path / "corrupt.onnx"
    artifact.write_bytes(b"not a protobuf")

    with pytest.raises(CalamariUnavailableError, match="unable to load Calamari ONNX artifact"):
        _load_session(str(artifact))


def test_empty_batch_is_a_client_error_even_when_the_weights_are_missing(tmp_path: Path) -> None:
    """422 beats 503: the request was unrunnable whatever is on disk.

    Ported from ``fix/remediation-runtime``, which fixed this ordering against
    the Torch adapter. ADR 0006 restored the ONNX adapter out of the archive and
    reintroduced the original ordering with it, so the guard is re-asserted here
    against the ONNX path rather than lost with the Torch one.
    """
    with pytest.raises(ValueError, match="at least one line image") as caught:
        run_calamari_transcribe_many([], checkpoint_path=tmp_path / "absent.onnx")

    assert not isinstance(caught.value, OSError)

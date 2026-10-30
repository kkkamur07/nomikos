"""Load a tensor-only Calamari checkpoint into the runtime Torch graph.

Loading a pickled checkpoint executes code, so this module never unpickles:
``torch.load`` runs with ``weights_only=True``, and the caller has already
verified the **artifact SHA-256** through ``architectures.artifact`` before the
path reaches here.

Export-time code under ADR 0006: the runtime graph runs as ``.onnx``, and this
loader, alongside the exporter beside it, supports that path rather than
serving inference directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from nomikos_inference.export.calamari.config import default_model_config
from nomikos_inference.export.calamari.model import CalamariTorchModel

# How deep the recurrent stack may be. Calamari checkpoints published before
# the two-layer models carry no ``lstm_layers`` key at all, and every one of
# them is a single BiLSTM, so an absent key means one layer rather than an
# error. Verified against the artifact rather than assumed: the Syriac
# checkpoint at Hub revision 5ff715e873f1ae3f325ebea4d2c4a95eb5094601 (best.pt
# sha256 ea711b91...) has payload keys ``format``/``classes``/``line_height``/
# ``charset``/``state_dict`` and a state dict that stops at ``layers.4.lstm.*``.
DEFAULT_LSTM_LAYERS = 1
SUPPORTED_LSTM_LAYERS = (1, 2)


class CalamariCheckpointError(ValueError):
    """A checkpoint this runtime cannot use.

    The three subclasses let callers tell failures apart by type instead of
    matching substrings of the message: "invalid Calamari checkpoint metadata
    or state dictionary" contains the words "state dictionary" and was once
    raised for a charset defect, misreporting it as a bad state dict. The
    distinction is what a deployment reads to know which half of the export
    to look at.
    """


class CalamariCheckpointUnreadableError(CalamariCheckpointError):
    """The file could not be read as a tensor-only checkpoint at all."""


class CalamariCheckpointMetadataError(CalamariCheckpointError):
    """The checkpoint's declared shape, codec, or format is wrong."""


class CalamariCheckpointStateDictError(CalamariCheckpointError):
    """The weights are absent, malformed, or do not fit the runtime graph."""


@dataclass(frozen=True)
class CalamariCheckpointMetadata:
    """Everything the decoder needs from a checkpoint that is not a weight."""

    classes: int
    line_height: int
    charset: tuple[str, ...]
    blank_index: int = 0
    temperature: float = -1.0
    #: Depth of the recurrent stack. This is the one metadata field the decoder
    #: does not read: it selects the graph the weights are loaded into, and a
    #: wrong value fails loudly at ``load_state_dict(strict=True)`` rather than
    #: quietly transcribing through the wrong topology.
    lstm_layers: int = DEFAULT_LSTM_LAYERS


def _read_checkpoint_doc(checkpoint_path: Path) -> Mapping:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CalamariCheckpointUnreadableError(
            f"unable to safely load Calamari checkpoint: {checkpoint_path}"
        ) from error
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != "calamari-pytorch-v1":
        raise CalamariCheckpointMetadataError("unsupported Calamari checkpoint format")
    return checkpoint


def _require_classes(doc: Mapping) -> int:
    classes = doc.get("classes")
    if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
        raise CalamariCheckpointMetadataError("invalid Calamari checkpoint metadata")
    return classes


def _require_line_height(doc: Mapping) -> int:
    line_height = doc.get("line_height", 48)
    if not isinstance(line_height, int) or isinstance(line_height, bool) or line_height < 1:
        raise CalamariCheckpointMetadataError("invalid Calamari checkpoint metadata")
    return line_height


def _require_charset(doc: Mapping, classes: int) -> tuple[str, ...]:
    charset = doc.get("charset")
    if (
        not isinstance(charset, list)
        or len(charset) != classes
        or not all(isinstance(character, str) for character in charset)
    ):
        raise CalamariCheckpointMetadataError("invalid Calamari checkpoint metadata")
    return tuple(charset)


def _require_temperature(doc: Mapping) -> float:
    temperature = doc.get("temperature", -1.0)
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
    ):
        raise CalamariCheckpointMetadataError("invalid Calamari checkpoint temperature")
    return float(temperature)


def _require_blank_index(doc: Mapping) -> int:
    blank_index = doc.get("blank_index", 0)
    if not isinstance(blank_index, int) or isinstance(blank_index, bool) or blank_index != 0:
        raise CalamariCheckpointMetadataError(
            "only blank-index zero is supported by the Calamari runtime"
        )
    return blank_index


def _require_lstm_layers(doc: Mapping) -> int:
    lstm_layers = doc.get("lstm_layers", DEFAULT_LSTM_LAYERS)
    if (
        not isinstance(lstm_layers, int)
        or isinstance(lstm_layers, bool)
        or lstm_layers not in SUPPORTED_LSTM_LAYERS
    ):
        # A metadata failure, not a state-dict one: the weights on disk may be
        # perfectly good, and what is wrong is the shape the checkpoint claims
        # to have. Reporting this as a state-dict defect would send a
        # deployment looking at the wrong half of the export.
        raise CalamariCheckpointMetadataError("invalid Calamari checkpoint lstm_layers")
    return lstm_layers


def _metadata_from_doc(doc: Mapping) -> CalamariCheckpointMetadata:
    classes = _require_classes(doc)
    return CalamariCheckpointMetadata(
        classes=classes,
        line_height=_require_line_height(doc),
        charset=_require_charset(doc, classes),
        blank_index=_require_blank_index(doc),
        temperature=_require_temperature(doc),
        lstm_layers=_require_lstm_layers(doc),
    )


def _state_dict_from_doc(doc: Mapping) -> Mapping:
    state_dict = doc.get("state_dict")
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or not all(
            isinstance(name, str) and isinstance(value, Tensor)
            for name, value in state_dict.items()
        )
    ):
        raise CalamariCheckpointStateDictError("invalid Calamari checkpoint state dictionary")
    return state_dict


def _materialize_model(
    metadata: CalamariCheckpointMetadata, state_dict: Mapping
) -> CalamariTorchModel:
    # ``default_model_config`` is the single definition of this topology. The
    # loader used to carry a private copy of it, which had drifted (a dropout
    # rate of 0.5 against 0.3) and knew only the single-BiLSTM stack, so every
    # two-layer checkpoint failed here at ``strict=True``. Dropout holds no
    # weights and is the identity in ``eval()``, so collapsing the two
    # definitions onto the 0.3 one changes neither the state dict nor a single
    # logit.
    model = CalamariTorchModel(
        default_model_config(
            classes=metadata.classes,
            temperature=metadata.temperature,
            lstm_layers=metadata.lstm_layers,
        )
    )
    model.eval()
    # Materialize LazyBiLSTM and LazyLinear before loading the state dict.  The
    # time width is deliberately arbitrary; weights do not depend on it.
    dummy = torch.zeros((1, 8, metadata.line_height, 1), dtype=torch.float32)
    # ``inference_mode`` would create inference tensors for the Lazy* parameters,
    # which cannot later receive a state-dict copy on recent Torch versions.
    with torch.no_grad():
        model(dummy, image_lengths=torch.tensor([8]))
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise CalamariCheckpointStateDictError(
            "Calamari checkpoint state dictionary is incompatible"
        ) from error
    # Second ``eval()``: materializing the lazy modules above ran a forward
    # pass, and dropout must be off for every inference call that follows.
    model.eval()
    return model


def load_calamari_checkpoint(
    checkpoint_path: Path,
) -> tuple[CalamariTorchModel, CalamariCheckpointMetadata]:
    """Load and materialize a tensor-only Calamari checkpoint."""
    doc = _read_checkpoint_doc(checkpoint_path)
    metadata = _metadata_from_doc(doc)
    model = _materialize_model(metadata, _state_dict_from_doc(doc))
    return model, metadata


__all__ = [
    "CalamariCheckpointError",
    "CalamariCheckpointMetadata",
    "CalamariCheckpointMetadataError",
    "CalamariCheckpointStateDictError",
    "CalamariCheckpointUnreadableError",
    "load_calamari_checkpoint",
]

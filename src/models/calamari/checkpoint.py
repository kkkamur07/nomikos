"""Tensor-only checkpoint persistence for PyTorch Calamari models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import default_model_config
from .model import CalamariTorchModel


class CalamariCheckpointError(ValueError):
    """Base error raised for an unusable Calamari checkpoint."""


@dataclass(frozen=True)
class CalamariCheckpointMetadata:
    classes: int
    line_height: int
    charset: tuple[str, ...]
    blank_index: int = 0
    temperature: float = -1.0
    lstm_layers: int = 1


def save_calamari_checkpoint(
    checkpoint_path: Path,
    model: CalamariTorchModel,
    *,
    charset: Sequence[str],
    line_height: int,
    temperature: float = -1.0,
) -> None:
    """Save a portable model checkpoint without executable Python objects."""
    if not charset or charset[0] != "":
        raise ValueError("Calamari charsets must reserve index zero as the blank string.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "calamari-pytorch-v1",
            "classes": len(charset),
            "line_height": line_height,
            "charset": list(charset),
            "blank_index": 0,
            "temperature": temperature,
            "lstm_layers": sum(layer.kind == "bilstm" for layer in model.config.layers),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )


def load_calamari_checkpoint(
    checkpoint_path: Path,
    *,
    dropout_rate: float = 0.3,
) -> tuple[CalamariTorchModel, CalamariCheckpointMetadata]:
    """Safely load and materialize a `calamari-pytorch-v1` checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CalamariCheckpointError(
            f"Unable to safely load Calamari checkpoint: {checkpoint_path}"
        ) from error

    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != "calamari-pytorch-v1":
        raise CalamariCheckpointError("Unsupported Calamari checkpoint format.")
    metadata = _metadata_from_checkpoint(checkpoint)
    state_dict = checkpoint.get("state_dict")
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or not all(
            isinstance(name, str) and isinstance(value, Tensor)
            for name, value in state_dict.items()
        )
    ):
        raise CalamariCheckpointError("Invalid Calamari checkpoint state dictionary.")

    conv0_filters, conv1_filters = _conv_filters_from_state_dict(state_dict)
    model = CalamariTorchModel(
        default_model_config(
            classes=metadata.classes,
            temperature=metadata.temperature,
            lstm_layers=metadata.lstm_layers,
            dropout_rate=dropout_rate,
            conv0_filters=conv0_filters,
            conv1_filters=conv1_filters,
        )
    )
    with torch.no_grad():
        model(
            torch.zeros((1, 8, metadata.line_height, 1), dtype=torch.float32),
            image_lengths=torch.tensor([8]),
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise CalamariCheckpointError(
            "Calamari checkpoint state dictionary is incompatible with the model."
        ) from error
    model.eval()
    return model, metadata


def _metadata_from_checkpoint(checkpoint: Mapping[str, object]) -> CalamariCheckpointMetadata:
    classes = checkpoint.get("classes")
    line_height = checkpoint.get("line_height")
    charset = checkpoint.get("charset")
    temperature = checkpoint.get("temperature", -1.0)
    blank_index = checkpoint.get("blank_index", 0)
    lstm_layers = checkpoint.get("lstm_layers", 1)
    if (
        not isinstance(classes, int)
        or isinstance(classes, bool)
        or classes < 2
        or not isinstance(line_height, int)
        or isinstance(line_height, bool)
        or line_height < 1
        or not isinstance(charset, list)
        or len(charset) != classes
        or not all(isinstance(character, str) for character in charset)
        or charset[0] != ""
        or blank_index != 0
        or not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or not isinstance(lstm_layers, int)
        or isinstance(lstm_layers, bool)
        or lstm_layers < 1
    ):
        raise CalamariCheckpointError("Invalid Calamari checkpoint metadata.")
    return CalamariCheckpointMetadata(
        classes=classes,
        line_height=line_height,
        charset=tuple(charset),
        blank_index=0,
        temperature=float(temperature),
        lstm_layers=lstm_layers,
    )


def _conv_filters_from_state_dict(state_dict: Mapping[str, Tensor]) -> tuple[int, int]:
    """Read CNN widths from a checkpoint so 40/60 and 60/80 nets both load."""
    first = state_dict.get("layers.0.conv.weight")
    second = state_dict.get("layers.2.conv.weight")
    if (
        not isinstance(first, Tensor)
        or not isinstance(second, Tensor)
        or first.ndim != 4
        or second.ndim != 4
        or first.shape[0] < 1
        or second.shape[0] < 1
    ):
        raise CalamariCheckpointError("Calamari checkpoint is missing convolution weights.")
    return int(first.shape[0]), int(second.shape[0])

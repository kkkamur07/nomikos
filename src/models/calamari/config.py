"""Configuration primitives for the PyTorch Calamari CNN–BiLSTM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


LayerKind = Literal["conv2d", "maxpool2d", "bilstm", "dropout"]


@dataclass(frozen=True)
class CalamariTorchLayerConfig:
    kind: LayerKind
    name: str
    filters: int | None = None
    kernel_size: tuple[int, int] | None = None
    strides: tuple[int, int] | None = None
    padding: str | None = None
    activation: str | None = None
    pool_size: tuple[int, int] | None = None
    hidden_nodes: int | None = None
    merge_mode: str | None = None
    rate: float | None = None


@dataclass(frozen=True)
class CalamariTorchConfig:
    layers: tuple[CalamariTorchLayerConfig, ...]
    classes: int
    temperature: float = -1.0

    def downscaled_sequence_lengths(self, sequence_lengths: Tensor) -> Tensor:
        lengths = sequence_lengths.to(dtype=torch.long)
        for layer in self.layers:
            if layer.kind == "conv2d":
                stride = require_tuple(layer.strides, layer.name, "strides")[0]
                lengths = torch.div(lengths + stride - 1, stride, rounding_mode="floor")
            elif layer.kind == "maxpool2d":
                stride = maxpool_strides(layer)[0]
                lengths = torch.div(lengths + stride - 1, stride, rounding_mode="floor")
        return lengths


def default_model_config(
    *,
    classes: int,
    temperature: float = -1.0,
    lstm_layers: int = 2,
    dropout_rate: float = 0.3,
    conv0_filters: int = 40,
    conv1_filters: int = 60,
) -> CalamariTorchConfig:
    """Return CNN blocks followed by ``lstm_layers`` BiLSTM–dropout stacks."""
    lstm_layers = require_lstm_layers(lstm_layers)
    dropout_rate = require_dropout_rate(dropout_rate)
    conv0_filters = require_int(conv0_filters, "conv2d_0", "filters")
    conv1_filters = require_int(conv1_filters, "conv2d_1", "filters")
    if conv0_filters < 1 or conv1_filters < 1:
        raise ValueError("Calamari conv filters must be at least one.")
    recurrent_layers = tuple(
        layer
        for index in range(lstm_layers)
        for layer in (
            CalamariTorchLayerConfig(
                "bilstm", f"lstm_{index}", hidden_nodes=200, merge_mode="concat"
            ),
            CalamariTorchLayerConfig("dropout", f"dropout_{index}", rate=dropout_rate),
        )
    )
    return CalamariTorchConfig(
        layers=(
            CalamariTorchLayerConfig("conv2d", "conv2d_0", conv0_filters, (3, 3), (1, 1), "same", "relu"),
            CalamariTorchLayerConfig(
                "maxpool2d", "maxpool2d_0", pool_size=(2, 2), strides=(-1, -1), padding="same"
            ),
            CalamariTorchLayerConfig("conv2d", "conv2d_1", conv1_filters, (3, 3), (1, 1), "same", "relu"),
            CalamariTorchLayerConfig(
                "maxpool2d", "maxpool2d_1", pool_size=(2, 2), strides=(-1, -1), padding="same"
            ),
            *recurrent_layers,
        ),
        classes=classes,
        temperature=temperature,
    )


def maxpool_strides(config: CalamariTorchLayerConfig) -> tuple[int, int]:
    pool_size = require_tuple(config.pool_size, config.name, "pool_size")
    raw_strides = require_tuple(config.strides, config.name, "strides")
    return tuple(
        pool if stride < 0 else stride for stride, pool in zip(raw_strides, pool_size, strict=True)
    )


def require_lstm_layers(lstm_layers: int) -> int:
    if isinstance(lstm_layers, bool) or not isinstance(lstm_layers, int) or lstm_layers < 1:
        raise ValueError("Calamari requires at least one bidirectional LSTM layer.")
    return lstm_layers


def require_dropout_rate(dropout_rate: float) -> float:
    if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float)):
        raise ValueError("Calamari dropout_rate must be a number.")
    rate = float(dropout_rate)
    if not 0.0 <= rate < 1.0:
        raise ValueError("Calamari dropout_rate must be in [0, 1).")
    return rate


def require_int(value: int | None, layer_name: str, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{layer_name}.{field_name} is required")
    return value


def require_tuple(
    value: tuple[int, int] | None, layer_name: str, field_name: str
) -> tuple[int, int]:
    if value is None:
        raise ValueError(f"{layer_name}.{field_name} is required")
    return value

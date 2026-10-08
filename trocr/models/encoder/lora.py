"""LoRA adapters for selected DeiT encoder attention projections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Add a trainable low-rank update to a frozen or trainable linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha_rank_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be positive; received {rank}.")
        if alpha_rank_ratio <= 0:
            raise ValueError(
                f"LoRA alpha/rank ratio must be positive; received {alpha_rank_ratio}."
            )

        self.base_layer = base_layer
        self.scaling = alpha_rank_ratio
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def enable_adapter_gradients(self) -> None:
        """Train only the LoRA matrices when the base encoder is frozen."""
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = self.lora_b(self.lora_a(self.dropout(hidden_states)))
        return self.base_layer(hidden_states) + update * self.scaling


def apply_lora_to_encoder(
    encoder: nn.Module,
    *,
    rank: int,
    alpha_rank_ratio: float,
    dropout: float,
    num_layers: int,
    target_modules: Sequence[str],
) -> int:
    """Attach LoRA to attention projections in the encoder's final layers."""
    if num_layers < 1:
        raise ValueError(f"LoRA num_layers must be positive; received {num_layers}.")

    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        # Small DeiT: encoder.encoder.layer[*].attention.attention.{query,key,value}
        layers = encoder.encoder.layer
        attention_for_layer = lambda layer: layer.attention.attention
        projection_names = {"query": "query", "key": "key", "value": "value"}
    elif hasattr(encoder, "layers"):
        # Base ViT: encoder.layers[*].attention.{q_proj,k_proj,v_proj}
        layers = encoder.layers
        attention_for_layer = lambda layer: layer.attention
        projection_names = {
            "query": "q_proj",
            "key": "k_proj",
            "value": "v_proj",
        }
    else:
        raise TypeError(
            "Unsupported encoder layout for LoRA; expected small DeiT "
            "encoder.encoder.layer or base ViT encoder.layers."
        )

    if num_layers > len(layers):
        raise ValueError(
            f"LoRA requested {num_layers} layers, but the encoder has only {len(layers)}."
        )

    supported_targets = {"query", "key", "value"}
    unknown_targets = set(target_modules) - supported_targets
    if unknown_targets:
        raise ValueError(
            f"Unsupported LoRA target modules: {sorted(unknown_targets)}. "
            f"Supported targets: {sorted(supported_targets)}."
        )

    adapter_count = 0
    for layer in layers[-num_layers:]:
        attention = attention_for_layer(layer)
        for target_name in target_modules:
            projection_name = projection_names[target_name]
            projection = getattr(attention, projection_name)
            if isinstance(projection, LoRALinear):
                projection.enable_adapter_gradients()
            elif isinstance(projection, nn.Linear):
                setattr(
                    attention,
                    projection_name,
                    LoRALinear(
                        projection,
                        rank=rank,
                        alpha_rank_ratio=alpha_rank_ratio,
                        dropout=dropout,
                    ),
                )
            else:
                raise TypeError(
                    f"Expected {target_name} to be nn.Linear or LoRALinear, "
                    f"received {type(projection).__name__}."
                )
            adapter_count += 1
    return adapter_count


def configure_encoder_lora(
    encoder: nn.Module,
    lora_config: Mapping[str, object] | None,
) -> int:
    """Apply enabled LoRA settings from a serializable configuration mapping."""
    if not lora_config or not bool(lora_config.get("enabled", False)):
        return 0

    return apply_lora_to_encoder(
        encoder,
        rank=int(lora_config["rank"]),
        alpha_rank_ratio=float(lora_config["alpha_rank_ratio"]),
        dropout=float(lora_config["dropout"]),
        num_layers=int(lora_config["num_layers"]),
        target_modules=tuple(str(name) for name in lora_config["target_modules"]),
    )

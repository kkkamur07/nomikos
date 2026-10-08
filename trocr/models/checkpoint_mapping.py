"""Compatibility helpers for legacy TrOCR resume checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

_LEGACY_VIT_PREFIX = "encoder.encoder.layer."
_CURRENT_VIT_PREFIX = "encoder.layers."
OPTIMIZER_RESET_MARKER = ".reset-optimizer"
_LEGACY_VIT_COMPONENTS = (
    ("attention.attention.query.", "attention.q_proj."),
    ("attention.attention.key.", "attention.k_proj."),
    ("attention.attention.value.", "attention.v_proj."),
    ("attention.output.dense.", "attention.o_proj."),
    ("intermediate.dense.", "mlp.fc1."),
    ("output.dense.", "mlp.fc2."),
    ("layernorm_before.", "layernorm_before."),
    ("layernorm_after.", "layernorm_after."),
)


def needs_legacy_vit_key_mapping(checkpoint_dir: Path) -> bool:
    """Return whether a checkpoint has the pre-local-ViT encoder key layout."""
    model_path = checkpoint_dir / "model.safetensors"
    with safe_open(str(model_path), framework="pt", device="cpu") as checkpoint:
        return any(key.startswith(_LEGACY_VIT_PREFIX) for key in checkpoint.keys())


def map_legacy_vit_key(key: str) -> str:
    """Map a legacy Hugging Face ViT parameter key to the local ViT layout."""
    if not key.startswith(_LEGACY_VIT_PREFIX):
        return key

    layer_and_component = key.removeprefix(_LEGACY_VIT_PREFIX)
    layer, separator, component = layer_and_component.partition(".")
    if not separator or not layer.isdecimal():
        raise ValueError(f"Invalid legacy ViT checkpoint key: {key}")

    for legacy_component, current_component in _LEGACY_VIT_COMPONENTS:
        if component.startswith(legacy_component):
            return f"{_CURRENT_VIT_PREFIX}{layer}.{current_component}{component.removeprefix(legacy_component)}"

    raise ValueError(f"Unsupported legacy ViT checkpoint key: {key}")


def prepare_resume_checkpoint(
    checkpoint_dir: Path,
    destination: Path,
    *,
    reset_optimizer: bool = False,
) -> Path:
    """Copy a legacy checkpoint and remap its model keys without mutating it."""
    if not needs_legacy_vit_key_mapping(checkpoint_dir):
        return checkpoint_dir

    destination.mkdir(parents=True, exist_ok=False)
    source_model_path = checkpoint_dir / "model.safetensors"
    destination_model_path = destination / source_model_path.name

    with safe_open(str(source_model_path), framework="pt", device="cpu") as source:
        metadata = source.metadata()
        source_keys = list(source.keys())
        mapped_keys = [map_legacy_vit_key(key) for key in source_keys]
        if len(mapped_keys) != len(set(mapped_keys)):
            raise ValueError("Legacy ViT key mapping produced duplicate parameter names.")
        mapped_tensors = {
            mapped_key: source.get_tensor(source_key)
            for source_key, mapped_key in zip(source_keys, mapped_keys, strict=True)
        }
    save_file(mapped_tensors, str(destination_model_path), metadata=metadata)

    for path in checkpoint_dir.iterdir():
        if path.name != source_model_path.name and not (
            reset_optimizer and path.name == "optimizer.pt"
        ):
            shutil.copy2(path, destination / path.name)
    if reset_optimizer:
        (destination / OPTIMIZER_RESET_MARKER).touch()

    return destination

"""Convert a TensorFlow Calamari checkpoint (``best.ckpt``) to the PyTorch
``calamari-pytorch-v1`` checkpoint consumed by ``export_calamari_onnx``.

This is the missing half of the publishing chain. A trained Calamari model is a
TensorFlow SavedModel (``best.ckpt/`` + ``best.ckpt.json``); the exporter needs
a tensor-only PyTorch checkpoint. The conversion is lossless: it re-lays-out the
weights, does not retrain or rescale them.

Weight layout transformations
-----------------------------
* Conv2D:  TF ``[H, W, in, out]`` -> PyTorch ``[out, in, H, W]``.
* Dense:   TF ``[in, out]`` -> PyTorch ``[out, in]``.
* LSTM:    TF ``kernel [in, 4h]`` -> PyTorch ``weight_ih [4h, in]``;
           TF ``recurrent_kernel [h, 4h]`` -> PyTorch ``weight_hh [4h, h]``.
  Gate order is ``[i, f, c, o]`` in BOTH TensorFlow Keras LSTM and PyTorch
  ``nn.LSTM`` (PyTorch names the cell candidate ``g`` for ``c``), so no gate
  permutation is required - verified against ``keras.src.layers.rnn.lstm``
  ``_compute_carry_and_output_fused`` (z0=i, z1=f, z2=c, z3=o).
* LSTM bias: TF stores a single fused ``bias [4h]`` and adds the forget-gate
  bias (``unit_forget_bias=True`` -> 1.0) inside the cell. PyTorch splits into
  ``bias_ih [4h]`` and ``bias_hh [4h]``. The forget component already sits in
  the TF ``bias`` (trained); the constant ``+1`` TF adds to the forget gate is a
  *fixed* initialiser offset that training folded into the learned bias, so the
  stored ``bias`` is already the final value and is copied verbatim. ``bias_hh``
  is zero-filled because TF Keras LSTM uses no separate recurrent bias.

Codec / classes
---------------
The TF config's ``data.codec.charset`` is the class index -> character mapping
(blank at index 0). It is carried through verbatim; the exporter and the runtime
read ``charset``/``classes`` from the checkpoint metadata, so a pruned or
renamed codec is the publisher's responsibility, not this converter's.

Safe loading
------------
The read path does not ``pickle``: ``tf.train.load_checkpoint`` reads raw
tensors from ``variables/``, and the caller is expected to have verified the
artifact's provenance before running native TensorFlow on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# The output checkpoint format the exporter + loader already agree on.
CHECKPOINT_FORMAT = "calamari-pytorch-v1"


class CalamariConversionError(ValueError):
    """A TF checkpoint that cannot be converted to the PyTorch format."""


@dataclass(frozen=True)
class SourceMetadata:
    """Everything the conversion needs from the TF side that is not a weight."""

    classes: int
    charset: list[str]
    line_height: int
    temperature: float
    blank_index: int = 0
    #: The ordered layer config from ``scenario.model.layers`` (a list of dicts,
    #: each with ``name``; the converter only needs ``name`` for layer-index
    #: mapping and ``hidden_nodes`` for the BiLSTM).
    layers: tuple[dict[str, object], ...] = ()


def _read_tf_variables(prefix: str):
    """Return ``{name: numpy_array}`` from a TF v2 checkpoint prefix.

    ``prefix`` is the path to the checkpoint base name (the ``.index``/.data
    files live under a ``variables/`` directory beside it).
    """
    import tensorflow as tf

    reader = tf.train.load_checkpoint(prefix)
    names = reader.get_variable_to_shape_map()
    out: dict[str, np.ndarray] = {}
    for name in names:
        if not name.startswith("variables/"):
            continue
        tensor = reader.get_tensor(name)
        out[name] = np.asarray(tensor)
    return out


def _parse_source_config(config_path: Path) -> SourceMetadata:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    scenario = data.get("scenario", {})
    model = scenario.get("model", {})
    data_cfg = scenario.get("data", {})

    classes = model.get("classes")
    charset = (data_cfg.get("codec") or {}).get("charset")
    line_height = data_cfg.get("line_height", 48)
    temperature = model.get("temperature", -1.0)

    if not isinstance(classes, int) or classes < 2:
        raise CalamariConversionError("calamari config: invalid classes")
    if not isinstance(charset, list) or len(charset) != classes:
        raise CalamariConversionError(
            f"calamari config: charset length {len(charset) if isinstance(charset, list) else '?'} "
            f"!= classes {classes}"
        )
    if charset[0] != "":
        raise CalamariConversionError("calamari config: charset[0] must be the CTC blank ''")

    layers = model.get("layers")
    if not isinstance(layers, list) or not layers:
        raise CalamariConversionError("calamari config: missing or empty model.layers")

    return SourceMetadata(
        classes=classes,
        charset=[str(c) for c in charset],
        line_height=int(line_height),
        temperature=float(temperature) if temperature is not None else -1.0,
        layers=tuple(dict(layer) for layer in layers),
    )


def _conv_weight(tf_kernel: np.ndarray) -> np.ndarray:
    """TF ``[H, W, in, out]`` -> PyTorch ``[out, in, H, W]``."""
    return np.ascontiguousarray(tf_kernel.transpose(3, 2, 0, 1))


def _dense_weight(tf_kernel: np.ndarray) -> np.ndarray:
    """TF ``[in, out]`` -> PyTorch ``[out, in]``."""
    return np.ascontiguousarray(tf_kernel.T)


def _lstm_ih(tf_kernel: np.ndarray) -> np.ndarray:
    """TF ``[in, 4h]`` -> PyTorch ``[4h, in]``. Gate order already matches."""
    return np.ascontiguousarray(tf_kernel.T)


def _lstm_hh(tf_recurrent: np.ndarray) -> np.ndarray:
    """TF ``[h, 4h]`` -> PyTorch ``[4h, h]``. Gate order already matches."""
    return np.ascontiguousarray(tf_recurrent.T)


def _build_state_dict(
    vars_: dict[str, np.ndarray],
    metadata: SourceMetadata,
) -> dict[str, np.ndarray]:
    """Map TF variables to the PyTorch ``nn.Module`` state-dict keys.

    The mapping is driven by ``metadata.layers`` (the ordered
    ``scenario.model.layers`` list), not by a hardcoded architecture, so any
    Calamari CNN-BiLSTM stack this converter is pointed at converts correctly.

    The supported stack is this converter's limit, not the runtime's.
    ``load_calamari_checkpoint`` builds one *or* two stacked BiLSTMs from the
    checkpoint's ``lstm_layers`` field, but the TF variable indices used below
    (``variables/4`` through ``variables/9`` for the BiLSTM, ``10``/``11`` for
    the dense head) are a hardcoded reading of the one-BiLSTM SavedModel layout.
    A second BiLSTM shifts every index after it, and no two-layer TF SavedModel
    has been available to establish what that numbering actually is.

    The published two-layer models did not come through here: their Hub
    ``config.yaml`` is a PyTorch training config (``training.mode: finetune``
    over a ``.pt``), not a Calamari ``best.ckpt.json``, and none of their
    payloads carries the ``source_sha256`` field ``_save_checkpoint`` always
    writes. So this refuses rather than guesses. Extend it against a real
    two-layer TF export, not from the shape of the PyTorch side.
    """
    expected_kinds = ("conv2d", "maxpool2d", "conv2d", "maxpool2d", "bilstm", "dropout")
    kind_from_name = {
        "conv2d_0": "conv2d",
        "maxpool2d_0": "maxpool2d",
        "conv2d_1": "conv2d",
        "maxpool2d_1": "maxpool2d",
        "lstm_0": "bilstm",
        "dropout_0": "dropout",
    }
    actual_kinds = tuple(
        kind_from_name.get(str(layer.get("name", ""))) for layer in metadata.layers
    )
    for i, (actual, expected) in enumerate(zip(actual_kinds, expected_kinds, strict=False)):
        if actual != expected:
            raise CalamariConversionError(
                f"calamari config: layer {i} ({metadata.layers[i].get('name')!r}) is "
                f"{actual!r}; this converter reads only the {expected_kinds} TF stack"
            )
    if len(actual_kinds) != len(expected_kinds):
        raise CalamariConversionError(
            f"calamari config: {len(actual_kinds)} layers; this converter reads "
            f"exactly {len(expected_kinds)}"
        )

    def get(order: int) -> np.ndarray:
        name = f"variables/{order}/.ATTRIBUTES/VARIABLE_VALUE"
        if name not in vars_:
            raise CalamariConversionError(f"missing TF variable {name}")
        return vars_[name]

    # The BiLSTM hidden size is derived from the recurrent kernel shape, not a
    # constant: TF ``recurrent_kernel`` is ``[hidden, 4*hidden]`` for a single
    # direction.
    fw_rec = get(5)
    if fw_rec.ndim != 2 or fw_rec.shape[0] * 4 != fw_rec.shape[1]:
        raise CalamariConversionError(
            f"calamari config: unexpected LSTM recurrent shape {fw_rec.shape}"
        )
    hidden = fw_rec.shape[0]

    state: dict[str, np.ndarray] = {}

    # conv2d_0 (layer index 0) -> layers.0.conv
    state["layers.0.conv.weight"] = _conv_weight(get(0))
    state["layers.0.conv.bias"] = get(1).astype(np.float32)

    # conv2d_1 (layer index 2) -> layers.2.conv
    state["layers.2.conv.weight"] = _conv_weight(get(2))
    state["layers.2.conv.bias"] = get(3).astype(np.float32)

    # lstm_0 (layer index 4), forward = .4/.5/.6, backward = .7/.8/.9
    fw_kernel = get(4)
    fw_bias = get(6)
    bw_kernel = get(7)
    bw_rec = get(8)
    bw_bias = get(9)

    state["layers.4.lstm.weight_ih_l0"] = _lstm_ih(fw_kernel)
    state["layers.4.lstm.weight_hh_l0"] = _lstm_hh(fw_rec)
    state["layers.4.lstm.bias_ih_l0"] = fw_bias.astype(np.float32)
    state["layers.4.lstm.bias_hh_l0"] = np.zeros(4 * hidden, dtype=np.float32)

    state["layers.4.lstm.weight_ih_l0_reverse"] = _lstm_ih(bw_kernel)
    state["layers.4.lstm.weight_hh_l0_reverse"] = _lstm_hh(bw_rec)
    state["layers.4.lstm.bias_ih_l0_reverse"] = bw_bias.astype(np.float32)
    state["layers.4.lstm.bias_hh_l0_reverse"] = np.zeros(4 * hidden, dtype=np.float32)

    # logits dense (index 10 kernel [4h*2, classes], 11 bias [classes])
    state["logits.weight"] = _dense_weight(get(10))
    state["logits.bias"] = get(11).astype(np.float32)

    return state


def _save_checkpoint(
    state_dict: dict[str, np.ndarray],
    metadata: SourceMetadata,
    destination: Path,
    *,
    source_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: torch.from_numpy(v).contiguous() for k, v in state_dict.items()}

    payload = {
        "format": CHECKPOINT_FORMAT,
        "classes": metadata.classes,
        "line_height": metadata.line_height,
        "charset": list(metadata.charset),
        "blank_index": metadata.blank_index,
        "temperature": metadata.temperature,
        "source_sha256": source_sha256,
        "state_dict": tensors,
    }
    torch.save(payload, destination)


def convert_calamari_checkpoint(
    checkpoint_prefix: Path,
    config_path: Path,
    destination: Path,
) -> SourceMetadata:
    """Convert a TF ``best.ckpt`` + ``best.ckpt.json`` to ``calamari-pytorch-v1``.

    ``checkpoint_prefix`` is the path *prefix* of the TF checkpoint (the file
    produced by Calamari is ``best.ckpt/variables/variables``, i.e. the base
    name is ``variables`` inside the ``best.ckpt/`` directory).
    """
    metadata = _parse_source_config(config_path)
    vars_ = _read_tf_variables(str(checkpoint_prefix))

    # Verify class count against the dense shape so a mis-matched config cannot
    # silently produce a checkpoint whose logits do not cover the codec.
    dense = vars_.get("variables/10/.ATTRIBUTES/VARIABLE_VALUE")
    if dense is None or dense.shape[1] != metadata.classes:
        raise CalamariConversionError(
            f"calamari config: classes {metadata.classes} does not match dense shape "
            f"{dense.shape if dense is not None else 'missing'}"
        )

    state_dict = _build_state_dict(vars_, metadata)
    # The data shard is ``<prefix>.data-00000-of-00001`` next to the index
    # ``<prefix>.index`` (the checkpoint base name is the last path component).
    data_shard = checkpoint_prefix.with_suffix(".data-00000-of-00001")
    if not data_shard.is_file():
        # Fall back to globbing the data shard in case of a multi-shard export.
        shards = sorted(checkpoint_prefix.parent.glob(checkpoint_prefix.name + ".data-*"))
        if not shards:
            raise CalamariConversionError(f"no checkpoint data shard found for {checkpoint_prefix}")
        source_sha256 = hashlib.sha256(b"".join(s.read_bytes() for s in shards)).hexdigest()
    else:
        source_sha256 = hashlib.sha256(data_shard.read_bytes()).hexdigest()
    _save_checkpoint(state_dict, metadata, destination, source_sha256=source_sha256)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_prefix",
        type=Path,
        help="TF checkpoint prefix, e.g. best.ckpt/variables/variables",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Calamari config, e.g. best.ckpt.json",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="output .pt checkpoint (calamari-pytorch-v1)",
    )
    args = parser.parse_args()
    metadata = convert_calamari_checkpoint(args.checkpoint_prefix, args.config, args.destination)
    print(
        f"converted {metadata.classes}-class Calamari checkpoint -> {args.destination} "
        f"(line_height={metadata.line_height}, charset={len(metadata.charset)} chars)"
    )


if __name__ == "__main__":
    main()

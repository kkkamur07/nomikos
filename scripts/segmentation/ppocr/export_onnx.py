#!/usr/bin/env python3
"""Export PP-OCRv6 detection to ONNX with dynamic height and width.

ADR 0006 serves models through onnxruntime on a CPU-only box, so the Paddle
graph is only the export-time oracle and this script is the artifact builder.
It converts one PaddleX model directory (``inference.json`` plus
``inference.pdiparams``) to a single ONNX file whose batch, height and width
axes stay dynamic, then checks the result loads in onnxruntime.

A previous model reached the Hub with an ONNX whose time axis was frozen at
the trace shape, which is why this script asserts dynamic axes instead of
assuming the exporter preserved them.

Example::

    LD_LIBRARY_PATH=<system-libs> venv/bin/python \\
        scripts/segmentation/ppocr/export_onnx.py \\
        --model-dir /root/ppocrv6-bench-20260919/models/medium \\
        --output /root/ppocrv6-onnx-parity-20260919/pp-ocrv6-medium-det.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.segmentation.ppocr._fixtures import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="PaddleX model directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx path")
    parser.add_argument(
        "--opset",
        type=int,
        default=21,
        help="ONNX opset to export at (default: 21, the highest paddle2onnx 2.1.0 "
        "converts this model at without crashing; 22 and 23 abort in the exporter)",
    )
    return parser.parse_args()


def _check_model_files(model_dir: Path) -> tuple[Path, Path]:
    """The PaddleX model directory must hold both halves of the graph."""
    model_file = model_dir / "inference.json"
    params_file = model_dir / "inference.pdiparams"
    if not model_file.is_file():
        raise FileNotFoundError(f"no such model file: {model_file}")
    if not params_file.is_file():
        raise FileNotFoundError(f"no such params file: {params_file}")
    return model_file, params_file


def _run_exporter(model_file: Path, params_file: Path, output: Path, opset: int) -> None:
    """Convert the Paddle graph; a silent no-file result is an error."""
    from paddle2onnx.convert import export as paddle2onnx_export

    output.parent.mkdir(parents=True, exist_ok=True)
    paddle2onnx_export(
        str(model_file),
        str(params_file),
        str(output),
        opset_version=opset,
        auto_upgrade_opset=False,
        verbose=False,
        enable_onnx_checker=True,
        enable_optimize=True,
        optimize_tool=None,
        deploy_backend="onnxruntime",
    )
    if not output.is_file():
        raise RuntimeError("paddle2onnx reported success but wrote no file")


def _tensor_dims(entry) -> list:
    """One graph tensor's dims, symbolic where the exporter kept them so."""
    return [
        dim.dim_param if dim.dim_param else dim.dim_value
        for dim in entry.type.tensor_type.shape.dim
    ]


def _graph_io(model) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Named inputs and outputs with their dims, for the report and the gate."""
    inputs = [{"name": entry.name, "dims": _tensor_dims(entry)} for entry in model.graph.input]
    outputs = [{"name": entry.name, "dims": _tensor_dims(entry)} for entry in model.graph.output]
    return inputs, outputs


def _require_dynamic_inputs(inputs: list) -> None:
    """The batch, height and width axes must stay symbolic.

    A frozen axis here is the failure mode that previously shipped, so it is
    an error, not a warning.
    """
    for entry in inputs:
        dims = entry["dims"]
        if len(dims) != 4 or dims[1] != 3:
            raise ValueError(f"unexpected detection input shape: {entry}")
        if all(isinstance(dim, int) and dim > 0 for dim in (dims[0], dims[2], dims[3])):
            raise ValueError(f"input axes look frozen: {entry}")


def _collect_facts(
    output: Path,
    opset_versions: dict,
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
) -> dict[str, object]:
    """Load the artifact in onnxruntime and record its measured facts."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    facts: dict[str, object] = {
        "opset": opset_versions,
        "inputs": inputs,
        "outputs": outputs,
        "session_inputs": [entry.name for entry in session.get_inputs()],
        "session_outputs": [entry.name for entry in session.get_outputs()],
        "size_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    return facts


def export(*, model_dir: Path, output: Path, opset: int) -> dict[str, object]:
    """Convert the model directory and return the measured artifact facts."""
    import onnx

    model_file, params_file = _check_model_files(model_dir)
    _run_exporter(model_file, params_file, output, opset)
    model = onnx.load(str(output))
    onnx.checker.check_model(model, full_check=True)
    opset_versions = {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}
    inputs, outputs = _graph_io(model)
    _require_dynamic_inputs(inputs)
    return _collect_facts(output, opset_versions, inputs, outputs)


def main() -> int:
    args = _parse_args()
    facts = export(model_dir=args.model_dir, output=args.output, opset=args.opset)
    print(json.dumps(facts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

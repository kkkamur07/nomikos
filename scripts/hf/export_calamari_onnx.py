#!/usr/bin/env python3
"""Export a Calamari ``best.pt`` checkpoint to the runtime ``best.onnx`` artifact.

This is the step between ``convert_calamari.py`` (or a PyTorch training run that
already writes ``calamari-pytorch-v1``) and ``publish_model.py``. The staging
tree wants a ``.onnx`` at the leaf and ``publish_model.py`` refuses a leaf that
holds only the checkpoint, so somebody had to produce the artifact by hand
before this script existed.

The input is either a local ``.pt`` or a Hub repo id. A Hub download is always
pinned to an explicit ``--revision``: the point of the report this prints is to
let the registry pin an exact commit alongside an exact digest, and resolving a
mutable branch would make the two disagree the next time somebody pushes.

The report is the operator's half of the registry entry:

* ``source_sha256`` identifies the checkpoint the artifact was traced from,
* ``artifact_sha256`` is what goes into ``registry.yaml`` as ``artifact_sha256``
  and is what the runtime verifies before opening the graph,
* ``classes`` / ``line_height`` / ``lstm_layers`` / ``charset_size`` describe
  the model the artifact actually contains, read out of the checkpoint rather
  than restated from a model card.

Example::

    PYTHONPATH=. python scripts/hf/export_calamari_onnx.py \\
        --repo-id nomikos-project/greek-htr-calamari \\
        --revision 182a3ae9e459630d31186c9b18976856b9301e13 \\
        --destination src/hf/staging/models/greek/calamari/v1/stable/best.onnx \\
        --report-json var/export/greek-calamari-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.hf._export_cli import add_destination_arguments, write_report_json

# The one file name the publishing chain agrees on, on both sides.
CHECKPOINT_FILENAME = "best.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Local calamari-pytorch-v1 checkpoint (best.pt)",
    )
    source.add_argument(
        "--repo-id",
        help="Hub model repo to download the checkpoint from (e.g. nomikos-project/greek-htr-calamari)",
    )
    parser.add_argument(
        "--revision",
        help="Hub revision to pin (required with --repo-id; a 40-character commit)",
    )
    parser.add_argument(
        "--filename",
        default=CHECKPOINT_FILENAME,
        help=f"Checkpoint file inside the Hub repo (default: {CHECKPOINT_FILENAME})",
    )
    add_destination_arguments(parser)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    # Chunked rather than ``read_bytes``: a checkpoint is tens of megabytes and
    # this runs on the same machines that hold several of them at once.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(
    *,
    checkpoint: Path | None,
    repo_id: str | None,
    revision: str | None,
    filename: str,
) -> Path:
    """Return a local checkpoint path, downloading from the Hub if asked."""
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"no such Calamari checkpoint: {checkpoint}")
        return checkpoint
    if repo_id is None:
        raise ValueError("either --checkpoint or --repo-id is required")
    if not revision:
        raise ValueError("--revision is required with --repo-id")

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, revision=revision))


def export(
    *,
    checkpoint_path: Path,
    destination: Path,
    opset_version: int,
    repo_id: str | None,
    revision: str | None,
) -> dict[str, object]:
    """Export the checkpoint and return the provenance report for it."""
    from nomikos_inference.export.calamari import export_calamari_onnx

    metadata = export_calamari_onnx(checkpoint_path, destination, opset_version=opset_version)
    return {
        "source_path": str(checkpoint_path),
        "source_repo_id": repo_id,
        "source_revision": revision,
        "source_sha256": _sha256(checkpoint_path),
        "artifact_path": str(destination),
        "artifact_sha256": _sha256(destination),
        "artifact_bytes": destination.stat().st_size,
        "classes": metadata.classes,
        "line_height": metadata.line_height,
        "lstm_layers": metadata.lstm_layers,
        "charset_size": len(metadata.charset),
        "blank_index": metadata.blank_index,
        "temperature": metadata.temperature,
        "opset": opset_version,
    }


def main() -> int:
    args = _parse_args()
    try:
        checkpoint_path = resolve_checkpoint(
            checkpoint=args.checkpoint,
            repo_id=args.repo_id,
            revision=args.revision,
            filename=args.filename,
        )
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    try:
        report = export(
            checkpoint_path=checkpoint_path,
            destination=args.destination,
            opset_version=args.opset,
            repo_id=args.repo_id,
            revision=args.revision,
        )
    except Exception as error:  # noqa: BLE001 - an operator wants the message, not a traceback
        print(f"export failed: {error}", file=sys.stderr)
        return 1

    write_report_json(args.report_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

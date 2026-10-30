"""Calamari OCR architecture: ONNX Runtime adapter and line preprocessing.

The Torch graph that used to be re-exported here (``CalamariTorchModel`` and
its config) is not part of the runtime under ADR 0006. It builds the artifact
rather than running it, so it lives in ``nomikos_inference/export/calamari/``
and is not in the
published wheel.
"""

from nomikos_inference.architectures.calamari.adapter import (
    CalamariUnavailableError,
    TranscribeLineFailure,
    run_calamari_transcribe,
    run_calamari_transcribe_many,
)

__all__ = [
    "CalamariUnavailableError",
    "TranscribeLineFailure",
    "run_calamari_transcribe",
    "run_calamari_transcribe_many",
]

"""Open an ONNX Runtime CPU session and report failures in the adapter's terms.

Every architecture opened its session the same way: import onnxruntime
lazily, build a CPU session, check the graph, and turn a missing runtime or
an unreadable file into the adapter's own ``...UnavailableError`` while
letting that error through untouched. This module is that wrapper once; the
adapters keep their ``lru_cache``-d loaders (the artifact fingerprint is
part of the cache key) and their graph checks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Protocol, TypeVar

import numpy as np

CheckedT = TypeVar("CheckedT")

CPU_PROVIDERS = ["CPUExecutionProvider"]


class OnnxNodeArg(Protocol):
    """A graph input or output as onnxruntime describes it."""

    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> Sequence[object]: ...


class OnnxModelMeta(Protocol):
    @property
    def custom_metadata_map(self) -> Mapping[str, str]: ...


class OnnxSession(Protocol):
    """The part of ``onnxruntime.InferenceSession`` the adapters use."""

    def run(
        self, output_names: list[str] | None, input_feed: Mapping[str, np.ndarray]
    ) -> Sequence[np.ndarray]: ...

    def get_inputs(self) -> Sequence[OnnxNodeArg]: ...

    def get_outputs(self) -> Sequence[OnnxNodeArg]: ...

    def get_modelmeta(self) -> OnnxModelMeta: ...


def open_checked_session(
    model_path: str,
    *,
    unavailable: type[Exception],
    missing_runtime_message: str,
    load_failure_message: str,
    check: Callable[[OnnxSession], CheckedT],
    session_options: Callable[[ModuleType], object] | None = None,
) -> CheckedT:
    """Open ``model_path`` on the CPU provider and return ``check(session)``.

    ``check`` raises ``unavailable`` for a graph the adapter cannot read; that
    error, and any ``unavailable`` raised while building ``session_options``,
    propagates as it is. A missing onnxruntime becomes ``unavailable`` with
    ``missing_runtime_message`` and every other failure ``unavailable`` with
    ``load_failure_message``, the original chained as the cause.
    """
    try:
        import onnxruntime as ort

        if session_options is None:
            session = ort.InferenceSession(model_path, providers=CPU_PROVIDERS)
        else:
            session = ort.InferenceSession(
                model_path, sess_options=session_options(ort), providers=CPU_PROVIDERS
            )
        return check(session)
    except unavailable:
        raise
    except ImportError as error:
        raise unavailable(missing_runtime_message) from error
    except Exception as error:
        raise unavailable(load_failure_message) from error


__all__ = [
    "OnnxModelMeta",
    "OnnxNodeArg",
    "OnnxSession",
    "open_checked_session",
]

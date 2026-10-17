"""Open ONNX Runtime sessions behind one typed helper.

Every architecture adapter opens its graph the same way: a CPU
``InferenceSession`` for a file path, with ``ImportError`` and any other
failure rewrapped into the architecture's own ``...UnavailableError``. This
module is the single ``InferenceSession(`` call site; the per-architecture
graph checks (format tag, codec, tensor names) stay in their adapters, and
the ``fingerprint`` cache key stays on each adapter's ``lru_cache`` loader.

Intended final home is ``nomikos_inference/architectures/shared/`` (owned by
another stream); until that move lands, adapters import it from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import onnxruntime as ort

ErrorT = TypeVar("ErrorT", bound=Exception)


def open_onnx_session(
    model_path: str,
    error_cls: type[ErrorT],
    open_message: str,
    missing_message: str,
    *,
    session_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """Open a CPU session for ``model_path``, rewrapping failures.

    ``open_message`` becomes the ``...UnavailableError`` for anything the
    runtime raises; ``missing_message`` is the one for a missing onnxruntime.
    """
    try:
        import onnxruntime as ort

        options: dict[str, object] = {"providers": ["CPUExecutionProvider"]}
        if session_options is not None:
            options["sess_options"] = session_options
        return ort.InferenceSession(model_path, **options)
    except ImportError as error:
        raise error_cls(missing_message) from error
    except Exception as error:
        raise error_cls(open_message) from error


__all__ = [
    "open_onnx_session",
]

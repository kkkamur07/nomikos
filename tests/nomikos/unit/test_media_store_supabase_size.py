"""SupabaseMediaStore.size and read against a fake Storage client.

``size`` lets finalize reject an over-cap blob from metadata alone; a missing object
must surface as ``FileNotFoundError`` whether Storage says so by status or by message,
and any other failure must propagate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from storage3.exceptions import StorageApiError

from backend.document.infrastructure.media_store.supabase import SupabaseMediaStore

_KEY = "parts/0b8d3c6e-0d7c-4f4e-9a59-3c1b7f1e2a10/page.webp"
_PREFIX, _, _NAME = _KEY.rpartition("/")


class _Bucket:
    def __init__(self, entries: object = None, error: Exception | None = None) -> None:
        self._entries = entries
        self._error = error
        self.list_calls: list[tuple[object, object]] = []

    def list(self, path: object = None, options: object = None) -> object:
        self.list_calls.append((path, options))
        if self._error is not None:
            raise self._error
        return self._entries

    def download(self, _key: str) -> bytes:
        if self._error is not None:
            raise self._error
        return b"bytes"


def _store(bucket: _Bucket) -> SupabaseMediaStore:
    storage = SimpleNamespace(from_=lambda _name: bucket)
    return SupabaseMediaStore(bucket="pages", client=SimpleNamespace(storage=storage))


def test_size_reads_metadata_of_the_exact_name_with_a_prefix_search() -> None:
    bucket = _Bucket(
        [
            {"name": f"{_NAME}.bak", "metadata": {"size": 1}},
            {"name": _NAME, "metadata": {"size": "2048"}},
        ]
    )

    assert _store(bucket).size(_KEY) == 2048
    assert bucket.list_calls == [(_PREFIX, {"search": _NAME, "limit": 100})]


def test_size_falls_back_to_content_length() -> None:
    bucket = _Bucket([{"name": _NAME, "metadata": {"contentLength": 77}}])

    assert _store(bucket).size(_KEY) == 77


@pytest.mark.parametrize(
    "entries",
    [None, [], [{"name": "other.webp", "metadata": {"size": 5}}], [{"name": _NAME}]],
)
def test_size_without_a_matching_sized_entry_is_not_found(entries: object) -> None:
    with pytest.raises(FileNotFoundError):
        _store(_Bucket(entries)).size(_KEY)


def test_size_rejects_an_invalid_key_before_listing() -> None:
    bucket = _Bucket([{"name": "x.webp", "metadata": {"size": 3}}])

    with pytest.raises(ValueError, match="Invalid image key"):
        _store(bucket).size("x.webp")
    assert bucket.list_calls == []


@pytest.mark.parametrize(
    "error",
    [
        StorageApiError("bucket gone", "NoSuchKey", 404),
        StorageApiError("Object not found", "InvalidRequest", 400),
        RuntimeError("HTTP 404"),
    ],
)
def test_missing_objects_are_file_not_found(error: Exception) -> None:
    with pytest.raises(FileNotFoundError):
        _store(_Bucket(error=error)).size(_KEY)
    with pytest.raises(FileNotFoundError):
        _store(_Bucket(error=error)).read(_KEY)


def test_other_storage_failures_propagate() -> None:
    error = StorageApiError("upstream timeout", "GatewayTimeout", 504)

    with pytest.raises(StorageApiError):
        _store(_Bucket(error=error)).size(_KEY)
    with pytest.raises(StorageApiError):
        _store(_Bucket(error=error)).read(_KEY)


def test_read_returns_the_downloaded_bytes() -> None:
    assert _store(_Bucket()).read(_KEY) == b"bytes"

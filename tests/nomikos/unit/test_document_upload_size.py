"""The multipart page upload refuses an over-cap file before reading it into memory.

Required QA finding: ``upload_part`` read up to ``MAX_UPLOAD_BYTES + 1`` (100 MB) into
memory per request before checking the size. Starlette's parser has spooled the body
and set ``UploadFile.size`` by then, so the size check now comes first.
"""

from __future__ import annotations

import uuid

import pytest

from backend.core.exceptions import ValidationError
from backend.document.api.documents import parts as parts_api


class _Upload:
    def __init__(self, size: int | None, data: bytes = b"") -> None:
        self.size = size
        self.filename = "page.png"
        self._data = data
        self.reads: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.reads.append(size)
        return self._data[: size if size >= 0 else None]


class _Parts:
    def __init__(self) -> None:
        self.uploads: list[bytes] = []

    async def upload_part(self, *_args: object, data: bytes, filename: str | None) -> str:
        self.uploads.append(data)
        return "part"


async def _upload(monkeypatch: pytest.MonkeyPatch, upload: _Upload) -> _Parts:
    fake = _Parts()
    monkeypatch.setattr(parts_api, "_parts", fake)
    monkeypatch.setattr(parts_api, "part_response", lambda part: part)
    await parts_api.upload_part(uuid.uuid4(), uuid.uuid4(), object(), object(), upload)
    return fake


@pytest.mark.asyncio
async def test_an_over_cap_upload_is_refused_without_a_read(monkeypatch) -> None:
    upload = _Upload(size=parts_api.MAX_UPLOAD_BYTES + 1)

    with pytest.raises(ValidationError, match="100 MB"):
        await _upload(monkeypatch, upload)

    assert upload.reads == []


@pytest.mark.asyncio
async def test_an_unknown_size_still_hits_the_bounded_read(monkeypatch) -> None:
    monkeypatch.setattr(parts_api, "MAX_UPLOAD_BYTES", 4)
    upload = _Upload(size=None, data=b"12345")

    with pytest.raises(ValidationError, match="100 MB"):
        await _upload(monkeypatch, upload)

    assert upload.reads == [5]


@pytest.mark.asyncio
async def test_an_empty_upload_is_refused(monkeypatch) -> None:
    with pytest.raises(ValidationError, match="empty"):
        await _upload(monkeypatch, _Upload(size=0))


@pytest.mark.asyncio
async def test_an_in_cap_upload_reaches_the_service(monkeypatch) -> None:
    fake = await _upload(monkeypatch, _Upload(size=3, data=b"png"))

    assert fake.uploads == [b"png"]

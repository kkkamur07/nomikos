"""Publish tag moves: create-first, delete only to move an existing tag."""

from __future__ import annotations

import httpx
import pytest
from huggingface_hub.utils import HfHubHTTPError

from nomikos_inference.publish.client import HuggingFacePublishClient


class _FakeApi:
    def __init__(self, calls: list[str], create_errors: list[Exception | None]) -> None:
        self.calls = calls
        self.create_errors = create_errors

    def create_tag(self, **kwargs: object) -> None:
        self.calls.append("create")
        error = self.create_errors.pop(0) if self.create_errors else None
        if error is not None:
            raise error

    def delete_tag(self, **kwargs: object) -> None:
        self.calls.append("delete")


def _conflict() -> HfHubHTTPError:
    request = httpx.Request("POST", "https://huggingface.co/api/models/tag")
    return HfHubHTTPError("409 Conflict", response=httpx.Response(409, request=request))


def test_create_tag_for_a_new_tag_never_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeApi(calls, []))
    HuggingFacePublishClient().create_tag(
        repo_id="nomikos/x", tag="stable", revision="a" * 40, repo_type="model"
    )
    assert calls == ["create"]


def test_create_tag_moves_an_existing_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeApi(calls, [_conflict(), None]))
    HuggingFacePublishClient().create_tag(
        repo_id="nomikos/x", tag="stable", revision="b" * 40, repo_type="model"
    )
    assert calls == ["create", "delete", "create"]


def test_create_tag_propagates_non_conflict_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    failure = RuntimeError("network down")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeApi(calls, [failure]))
    with pytest.raises(RuntimeError, match="network down"):
        HuggingFacePublishClient().create_tag(
            repo_id="nomikos/x", tag="stable", revision="a" * 40, repo_type="model"
        )
    assert calls == ["create"]

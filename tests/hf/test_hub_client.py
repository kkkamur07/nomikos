"""Hub download error mapping: every branch pins its exact message."""

from __future__ import annotations

import pytest

from nomikos_inference.hub.client import _hub_error_message


def _error(name: str, message: str = "") -> Exception:
    return type(name, (Exception,), {})(message)


def test_repo_not_found_names_the_repo() -> None:
    message = _hub_error_message(
        _error("RepositoryNotFoundError"), repo_id="nomikos/x", revision="a" * 40
    )
    assert "nomikos/x" in message
    assert "namespace" in message


def test_revision_not_found_names_repo_and_revision() -> None:
    message = _hub_error_message(
        _error("RevisionNotFoundError"), repo_id="nomikos/x", revision="b" * 40
    )
    assert "nomikos/x" in message
    assert "b" * 40 in message


@pytest.mark.parametrize("name", ["GatedRepoError", "HfHubHTTPError"])
@pytest.mark.parametrize("detail", ["401 Unauthorized", "403 Forbidden", "not authorized"])
def test_access_denied_points_at_hf_token(name: str, detail: str) -> None:
    message = _hub_error_message(_error(name, detail), repo_id="nomikos/x", revision="a" * 40)
    assert "access denied" in message
    assert "HF_TOKEN" in message


def test_offline_mode_points_at_fetch_model() -> None:
    message = _hub_error_message(
        _error("OfflineModeIsEnabled"), repo_id="nomikos/x", revision="a" * 40
    )
    assert "offline" in message
    assert "fetch_model.py" in message


def test_generic_error_keeps_the_detail() -> None:
    message = _hub_error_message(
        _error("ConnectionError", "connection reset"), repo_id="nomikos/x", revision="a" * 40
    )
    assert "connection reset" in message


def test_generic_error_without_detail_falls_back_to_the_name() -> None:
    message = _hub_error_message(_error("TimeoutError"), repo_id="nomikos/x", revision="a" * 40)
    assert "TimeoutError" in message

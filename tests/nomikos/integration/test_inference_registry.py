"""Hosted inference registry endpoint for helper sync."""

import pytest
import yaml
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_inference_registry_returns_yaml(client: TestClient):
    response = client.get("/inference/v1/registry")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/yaml")
    assert "models:" in response.text
    document = yaml.safe_load(response.text)
    # greek-calamari-v1 was asserted *absent* here while its Hub repo 404'd and
    # its registry block was commented out. All three scripts are published now,
    # so the endpoint is expected to serve the whole transcribe family.
    assert {
        "greek-calamari-v1",
        "armenian-calamari-v1",
        "syriac-calamari-v2",
        "coptic-calamari-v1",
    } <= set(document["models"])
    assert "etag" in response.headers


def test_inference_registry_returns_304_when_etag_matches(client: TestClient):
    first = client.get("/inference/v1/registry")
    etag = first.headers["etag"]
    second = client.get("/inference/v1/registry", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.text == ""

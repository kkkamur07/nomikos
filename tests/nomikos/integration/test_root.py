"""Root endpoint security headers."""

# --- Security headers ---
# Tests baseline HTTP security headers on responses. Does not test CSP for the SPA.

import pytest

pytestmark = pytest.mark.integration


def test_security_headers_are_present(client):
    response = client.get("/")

    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert (
        response.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
    )
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

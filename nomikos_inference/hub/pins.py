"""Digest patterns for pinned Hub provenance.

The registry validates the *shape* of ``hub_revision`` and ``artifact_sha256``
when the YAML is edited; the Hub cache re-checks the same shapes at resolve
time. One definition so the two cannot drift apart.
"""

from __future__ import annotations

import re

COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

__all__ = [
    "COMMIT_SHA_PATTERN",
    "SHA256_PATTERN",
]

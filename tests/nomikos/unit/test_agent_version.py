"""The version comparison itself, and the two dials that drive it.

No database and no HTTP here on purpose: this file is about arithmetic and
configuration validity, and both are answered exactly by calling the real
functions. The behaviour over the wire - who is refused, who is served, who is
told - is proved against the real app and live Postgres in
``tests/nomikos/integration/test_agent_version_floor.py``.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")

from backend.core.settings.device import DeviceSettings
from backend.ml.domain.agent_version import (
    AgentVersion,
    AgentVersionStatus,
    MalformedAgentVersionError,
    evaluate_agent_version,
)

# ---------------------------------------------------------------------------
# Ordering: the trap this whole module exists to avoid
# ---------------------------------------------------------------------------


def test_zero_ten_is_newer_than_zero_nine() -> None:
    """``"0.10.0" > "0.9.0"`` is False. That is the bug, written down.

    A string comparison would have refused every agent on the tenth minor
    release the day it shipped, and refused it for being "too old".
    """
    assert "0.10.0" < "0.9.0", "the string comparison this test exists to rule out"

    assert AgentVersion.parse("0.10.0") > AgentVersion.parse("0.9.0")
    assert AgentVersion.parse("1.0.0") > AgentVersion.parse("0.99.99")
    assert AgentVersion.parse("0.2.10") > AgentVersion.parse("0.2.9")


def test_versions_that_are_equal_compare_equal() -> None:
    assert AgentVersion.parse("1.2.3") == AgentVersion.parse("1.2.3")
    assert AgentVersion.parse(" 1.2.3 ") == AgentVersion.parse("1.2.3")


def test_a_pre_release_sorts_below_the_release_it_leads_to() -> None:
    """A release candidate is not the release, so it is below a floor set at it."""
    assert AgentVersion.parse("0.4.0rc1") < AgentVersion.parse("0.4.0")
    assert AgentVersion.parse("0.4.0a1") < AgentVersion.parse("0.4.0b1")
    assert AgentVersion.parse("0.4.0.dev3") < AgentVersion.parse("0.4.0a1")
    assert evaluate_agent_version("0.4.0rc1", minimum="0.4.0", latest="0.4.0").refused


def test_local_build_metadata_does_not_change_the_comparison() -> None:
    assert AgentVersion.parse("0.4.0+g1234abc") == AgentVersion.parse("0.4.0")


# ---------------------------------------------------------------------------
# What counts as a version at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "latest",
        "v",
        "1",
        "1.2",
        "1.2.3.4",
        "0.1.x",
        "1.2.-3",
        "01.2.3",
        "1.2.3-unstable",
        "1.2.3 ; DROP TABLE jobs",
        "0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.1",
        None,
    ],
)
def test_a_version_the_platform_cannot_compare_is_not_a_version(raw: str | None) -> None:
    with pytest.raises(MalformedAgentVersionError):
        AgentVersion.parse(raw)


# ---------------------------------------------------------------------------
# Five states, three of which refuse
# ---------------------------------------------------------------------------


def test_below_the_floor_refuses() -> None:
    verdict = evaluate_agent_version("0.3.0", minimum="0.4.0", latest="0.6.0")

    assert verdict.status is AgentVersionStatus.below_floor
    assert verdict.refused
    assert "0.4.0" in verdict.message()


def test_at_the_floor_is_served() -> None:
    verdict = evaluate_agent_version("0.4.0", minimum="0.4.0", latest="0.4.0")

    assert verdict.status is AgentVersionStatus.current
    assert not verdict.refused
    assert not verdict.outdated


def test_above_the_floor_but_behind_the_latest_is_served_and_outdated() -> None:
    """The state that must not collapse into refusal: it is a notice."""
    verdict = evaluate_agent_version("0.4.0", minimum="0.4.0", latest="0.6.0")

    assert verdict.status is AgentVersionStatus.outdated
    assert not verdict.refused
    assert verdict.outdated


def test_missing_and_malformed_are_refused_and_told_apart() -> None:
    """Both refuse. An agent that does not say what it is is exactly the
    population the floor exists to stop, so it is never assumed current."""
    missing = evaluate_agent_version(None, minimum="0.4.0", latest="0.6.0")
    malformed = evaluate_agent_version("not-a-version", minimum="0.4.0", latest="0.6.0")

    assert missing.status is AgentVersionStatus.missing
    assert missing.refused
    assert missing.presented is None
    assert malformed.status is AgentVersionStatus.malformed
    assert malformed.refused
    assert malformed.presented == "not-a-version"


def test_an_absurdly_long_header_is_malformed_rather_than_stored() -> None:
    verdict = evaluate_agent_version("9" * 500, minimum="0.4.0", latest="0.6.0")

    assert verdict.status is AgentVersionStatus.malformed
    assert verdict.presented is not None
    assert len(verdict.presented) <= 32


# ---------------------------------------------------------------------------
# The floor is configuration, and bad configuration does not start
# ---------------------------------------------------------------------------


def test_the_floor_is_read_from_the_environment() -> None:
    settings = DeviceSettings(INFERENCE_AGENT_MIN_VERSION="2.3.4")

    assert settings.inference_agent_min_version == "2.3.4"
    # Raising the floor alone is a one-variable turn: latest follows it rather
    # than staying at a stale default and refusing to start.
    assert settings.agent_latest_version() == "2.3.4"
    assert (
        DeviceSettings(
            INFERENCE_AGENT_MIN_VERSION="2.3.4", INFERENCE_AGENT_LATEST_VERSION="9.9.9"
        ).agent_latest_version()
        == "9.9.9"
    )


def test_an_unparseable_floor_refuses_to_start_rather_than_failing_every_claim() -> None:
    """It would otherwise surface as a 500 on the endpoint all inference runs
    through, on the first agent to poll rather than on the deploy that broke it."""
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        DeviceSettings(INFERENCE_AGENT_MIN_VERSION="newest")
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        DeviceSettings(INFERENCE_AGENT_LATEST_VERSION="nightly")


def test_a_latest_below_the_floor_is_incoherent_and_refused() -> None:
    with pytest.raises(ValueError, match="must not be below"):
        DeviceSettings(INFERENCE_AGENT_MIN_VERSION="0.5.0", INFERENCE_AGENT_LATEST_VERSION="0.4.0")

    # And the numeric comparison holds here too: 0.10.0 is a valid latest for a
    # 0.9.0 floor, which a string compare would have rejected.
    DeviceSettings(INFERENCE_AGENT_MIN_VERSION="0.9.0", INFERENCE_AGENT_LATEST_VERSION="0.10.0")

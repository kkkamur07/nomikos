"""How old an **inference agent** is allowed to be, and how that is decided.

ADR 0002 gives the CLI a launch moment with no in-flight work: it asks the
platform what it must be running, upgrades itself if it is too old, prints a
notice if it is merely behind, and only then starts claiming. This module is the
platform half of that exchange - the comparison itself, with no HTTP and no
settings object in it.

Two states, and collapsing them would defeat the point
------------------------------------------------------
* **Below the floor** - refused. The agent cannot claim at all until it upgrades.
  This is the thing frozen installers made impossible: stopping a known-bad
  agent from taking work without waiting for anyone to install anything.
* **Outdated** - at or above the floor but behind the latest. Served normally
  *and told*, because most upgrades are not urgent and refusing them would turn
  every release into an outage for anyone who had not restarted.

A version nobody stated is refused
----------------------------------
Missing and malformed both refuse. Treating an absent version as current would
exempt precisely the population this exists to stop: an agent old enough to
predate the header, or one whose version string we cannot compare, is not one we
can vouch for. The refusal is the same either way, and the ``reason`` tells the
CLI which it was.

Ordering is numeric, never lexicographic
----------------------------------------
``0.10.0`` is newer than ``0.9.0``, and ``"0.10.0" > "0.9.0"`` is ``False``. That
one comparison is the whole reason this is a parsed tuple rather than a string
compare, and it is what a floor of ``0.9.0`` would silently get wrong on the day
the tenth minor release shipped.

The accepted grammar
--------------------
``MAJOR.MINOR.PATCH``, with an optional pre-release marker (``a``/``b``/``rc``
/``dev`` plus a number) and an optional ``+local`` segment that is ignored for
ordering. A pre-release sorts *below* the release it leads to, so an agent on
``0.4.0rc1`` is below a floor of ``0.4.0`` - which is correct: a release
candidate is not the release.

That is narrower than PEP 440 on purpose. This value arrives in a header from an
unauthenticated caller, so the grammar is small enough to read in one sitting,
and anything outside it is refused rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

MAX_AGENT_VERSION_LENGTH = 32
"""Matches ``helper_devices.helper_version``. A header longer than the column
that would record it is not a version, it is someone probing."""

_VERSION_PATTERN = re.compile(
    r"""
    ^
    (?P<major>0|[1-9][0-9]*)
    \.(?P<minor>0|[1-9][0-9]*)
    \.(?P<patch>0|[1-9][0-9]*)
    (?:
        [-.]?
        (?P<pre_label>a|b|rc|alpha|beta|dev)
        \.?
        (?P<pre_number>0|[1-9][0-9]*)?
    )?
    (?:\+[0-9a-zA-Z.]+)?          # local/build metadata, ignored for ordering
    $
    """,
    re.VERBOSE,
)

# Ordering among pre-release kinds. Everything here is negative so that any
# pre-release sorts below the plain release it leads to, which is the ``0`` rank
# a final release carries.
_PRE_RANKS: dict[str, int] = {
    "dev": -4,
    "alpha": -3,
    "a": -3,
    "beta": -2,
    "b": -2,
    "rc": -1,
}


class MalformedAgentVersionError(ValueError):
    """The presented string is not a version this platform can compare."""


@dataclass(frozen=True, order=True)
class AgentVersion:
    """One comparable agent version.

    ``order=True`` on a field order of major, minor, patch, pre-rank, pre-number
    is the comparison: tuples of integers, so ``0.10.0`` beats ``0.9.0`` by
    arithmetic rather than by alphabet.
    """

    major: int
    minor: int
    patch: int
    pre_rank: int = 0
    pre_number: int = 0

    @classmethod
    def parse(cls, raw: str | None) -> AgentVersion:
        """Parse, or raise :class:`MalformedAgentVersionError`.

        ``None`` and blank are malformed here; the caller distinguishes *missing*
        from *malformed* because the two mean different things to a human reading
        an agent's logs, even though both refuse.
        """
        if raw is None:
            raise MalformedAgentVersionError("no version was presented")
        candidate = raw.strip()
        if not candidate:
            raise MalformedAgentVersionError("empty version")
        if len(candidate) > MAX_AGENT_VERSION_LENGTH:
            raise MalformedAgentVersionError("version is too long to be a version")
        match = _VERSION_PATTERN.match(candidate)
        if match is None:
            raise MalformedAgentVersionError(f"{candidate!r} is not MAJOR.MINOR.PATCH")
        label = match.group("pre_label")
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            pre_rank=_PRE_RANKS[label] if label else 0,
            pre_number=int(match.group("pre_number") or 0),
        )


class AgentVersionStatus(StrEnum):
    """What the platform makes of the version an agent presented.

    The first three refuse and the last two serve. Nothing collapses them: an
    agent that is merely behind keeps working, and an agent that is too old stops
    - which is the distinction the whole feature exists for.
    """

    missing = "missing"
    malformed = "malformed"
    below_floor = "below_floor"
    outdated = "outdated"
    current = "current"

    @property
    def refuses(self) -> bool:
        return self in _REFUSING_STATUSES


_REFUSING_STATUSES = frozenset(
    {
        AgentVersionStatus.missing,
        AgentVersionStatus.malformed,
        AgentVersionStatus.below_floor,
    }
)


@dataclass(frozen=True)
class AgentVersionVerdict:
    """The answer, plus everything the agent needs to act on it.

    ``presented`` is the raw header, kept verbatim even when it did not parse, so
    the refusal can quote back what the agent actually sent rather than a
    normalised guess at it.
    """

    status: AgentVersionStatus
    presented: str | None
    minimum: str
    latest: str

    @property
    def refused(self) -> bool:
        return self.status.refuses

    @property
    def outdated(self) -> bool:
        return self.status is AgentVersionStatus.outdated

    def message(self) -> str:
        """One human sentence, for a terminal rather than for a log aggregator."""
        if self.status is AgentVersionStatus.missing:
            return (
                "This request did not say which agent version it is. The platform "
                f"requires {self.minimum} or newer and will not hand work to an "
                "agent it cannot identify."
            )
        if self.status is AgentVersionStatus.malformed:
            return (
                f"{self.presented!r} is not a version the platform can compare. "
                f"It requires {self.minimum} or newer."
            )
        if self.status is AgentVersionStatus.below_floor:
            return (
                f"This agent is {self.presented}, and the platform no longer accepts "
                f"anything below {self.minimum}. Upgrade and start it again."
            )
        if self.status is AgentVersionStatus.outdated:
            return (
                f"This agent is {self.presented}; {self.latest} is available. Work is "
                "still being handed over, but upgrade when convenient."
            )
        return f"This agent is {self.presented}, which is current."


def evaluate_agent_version(
    presented: str | None, *, minimum: str, latest: str
) -> AgentVersionVerdict:
    """Judge one presented version against the configured floor and latest.

    ``minimum`` and ``latest`` come from device settings and are therefore
    turnable without a release - that is the point of asking the platform rather
    than PyPI, and it is what lets a known-bad agent be stopped from claiming in
    the time it takes to change one environment variable.
    """
    if presented is None or not presented.strip():
        return AgentVersionVerdict(
            status=AgentVersionStatus.missing, presented=None, minimum=minimum, latest=latest
        )
    try:
        version = AgentVersion.parse(presented)
    except MalformedAgentVersionError:
        return AgentVersionVerdict(
            status=AgentVersionStatus.malformed,
            presented=presented.strip()[:MAX_AGENT_VERSION_LENGTH],
            minimum=minimum,
            latest=latest,
        )

    normalized = presented.strip()
    if version < AgentVersion.parse(minimum):
        status = AgentVersionStatus.below_floor
    elif version < AgentVersion.parse(latest):
        status = AgentVersionStatus.outdated
    else:
        status = AgentVersionStatus.current
    return AgentVersionVerdict(status=status, presented=normalized, minimum=minimum, latest=latest)

"""Failures of the platform surface the CLI talks to.

Every request the CLI makes can fail in only a few ways that change what the
caller does next, so each of those ways is its own exception. Anything else is
a `PlatformError`: unreachable, unusable, try again later.
"""

from __future__ import annotations


class PlatformError(RuntimeError):
    """The platform could not be reached, or answered something unusable."""


class InsecurePlatformURL(PlatformError):
    """A platform URL this CLI will not send a credential to."""


class CredentialRefused(PlatformError):
    """The platform will not accept this agent's credential (401).

    Its own exception because it's the one **claim** failure that backing off
    can't fix: a revoked, expired, or unknown credential answers 401 forever.
    A loop that retried it would hammer the platform until someone noticed;
    stopping and saying `nomikos pair` is the only useful response.
    """


class PageLeaseLost(PlatformError):
    """The platform says this agent is no longer holding the page it is reporting.

    Its own exception, mirroring the reason above: a 403 on the callback is
    terminal for *this* agent. The **lease** expired, the page went back to the
    queue, and someone else may already own it, so retrying the report is at
    best noise and at worst a race with the agent that now holds it.
    """


class AgentVersionRefused(PlatformError):
    """This build is below the **version floor** and may not claim (issue 055).

    Its own exception rather than a status code the loop inspects. `retryable`
    is always false on the wire: no amount of backing off turns this into
    work, so a claim loop that treated it as transient would spin forever
    against a platform that already said what to do.
    """

    def __init__(
        self,
        *,
        message: str,
        agent_version: str | None,
        minimum_version: str,
        latest_version: str,
        upgrade_command: str,
    ) -> None:
        super().__init__(message)
        self.agent_version = agent_version
        self.minimum_version = minimum_version
        self.latest_version = latest_version
        self.upgrade_command = upgrade_command

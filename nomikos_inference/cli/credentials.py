"""Where the device token lives on a researcher's machine, and who may read it.

ADR 0001: a long-lived credential now sits on a laptop where none existed
before, and whoever can read it can claim that researcher's `local` work until
the device is revoked. The control this layer owns is the mode bits (`0600` in
a `0700` directory), so another account on a shared machine cannot claim jobs
as them. Everything else (per-user scope, revocation on the next call) is
enforced by the platform.

The file is written through a temp file in the same directory, flushed and
`fsync`ed, then renamed, so an interrupted write leaves the previous credential
intact rather than a truncated one. That promise only covers a killed process,
not a lost power supply: `os.replace` orders the rename against the file's
metadata, so a rename that reaches disk before the bytes do leaves a
zero-length `device.json`, which reads as a corrupt credential rather than an
unpaired machine. `os.open` with `0o600` alone isn't enough either, since the
process umask is subtracted from that mode, hence the `fchmod` that follows it.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nomikos_inference.cli.api import InsecurePlatformURL, is_secure_platform_url
from nomikos_inference.cli.dto import _parse_datetime

# The same root the **Hub cache** uses (`nomikos_inference/hub/cache.py`). One directory
# in the researcher's home for everything the installed package writes, rather
# than one per subsystem.
DEFAULT_NOMIKOS_HOME = Path.home() / ".nomikos"
NOMIKOS_HOME_ENV = "NOMIKOS_HOME"

CREDENTIAL_FILENAME = "device.json"

CREDENTIAL_FILE_MODE = 0o600
CREDENTIAL_DIR_MODE = 0o700


def nomikos_home() -> Path:
    """The directory the CLI keeps its state in, overridable for tests.

    Read from the environment on every call rather than at import time, for the
    same reason `default_cache_root()` is: the process that sets it is often not
    the one that imported this module.
    """
    override = os.environ.get(NOMIKOS_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_NOMIKOS_HOME


def credential_path() -> Path:
    return nomikos_home() / CREDENTIAL_FILENAME


@dataclass(frozen=True)
class DeviceCredential:
    """One paired machine's **device token** and what it is bound to.

    `platform_url` is part of the record because the credential is only
    meaningful against the platform that minted it; pairing against a second one
    is a different device row, not a replacement for this one.
    """

    platform_url: str
    device_id: str
    device_token: str
    account_email: str
    device_name: str
    token_expires_at: datetime | None
    paired_at: datetime

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Whether the 180-day TTL has run out on this machine's clock.

        Only ever used to *explain* a rejection, never to pre-empt one: the
        platform is the authority on whether a credential is live, and a laptop
        with a wrong clock must not talk itself out of a working token.
        """
        if self.token_expires_at is None:
            return False
        return self.token_expires_at <= (now or datetime.now(UTC))

    def to_json(self) -> dict:
        return {
            "platform_url": self.platform_url,
            "device_id": self.device_id,
            "device_token": self.device_token,
            "account_email": self.account_email,
            "device_name": self.device_name,
            "token_expires_at": _isoformat(self.token_expires_at),
            "paired_at": _isoformat(self.paired_at),
        }


class CredentialError(RuntimeError):
    """The credential file exists but cannot be used as one."""


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _missing_fields(payload: dict) -> list[str]:
    """Required credential fields that are absent or empty."""
    return [
        field
        for field in ("platform_url", "device_id", "device_token")
        if not isinstance(payload.get(field), str) or not payload[field]
    ]


def load_credential(path: Path | None = None) -> DeviceCredential | None:
    """The stored credential, or `None` when this machine is not paired.

    A malformed file raises rather than reading as "not paired": silently
    starting a second pairing because a byte flipped would leave an orphan
    device row on the account and no explanation for it.

    The group/other readability check below applies on POSIX only: on Windows
    `stat` reports 0o666 for ordinary files, so the check would refuse every
    paired Windows helper.
    """
    target = path or credential_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CredentialError(f"Cannot read {target}: {exc}") from exc

    mode = file_mode(target)
    if os.name == "posix" and mode & 0o077:
        raise CredentialError(
            f"{target} is readable by others (mode {mode:04o}), so it cannot be "
            "trusted to hold only this machine's device token. "
            "Delete it and run `nomikos pair` again to replace it."
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CredentialError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CredentialError(f"{target} does not contain a device credential")

    if missing := _missing_fields(payload):
        raise CredentialError(f"{target} is missing {', '.join(missing)}")

    return DeviceCredential(
        platform_url=payload["platform_url"],
        device_id=payload["device_id"],
        device_token=payload["device_token"],
        account_email=payload.get("account_email") or "",
        device_name=payload.get("device_name") or "",
        token_expires_at=_parse_datetime(payload.get("token_expires_at")),
        paired_at=_parse_datetime(payload.get("paired_at")) or datetime.now(UTC),
    )


def save_credential(credential: DeviceCredential, path: Path | None = None) -> Path:
    """Write the credential owner-only, and return where it landed.

    Refuses to persist a token bound to a platform this CLI is not allowed to
    send it to. `nomikos run` reads `platform_url` back and claims against it,
    so writing one for a cleartext remote host would turn a URL rejected at
    pairing time into one trusted on every run afterward.
    """
    if not is_secure_platform_url(credential.platform_url):
        raise InsecurePlatformURL(
            f"Refusing to store a device token for {credential.platform_url}. "
            "A stored credential is claimed with on every later run, so the "
            "platform it names must be https (or http on localhost)."
        )
    target = path or credential_path()
    directory = target.parent
    directory.mkdir(mode=CREDENTIAL_DIR_MODE, parents=True, exist_ok=True)
    # `mkdir(mode=...)` does nothing when the directory already exists, and the
    # **Hub cache** may well have created it first. Narrow it either way: this
    # directory holds a bearer credential for the researcher's account.
    os.chmod(directory, CREDENTIAL_DIR_MODE)

    # A unique temporary per write, created exclusively: two concurrent
    # `nomikos pair` runs no longer truncate each other's file mid-write.
    descriptor, temporary_name = tempfile.mkstemp(dir=directory, prefix=f".{target.name}.")
    temporary = Path(temporary_name)
    try:
        # Belt and braces over `mkstemp`'s own 0600: the mode this file must
        # have should not depend on which call created it.
        os.fchmod(descriptor, CREDENTIAL_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(credential.to_json(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            # Before the `with` closes, and before the rename below: the bytes
            # have to be on the disk for the rename to be the atomic swap this
            # module advertises.
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(directory)
    return target


def _fsync_directory(directory: Path) -> None:
    """Persist the rename itself, where the platform supports it.

    Best effort on purpose. Directory `fsync` is meaningless on Windows and can
    be refused on some network filesystems, and neither is a reason to fail a
    pairing that has already written a good file.
    """
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms that cannot open a directory
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - filesystems that refuse it
        pass
    finally:
        os.close(descriptor)


def file_mode(path: Path) -> int:
    """Permission bits of `path`, for reporting and for tests to assert on."""
    return stat.S_IMODE(path.stat().st_mode)

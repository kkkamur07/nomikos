"""The launch check, the upgrade it may trigger, and the entry point around them.

`check_before_claiming` runs before every claim-capable command: it asks the
floor, upgrades through it, or gets out of the way when the platform cannot be
reached. `version` stays offline on purpose, and `main` dispatches. The
installer only ever names this distribution at a stated version; the
end-to-end upgrade needs a platform and an index, so it lives in the
integration lane.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nomikos_inference.cli import console as ui
from nomikos_inference.cli import upgrade as upgrade_module
from nomikos_inference.cli import version as version_module
from nomikos_inference.cli.api import AgentFloor, InsecurePlatformURL, PlatformError
from nomikos_inference.cli.credentials import CredentialError
from nomikos_inference.cli.main import main as cli_main
from nomikos_inference.cli.version import DISTRIBUTION_NAME


def _floor(**overrides: object) -> AgentFloor:
    fields: dict[str, object] = {
        "agent_version": "0.4.0",
        "minimum_version": "0.4.0",
        "latest_version": "0.5.0",
        "package": DISTRIBUTION_NAME,
        "refused": False,
        "outdated": False,
        "message": "",
        "upgrade_command": "",
    }
    fields.update(overrides)
    return AgentFloor(
        agent_version=str(fields["agent_version"]),
        minimum_version=str(fields["minimum_version"]),
        latest_version=str(fields["latest_version"]),
        package=str(fields["package"]),
        refused=bool(fields["refused"]),
        outdated=bool(fields["outdated"]),
        message=str(fields["message"]),
        upgrade_command=str(fields["upgrade_command"]),
    )


def _args(**overrides: object) -> argparse.Namespace:
    fields: dict[str, object] = {"api_url": None}
    fields.update(overrides)
    return argparse.Namespace(**fields)


# ---------------------------------------------------------------------------
# check_before_claiming
# ---------------------------------------------------------------------------
def test_an_insecure_platform_fails_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def _insecure(args: object, agent_version: str) -> AgentFloor:
        raise InsecurePlatformURL("no")

    monkeypatch.setattr(upgrade_module, "_ask", _insecure)

    assert upgrade_module.check_before_claiming(_args()) == ui.EXIT_FAILED


def test_upgrade_delegates_to_the_launch_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upgrade_module, "check_before_claiming", lambda args: 4)

    assert upgrade_module.run(_args()) == 4


def test_a_source_checkout_cannot_be_upgraded_in_place(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from nomikos_inference.cli.version import SOURCE_CHECKOUT_VERSION

    floor = _floor(refused=True, minimum_version="9.9.9")
    assert (
        upgrade_module._upgrade_or_stop(ui.out(), ui.err(), floor, SOURCE_CHECKOUT_VERSION)
        == ui.EXIT_FAILED
    )
    assert "source checkout" in capsys.readouterr().err


def test_a_repeated_failed_upgrade_stops(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(upgrade_module.UPGRADED_FROM_ENV, "0.1.0")
    floor = _floor(refused=True, minimum_version="9.9.9")

    assert upgrade_module._upgrade_or_stop(ui.out(), ui.err(), floor, "0.2.0") == ui.EXIT_FAILED
    assert "did not fix this" in capsys.readouterr().err


def test_an_upgrade_without_an_installer_stops(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(upgrade_module.UPGRADED_FROM_ENV, raising=False)
    monkeypatch.setattr(upgrade_module, "_installer_command", lambda requirement: None)
    floor = _floor(refused=True, minimum_version="9.9.9")

    assert upgrade_module._upgrade_or_stop(ui.out(), ui.err(), floor, "0.1.0") == ui.EXIT_FAILED
    assert "No installer is available" in capsys.readouterr().err


def test_a_failed_install_prints_the_manual_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    monkeypatch.delenv(upgrade_module.UPGRADED_FROM_ENV, raising=False)
    monkeypatch.setattr(
        upgrade_module, "_installer_command", lambda requirement: ["pip", "install", "x"]
    )
    monkeypatch.setattr(
        upgrade_module,
        "_install",
        lambda command: subprocess.CompletedProcess(command, 1, stdout="", stderr="nope"),
    )
    floor = _floor(refused=True, minimum_version="9.9.9")

    assert upgrade_module._upgrade_or_stop(ui.out(), ui.err(), floor, "0.1.0") == ui.EXIT_FAILED
    err = capsys.readouterr().err
    assert "failed" in err
    assert "nope" in err


def test_a_successful_install_that_cannot_restart_stops(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    monkeypatch.delenv(upgrade_module.UPGRADED_FROM_ENV, raising=False)
    monkeypatch.setattr(
        upgrade_module, "_installer_command", lambda requirement: ["pip", "install", "x"]
    )
    monkeypatch.setattr(
        upgrade_module,
        "_install",
        lambda command: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(upgrade_module, "_re_exec", lambda previous_version: None)
    floor = _floor(refused=True, minimum_version="9.9.9")

    assert upgrade_module._upgrade_or_stop(ui.out(), ui.err(), floor, "0.1.0") == ui.EXIT_FAILED
    assert "could not restart" in capsys.readouterr().err


def test_an_unreachable_platform_does_not_stop_claiming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _down(args: object, agent_version: str) -> AgentFloor:
        raise PlatformError("offline")

    monkeypatch.setattr(upgrade_module, "_ask", _down)

    assert upgrade_module.check_before_claiming(_args()) == ui.EXIT_OK


def test_a_current_agent_may_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upgrade_module, "_ask", lambda args, agent_version: _floor())

    assert upgrade_module.check_before_claiming(_args()) == ui.EXIT_OK


def test_an_outdated_agent_is_told_and_may_claim(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        upgrade_module,
        "_ask",
        lambda args, agent_version: _floor(outdated=True, latest_version="0.9.0"),
    )

    assert upgrade_module.check_before_claiming(_args()) == ui.EXIT_OK
    assert "0.9.0" in capsys.readouterr().out


def test_ask_queries_the_floor_without_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _FloorClient:
        def __init__(self, base_url: str, **kwargs: object) -> None:
            seen["url"] = base_url

        def read_agent_floor(self, *, agent_version: str) -> AgentFloor:
            seen["version"] = agent_version
            return _floor()

    monkeypatch.setattr(upgrade_module, "PlatformClient", _FloorClient)

    floor = upgrade_module._ask(_args(api_url="https://floor.example.com"), "0.4.0")

    assert seen == {"url": "https://floor.example.com", "version": "0.4.0"}
    assert floor.outdated is False


def test_a_credential_without_an_expiry_never_counts_as_expired() -> None:
    from datetime import UTC, datetime

    from nomikos_inference.cli.credentials import DeviceCredential

    credential = DeviceCredential(
        platform_url="https://platform.example.com",
        device_id="d",
        device_token="t",
        account_email="",
        device_name="",
        token_expires_at=None,
        paired_at=datetime.now(UTC),
    )

    assert credential.is_expired() is False


def test_a_refused_agent_goes_through_the_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    floor = _floor(refused=True, minimum_version="9.9.9")
    monkeypatch.setattr(upgrade_module, "_ask", lambda args, agent_version: floor)
    monkeypatch.setattr(upgrade_module, "_upgrade_or_stop", lambda *args: seen.append(args) or 5)

    assert upgrade_module.check_before_claiming(_args()) == 5
    assert seen[0][2] is floor


# ---------------------------------------------------------------------------
# platform_url_for
# ---------------------------------------------------------------------------
def test_an_explicit_flag_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        upgrade_module.platform_url_for(_args(api_url="https://staging.example.com/"))
        == "https://staging.example.com"
    )


def test_the_paired_platform_is_next(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from nomikos_inference.cli.credentials import DeviceCredential

    credential = DeviceCredential(
        platform_url="https://paired.example.com",
        device_id="d",
        device_token="t",
        account_email="",
        device_name="",
        token_expires_at=None,
        paired_at=datetime.now(UTC),
    )
    monkeypatch.setattr(upgrade_module, "load_credential", lambda: credential)

    assert upgrade_module.platform_url_for(_args()) == "https://paired.example.com"


def test_an_unreadable_credential_falls_through_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unreadable() -> None:
        raise CredentialError("bad")

    monkeypatch.setattr(upgrade_module, "load_credential", _unreadable)

    from nomikos_inference.cli.api import default_platform_url

    assert upgrade_module.platform_url_for(_args()) == default_platform_url()


# ---------------------------------------------------------------------------
# _install and _installer_command
# ---------------------------------------------------------------------------
def test_a_timed_out_installer_is_a_failed_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _hang(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 600.0)  # type: ignore[arg-type]

    monkeypatch.setattr(upgrade_module.subprocess, "run", _hang)

    completed = upgrade_module._install(["pip", "install", "x"])

    assert completed.returncode == 1
    assert "did not finish" in (completed.stderr or "")


def test_pip_is_preferred_when_present() -> None:
    command = upgrade_module._installer_command("nomikos-inference>=0.4.0")

    assert command is not None
    assert command[:3] == [sys.executable, "-m", "pip"]


def test_uv_is_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upgrade_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(upgrade_module.shutil, "which", lambda name: "/bin/uv")

    command = upgrade_module._installer_command("nomikos-inference>=0.4.0")

    assert command is not None
    assert command[:2] == ["/bin/uv", "pip"]


def test_no_installer_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upgrade_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(upgrade_module.shutil, "which", lambda name: None)

    assert upgrade_module._installer_command("nomikos-inference>=0.4.0") is None


# ---------------------------------------------------------------------------
# _re_exec
# ---------------------------------------------------------------------------
def test_re_exec_through_the_console_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}
    script = tmp_path / "nomikos"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)

    def _capture(path: str, args: list[str], env: dict[str, str]) -> None:
        seen["path"] = path
        seen["args"] = args
        seen["env"] = env
        raise OSError("no exec in tests")

    monkeypatch.setattr(upgrade_module.os, "execve", _capture)
    monkeypatch.setattr(sys, "argv", [str(script), "run", "--exit-when-empty"])

    upgrade_module._re_exec("0.4.0")

    assert seen["path"] == str(script)
    assert seen["args"] == [str(script), "run", "--exit-when-empty"]
    assert seen["env"]["NOMIKOS_UPGRADED_FROM"] == "0.4.0"


def test_re_exec_without_a_script_uses_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _capture(path: str, args: list[str], env: dict[str, str]) -> None:
        seen["path"] = path
        seen["args"] = args
        raise OSError("no exec in tests")

    monkeypatch.setattr(upgrade_module.os, "execve", _capture)
    monkeypatch.setattr(sys, "argv", [""])

    upgrade_module._re_exec("0.4.0")

    assert seen["path"] == sys.executable
    assert seen["args"][:3] == [sys.executable, "-m", "nomikos_inference.cli"]


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
def test_version_reports_the_installation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from nomikos_inference.cli import credentials as credentials_module

    monkeypatch.setattr(version_module, "installed_version", lambda: "0.4.0")
    monkeypatch.setattr(credentials_module, "load_credential", lambda: None)

    assert version_module.run(argparse.Namespace()) == ui.EXIT_OK
    assert "0.4.0" in capsys.readouterr().out


def test_version_reports_a_pairing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from datetime import UTC, datetime

    from nomikos_inference.cli import credentials as credentials_module
    from nomikos_inference.cli.credentials import DeviceCredential

    credential = DeviceCredential(
        platform_url="https://platform.example.com",
        device_id="d",
        device_token="t",
        account_email="r@example.com",
        device_name="laptop",
        token_expires_at=None,
        paired_at=datetime.now(UTC),
    )
    monkeypatch.setattr(version_module, "installed_version", lambda: "0.4.0")
    monkeypatch.setattr(credentials_module, "load_credential", lambda: credential)

    assert version_module.run(argparse.Namespace()) == ui.EXIT_OK
    assert "r@example.com" in capsys.readouterr().out


def test_version_reports_an_unreadable_credential(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from nomikos_inference.cli import credentials as credentials_module

    def _unreadable() -> None:
        raise CredentialError("bad")

    monkeypatch.setattr(version_module, "installed_version", lambda: "0.4.0")
    monkeypatch.setattr(credentials_module, "load_credential", _unreadable)

    assert version_module.run(argparse.Namespace()) == ui.EXIT_OK
    assert "unreadable credential" in capsys.readouterr().out


def test_version_names_a_source_checkout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from nomikos_inference.cli import credentials as credentials_module
    from nomikos_inference.cli.version import SOURCE_CHECKOUT_VERSION

    monkeypatch.setattr(version_module, "installed_version", lambda: SOURCE_CHECKOUT_VERSION)
    monkeypatch.setattr(credentials_module, "load_credential", lambda: None)

    assert version_module.run(argparse.Namespace()) == ui.EXIT_OK
    assert "source checkout" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def test_main_without_a_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main([]) == ui.EXIT_OK
    assert "nomikos" in capsys.readouterr().out


def test_main_runs_version() -> None:
    assert cli_main(["version"]) == ui.EXIT_OK


def test_main_gates_claiming_on_the_launch_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.cli import upgrade as upgrade_command

    monkeypatch.setattr(upgrade_command, "check_before_claiming", lambda args: 1)

    assert cli_main(["run", "--exit-when-empty"]) == 1


def test_main_claims_when_the_check_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from nomikos_inference.cli import upgrade as upgrade_command
    from nomikos_inference.cli.main import _COMMANDS

    monkeypatch.setattr(upgrade_command, "check_before_claiming", lambda args: ui.EXIT_OK)
    summary, add_args, _ = _COMMANDS["run"]
    monkeypatch.setitem(_COMMANDS, "run", (summary, add_args, lambda args: 3))

    assert cli_main(["run", "--exit-when-empty"]) == 3


def test_main_turns_an_interrupt_into_130(monkeypatch: pytest.MonkeyPatch) -> None:
    from nomikos_inference.cli.main import _COMMANDS

    def _interrupted(args: argparse.Namespace) -> int:
        raise KeyboardInterrupt

    summary, add_args, _ = _COMMANDS["pair"]
    monkeypatch.setitem(_COMMANDS, "pair", (summary, add_args, _interrupted))

    assert cli_main(["pair"]) == ui.EXIT_INTERRUPTED


# ---------------------------------------------------------------------------
# credentials: the mode check and the rest of load_credential
# ---------------------------------------------------------------------------
def _write_credential(tmp_path: Path, payload: object, *, mode: int = 0o600) -> Path:
    import json
    import os

    target = tmp_path / "device.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(target, mode)
    return target


def _credential_payload() -> dict[str, object]:
    return {
        "platform_url": "https://platform.example.com",
        "device_id": "device-1",
        "device_token": "secret-token",
    }


def test_a_world_readable_credential_is_refused(tmp_path: Path) -> None:
    from nomikos_inference.cli.credentials import CredentialError, load_credential

    target = _write_credential(tmp_path, _credential_payload(), mode=0o644)

    with pytest.raises(CredentialError, match="readable by others"):
        load_credential(target)


def test_a_group_readable_credential_is_refused(tmp_path: Path) -> None:
    from nomikos_inference.cli.credentials import CredentialError, load_credential

    target = _write_credential(tmp_path, _credential_payload(), mode=0o640)

    with pytest.raises(CredentialError, match="readable by others"):
        load_credential(target)


def test_an_owner_only_credential_loads(tmp_path: Path) -> None:
    from nomikos_inference.cli.credentials import load_credential

    stored = load_credential(_write_credential(tmp_path, _credential_payload()))

    assert stored is not None
    assert stored.device_token == "secret-token"


def test_a_missing_credential_is_not_paired(tmp_path: Path) -> None:
    from nomikos_inference.cli.credentials import load_credential

    assert load_credential(tmp_path / "device.json") is None


def test_a_non_json_credential_is_a_credential_error(tmp_path: Path) -> None:
    import os

    from nomikos_inference.cli.credentials import CredentialError, load_credential

    target = tmp_path / "device.json"
    target.write_text("not json", encoding="utf-8")
    os.chmod(target, 0o600)

    with pytest.raises(CredentialError, match="not valid JSON"):
        load_credential(target)


def test_a_non_dict_credential_is_a_credential_error(tmp_path: Path) -> None:
    from nomikos_inference.cli.credentials import CredentialError, load_credential

    with pytest.raises(CredentialError, match="does not contain a device credential"):
        load_credential(_write_credential(tmp_path, [1, 2]))


def test_a_credential_missing_fields_names_them(tmp_path: Path) -> None:
    from nomikos_inference.cli.credentials import CredentialError, load_credential

    with pytest.raises(CredentialError, match="missing device_token"):
        load_credential(_write_credential(tmp_path, {"platform_url": "x", "device_id": "y"}))


def test_an_unreadable_credential_is_a_credential_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nomikos_inference.cli.credentials import CredentialError, load_credential

    def _unreadable(self: Path, *args: object, **kwargs: object) -> str:
        raise OSError("denied")

    monkeypatch.setattr(Path, "read_text", _unreadable)
    target = _write_credential(tmp_path, _credential_payload())

    with pytest.raises(CredentialError, match="Cannot read"):
        load_credential(target)


def test_a_failed_save_leaves_no_temporary_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nomikos_inference.cli import credentials as credentials_module
    from nomikos_inference.cli.credentials import DeviceCredential, save_credential

    credential = DeviceCredential(
        platform_url="https://platform.example.com",
        device_id="device-1",
        device_token="secret-token",
        account_email="",
        device_name="",
        token_expires_at=None,
        paired_at=datetime.now(UTC),
    )
    target = tmp_path / "device.json"

    def _die(source: object, destination: object) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(credentials_module.os, "replace", _die)
    with pytest.raises(OSError):
        save_credential(credential, target)

    assert list(tmp_path.glob(".device.json.*")) == []
    assert not target.exists()


def test_a_saved_credential_leaves_no_temporary_behind(tmp_path: Path) -> None:
    from nomikos_inference.cli.credentials import (
        CREDENTIAL_FILE_MODE,
        DeviceCredential,
        file_mode,
        load_credential,
        save_credential,
    )

    credential = DeviceCredential(
        platform_url="https://platform.example.com",
        device_id="device-1",
        device_token="secret-token",
        account_email="",
        device_name="",
        token_expires_at=None,
        paired_at=datetime.now(UTC),
    )
    target = tmp_path / "device.json"

    save_credential(credential, target)

    assert list(tmp_path.glob(".device.json.*")) == []
    assert file_mode(target) == CREDENTIAL_FILE_MODE
    assert load_credential(target) is not None


def test_nomikos_home_prefers_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from nomikos_inference.cli.credentials import credential_path, nomikos_home

    monkeypatch.delenv("NOMIKOS_HOME", raising=False)
    assert nomikos_home().name == ".nomikos"

    monkeypatch.setenv("NOMIKOS_HOME", "~/custom-nomikos")
    assert str(nomikos_home()).endswith("custom-nomikos")
    assert credential_path().name == "device.json"

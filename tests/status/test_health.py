"""Tests para src/status/health.py — probes + cascada de decisión de estado."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from src.status.health import (
    check_calibration_loadable,
    check_cpu_temp_ok,
    check_disk_ok,
    check_hailo_temp_ok,
    check_internet,
    decide_state,
)
from src.status.led import LedState


# ---------------------------------------------------------------------------
# decide_state — priority cascade
# ---------------------------------------------------------------------------


def test_all_ok_returns_ok():
    assert decide_state() is LedState.OK


def test_init_failed_maps_to_hardware_fault_and_dominates():
    assert (
        decide_state(
            init_failed=True,
            hardware_ok=True,
            pipeline_ok=False,
            internet_ok=False,
            cloud_connected=False,
            provisioned=False,
        )
        is LedState.HARDWARE_FAULT
    )


def test_hardware_fault_dominates_pipeline_and_below():
    assert (
        decide_state(
            hardware_ok=False,
            pipeline_ok=False,
            internet_ok=False,
            cloud_connected=False,
        )
        is LedState.HARDWARE_FAULT
    )


def test_pipeline_dominates_internet():
    assert decide_state(pipeline_ok=False, internet_ok=False) is LedState.SOFTWARE_FAULT


def test_no_internet_dominates_cloud():
    assert (
        decide_state(internet_ok=False, cloud_connected=False) is LedState.NO_INTERNET
    )


def test_no_cloud_when_internet_ok():
    assert decide_state(cloud_connected=False) is LedState.NO_CLOUD


def test_unprovisioned_lowest_priority_after_cloud():
    assert decide_state(provisioned=False) is LedState.UNPROVISIONED


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, LedState.OK),
        ({"hardware_ok": False}, LedState.HARDWARE_FAULT),
        ({"pipeline_ok": False}, LedState.SOFTWARE_FAULT),
        ({"internet_ok": False}, LedState.NO_INTERNET),
        ({"cloud_connected": False}, LedState.NO_CLOUD),
        ({"provisioned": False}, LedState.UNPROVISIONED),
        ({"init_failed": True}, LedState.HARDWARE_FAULT),
    ],
)
def test_decide_state_table(kwargs, expected):
    assert decide_state(**kwargs) is expected


# ---------------------------------------------------------------------------
# check_disk_ok
# ---------------------------------------------------------------------------


def test_check_disk_ok_above_threshold(monkeypatch):
    class _Usage:
        total = 100 * 1024 * 1024 * 1024  # 100 GB
        free = 50 * 1024 * 1024 * 1024  # 50 GB free → 50%

    monkeypatch.setattr("shutil.disk_usage", lambda _p: _Usage())
    assert check_disk_ok() is True


def test_check_disk_ok_below_threshold(monkeypatch):
    class _Usage:
        total = 100 * 1024 * 1024 * 1024
        free = 5 * 1024 * 1024 * 1024  # 5% — below 10% threshold

    monkeypatch.setattr("shutil.disk_usage", lambda _p: _Usage())
    assert check_disk_ok() is False


def test_check_disk_ok_fails_open_on_error(monkeypatch):
    """A probe error must NOT trigger a fault — fail-open semantics."""
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda _p: (_ for _ in ()).throw(OSError("no such path")),
    )
    assert check_disk_ok() is True


# ---------------------------------------------------------------------------
# check_cpu_temp_ok
# ---------------------------------------------------------------------------


def test_check_cpu_temp_ok_below_threshold(tmp_path, monkeypatch):
    f = tmp_path / "temp"
    f.write_text("65000\n")  # 65 °C

    real_open = open

    def _fake_open(path, *a, **k):
        if str(path) == "/sys/class/thermal/thermal_zone0/temp":
            return real_open(f, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert check_cpu_temp_ok() is True


def test_check_cpu_temp_ok_above_threshold(tmp_path, monkeypatch):
    f = tmp_path / "temp"
    f.write_text("85000\n")  # 85 °C — above 80 °C threshold

    real_open = open

    def _fake_open(path, *a, **k):
        if str(path) == "/sys/class/thermal/thermal_zone0/temp":
            return real_open(f, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert check_cpu_temp_ok() is False


def test_check_cpu_temp_ok_fails_open_when_unreadable(monkeypatch):
    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError),
    )
    assert check_cpu_temp_ok() is True


# ---------------------------------------------------------------------------
# check_hailo_temp_ok
# ---------------------------------------------------------------------------


def test_check_hailo_temp_ok_parses_below_threshold():
    fake_output = "Identifying board\nDie Temperature: 50.2 C\nFW: 4.23"
    with patch("subprocess.run") as mock:
        mock.return_value.stdout = fake_output
        mock.return_value.stderr = ""
        assert check_hailo_temp_ok() is True


def test_check_hailo_temp_ok_parses_above_threshold():
    fake_output = "Die Temperature: 90.5 C"
    with patch("subprocess.run") as mock:
        mock.return_value.stdout = fake_output
        mock.return_value.stderr = ""
        assert check_hailo_temp_ok() is False


def test_check_hailo_temp_ok_no_cli_returns_true():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        # Running off-RPi: no hailortcli — must not light a fault.
        assert check_hailo_temp_ok() is True


def test_check_hailo_temp_ok_unparseable_returns_true():
    with patch("subprocess.run") as mock:
        mock.return_value.stdout = "no temperature line in this output"
        mock.return_value.stderr = ""
        assert check_hailo_temp_ok() is True


# ---------------------------------------------------------------------------
# check_calibration_loadable
# ---------------------------------------------------------------------------


def test_check_calibration_no_path_is_ok():
    assert check_calibration_loadable(None) is True
    assert check_calibration_loadable("") is True


def test_check_calibration_path_exists(tmp_path):
    f = tmp_path / "calib.npz"
    f.write_bytes(b"placeholder")
    assert check_calibration_loadable(str(f)) is True


def test_check_calibration_path_missing(tmp_path):
    missing = tmp_path / "does-not-exist.npz"
    assert check_calibration_loadable(str(missing)) is False


# ---------------------------------------------------------------------------
# check_internet
# ---------------------------------------------------------------------------


def test_check_internet_success(monkeypatch):
    class _Sock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _Sock())
    assert check_internet() is True


def test_check_internet_failure(monkeypatch):
    def _raise(*_a, **_k):
        raise OSError("network unreachable")

    monkeypatch.setattr(socket, "create_connection", _raise)
    assert check_internet() is False

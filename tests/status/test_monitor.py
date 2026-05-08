"""Tests para src/status/monitor.py — HealthMonitor end-to-end."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.status.led import LedState, StatusLED
from src.status.monitor import HealthMonitor, HealthSignals


class _NullLed(StatusLED):
    """LED that records state transitions without touching GPIO or threads."""

    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        # Bypass StatusLED.__init__ — we don't need its worker thread for the
        # monitor tests; just record set_state calls.
        self.history: list[LedState] = [LedState.OFF]
        self._state = LedState.OFF

    def set_state(self, state: LedState) -> None:
        self._state = state
        self.history.append(state)

    def get_state(self) -> LedState:
        return self._state

    def close(self) -> None:
        return None


@pytest.fixture
def all_probes_ok():
    """Patch every health probe so only ``HealthSignals`` drives the result."""
    with patch("src.status.monitor.check_cpu_temp_ok", return_value=True), \
         patch("src.status.monitor.check_hailo_temp_ok", return_value=True), \
         patch("src.status.monitor.check_disk_ok", return_value=True), \
         patch("src.status.monitor.check_calibration_loadable", return_value=True), \
         patch("src.status.monitor.check_internet", return_value=True):
        yield


def test_evaluate_once_ok(all_probes_ok):
    led = _NullLed()
    signals = HealthSignals(
        last_loop_ts=time.time(),
        capture_ok=True,
        detect_ok=True,
        mqtt_connected=True,
        provisioned=True,
        boot_complete=True,
    )
    monitor = HealthMonitor(led=led, signals=signals)
    assert monitor.evaluate_once() is LedState.OK
    assert led.get_state() is LedState.OK


def test_evaluate_once_hardware_fault_when_capture_failed(all_probes_ok):
    led = _NullLed()
    signals = HealthSignals(
        last_loop_ts=time.time(),
        capture_ok=False,
        mqtt_connected=True,
    )
    monitor = HealthMonitor(led=led, signals=signals)
    assert monitor.evaluate_once() is LedState.HARDWARE_FAULT


def test_evaluate_once_pipeline_stalled_after_threshold(all_probes_ok):
    led = _NullLed()
    # Loop ts well in the past, beyond PIPELINE_STALL_S.
    signals = HealthSignals(
        last_loop_ts=time.time() - 30.0,
        mqtt_connected=True,
    )
    monitor = HealthMonitor(led=led, signals=signals)
    assert monitor.evaluate_once() is LedState.SOFTWARE_FAULT


def test_evaluate_once_no_pipeline_check_during_boot(all_probes_ok):
    """Before the first loop iteration ``last_loop_ts`` is 0 — don't fault."""
    led = _NullLed()
    signals = HealthSignals(last_loop_ts=0.0, mqtt_connected=True)
    monitor = HealthMonitor(led=led, signals=signals)
    assert monitor.evaluate_once() is LedState.OK


def test_evaluate_once_no_internet():
    led = _NullLed()
    signals = HealthSignals(
        last_loop_ts=time.time(),
        mqtt_connected=False,
    )
    with patch("src.status.monitor.check_cpu_temp_ok", return_value=True), \
         patch("src.status.monitor.check_hailo_temp_ok", return_value=True), \
         patch("src.status.monitor.check_disk_ok", return_value=True), \
         patch("src.status.monitor.check_calibration_loadable", return_value=True), \
         patch("src.status.monitor.check_internet", return_value=False):
        monitor = HealthMonitor(led=led, signals=signals)
        assert monitor.evaluate_once() is LedState.NO_INTERNET


def test_evaluate_once_no_cloud_when_internet_ok(all_probes_ok):
    led = _NullLed()
    signals = HealthSignals(
        last_loop_ts=time.time(),
        mqtt_connected=False,  # MQTT not connected
        provisioned=True,
    )
    monitor = HealthMonitor(led=led, signals=signals)
    assert monitor.evaluate_once() is LedState.NO_CLOUD


def test_internet_probe_is_cached(all_probes_ok, monkeypatch):
    """Repeated ticks within the cache window must not re-probe."""
    led = _NullLed()
    signals = HealthSignals(last_loop_ts=time.time(), mqtt_connected=True)
    monitor = HealthMonitor(
        led=led, signals=signals, internet_probe_interval_s=60.0,
    )
    call_count = {"n": 0}

    def _counting_probe():
        call_count["n"] += 1
        return True

    with patch("src.status.monitor.check_internet", side_effect=_counting_probe):
        for _ in range(5):
            monitor.evaluate_once()
    assert call_count["n"] == 1  # only the first tick probed


def test_start_stop_runs_at_least_one_tick(all_probes_ok):
    led = _NullLed()
    signals = HealthSignals(
        last_loop_ts=time.time(),
        mqtt_connected=True,
        boot_complete=True,
    )
    monitor = HealthMonitor(led=led, signals=signals, poll_interval_s=0.02)
    monitor.start()
    deadline = time.time() + 1.0
    while time.time() < deadline and led.get_state() is LedState.OFF:
        time.sleep(0.01)
    monitor.stop()
    assert led.get_state() is LedState.OK


def test_stop_is_idempotent():
    led = _NullLed()
    signals = HealthSignals()
    monitor = HealthMonitor(led=led, signals=signals)
    monitor.stop()  # never started
    monitor.start()
    monitor.stop()
    monitor.stop()  # safe to call repeatedly

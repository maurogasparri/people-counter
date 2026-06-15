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
    with (
        patch("src.status.monitor.check_cpu_temp_ok", return_value=True),
        patch("src.status.monitor.check_hailo_temp_ok", return_value=True),
        patch("src.status.monitor.check_disk_ok", return_value=True),
        patch("src.status.monitor.check_calibration_loadable", return_value=True),
        patch("src.status.monitor.check_internet", return_value=True),
    ):
        yield


def test_evaluate_once_ok(all_probes_ok):
    led = _NullLed()
    signals = HealthSignals(
        last_loop_ts=time.time(),
        capture_ok=True,
        detect_ok=True,
        mqtt_connected=True,
        provisioned=True,
        init_complete=True,
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


def test_init_failed_flag_shows_hardware_fault(all_probes_ok):
    """main() setea init_failed en el except de init antes de morir — el
    tick tiene que mostrar rojo fijo. Un crash/wedge de init se reporta como
    HARDWARE_FAULT (no hay estado 'boot' dedicado: el LED no puede señalizar
    un fallo previo a su propio init). Regresión: sin este wiring un crash de
    init mostraba verde (NO_CLOUD) — el operador leía 'problema de internet'
    frente a una falla de arranque."""
    led = _NullLed()
    signals = HealthSignals(init_failed=True)
    monitor = HealthMonitor(led=led, signals=signals)
    assert monitor.evaluate_once() is LedState.HARDWARE_FAULT


def test_init_grace_expired_without_init_complete_shows_hardware_fault(all_probes_ok):
    """Init wedgeado (load_model/capture.open colgados): si venció el grace
    sin init_complete, el LED pasa a rojo (HARDWARE_FAULT) en vez de quedarse
    verde para siempre."""
    led = _NullLed()
    signals = HealthSignals(init_complete=False)
    monitor = HealthMonitor(led=led, signals=signals, init_grace_s=0.0)
    assert monitor.evaluate_once() is LedState.HARDWARE_FAULT


def test_init_grace_not_expired_keeps_normal_cascade(all_probes_ok):
    """Durante el grace (init normal en curso), la cascada sigue normal —
    no flaggear falla de init mientras el operador espera el arranque."""
    led = _NullLed()
    signals = HealthSignals(init_complete=False, mqtt_connected=True)
    monitor = HealthMonitor(led=led, signals=signals, init_grace_s=120.0)
    state = monitor.evaluate_once()
    assert state is not LedState.HARDWARE_FAULT


def test_init_complete_disables_grace_path(all_probes_ok):
    """Con init_complete=True el grace nunca dispara, aunque haya pasado."""
    led = _NullLed()
    signals = HealthSignals(
        last_loop_ts=time.time(),
        init_complete=True,
        mqtt_connected=True,
    )
    monitor = HealthMonitor(led=led, signals=signals, init_grace_s=0.0)
    assert monitor.evaluate_once() is LedState.OK


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
    with (
        patch("src.status.monitor.check_cpu_temp_ok", return_value=True),
        patch("src.status.monitor.check_hailo_temp_ok", return_value=True),
        patch("src.status.monitor.check_disk_ok", return_value=True),
        patch("src.status.monitor.check_calibration_loadable", return_value=True),
        patch("src.status.monitor.check_internet", return_value=False),
    ):
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
        led=led,
        signals=signals,
        internet_probe_interval_s=60.0,
    )
    call_count = {"n": 0}

    def _counting_probe():
        call_count["n"] += 1
        return True

    with patch("src.status.monitor.check_internet", side_effect=_counting_probe):
        for _ in range(5):
            monitor.evaluate_once()
    assert call_count["n"] == 1  # only the first tick probed


def test_hailo_probe_is_cached():
    """El probe de Hailo (subprocess hailortcli, hasta 5s) NO debe correr en
    cada tick — se cachea en su propia cadencia. Sin esto clavaria el thread
    monitor y la cadencia del probe de internet."""
    led = _NullLed()
    signals = HealthSignals(last_loop_ts=time.time(), mqtt_connected=True)
    monitor = HealthMonitor(
        led=led,
        signals=signals,
        hailo_probe_interval_s=60.0,
    )
    call_count = {"n": 0}

    def _counting_hailo():
        call_count["n"] += 1
        return True

    with (
        patch("src.status.monitor.check_cpu_temp_ok", return_value=True),
        patch("src.status.monitor.check_disk_ok", return_value=True),
        patch("src.status.monitor.check_calibration_loadable", return_value=True),
        patch("src.status.monitor.check_internet", return_value=True),
        patch("src.status.monitor.check_hailo_temp_ok", side_effect=_counting_hailo),
    ):
        for _ in range(5):
            monitor.evaluate_once()
    assert call_count["n"] == 1  # solo el primer tick probó el subprocess


def test_hailo_fault_surfaces_via_cache():
    """Un Hailo sobre-temperatura cacheado igual dispara HARDWARE_FAULT."""
    led = _NullLed()
    signals = HealthSignals(last_loop_ts=time.time(), mqtt_connected=True)
    monitor = HealthMonitor(led=led, signals=signals)
    with (
        patch("src.status.monitor.check_cpu_temp_ok", return_value=True),
        patch("src.status.monitor.check_disk_ok", return_value=True),
        patch("src.status.monitor.check_calibration_loadable", return_value=True),
        patch("src.status.monitor.check_internet", return_value=True),
        patch("src.status.monitor.check_hailo_temp_ok", return_value=False),
    ):
        assert monitor.evaluate_once() is LedState.HARDWARE_FAULT


def test_start_stop_runs_at_least_one_tick(all_probes_ok):
    led = _NullLed()
    signals = HealthSignals(
        last_loop_ts=time.time(),
        mqtt_connected=True,
        init_complete=True,
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

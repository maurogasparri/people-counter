"""Tests para src/status/led.py — driver del status LED RGB."""

from __future__ import annotations

import threading
import time

from src.status.led import LedState, StatusLED


class _FakeLed:
    """Records on/off/close calls; substitute for ``gpiozero.LED``."""

    def __init__(self) -> None:
        self.is_on = False
        self.closed = False
        self.history: list[bool] = []

    def on(self) -> None:
        self.is_on = True
        self.history.append(True)

    def off(self) -> None:
        self.is_on = False
        self.history.append(False)

    def close(self) -> None:
        self.closed = True


def _make_led(
    blink_period_s: float = 0.04,
) -> tuple[StatusLED, _FakeLed, _FakeLed, _FakeLed]:
    r, g, b = _FakeLed(), _FakeLed(), _FakeLed()
    led = StatusLED(blink_period_s=blink_period_s, backend=(r, g, b))
    return led, r, g, b


def _wait_for(predicate, timeout_s: float = 1.0) -> bool:
    """Poll until ``predicate()`` is true or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_initial_state_is_off():
    led, r, g, b = _make_led()
    try:
        assert led.get_state() is LedState.OFF
        # Allow the worker thread one tick to apply OFF outputs.
        assert _wait_for(lambda: not (r.is_on or g.is_on or b.is_on))
    finally:
        led.close()


def test_solid_blue_for_ok():
    led, r, g, b = _make_led()
    try:
        led.set_state(LedState.OK)
        assert _wait_for(lambda: b.is_on and not r.is_on and not g.is_on)
    finally:
        led.close()


def test_solid_red_for_hardware_fault():
    led, r, g, b = _make_led()
    try:
        led.set_state(LedState.HARDWARE_FAULT)
        # Rojo puro fijo (R on, G+B off). Regresión: antes era amarillo (R+G),
        # que se veía verdoso por el balance de brillo. Sin verde nunca.
        assert _wait_for(lambda: r.is_on and not g.is_on and not b.is_on)
    finally:
        led.close()


def test_software_fault_is_red_not_yellow():
    led, r, g, b = _make_led(blink_period_s=0.04)
    try:
        led.set_state(LedState.SOFTWARE_FAULT)  # rojo parpadeante
        # El verde NUNCA debe encenderse (antes era amarillo blink = R+G).
        time.sleep(0.2)
        assert not any(g.history)
        assert not any(b.history)
        assert any(r.history)  # el rojo sí togglea
    finally:
        led.close()


def test_green_solid_for_no_cloud():
    led, r, g, b = _make_led()
    try:
        led.set_state(LedState.NO_CLOUD)
        assert _wait_for(lambda: g.is_on and not r.is_on and not b.is_on)
    finally:
        led.close()


def test_blink_state_toggles_outputs():
    led, r, g, b = _make_led(blink_period_s=0.04)
    try:
        led.set_state(LedState.SOFTWARE_FAULT)  # rojo parpadeante
        # Sample the worker for several blink half-periods.
        time.sleep(0.25)
        # During SOFTWARE_FAULT we expect the worker to have toggled R between
        # on and off at least once (rojo puro — el verde queda apagado).
        assert any(r.history) and any(not v for v in r.history)
    finally:
        led.close()


def test_set_state_idempotent_does_not_re_log():
    led, _r, _g, _b = _make_led()
    try:
        led.set_state(LedState.OK)
        first = led.get_state()
        # A second identical call should be a no-op (no exceptions, same state).
        led.set_state(LedState.OK)
        assert led.get_state() is first
    finally:
        led.close()


def test_state_transition_red_to_blue():
    led, r, g, b = _make_led()
    try:
        led.set_state(LedState.HARDWARE_FAULT)
        assert _wait_for(lambda: r.is_on and not b.is_on)
        led.set_state(LedState.OK)
        assert _wait_for(lambda: b.is_on and not r.is_on)
    finally:
        led.close()


def test_close_turns_off_and_releases():
    led, r, g, b = _make_led()
    led.set_state(LedState.OK)
    assert _wait_for(lambda: b.is_on)
    led.close()
    assert not (r.is_on or g.is_on or b.is_on)
    assert r.closed and g.closed and b.closed


def test_close_is_idempotent():
    led, _r, _g, _b = _make_led()
    led.close()
    led.close()  # second call must not raise


def test_disabled_when_backend_none(monkeypatch):
    """If gpiozero can't be opened, LED degrades to a logging no-op."""
    # Force the default-backend opener to return None and verify we still
    # construct + transition + close without raising.
    from src.status import led as led_mod

    monkeypatch.setattr(
        led_mod,
        "_open_default_backend",
        lambda *_a, **_k: None,
    )
    s = StatusLED()
    try:
        s.set_state(LedState.OK)
        s.set_state(LedState.HARDWARE_FAULT)
    finally:
        s.close()


def test_thread_is_daemon():
    led, _r, _g, _b = _make_led()
    try:
        # Daemon ensures interpreter exit doesn't hang on a stuck LED thread.
        assert led._thread.daemon is True  # noqa: SLF001 (private intentionally)
    finally:
        led.close()


def test_no_backend_calls_after_close():
    led, r, g, b = _make_led()
    led.set_state(LedState.OK)
    assert _wait_for(lambda: b.is_on)
    led.close()
    history_len_after_close = len(r.history) + len(g.history) + len(b.history)
    # Calling set_state after close should not produce more writes.
    led.set_state(LedState.HARDWARE_FAULT)
    time.sleep(0.05)
    assert len(r.history) + len(g.history) + len(b.history) == history_len_after_close


def test_concurrent_state_changes_are_safe():
    led, _r, _g, _b = _make_led()
    states = [
        LedState.OK,
        LedState.NO_INTERNET,
        LedState.HARDWARE_FAULT,
        LedState.SOFTWARE_FAULT,
    ]

    def worker(s: LedState):
        for _ in range(20):
            led.set_state(s)

    try:
        threads = [threading.Thread(target=worker, args=(s,)) for s in states]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Final state must be one of the inputs (no corruption).
        assert led.get_state() in states
    finally:
        led.close()

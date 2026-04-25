"""RGB status LED for field diagnosis.

8-state pattern aligned with FootfallCam's operator codes — retail staff
can interpret device health from the LED color/blink without SSH access.

Hardware: 3mm common-cathode RGB on the GPIO header.
    GPIO 17 (pin 11) -> R via 150 ohm
    GPIO 18 (pin 12) -> G via 220 ohm
    GPIO 27 (pin 13) -> B via 220 ohm
    pin 14 (GND)     -> common cathode

State priority cascade (worst-first), used by ``decide_state`` in
``src/status/health.py``::

    BOOT_FAILURE > HARDWARE_FAULT > SOFTWARE_FAULT
        > NO_INTERNET > NO_CLOUD > UNPROVISIONED > OK
"""
from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class LedState(Enum):
    """Logical device state expressed by the LED."""

    OFF = "off"
    BOOT_FAILURE = "boot_failure"        # red solid
    HARDWARE_FAULT = "hardware_fault"    # yellow solid
    SOFTWARE_FAULT = "software_fault"    # yellow blink
    NO_INTERNET = "no_internet"          # green blink
    NO_CLOUD = "no_cloud"                # green solid
    UNPROVISIONED = "unprovisioned"      # blue blink
    OK = "ok"                            # blue solid


# (red_on, green_on, blue_on, blinking) per state.
_PATTERN: dict[LedState, tuple[bool, bool, bool, bool]] = {
    LedState.OFF:            (False, False, False, False),
    LedState.BOOT_FAILURE:   (True,  False, False, False),
    LedState.HARDWARE_FAULT: (True,  True,  False, False),
    LedState.SOFTWARE_FAULT: (True,  True,  False, True),
    LedState.NO_INTERNET:    (False, True,  False, True),
    LedState.NO_CLOUD:       (False, True,  False, False),
    LedState.UNPROVISIONED:  (False, False, True,  True),
    LedState.OK:             (False, False, True,  False),
}


class _GpioBackend(Protocol):
    """Minimum interface expected of a GPIO output. ``gpiozero.LED`` matches."""

    def on(self) -> None: ...
    def off(self) -> None: ...
    def close(self) -> None: ...


def _open_default_backend(
    pin_red: int, pin_green: int, pin_blue: int,
) -> tuple[_GpioBackend, _GpioBackend, _GpioBackend] | None:
    """Open ``gpiozero.LED`` handles, or return None if unavailable.

    Used as the default backend on the RPi. Tests inject their own backend
    and never hit this path.
    """
    try:
        from gpiozero import LED
    except Exception as e:
        logger.warning("gpiozero unavailable, status LED disabled: %s", e)
        return None
    try:
        return LED(pin_red), LED(pin_green), LED(pin_blue)
    except Exception:
        logger.exception("Failed to open LED GPIO pins (R=%d G=%d B=%d)",
                         pin_red, pin_green, pin_blue)
        return None


class StatusLED:
    """RGB status LED driver with blink loop in a background thread.

    Off-RPi and on init failure the LED degrades to a no-op and only logs
    state transitions — main.py keeps running. Tests inject ``backend`` to
    exercise transitions without touching real GPIO.

    Args:
        pin_red: BCM pin number for the R anode (default GPIO 17).
        pin_green: BCM pin number for the G anode (default GPIO 18).
        pin_blue: BCM pin number for the B anode (default GPIO 27).
        blink_period_s: Full on/off period for blinking states (default 1 s).
        backend: Optional injected ``(red, green, blue)`` GPIO handles.
    """

    def __init__(
        self,
        pin_red: int = 17,
        pin_green: int = 18,
        pin_blue: int = 27,
        blink_period_s: float = 1.0,
        backend: Optional[
            tuple[_GpioBackend, _GpioBackend, _GpioBackend]
        ] = None,
    ) -> None:
        self._blink_period_s = float(blink_period_s)
        self._state = LedState.OFF
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._tick_event = threading.Event()

        if backend is None:
            backend = _open_default_backend(pin_red, pin_green, pin_blue)
        self._backend = backend
        self._enabled = backend is not None

        self._thread = threading.Thread(
            target=self._run, name="status-led", daemon=True,
        )
        self._thread.start()
        logger.info(
            "Status LED %s (pins R=%d G=%d B=%d)",
            "active" if self._enabled else "disabled",
            pin_red, pin_green, pin_blue,
        )

    def set_state(self, state: LedState) -> None:
        """Switch to ``state``. Idempotent: same state is a no-op."""
        with self._state_lock:
            if state is self._state:
                return
            self._state = state
        logger.info("LED state -> %s", state.value)
        self._tick_event.set()

    def get_state(self) -> LedState:
        with self._state_lock:
            return self._state

    def close(self) -> None:
        """Turn off and release GPIO. Safe to call repeatedly."""
        self._stop_event.set()
        self._tick_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._set_outputs(False, False, False)
        if self._backend is not None:
            for handle in self._backend:
                try:
                    handle.close()
                except Exception:
                    logger.exception("Closing LED handle failed")
            self._backend = None
            self._enabled = False

    def _set_outputs(self, r: bool, g: bool, b: bool) -> None:
        if self._backend is None:
            return
        red, green, blue = self._backend
        try:
            (red.on if r else red.off)()
            (green.on if g else green.off)()
            (blue.on if b else blue.off)()
        except Exception:
            logger.exception("LED GPIO write failed")

    def _run(self) -> None:
        half = self._blink_period_s / 2.0
        on_phase = True
        while not self._stop_event.is_set():
            with self._state_lock:
                state = self._state
            r, g, b, blink = _PATTERN[state]
            if blink:
                self._set_outputs(r, g, b) if on_phase else self._set_outputs(
                    False, False, False
                )
                self._tick_event.wait(timeout=half)
            else:
                self._set_outputs(r, g, b)
                self._tick_event.wait()  # block until state change or stop
            self._tick_event.clear()
            on_phase = not on_phase if blink else True

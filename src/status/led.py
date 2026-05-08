"""LED RGB de status para diagnóstico en sitio.

Patrón de 8 estados (apagado / rojo / amarillo / amarillo parpadeante /
verde parpadeante / verde / azul / azul parpadeante) — el operador del local
puede interpretar la salud del dispositivo por color/parpadeo del LED sin
acceso SSH.

Hardware: RGB de 3mm common-cathode sobre el header de GPIO.
    GPIO 17 (pin 11) -> R vía 150 ohm
    GPIO 18 (pin 12) -> G vía 100 ohm
    GPIO 27 (pin 13) -> B vía 100 ohm
    pin 14 (GND)     -> common cathode

Cascada de prioridad de estados (worst-first), usada por ``decide_state`` en
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
    """Estado lógico del dispositivo expresado por el LED."""

    OFF = "off"
    BOOT_FAILURE = "boot_failure"        # rojo fijo
    HARDWARE_FAULT = "hardware_fault"    # amarillo fijo
    SOFTWARE_FAULT = "software_fault"    # amarillo parpadeante
    NO_INTERNET = "no_internet"          # verde parpadeante
    NO_CLOUD = "no_cloud"                # verde fijo
    UNPROVISIONED = "unprovisioned"      # azul parpadeante
    OK = "ok"                            # azul fijo


# (red_on, green_on, blue_on, blinking) por estado.
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
    """Interfaz mínima esperada de un output GPIO. ``gpiozero.LED`` matchea."""

    def on(self) -> None: ...
    def off(self) -> None: ...
    def close(self) -> None: ...


def _open_default_backend(
    pin_red: int, pin_green: int, pin_blue: int,
) -> tuple[_GpioBackend, _GpioBackend, _GpioBackend] | None:
    """Abre los handles ``gpiozero.LED``, o devuelve None si no está disponible.

    Se usa como backend default en la RPi. Los tests inyectan su propio backend
    y nunca pegan en este path.
    """
    try:
        from gpiozero import LED
    except Exception as e:
        logger.warning("gpiozero no disponible, status LED deshabilitado: %s", e)
        return None
    try:
        return LED(pin_red), LED(pin_green), LED(pin_blue)
    except Exception:
        logger.exception("Falló abrir los pines GPIO del LED (R=%d G=%d B=%d)",
                         pin_red, pin_green, pin_blue)
        return None


class StatusLED:
    """Driver del LED RGB de status con loop de blink en thread background.

    Fuera de la RPi y ante fallas de init el LED degrada a no-op y solo loggea
    las transiciones de estado — main.py sigue corriendo. Los tests inyectan
    ``backend`` para ejercitar transiciones sin tocar el GPIO real.

    Args:
        pin_red: Número de pin BCM para el ánodo R (default GPIO 17).
        pin_green: Número de pin BCM para el ánodo G (default GPIO 18).
        pin_blue: Número de pin BCM para el ánodo B (default GPIO 27).
        blink_period_s: Período completo on/off para los estados que parpadean (default 1 s).
        backend: Handles GPIO ``(red, green, blue)`` inyectados opcionalmente.
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
            "activo" if self._enabled else "deshabilitado",
            pin_red, pin_green, pin_blue,
        )

    def set_state(self, state: LedState) -> None:
        """Cambia a ``state``. Idempotente: el mismo estado es no-op."""
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
        """Apaga y libera los GPIO. Seguro llamarlo varias veces."""
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
                self._tick_event.wait()  # bloquea hasta cambio de estado o stop
            self._tick_event.clear()
            on_phase = not on_phase if blink else True

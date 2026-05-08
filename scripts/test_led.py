#!/usr/bin/env python3
"""Recorre los 8 estados del status LED para validación de bench.

Útil para testear el wiring sobre un breadboard antes del armado final.
Cada estado estático se sostiene por STATIC_S; los estados que parpadean
corren por BLINK_S así se ven ~4 ciclos de blink (período default es 1 s).

Uso:
    PYTHONPATH=. python3 scripts/test_led.py
"""
from __future__ import annotations

import time

from src.status.led import LedState, StatusLED

STATIC_S = 2.0
BLINK_S = 4.0

SEQUENCE: list[tuple[LedState, float, str]] = [
    (LedState.OFF,            STATIC_S, "OFF              — sin power"),
    (LedState.BOOT_FAILURE,   STATIC_S, "Rojo fijo        — boot failure"),
    (LedState.HARDWARE_FAULT, STATIC_S, "Amarillo fijo    — hardware fault"),
    (LedState.SOFTWARE_FAULT, BLINK_S,  "Amarillo blink   — software fault"),
    (LedState.NO_INTERNET,    BLINK_S,  "Verde blink      — sin internet"),
    (LedState.NO_CLOUD,       STATIC_S, "Verde fijo       — sin cloud"),
    (LedState.UNPROVISIONED,  BLINK_S,  "Azul blink       — sin provisioning"),
    (LedState.OK,             STATIC_S, "Azul fijo        — operación normal"),
]


def main() -> None:
    led = StatusLED()
    try:
        for state, hold_s, description in SEQUENCE:
            print(f"  {description}")
            led.set_state(state)
            time.sleep(hold_s)
        print("\nDone. Apagando LED.")
    finally:
        led.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Walk through the 8 status LED states for bench validation.

Useful when testing the wiring on a breadboard before final assembly.
Each static state is held for STATIC_S; blinking states run for BLINK_S
so you see ~4 blink cycles (default period is 1 s).

Usage:
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

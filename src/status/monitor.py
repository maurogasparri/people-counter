"""Background health monitor that drives the status LED.

The monitor reads pipeline-side signals from a shared :class:`HealthSignals`
dataclass (written by main.py during the loop) and OS/network probes from
``health.py``, then maps the aggregate into a :class:`LedState` via
``decide_state``.

The monitor runs in its own thread because the internet probe is blocking
I/O (``socket.create_connection`` with a 3-second timeout) - we don't want
that on the pipeline's hot path. Internet probes also run at a slower
cadence than the LED update tick to avoid flooding the network.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from src.status.health import (
    PIPELINE_STALL_S,
    check_calibration_loadable,
    check_cpu_temp_ok,
    check_disk_ok,
    check_hailo_temp_ok,
    check_internet,
    decide_state,
)
from src.status.led import LedState, StatusLED

logger = logging.getLogger(__name__)


@dataclass
class HealthSignals:
    """Mutable signals the pipeline writes; the monitor reads them.

    Field assignments are atomic under CPython's GIL for primitive types,
    so reads in the monitor thread don't need locking. Don't add mutable
    containers without adding a lock.
    """

    last_loop_ts: float = 0.0
    capture_ok: bool = True
    detect_ok: bool = True
    mqtt_connected: bool = False
    provisioned: bool = True
    boot_complete: bool = False
    calibration_path: Optional[str] = None


class HealthMonitor:
    """Background thread that maps :class:`HealthSignals` to LED state.

    Args:
        led: The :class:`StatusLED` to drive.
        signals: Shared signals object updated by the pipeline.
        poll_interval_s: How often to re-evaluate state (default 2 s).
        internet_probe_interval_s: How often to run the blocking internet
            probe (default 30 s; cached between probes).
    """

    def __init__(
        self,
        led: StatusLED,
        signals: HealthSignals,
        poll_interval_s: float = 2.0,
        internet_probe_interval_s: float = 30.0,
    ) -> None:
        self._led = led
        self._signals = signals
        self._poll_interval_s = float(poll_interval_s)
        self._internet_probe_interval_s = float(internet_probe_interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_internet_probe = 0.0
        self._cached_internet_ok = True

    def start(self) -> None:
        """Start the monitor thread. Idempotent."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="health-monitor", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop and join the monitor thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def evaluate_once(self) -> LedState:
        """Run one tick synchronously; mainly for tests."""
        return self._tick()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Health monitor tick failed")
            self._stop_event.wait(timeout=self._poll_interval_s)

    def _tick(self) -> LedState:
        s = self._signals
        now = time.time()

        hw_ok = (
            s.capture_ok
            and s.detect_ok
            and check_cpu_temp_ok()
            and check_hailo_temp_ok()
            and check_disk_ok()
            and check_calibration_loadable(s.calibration_path)
        )

        # Pipeline freshness: only judged once the loop has produced at least
        # one iteration. A zero ts means the pipeline hasn't started yet
        # (boot phase) - don't fault while the operator is still waiting.
        if s.last_loop_ts == 0.0:
            pipeline_ok = True
        else:
            pipeline_ok = (now - s.last_loop_ts) <= PIPELINE_STALL_S

        if now - self._last_internet_probe >= self._internet_probe_interval_s:
            self._cached_internet_ok = check_internet()
            self._last_internet_probe = now
        internet_ok = self._cached_internet_ok

        cloud_connected = bool(s.mqtt_connected)

        state = decide_state(
            hardware_ok=hw_ok,
            pipeline_ok=pipeline_ok,
            internet_ok=internet_ok,
            cloud_connected=cloud_connected,
            provisioned=s.provisioned,
        )
        self._led.set_state(state)
        return state

"""Health probes and aggregation logic for the status LED.

The probes are pure (no side effects beyond reads of /proc, /sys, sockets)
and individually unit-testable. ``decide_state`` applies the priority
cascade defined in ``led.py``.

Threshold rationale:
    * CPU temp 80 C: RPi5 starts thermal throttling at this point, so the
      pipeline is at risk of frame drops.
    * Hailo temp 85 C: Hailo-8L stays well below this in normal operation;
      crossing it indicates airflow blocked or fan failure.
    * Disk free 10 %: SQLite buffer needs headroom to retain events across
      a multi-day MQTT outage; below this we may lose data on next disconnect.
    * Pipeline stall 5 s: at 15 FPS the loop iterates every ~67 ms, so 5 s
      represents ~75 missed iterations - well past any plausible single-frame
      hiccup.
"""
from __future__ import annotations

import logging
import re
import shutil
import socket
import subprocess
from pathlib import Path

from src.status.led import LedState

logger = logging.getLogger(__name__)

CPU_TEMP_CRITICAL_C = 80.0
HAILO_TEMP_CRITICAL_C = 85.0
DISK_FREE_MIN_PCT = 10.0
PIPELINE_STALL_S = 5.0

_HAILO_TEMP_RE = re.compile(
    r"(?:die|chip)\s+temperature\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def check_cpu_temp_ok() -> bool:
    """Return True when CPU is below the throttling threshold or unreadable.

    Fail-open: an unreadable thermal zone (off-RPi tests, kernel changes)
    must not light the fault LED.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read().strip()) / 1000.0
        return temp < CPU_TEMP_CRITICAL_C
    except Exception:
        return True


def check_hailo_temp_ok() -> bool:
    """Probe Hailo die temperature via ``hailortcli``. Fail-open on error."""
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return True
    except Exception:
        logger.exception("hailortcli invocation failed")
        return True

    try:
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        match = _HAILO_TEMP_RE.search(output)
        if match is None:
            return True  # no temp parsed - don't trip
        return float(match.group(1)) < HAILO_TEMP_CRITICAL_C
    except Exception:
        logger.exception("Failed to parse hailortcli output")
        return True


def check_disk_ok() -> bool:
    """Return True when free disk space is above ``DISK_FREE_MIN_PCT``."""
    try:
        usage = shutil.disk_usage("/")
        free_pct = usage.free / usage.total * 100.0
        return free_pct >= DISK_FREE_MIN_PCT
    except Exception:
        logger.exception("Disk check failed")
        return True


def check_calibration_loadable(path: str | None) -> bool:
    """Return True if no calibration is configured or the file exists.

    A missing file is a hardware-class fault because the pipeline can't
    rectify frames without it.
    """
    if not path:
        return True
    return Path(path).exists()


def check_internet(
    host: str = "1.1.1.1", port: int = 53, timeout_s: float = 3.0,
) -> bool:
    """TCP-connect probe for internet reachability.

    Cloudflare DNS on port 53 answers from anywhere and doesn't require
    ICMP (often firewalled). The probe is blocking; callers should run it
    on a thread with a slower cadence than the main loop.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def check_cloud_endpoint(
    endpoint: str, port: int = 8883, timeout_s: float = 5.0,
) -> bool:
    """TCP probe to the AWS IoT broker (no TLS validation).

    Used as a cold-start fallback before the MQTT client has a connected
    flag. Once MQTT is up, ``MQTTClient.is_connected()`` is the source of
    truth.
    """
    try:
        with socket.create_connection((endpoint, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def decide_state(
    *,
    boot_failure: bool = False,
    hardware_ok: bool = True,
    pipeline_ok: bool = True,
    internet_ok: bool = True,
    cloud_connected: bool = True,
    provisioned: bool = True,
) -> LedState:
    """Apply the priority cascade and return the worst applicable state.

    Worst-first: a single hardware fault masks downstream issues, since the
    operator's first action (power cycle, check cables) is the same.
    """
    if boot_failure:
        return LedState.BOOT_FAILURE
    if not hardware_ok:
        return LedState.HARDWARE_FAULT
    if not pipeline_ok:
        return LedState.SOFTWARE_FAULT
    if not internet_ok:
        return LedState.NO_INTERNET
    if not cloud_connected:
        return LedState.NO_CLOUD
    if not provisioned:
        return LedState.UNPROVISIONED
    return LedState.OK

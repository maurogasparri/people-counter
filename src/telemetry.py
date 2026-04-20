"""Device telemetry collection.

Aggregates OS-level metrics (CPU/Hailo temps, memory, disk, uptime) plus
runtime pipeline state (frame latency percentiles, detection rate, tracker
state counts, MQTT connectivity, buffer backlog) into a single flat dict
suitable for JSON publishing.

Every probe is wrapped in try/except and emits ``None`` on failure so the
downstream schema stays stable — consumers can tell the difference between
"sensor queried, failed" and "field not emitted".
"""
from __future__ import annotations

import logging
import re
import shutil
import statistics
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# Keys whose presence downstream (Timestream telemetry rule) depends on.
# Adding fields is safe; renaming would break consumers.
_STATE_DEPENDENT_KEYS = (
    "frame_latency_p50_ms",
    "frame_latency_p95_ms",
    "detection_rate_per_min",
    "tracker_confirmed_count",
    "tracker_pending_count",
    "mqtt_disconnect_count",
    "seconds_since_last_reconnect",
    "buffer_backlog_messages",
)


def _read_uptime() -> float | None:
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return None


def _read_cpu_temp() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def _read_disk_free_mb() -> int | None:
    try:
        usage = shutil.disk_usage("/")
        return usage.free // (1024 * 1024)
    except Exception:
        return None


def _read_mem_available_mb() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


# Hailo die temperature appears in `hailortcli fw-control identify` output
# on a line similar to:  "Die temperature: 45.3 C".  The label varies across
# HailoRT versions ("Die Temperature", "Chip Temperature") so match loosely.
_HAILO_TEMP_RE = re.compile(
    r"(?:die|chip)\s+temperature\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def _read_hailo_temp() -> float | None:
    """Query hailortcli for die temperature. Returns None on any failure."""
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("hailortcli not available: %s", e)
        return None
    except Exception:
        logger.exception("hailortcli invocation failed")
        return None

    try:
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        match = _HAILO_TEMP_RE.search(output)
        if match is None:
            return None
        return float(match.group(1))
    except Exception:
        logger.exception("Failed to parse hailortcli output")
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. Returns None for empty input."""
    if not values:
        return None
    try:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        # Nearest-rank: rank = ceil(pct/100 * n), index = rank - 1
        rank = max(1, int(-(-pct * n // 100)))  # ceil via floor-div trick
        rank = min(rank, n)
        return float(sorted_vals[rank - 1])
    except Exception:
        return None


def _p50(values: list[float]) -> float | None:
    if not values:
        return None
    try:
        return float(statistics.median(values))
    except Exception:
        return None


def _detection_rate_per_min(state: dict[str, Any]) -> float | None:
    """Persons-per-minute over the sampling window.

    Prefers a wall-clock window derived from ``detection_window_start_ts``.
    Falls back to ``len(counts) / fps`` if ``fps`` is provided. Returns
    ``None`` if neither is available or the window is degenerate.
    """
    counts = state.get("detection_counts")
    if not counts:
        return None
    try:
        total = float(sum(counts))
    except Exception:
        return None

    start_ts = state.get("detection_window_start_ts")
    if start_ts is not None:
        try:
            elapsed = time.time() - float(start_ts)
            if elapsed > 0:
                return total / elapsed * 60.0
        except Exception:
            pass

    fps = state.get("fps")
    if fps:
        try:
            elapsed_s = len(counts) / float(fps)
            if elapsed_s > 0:
                return total / elapsed_s * 60.0
        except Exception:
            pass

    return None


def collect_telemetry(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect device + runtime telemetry into a flat dict.

    All keys are always present; failing probes emit ``None``.

    Args:
        state: Runtime counters tracked by main.py. Recognized keys:
            - ``frame_latencies_ms``: iterable of recent per-frame latencies
            - ``detection_counts``: iterable of persons-per-frame counts
            - ``detection_window_start_ts``: epoch seconds for the start of
              the detection counting window (used to compute rate)
            - ``fps``: fallback for rate computation if no start ts
            - ``tracker_confirmed``: int count of CONFIRMED tracks
            - ``tracker_pending``: int count of PENDING tracks
            - ``mqtt_disconnect_count``: cumulative disconnect count
            - ``mqtt_reconnect_ts``: epoch of last (re)connect, or None
            - ``buffer_backlog``: count of unsent messages in the buffer
    """
    state = state or {}
    telemetry: dict[str, Any] = {}

    # --- OS probes (keep names stable for Timestream consumers) ---
    telemetry["uptime_s"] = _read_uptime()
    telemetry["cpu_temp_c"] = _read_cpu_temp()
    telemetry["disk_free_mb"] = _read_disk_free_mb()
    telemetry["mem_available_mb"] = _read_mem_available_mb()
    telemetry["hailo_temp_c"] = _read_hailo_temp()

    # --- Frame latency percentiles ---
    try:
        latencies = list(state.get("frame_latencies_ms") or [])
    except Exception:
        latencies = []
    telemetry["frame_latency_p50_ms"] = _p50(latencies)
    telemetry["frame_latency_p95_ms"] = _percentile(latencies, 95)

    # --- Detection rate ---
    try:
        telemetry["detection_rate_per_min"] = _detection_rate_per_min(state)
    except Exception:
        logger.exception("detection_rate_per_min computation failed")
        telemetry["detection_rate_per_min"] = None

    # --- Tracker state counts ---
    telemetry["tracker_confirmed_count"] = state.get("tracker_confirmed")
    telemetry["tracker_pending_count"] = state.get("tracker_pending")

    # --- MQTT health ---
    telemetry["mqtt_disconnect_count"] = state.get("mqtt_disconnect_count")
    reconnect_ts = state.get("mqtt_reconnect_ts")
    if reconnect_ts is None:
        telemetry["seconds_since_last_reconnect"] = None
    else:
        try:
            telemetry["seconds_since_last_reconnect"] = max(
                0.0, time.time() - float(reconnect_ts)
            )
        except Exception:
            telemetry["seconds_since_last_reconnect"] = None

    # --- Buffer backlog ---
    telemetry["buffer_backlog_messages"] = state.get("buffer_backlog")

    # Guarantee every state-dependent key exists even if something above
    # raised before assigning it.
    for key in _STATE_DEPENDENT_KEYS:
        telemetry.setdefault(key, None)

    return telemetry

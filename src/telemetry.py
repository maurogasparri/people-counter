"""Recolección de telemetría del dispositivo.

Agrega métricas a nivel de OS (temps de CPU/Hailo, memoria, disco, uptime) más
estado runtime del pipeline (percentiles de latencia por frame, detection rate,
counts de estado del tracker, conectividad MQTT, backlog del buffer) en un único
dict plano apto para publicar como JSON.

Cada probe está wrappeado en try/except y emite ``None`` si falla, así el
schema downstream se mantiene estable — los consumers pueden distinguir entre
"sensor consultado, falló" y "campo no emitido".
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

# Keys de cuya presencia depende el downstream (regla de telemetría de Timestream).
# Agregar campos es seguro; renombrar rompería los consumers.
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


# La temperatura del die de Hailo aparece en el output de `hailortcli fw-control identify`
# en una línea similar a: "Die temperature: 45.3 C". El label varía entre versiones
# de HailoRT ("Die Temperature", "Chip Temperature") así que matcheamos con flexibilidad.
_HAILO_TEMP_RE = re.compile(
    r"(?:die|chip)\s+temperature\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def _read_hailo_temp() -> float | None:
    """Consulta hailortcli por la temperatura del die. Devuelve None ante cualquier falla."""
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
    """Percentil nearest-rank. Devuelve None si la entrada está vacía."""
    if not values:
        return None
    try:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        # Nearest-rank: rank = ceil(pct/100 * n), index = rank - 1
        rank = max(1, int(-(-pct * n // 100)))  # ceil vía truco de floor-div
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
    """Personas-por-minuto sobre la ventana de sampling.

    Prefiere una ventana wall-clock derivada de ``detection_window_start_ts``.
    Falla a ``len(counts) / fps`` si se provee ``fps``. Devuelve ``None`` si
    ninguno está disponible o si la ventana es degenerada.
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
    """Recolecta telemetría de dispositivo + runtime en un dict plano.

    Todas las keys están siempre presentes; los probes que fallan emiten ``None``.

    Args:
        state: Counters runtime trackeados por main.py. Keys reconocidas:
            - ``frame_latencies_ms``: iterable de latencias por frame recientes
            - ``detection_counts``: iterable de counts de personas por frame
            - ``detection_window_start_ts``: epoch seconds del inicio de la
              ventana de conteo de detecciones (se usa para computar la rate)
            - ``fps``: fallback para computar rate si no hay start ts
            - ``tracker_confirmed``: count int de tracks CONFIRMED
            - ``tracker_pending``: count int de tracks PENDING
            - ``mqtt_disconnect_count``: count acumulado de disconnects
            - ``mqtt_reconnect_ts``: epoch del último (re)connect, o None
            - ``buffer_backlog``: count de mensajes no enviados en el buffer
    """
    state = state or {}
    telemetry: dict[str, Any] = {}

    # --- Probes de OS (mantener nombres estables para los consumers de Timestream) ---
    telemetry["uptime_s"] = _read_uptime()
    telemetry["cpu_temp_c"] = _read_cpu_temp()
    telemetry["disk_free_mb"] = _read_disk_free_mb()
    telemetry["mem_available_mb"] = _read_mem_available_mb()
    telemetry["hailo_temp_c"] = _read_hailo_temp()

    # --- Percentiles de latencia por frame ---
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

    # --- Counts de estado del tracker ---
    telemetry["tracker_confirmed_count"] = state.get("tracker_confirmed")
    telemetry["tracker_pending_count"] = state.get("tracker_pending")

    # --- Health MQTT ---
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

    # --- Backlog del buffer ---
    telemetry["buffer_backlog_messages"] = state.get("buffer_backlog")

    # Garantiza que toda key dependiente de state exista aunque algo arriba
    # haya raiseado antes de asignarla.
    for key in _STATE_DEPENDENT_KEYS:
        telemetry.setdefault(key, None)

    return telemetry

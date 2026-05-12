"""Probes de health y lógica de agregación para el status LED.

Los probes son puros (sin side effects más allá de lecturas a /proc, /sys,
sockets) y unit-testables individualmente. ``decide_state`` aplica la cascada
de prioridad definida en ``led.py``.

Justificación de thresholds:
    * CPU temp 80 C: la RPi5 empieza a throttlear térmicamente acá, así que
      el pipeline corre riesgo de drop de frames.
    * Hailo temp 85 C: el Hailo-8L se mantiene bien debajo en operación normal;
      cruzarlo indica airflow bloqueado o falla del fan.
    * Disk free 10 %: el buffer SQLite necesita headroom para retener eventos
      durante un outage MQTT multi-día; por debajo podemos perder data en el
      próximo disconnect.
    * Pipeline stall 5 s: a 15 FPS el loop itera cada ~67 ms, así que 5 s
      representa ~75 iteraciones perdidas — muy por encima de cualquier
      hiccup plausible de un solo frame.
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
    """Devuelve True cuando la CPU está debajo del threshold de throttling o es ilegible.

    Fail-open: una zona térmica ilegible (tests fuera de la RPi, cambios de
    kernel) no debe prender el LED de falla.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read().strip()) / 1000.0
        return temp < CPU_TEMP_CRITICAL_C
    except Exception:
        return True


def check_hailo_temp_ok() -> bool:
    """Probea la temperatura del die de Hailo vía ``hailortcli``. Fail-open en error."""
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
            return True  # no se parseó la temp — no disparar
        return float(match.group(1)) < HAILO_TEMP_CRITICAL_C
    except Exception:
        logger.exception("Falló parsear output de hailortcli")
        return True


def check_disk_ok() -> bool:
    """Devuelve True cuando el espacio libre en disco está sobre ``DISK_FREE_MIN_PCT``."""
    try:
        usage = shutil.disk_usage("/")
        free_pct = usage.free / usage.total * 100.0
        return free_pct >= DISK_FREE_MIN_PCT
    except Exception:
        logger.exception("Falló el check de disco")
        return True


def check_calibration_loadable(path: str | None) -> bool:
    """Devuelve True si no hay calibración configurada o si el archivo existe.

    Un archivo faltante es una falla de clase hardware porque el pipeline
    no puede rectificar frames sin él.
    """
    if not path:
        return True
    return Path(path).exists()


def check_internet(
    host: str = "1.1.1.1", port: int = 53, timeout_s: float = 3.0,
) -> bool:
    """Probe TCP-connect para reachability de internet.

    Cloudflare DNS en el puerto 53 responde desde cualquier lado y no requiere
    ICMP (a menudo firewalled). El probe es blocking; los callers deberían
    correrlo en un thread con cadencia más lenta que la del main loop.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
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
    """Aplica la cascada de prioridad y devuelve el peor estado aplicable.

    Worst-first: una sola falla de hardware enmascara problemas downstream,
    porque la primera acción del operador (power cycle, chequear cables) es
    la misma.
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

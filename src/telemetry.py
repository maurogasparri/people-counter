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

import glob
import logging
import re
import shutil
import statistics
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# Keys de cuya presencia depende el downstream (IoT Rule → Lambda persist_event
# → Postgres). Agregar campos es seguro; renombrar rompería los consumers.
_STATE_DEPENDENT_KEYS = (
    "frame_latency_p50_ms",
    "frame_latency_p95_ms",
    "detection_rate_per_min",
    "tracker_confirmed_count",
    "tracker_pending_count",
    "mqtt_disconnect_count",
    "seconds_since_last_reconnect",
    "buffer_backlog_messages",
    "wifi_probe_ok",
    "ble_scanner_ok",
    "wifi_ble_stitching_ratio",
    "cam_left_ok",
    "cam_right_ok",
    "wifi_probe_rate_per_min",
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


def _read_hailo_temp() -> float | None:
    """Lee la temperatura del chip Hailo via la SDK Python.

    Usa ``hailo_platform.Device().control.get_chip_temperature()`` que
    devuelve un ``TemperatureInfo`` con ``ts0_temperature`` y
    ``ts1_temperature`` (los dos sensores del die). Tomamos el max
    como métrica conservadora — eso es lo que pega throttling primero.

    El CLI ``hailortcli fw-control identify`` NO incluye temperatura en
    HailoRT 4.23+ (cambió en versiones anteriores), así que no lo
    usamos como fallback. Devuelve None ante cualquier falla así un
    sample de telemetría malo no rompe la emisión.
    """
    try:
        import hailo_platform as hp
    except ImportError:
        logger.debug("hailo_platform not available")
        return None
    try:
        device = hp.Device()
        info = device.control.get_chip_temperature()
        ts0 = float(info.ts0_temperature)
        ts1 = float(info.ts1_temperature)
        return max(ts0, ts1)
    except Exception:
        logger.exception("Failed to read Hailo chip temperature")
        return None


# --- Probes de salud de hardware (S12) ---------------------------------------
# Detectan fallas que en un device de techo, desatendido, pasan en silencio:
# fan clavado / airflow bloqueado (throttle térmico), fuente o cable PoE/USB-C
# flojo (undervoltage + sag de EXT5V), y consumo anómalo. Todos son reads
# rápidos (sysfs / vcgencmd / PMIC) → samplean a la cadencia de telemetría sin
# caché. Fail → None, igual que el resto de los probes.


def _vcgencmd(*args: str) -> str | None:
    """Corre ``vcgencmd <args>`` y devuelve stdout, o None si falla/no existe."""
    try:
        r = subprocess.run(
            ["vcgencmd", *args], capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except Exception:
        return None


def _read_throttled() -> int | None:
    """Bitmask de ``vcgencmd get_throttled``. Bits: 0x1 undervoltage actual,
    0x2 ARM freq capped, 0x4 throttled actual, 0x8 soft-temp-limit actual;
    0x10000/0x20000/0x40000/0x80000 = los mismos "ocurrió desde el boot"
    (sticky). 0 = todo sano. Se persiste crudo; el classify es server-side."""
    out = _vcgencmd("get_throttled")
    if not out:
        return None
    try:
        return int(out.split("=")[1], 16)
    except Exception:
        return None


def _read_arm_mhz() -> int | None:
    """Frecuencia actual del ARM en MHz (``vcgencmd measure_clock arm``).
    Cae bajo el nominal (2400 en Pi 5) cuando hay throttle — corrobora flags."""
    out = _vcgencmd("measure_clock", "arm")
    if not out:
        return None
    try:
        return int(out.split("=")[1]) // 1_000_000
    except Exception:
        return None


def _read_fan_rpm() -> int | None:
    """RPM del fan del Active Cooler vía el hwmon ``pwmfan`` (índice no estable
    entre boots → se busca por name). 0 con temp alta = fan clavado."""
    try:
        for hw in glob.glob("/sys/class/hwmon/hwmon*"):
            try:
                with open(f"{hw}/name") as f:
                    if f.read().strip() != "pwmfan":
                        continue
                with open(f"{hw}/fan1_input") as f:
                    return int(f.read().strip())
            except Exception:
                continue
    except Exception:
        return None
    return None


_PMIC_CUR_RE = re.compile(r"(\S+)_A current\(\d+\)=([\d.]+)A")
_PMIC_VOLT_RE = re.compile(r"(\S+)_V volt\(\d+\)=([\d.]+)V")


def _read_pmic() -> tuple[float | None, float | None]:
    """``(power_w, ext5v_v)`` de un solo ``vcgencmd pmic_read_adc``.

    ``power_w`` = suma de V×I de todos los rieles (output-side; incluye placa +
    Hailo vía PCIe + cámaras). ``ext5v_v`` = tensión de entrada (sag = fuente o
    cable flojo). ``(None, None)`` si falla."""
    out = _vcgencmd("pmic_read_adc")
    if not out:
        return None, None
    try:
        cur: dict[str, float] = {}
        vol: dict[str, float] = {}
        for line in out.splitlines():
            m = _PMIC_CUR_RE.search(line)
            if m:
                cur[m.group(1)] = float(m.group(2))
            m = _PMIC_VOLT_RE.search(line)
            if m:
                vol[m.group(1)] = float(m.group(2))
        power = round(sum(cur[k] * vol[k] for k in cur if k in vol), 3) if cur else None
        return power, vol.get("EXT5V")
    except Exception:
        return None, None


# --- Probes Tier 2/3: fallas silenciosas "corre pero roto" ---------------------
# Detectan estados donde el device figura up pero está degradado o produciendo
# data mala: microSD fallando (fs read-only), pipeline en crash-loop, reloj
# desincronizado (timestamps corruptos). Reads baratos; fail → None.


# Path escribible del device (ReadWritePaths del .service; outbox del buffer).
_WRITABLE_DATA_DIR = "/var/lib/people-counter"


def _mount_contains(mount_point: str, path: str) -> bool:
    """True si ``mount_point`` es prefijo de directorio de ``path``."""
    if mount_point == "/":
        return True
    return path == mount_point or path.startswith(mount_point.rstrip("/") + "/")


def _read_fs_readonly() -> bool | None:
    """True si el filesystem que respalda el dir de datos escribible está
    montado read-only — microSD fallando: ext4 con ``errors=remount-ro``
    flipea el superblock a ro y las escrituras (outbox/SQLite) mueren.

    Chequea el mount que cubre ``_WRITABLE_DATA_DIR`` (prefijo más largo), NO
    ``/``: el service corre con ``ProtectSystem=strict``, que monta ``/``
    read-only en su mount namespace → ``/`` daría True permanentemente (falso
    positivo). El bind de ReadWritePaths es rw salvo que el superblock real
    (mismo /dev/mmcblk0p2) se vuelva ro por error de la SD."""
    try:
        best_mp = None
        best_ro = None
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                mp = parts[1]
                if not _mount_contains(mp, _WRITABLE_DATA_DIR):
                    continue
                if best_mp is None or len(mp) > len(best_mp):
                    best_mp = mp
                    best_ro = "ro" in parts[3].split(",")
        return best_ro
    except Exception:
        return None


def _read_service_restarts() -> int | None:
    """``NRestarts`` del service (auto-restarts acumulados desde el último start
    limpio). Crece rápido en crash-loop. Se resetea en restart manual."""
    try:
        r = subprocess.run(
            [
                "systemctl",
                "show",
                "people-counter.service",
                "-p",
                "NRestarts",
                "--value",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())
    except Exception:
        return None


def _read_clock_synchronized() -> bool | None:
    """True si el reloj del sistema está sincronizado por NTP. Sin sync (y RTC
    drifteado tras un corte) → timestamps de eventos incorrectos."""
    try:
        r = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        v = r.stdout.strip().lower()
        if v in ("yes", "true", "1"):
            return True
        if v in ("no", "false", "0"):
            return False
    except Exception:
        return None
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

    # --- Probes de OS (mantener nombres estables para Lambda persist_event → Postgres) ---
    telemetry["uptime_s"] = _read_uptime()
    telemetry["cpu_temp_c"] = _read_cpu_temp()
    telemetry["disk_free_mb"] = _read_disk_free_mb()
    telemetry["mem_available_mb"] = _read_mem_available_mb()
    telemetry["hailo_temp_c"] = _read_hailo_temp()

    # --- Salud de hardware (throttle, clock, fan, power delivery) ---
    telemetry["throttled_flags"] = _read_throttled()
    telemetry["arm_clock_mhz"] = _read_arm_mhz()
    telemetry["fan_rpm"] = _read_fan_rpm()
    power_w, ext5v_v = _read_pmic()
    telemetry["power_w"] = power_w
    telemetry["ext5v_v"] = ext5v_v

    # --- Salud "corre pero roto" (SD / crash-loop / reloj) ---
    telemetry["fs_readonly"] = _read_fs_readonly()
    telemetry["service_restarts"] = _read_service_restarts()
    telemetry["clock_synchronized"] = _read_clock_synchronized()

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
    telemetry["mqtt_connected"] = state.get("mqtt_connected")
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

    # --- Health de subsistemas WiFi/BLE ---
    # ``None`` cuando el subsistema está deshabilitado en config (wifi_ble.enabled
    # = false); ``True``/``False`` cuando está habilitado y reportando salud.
    # Permite distinguir "no aplica" de "está caído" en las alertas de Grafana.
    telemetry["wifi_probe_ok"] = state.get("wifi_probe_ok")
    telemetry["ble_scanner_ok"] = state.get("ble_scanner_ok")

    # --- Stitching ratio del dedup WiFi/BLE ---
    # groups / hashes en la ventana del dia: 1.0 = sin stitch (cada MAC su
    # propio "visitor"), 0.5 = mitad de los hashes se mergearon (avg 2/grupo).
    # Canary para ver si la flota corre con OS que defeatean las reglas (Apple
    # H1+ con seqnum reset, BLE off, etc) — en ese caso ratio == 1 sostenido y
    # los counts estaran inflados; calibrar contra la cam.
    telemetry["wifi_ble_stitching_ratio"] = state.get("wifi_ble_stitching_ratio")

    # --- Health por cámara (atribución del par estéreo) + rate de probes WiFi ---
    telemetry["cam_left_ok"] = state.get("cam_left_ok")
    telemetry["cam_right_ok"] = state.get("cam_right_ok")
    telemetry["wifi_probe_rate_per_min"] = state.get("wifi_probe_rate_per_min")

    # Garantiza que toda key dependiente de state exista aunque algo arriba
    # haya raiseado antes de asignarla.
    for key in _STATE_DEPENDENT_KEYS:
        telemetry.setdefault(key, None)

    return telemetry

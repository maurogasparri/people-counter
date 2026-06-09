"""Monitor de health en background que maneja el status LED.

El monitor lee señales pipeline-side de un dataclass :class:`HealthSignals`
compartido (escrito por main.py durante el loop) más probes OS/red de
``health.py``, y mapea la agregación a un :class:`LedState` vía ``decide_state``.

El monitor corre en su propio thread porque dos de sus probes son blocking:
el de internet (``socket.create_connection`` con timeout de 3 s) y el de
temperatura del Hailo (``hailortcli`` como subprocess, hasta 5 s). No queremos
eso en el hot path del pipeline. Ambos probes blocking corren además con
cadencia más lenta que el tick de update del LED (cacheados entre probes), así
un subprocess lento no clava las actualizaciones del LED ni satura la red.
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
    """Señales mutables que el pipeline escribe; el monitor las lee.

    Las asignaciones de campos son atómicas bajo el GIL de CPython para tipos
    primitivos, así las lecturas en el thread monitor no necesitan locking.
    No agregar containers mutables sin agregar un lock.
    """

    last_loop_ts: float = 0.0
    capture_ok: bool = True
    detect_ok: bool = True
    mqtt_connected: bool = False
    provisioned: bool = True
    boot_complete: bool = False
    # Init crasheó: lo setea main() en el except alrededor de run_pipeline
    # ANTES de morir, así el tick del monitor muestra BOOT_FAILURE (rojo)
    # en vez del estado engañoso que dan los defaults (verde/NO_CLOUD).
    boot_failed: bool = False
    calibration_path: Optional[str] = None


class HealthMonitor:
    """Thread background que mapea :class:`HealthSignals` a estado del LED.

    Args:
        led: El :class:`StatusLED` a manejar.
        signals: Objeto de señales compartido que el pipeline actualiza.
        poll_interval_s: Cada cuánto re-evaluar el estado (default 2 s).
        internet_probe_interval_s: Cada cuánto correr el probe de internet
            blocking (default 30 s; cacheado entre probes).
        hailo_probe_interval_s: Cada cuánto correr el probe de temperatura del
            Hailo vía ``hailortcli`` subprocess (default 30 s; cacheado). No
            necesita resolución de 2 s — el die térmico cambia lento — y a 5 s
            de timeout cada tick clavaría el thread monitor.
        boot_grace_s: Cuánto esperar a que el pipeline declare
            ``boot_complete`` antes de considerar el init wedgeado y mostrar
            BOOT_FAILURE (rojo fijo). El init normal (load del HEF + open de
            cámaras + MQTT) toma 10-30 s; 120 s da margen amplio. Sin este
            gate, un ``load_model``/``capture.open`` colgado mostraba verde
            (NO_CLOUD) para siempre — el operador leía "problema de internet"
            frente a una falla de boot.
    """

    def __init__(
        self,
        led: StatusLED,
        signals: HealthSignals,
        poll_interval_s: float = 2.0,
        internet_probe_interval_s: float = 30.0,
        hailo_probe_interval_s: float = 30.0,
        boot_grace_s: float = 120.0,
    ) -> None:
        self._led = led
        self._signals = signals
        self._poll_interval_s = float(poll_interval_s)
        self._internet_probe_interval_s = float(internet_probe_interval_s)
        self._hailo_probe_interval_s = float(hailo_probe_interval_s)
        self._boot_grace_s = float(boot_grace_s)
        self._created_ts = time.time()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_internet_probe = 0.0
        self._cached_internet_ok = True
        self._last_hailo_probe = 0.0
        self._cached_hailo_ok = True

    def start(self) -> None:
        """Arranca el thread monitor. Idempotente."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="health-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Detiene y joinea el thread monitor."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def evaluate_once(self) -> LedState:
        """Corre un tick sincrónicamente; principalmente para tests."""
        return self._tick()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Falló tick del health monitor")
            self._stop_event.wait(timeout=self._poll_interval_s)

    def _tick(self) -> LedState:
        s = self._signals
        now = time.time()

        # Probe de Hailo cacheado en su propia cadencia: el subprocess
        # hailortcli puede tardar hasta 5 s y NO debe correr en cada tick
        # (clavaría el LED y la cadencia del probe de internet). Se computa
        # fuera del short-circuit de hw_ok para que la cadencia sea estable.
        if now - self._last_hailo_probe >= self._hailo_probe_interval_s:
            self._cached_hailo_ok = check_hailo_temp_ok()
            self._last_hailo_probe = now
        hailo_ok = self._cached_hailo_ok

        hw_ok = (
            s.capture_ok
            and s.detect_ok
            and check_cpu_temp_ok()
            and hailo_ok
            and check_disk_ok()
            and check_calibration_loadable(s.calibration_path)
        )

        # Freshness del pipeline: se juzga solo una vez que el loop produjo
        # al menos una iteración. Un ts en cero significa que el pipeline
        # todavía no arrancó (fase de boot) — no marcar falla mientras el
        # operador sigue esperando.
        if s.last_loop_ts == 0.0:
            pipeline_ok = True
        else:
            pipeline_ok = (now - s.last_loop_ts) <= PIPELINE_STALL_S

        if now - self._last_internet_probe >= self._internet_probe_interval_s:
            self._cached_internet_ok = check_internet()
            self._last_internet_probe = now
        internet_ok = self._cached_internet_ok

        cloud_connected = bool(s.mqtt_connected)

        # Boot failure: explícito (main marcó boot_failed en el except de
        # init antes de morir) o implícito (init wedgeado: venció el grace
        # sin que el pipeline declare boot_complete). Sin este wiring,
        # decide_state aceptaba boot_failure pero nadie se lo pasaba —
        # BOOT_FAILURE (rojo) era inalcanzable en producción.
        boot_failure = s.boot_failed or (
            not s.boot_complete and (now - self._created_ts) >= self._boot_grace_s
        )

        state = decide_state(
            boot_failure=boot_failure,
            hardware_ok=hw_ok,
            pipeline_ok=pipeline_ok,
            internet_ok=internet_ok,
            cloud_connected=cloud_connected,
            provisioned=s.provisioned,
        )
        self._led.set_state(state)
        return state

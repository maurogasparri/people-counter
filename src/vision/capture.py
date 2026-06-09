"""Adquisición de frames estéreo desde dos cámaras CSI.

Soporta tres modos:
  - picamera2: cámaras CSI de la RPi5 vía libcamera/picamera2 (producción).
  - opencv: cámaras USB o V4L2 vía OpenCV VideoCapture (fallback).
  - file: replay desde pares de imágenes guardados (testing/desarrollo).
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Sensor mode canónico del IMX708. Picamera2, sin hint de raw=,
# elige Mode 0 (1536×864 = crop central del 67% del sensor, HFOV
# efectivo ~80°) en vez del modo full-FOV. Eso silenciosamente
# reduce la cobertura de piso a ~50% del diseño (120° HFOV). Forzar
# raw={"size": CANONICAL_RAW_SIZE} selecciona Mode 1 — full-FOV
# binned 2×2, 56fps, HFOV 120°. Cambiar este valor invalida toda
# calibración previa.
CANONICAL_RAW_SIZE = (2304, 1296)

# Capturas consecutivas sin que avance el SensorTimestamp de una cámara antes
# de marcarla como caída. ~10 ≈ 0.3-0.4s a 25-30fps: un duplicado aislado (raro)
# no la marca, una cámara congelada / cable CSI muerto sí.
CAMERA_STALL_THRESHOLD = 10

# Rate-limit del log de errores del stream loop ante falla PERSISTENTE de
# captura (cable CSI muerto): el primer error va con traceback completo;
# después 1 línea cada N consecutivos. Sin esto eran ~20 tracebacks/seg
# sostenidos (~1.7M líneas/día) — wear de la microSD + journald saturado en
# un device desatendido. El backoff también escala: corto al principio
# (glitch transitorio se recupera rápido), 0.5s cuando es persistente.
_STREAM_ERROR_LOG_EVERY = 200
_STREAM_ERROR_BACKOFF_AFTER = 20


def _next_stall(ts: int, last_ts: int, stall: int) -> int:
    """Contador de stall del SensorTimestamp por cámara. ts==0 (libcamera sin
    SensorTimestamp) no es evaluable → resetea (no marca falla espuria)."""
    if ts == 0:
        return 0
    return stall + 1 if ts == last_ts else 0


class StereoCapture:
    """Maneja la captura simultánea desde las cámaras CSI izquierda y derecha vía picamera2."""

    def __init__(
        self,
        cam_left_id: int,
        cam_right_id: int,
        resolution: tuple[int, int],
        fps: int = 15,
        meter_mode: str = "matrix",
        lock_ae: bool = False,
        max_exposure_us: Optional[int] = None,
        sensor_raw_size: tuple[int, int] = CANONICAL_RAW_SIZE,
        initial_settle_seconds: float = 2.0,
        resettle_seconds: float = 1.5,
        async_capture: bool = True,
    ) -> None:
        """Inicializa la captura estéreo.

        Args:
            cam_left_id: Índice de la cámara izquierda según lista rpicam-hello --list-cameras.
            cam_right_id: Índice de la cámara derecha.
            resolution: Resolución de captura (width, height).
            fps: Frame rate objetivo.
            meter_mode: Modo de AE metering ('matrix', 'centre', 'spot'). Centre/spot
                ignoran la periferia del frame durante el settle del AE, útil cuando
                hay zonas brillantes fuera del área de trabajo que bajan la exposición
                sobre el target (board de calibración). Default 'matrix' matchea la
                ponderación whole-frame default de picamera2.
            lock_ae: Cuando es True, AE/AWB se lockean a los valores donde el AE
                hizo settle tras una espera de 1 segundo. Útil cuando la escena de
                calibración tiene luz fluctuante (luz natural, puertas que abren) —
                el lock evita drift independiente de AE entre L/R mid-session.
                Default False — para iluminación interior estable, AE auto es más
                simple y produce imágenes representativas para el reporte de ground-truth.
            max_exposure_us: Cap de exposure time en microsegundos. Cuando se setea,
                se aplica vía FrameDurationLimits=(max_exposure_us, max_exposure_us),
                que indirectamente cap-ea el shutter (ExposureTime ≤ FrameDuration).
                AE sigue activo dentro de ese límite, compensando con AnalogueGain
                cuando la luz baja. Default None (sin cap, usa FrameRate=fps con
                shutter hasta el frame interval ~1/fps). Pensado para limitar motion
                blur en runtime de detección — 16000us (16ms) reduce blur a ~5cm a
                3 m/s, manteniendo distribución de blur del modelo entrenado.
                Override fps efectivo: con max_exposure_us=16000 la cámara corre a
                ~60 FPS internamente; el pipeline siempre consume el frame más
                reciente (latest-only semantics del request loop).
            sensor_raw_size: Tamaño del stream RAW del sensor (w, h). Forza el
                sensor mode de picamera2 vía raw={"size": ...}. Default
                (2304, 1296) = Mode 1 IMX708, full-FOV binned 2×2 (120° HFOV).
                Sin override, picamera2 elige Mode 0 (1536×864 cropeado) que
                reduce el FOV a ~80°. Cambiar invalida la calibración existente.
            initial_settle_seconds: Segundos de espera entre ``cam.start()``
                y el lock provisional de AE (cuando ``lock_ae=True``). En
                sites con luz fluctuante puede necesitar 3-4s para
                convergencia.
            resettle_seconds: Segundos de espera entre re-habilitar AE y
                re-lockear en ``resettle_and_lock()``. Patrón del wizard:
                lock inicial → operador apreta Comenzar → re-settle →
                re-lock.
            async_capture: Cuando es True (default), ``read_with_metadata``
                NO bloquea esperando el frame: un thread productor captura
                el par sincronizado en background a un "latest slot" y la
                lectura devuelve el último par disponible. Solapa la captura
                (~22ms, bloqueante en picamera2) con el cómputo del pipeline
                → la captura sale del critical path. False = lectura
                bloqueante clásica (fallback / setup tools).
        """
        self.cam_left_id = cam_left_id
        self.cam_right_id = cam_right_id
        self.resolution = resolution
        self.fps = fps
        self.meter_mode = meter_mode
        self.lock_ae = lock_ae
        self.max_exposure_us = max_exposure_us
        self.sensor_raw_size = sensor_raw_size
        self.initial_settle_seconds = float(initial_settle_seconds)
        self.resettle_seconds = float(resettle_seconds)
        self._cam_left = None
        self._cam_right = None
        self._executor: Optional[ThreadPoolExecutor] = None
        # Captura async (double-buffering): thread productor + latest-slot.
        self.async_capture = bool(async_capture)
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop = threading.Event()
        self._slot_ready = threading.Event()  # thread -> consumidor: par listo
        self._slot_consumed = threading.Event()  # consumidor -> thread: backpressure
        self._stream_lock = threading.Lock()
        self._stream_slot: Optional[
            tuple[np.ndarray, np.ndarray, int, int, Optional[float], Optional[float]]
        ] = None
        self._stream_error: Optional[str] = None
        # Health por cámara: stall del SensorTimestamp (no avanza = cámara
        # congelada/muerta). Lo escribe el stream loop, lo lee la telemetría.
        self._cam_l_last_ts = 0
        self._cam_r_last_ts = 0
        self._cam_l_stall = 0
        self._cam_r_stall = 0

    def open(self) -> None:
        """Abre ambos streams de cámara vía picamera2.

        Raises:
            RuntimeError: Si alguna cámara no puede abrirse.
        """
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise RuntimeError(
                "picamera2 no instalado. " "Instalar con: pip install picamera2"
            )

        try:
            self._cam_left = Picamera2(self.cam_left_id)
            self._cam_right = Picamera2(self.cam_right_id)
        except Exception as e:
            self.close()
            raise RuntimeError(f"Falló abrir las cámaras: {e}") from e

        w, h = self.resolution
        # Si max_exposure_us está seteado, usamos FrameDurationLimits en lugar de
        # FrameRate. El cap de FrameDuration indirectamente cap-ea ExposureTime
        # (en libcamera, ExposureTime ≤ FrameDuration). AE sigue activo y
        # compensa con AnalogueGain cuando la luz baja. Sin esto, AE puede
        # subir el shutter hasta ~30ms en interior y generar motion blur OOD
        # del training distribution para personas en movimiento rápido.
        if self.max_exposure_us is not None:
            initial_controls = {
                "FrameDurationLimits": (self.max_exposure_us, self.max_exposure_us),
            }
        else:
            initial_controls = {"FrameRate": self.fps}
        for cam, name in [
            (self._cam_left, "left"),
            (self._cam_right, "right"),
        ]:
            # Formato "RGB888": en los builds de RPi OS Trixie / libcamera con
            # los que shippeamos, los nombres de formato de picamera2 están
            # invertidos empíricamente — "BGR888" entrega RGB y "RGB888"
            # entrega BGR. Pedir "RGB888" da BGR directo del ISP y elimina
            # los 2 cv2.cvtColor por par que había en el hot path (~28 fps).
            # Verificado en hardware 2026-06-09: diff medio 1.77 (ruido de AE
            # entre capturas) contra el output del pipeline anterior
            # (BGR888 + cvtColor), con los means por canal espejados exactos.
            config = cam.create_still_configuration(
                main={"size": (w, h), "format": "RGB888"},
                raw={"size": self.sensor_raw_size},
                controls=initial_controls,
            )
            cam.configure(config)
            cam.start()

        # Setea el modo de AE metering antes del settle, así la exposición
        # lockeada refleja el weighting elegido (matrix/centre/spot).
        if self.meter_mode != "matrix":
            try:
                from libcamera import controls as _libcam_controls

                meter_map = {
                    "centre": _libcam_controls.AeMeteringModeEnum.CentreWeighted,
                    "spot": _libcam_controls.AeMeteringModeEnum.Spot,
                }
                meter_value = meter_map[self.meter_mode]
                for cam in [self._cam_left, self._cam_right]:
                    cam.set_controls({"AeMeteringMode": meter_value})
                logger.info(
                    "ae_metering_mode_set",
                    extra={"mode": self.meter_mode},
                )
            except (ImportError, KeyError, Exception) as e:
                logger.warning(
                    "ae_metering_mode_failed",
                    extra={"mode": self.meter_mode, "error": str(e)},
                )

        # Opcionalmente lockea exposición, gain y white balance tras un settle
        # provisional de 2s. El comportamiento default (lock_ae=False) mantiene
        # AE auto durante toda la sesión — más simple, produce imágenes
        # ground-truth representativas y anda bien para iluminación interior
        # estable. Activar lock_ae=True cuando la escena tiene luz fluctuante
        # (luz natural, puertas que abren, iluminación mixta) — el lock evita
        # AE independiente entre L/R durante la sesión.
        if self.lock_ae:
            import time as _time

            _time.sleep(self.initial_settle_seconds)
            self._lock_ae_to_current_metadata(event="initial")

        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="stereo-cap"
        )

        logger.info(
            "stereo_capture_opened",
            extra={
                "left_id": self.cam_left_id,
                "right_id": self.cam_right_id,
                "resolution": list(self.resolution),
                "fps": self.fps,
            },
        )

    def _lock_ae_to_current_metadata(self, event: str = "lock") -> None:
        """Lockea AE/AWB en cada cámara con el snapshot actual de metadata.

        Asume que las cámaras ya están abiertas y settleadas. Idempotente —
        si las cámaras ya están lockeadas, capture_metadata devuelve los
        valores actuales y los re-aplica, sin side effect.
        """
        if self._cam_left is None or self._cam_right is None:
            raise RuntimeError("Cámaras no abiertas. Llamar open() primero.")
        for cam, name in [
            (self._cam_left, "left"),
            (self._cam_right, "right"),
        ]:
            metadata = cam.capture_metadata()
            cam.set_controls(
                {
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": metadata.get("ExposureTime", 30000),
                    "AnalogueGain": metadata.get("AnalogueGain", 1.0),
                    "ColourGains": metadata.get("ColourGains", (1.0, 1.0)),
                }
            )
            logger.info(
                "camera_controls_locked",
                extra={
                    "camera": name,
                    "event": event,
                    "exposure_us": metadata.get("ExposureTime", 0),
                    "analogue_gain": metadata.get("AnalogueGain", 0),
                },
            )

    def resettle_and_lock(
        self,
        settle_seconds: Optional[float] = None,
    ) -> None:
        """Re-habilita AE/AWB, espera ``settle_seconds`` para que reconverja
        a la escena actual, y vuelve a lockear con el snapshot fresco.

        Patrón canónico para setup tools browser-driven: el lock inicial
        del ``open()`` ocurre cuando el operador acaba de lanzar el script
        (escena todavía sin board en posición); cuando el operador apreta
        "Comenzar" con todo armado, este método refresca el lock con la
        escena real de medición.

        No-op si ``lock_ae=False`` en el constructor — la sesión corre con
        AE auto a propósito.

        Args:
            settle_seconds: Override del settle por llamada. Si es None,
                usa ``self.resettle_seconds`` (default del constructor /
                config).
        """
        if not self.lock_ae:
            return
        if self._cam_left is None or self._cam_right is None:
            raise RuntimeError("Cámaras no abiertas. Llamar open() primero.")
        import time as _time

        dt = (
            float(settle_seconds)
            if settle_seconds is not None
            else self.resettle_seconds
        )
        for cam in [self._cam_left, self._cam_right]:
            cam.set_controls({"AeEnable": True, "AwbEnable": True})
        _time.sleep(dt)
        self._lock_ae_to_current_metadata(event="resettle")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Lee un par de frames.

        Returns:
            (left_frame, right_frame) como arrays BGR de numpy.

        Raises:
            RuntimeError: Si las cámaras no están abiertas o falla la lectura.
        """
        if self._cam_left is None or self._cam_right is None or self._executor is None:
            raise RuntimeError("Cámaras no abiertas. Llamar open() primero.")

        if self.async_capture:
            slot = self._read_async_slot()
            return slot[0], slot[1]

        try:
            fut_l = self._executor.submit(self._cam_left.capture_array, "main")
            fut_r = self._executor.submit(self._cam_right.capture_array, "main")
            frame_l = fut_l.result()
            frame_r = fut_r.result()
        except Exception as e:
            raise RuntimeError(f"Falló la captura de frame: {e}") from e

        # Sin cvtColor: el formato "RGB888" pedido en open() ya entrega BGR
        # directo del ISP (nombres invertidos empíricamente en picamera2 —
        # ver el comentario en la configuración). Los consumers downstream
        # (OpenCV, preproceso YOLO) ven el mismo orden de canales de siempre.
        return frame_l, frame_r

    def read_with_metadata(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, Optional[float], Optional[float]]:
        """Lee par de frames con timestamps de sensor (ns) y temperaturas (°C).

        Con ``async_capture`` (default) devuelve el último par capturado por
        el thread productor SIN bloquear; si no, captura de forma bloqueante.

        La temperatura viene de la key SensorTemperature de la metadata de
        picamera2 cuando está disponible (el IMX708 en RPi5 la expone). Devuelve
        None por cámara si la key no está — builds más viejos de libcamera o
        sensores no-IMX708.
        """
        if self._cam_left is None or self._cam_right is None or self._executor is None:
            raise RuntimeError("Cámaras no abiertas. Llamar open() primero.")

        if self.async_capture:
            return self._read_async_slot()

        return self._capture_pair_with_metadata()

    def _read_async_slot(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, Optional[float], Optional[float]]:
        """Devuelve el último par del thread productor (frames + metadata) SIN
        bloquear, y libera al thread para capturar el siguiente. Consumidor
        único (el pipeline usa read() O read_with_metadata, no ambos)."""
        self._ensure_stream_started()
        if not self._slot_ready.wait(timeout=5.0):
            with self._stream_lock:
                err = self._stream_error
            raise RuntimeError(
                f"Timeout esperando el frame del stream de captura: {err}"
            )
        with self._stream_lock:
            slot = self._stream_slot
            err = self._stream_error
        if slot is None:
            raise RuntimeError(f"Captura async sin frame disponible: {err}")
        # Liberá al thread para que capture el SIGUIENTE par mientras el
        # pipeline procesa éste (overlap one-ahead, sin free-run).
        self._slot_ready.clear()
        self._slot_consumed.set()
        return slot

    def _capture_pair_with_metadata(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, Optional[float], Optional[float]]:
        """Captura BLOQUEANTE de un par sincronizado + metadata. Lógica core
        compartida por la lectura síncrona y el thread de streaming. Las dos
        cámaras se capturan en paralelo (executor)."""

        def _grab(cam):
            req = cam.capture_request()
            try:
                frame = req.make_array("main")
                metadata = req.get_metadata()
                ts = int(metadata.get("SensorTimestamp", 0))
                temp_raw = metadata.get("SensorTemperature")
                temp = float(temp_raw) if temp_raw is not None else None
                return frame, ts, temp
            finally:
                req.release()

        try:
            fut_l = self._executor.submit(_grab, self._cam_left)
            fut_r = self._executor.submit(_grab, self._cam_right)
            frame_l, ts_l, temp_l = fut_l.result()
            frame_r, ts_r, temp_r = fut_r.result()
        except Exception as e:
            raise RuntimeError(f"Falló la captura de frame: {e}") from e

        # Sin cvtColor — "RGB888" entrega BGR directo (ver open()).
        return frame_l, frame_r, ts_l, ts_r, temp_l, temp_r

    def _ensure_stream_started(self) -> None:
        """Lanza (idempotente) el thread productor de captura. Se llama lazy
        en el primer read async — así todo el setup de AE (settle/lock, que
        usa capture_metadata) ya terminó y no compite con el thread."""
        if self._stream_thread is not None:
            return
        self._stream_stop.clear()
        self._slot_ready.clear()
        self._slot_consumed.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="stereo-stream", daemon=True
        )
        self._stream_thread.start()
        logger.info("stereo_async_capture_started")

    def _stream_loop(self) -> None:
        """Captura pares de forma continua a ``_stream_slot``. Corre en su
        propio thread; el main loop consume el último par sin bloquear. La
        cámara corre más rápido que el cómputo del pipeline, así que siempre
        hay un frame fresco listo y el consumidor nunca espera."""
        consecutive_errors = 0
        while not self._stream_stop.is_set():
            try:
                pair = self._capture_pair_with_metadata()
            except Exception as e:
                consecutive_errors += 1
                with self._stream_lock:
                    self._stream_error = str(e)
                if consecutive_errors == 1:
                    logger.exception("stereo_stream_capture_failed")
                elif consecutive_errors % _STREAM_ERROR_LOG_EVERY == 0:
                    logger.error(
                        "stereo_stream_capture_failed persistente "
                        "(%d consecutivos): %s",
                        consecutive_errors,
                        e,
                    )
                delay = (
                    0.05 if consecutive_errors < _STREAM_ERROR_BACKOFF_AFTER else 0.5
                )
                self._stream_stop.wait(delay)  # backoff: no spinear en error
                continue
            consecutive_errors = 0
            ts_l, ts_r = pair[2], pair[3]
            with self._stream_lock:
                self._cam_l_stall = _next_stall(
                    ts_l, self._cam_l_last_ts, self._cam_l_stall
                )
                self._cam_r_stall = _next_stall(
                    ts_r, self._cam_r_last_ts, self._cam_r_stall
                )
                self._cam_l_last_ts = ts_l
                self._cam_r_last_ts = ts_r
                self._stream_slot = pair
                self._stream_error = None
            self._slot_ready.set()
            # Backpressure: esperar a que el consumidor tome este par antes de
            # capturar el siguiente. Sin esto el thread free-runea (captura más
            # rápido que el consumo) y desperdicia CPU/ancho de banda con el
            # cvtColor, compitiéndole al main loop (medido: bajaba los FPS).
            self._slot_consumed.wait()
            self._slot_consumed.clear()

    def camera_health(self) -> tuple[bool, bool]:
        """``(left_ok, right_ok)``. Una cámara está NOT ok si su SensorTimestamp
        dejó de avanzar por ≥ ``CAMERA_STALL_THRESHOLD`` capturas (congelada o
        cable CSI muerto). Permite atribuir una caída de FPS a la cámara
        culpable sin subir al techo a adivinar. Ambas True hasta que el stream
        loop corra (o si libcamera no expone SensorTimestamp)."""
        with self._stream_lock:
            return (
                self._cam_l_stall < CAMERA_STALL_THRESHOLD,
                self._cam_r_stall < CAMERA_STALL_THRESHOLD,
            )

    def close(self) -> None:
        """Libera los recursos de las cámaras."""
        # Frenar el thread productor ANTES del executor (el thread lo usa).
        stream_wedged = False
        if self._stream_thread is not None:
            self._stream_stop.set()
            self._slot_consumed.set()  # desbloquea el thread si espera backpressure
            self._stream_thread.join(timeout=2.0)
            stream_wedged = self._stream_thread.is_alive()
            if stream_wedged:
                logger.warning("stereo_stream_thread_join_timeout_wedged")
            self._stream_thread = None
        if self._executor is not None:
            # wait=False si el productor sigue wedgeado (típicamente colgado
            # dentro de un capture_request con la cámara stalleada): un
            # shutdown(wait=True) esperaría para siempre a ese future y
            # close() nunca llegaría a cam.stop() — el shutdown "limpio"
            # terminaba en kill de systemd por TimeoutStopSec, justo en el
            # escenario de falla que CAMERA_STALL_THRESHOLD detecta.
            self._executor.shutdown(wait=not stream_wedged)
            self._executor = None
        for cam, name in [
            (self._cam_left, "left"),
            (self._cam_right, "right"),
        ]:
            if cam is not None:
                try:
                    cam.stop()
                    cam.close()
                except Exception:
                    logger.warning("camera_close_failed", extra={"camera": name})
        self._cam_left = None
        self._cam_right = None
        logger.info("stereo_capture_closed")

    def __enter__(self) -> "StereoCapture":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FileCapture:
    """Reproduce pares de frames estéreo desde archivos de imagen guardados.

    Busca archivos llamados left_NNN.png y right_NNN.png en el directorio
    dado. Útil para desarrollo y testing sin hardware.
    """

    def __init__(
        self,
        directory: str,
        loop: bool = True,
        fps: int = 15,
    ) -> None:
        """Inicializa la captura desde archivos.

        Args:
            directory: Path al directorio que contiene imágenes left_*/right_*.
            loop: Si reiniciar desde el principio luego de todos los pares.
            fps: Frame rate simulado (controla el delay de lectura).
        """
        self.directory = Path(directory)
        self.loop = loop
        self.fps = fps
        self._pairs: list[tuple[Path, Path]] = []
        self._index = 0
        self._frame_interval = 1.0 / fps if fps > 0 else 0
        self._last_read = 0.0

    def open(self) -> None:
        """Escanea el directorio en busca de pares de imágenes.

        Raises:
            RuntimeError: Si no se encuentra ningún par válido.
        """
        left_files = sorted(self.directory.glob("left_*.png"))
        self._pairs = []

        for lf in left_files:
            rf = lf.parent / lf.name.replace("left_", "right_")
            if rf.exists():
                self._pairs.append((lf, rf))

        if not self._pairs:
            # Probar también con .jpg
            left_files = sorted(self.directory.glob("left_*.jpg"))
            for lf in left_files:
                rf = lf.parent / lf.name.replace("left_", "right_")
                if rf.exists():
                    self._pairs.append((lf, rf))

        if not self._pairs:
            raise RuntimeError(
                f"No stereo pairs encontrados en {self.directory}. "
                "Se esperaban archivos llamados left_NNN.png y right_NNN.png"
            )

        self._index = 0
        logger.info(
            "file_capture_opened",
            extra={"pairs": len(self._pairs), "path": str(self.directory)},
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Lee el siguiente par de frames.

        Returns:
            (left_frame, right_frame) como arrays BGR de numpy.

        Raises:
            StopIteration: Si se consumieron todos los pares y loop=False.
            RuntimeError: Si los pares no fueron cargados.
        """
        if not self._pairs:
            raise RuntimeError("No pairs loaded. Llamar open() primero.")

        if self._index >= len(self._pairs):
            if self.loop:
                self._index = 0
            else:
                raise StopIteration("Todos los pares de frames consumidos")

        # Simula el frame rate
        now = time.monotonic()
        elapsed = now - self._last_read
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)
        self._last_read = time.monotonic()

        lf, rf = self._pairs[self._index]
        img_l = cv2.imread(str(lf))
        img_r = cv2.imread(str(rf))

        if img_l is None or img_r is None:
            raise RuntimeError(f"Falló la lectura del par en el índice {self._index}")

        self._index += 1
        return img_l, img_r

    @property
    def total_pairs(self) -> int:
        return len(self._pairs)

    @property
    def current_index(self) -> int:
        return self._index

    def close(self) -> None:
        """Resetea el estado."""
        self._pairs = []
        self._index = 0

    def __enter__(self) -> "FileCapture":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

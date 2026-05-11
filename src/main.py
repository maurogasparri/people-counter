"""People Counter — entry point principal.

Orquesta el pipeline edge completo:
    1. Captura estéreo → rectificación
    2. Disparidad → depth map
    3. Detección de personas YOLOv8n
    4. Tracking 3D + conteo por línea virtual
    5. Publicación de eventos vía MQTT (con buffer)
    6. Reporte de telemetría
    7. Probing WiFi/BLE (opcional)
"""

import argparse
import logging
import math
import os
import queue
import signal
import socket
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from src.config.loader import (
    DEFAULT_DEVICE_CONFIG_PATH,
    apply_shadow_delta,
    build_reported_state,
    get_invalid_schedule_mode,
    get_scaling_factor,
    has_schedule_error,
    is_counting_enabled,
    is_within_operating_hours,
    load_config,
    merge_cloud_config,
)
from src.mqtt.buffer import MessageBuffer
from src.mqtt.client import MQTTClient
from src.status.led import StatusLED
from src.status.monitor import HealthMonitor, HealthSignals
from src.telemetry import collect_telemetry
from src.tracking.counter import Counter, build_counter
from src.tracking.tracker import EuclideanTracker
from src.vision.calibration import load_calibration, rectify_pair
from src.vision.capture import FileCapture, StereoCapture
from src.vision.depth import (
    compute_disparity,
    create_sgbm,
    depth_at_bbox,
    disparity_to_depth,
    enable_depth_debug,
    head_depth_in_bbox,
)
from src.vision.best_frame import BestFrameManager
from src.vision.world_coords import classify_height, head_height_above_floor
from src.vision.detect import detect_persons, load_model
from src.vision.static_suppressor import StaticSuppressor
from src.web.annotate import annotate_left, compose_3panel, depth_to_colormap
from src.web.viewer import WebViewer

# Tamaño de las ventanas rolling usadas para los percentiles de latencia por
# frame y los cálculos de detection-rate. 100 cubre ~7s a 15 FPS — suficiente
# para suavizar ruido sin ahogar stalls breves.
TELEMETRY_WINDOW_SIZE = 100

logger = logging.getLogger(__name__)


def sd_notify(message: str) -> None:
    """Manda una notificación a systemd vía NOTIFY_SOCKET.

    No-op si la env var del socket no está seteada (es decir, corriendo
    fuera de systemd). Se usa para mandar READY=1, WATCHDOG=1, STOPPING=1
    en servicios Type=notify.
    """
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    if sock_path.startswith("@"):  # Namespace abstracto en Linux
        sock_path = "\0" + sock_path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(sock_path)
            s.sendall(message.encode())
    except OSError as e:
        logger.debug("sd_notify(%s) failed: %s", message, e)


def setup_logging(config: dict[str, Any]) -> None:
    """Configura logging a partir del config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))

    if log_cfg.get("format") == "json":
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_file = log_cfg.get("file")
    if log_file:
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def _runtime_resolution(config: dict[str, Any]) -> tuple[int, int]:
    """Elige la resolución de runtime desde el config.

    ``config.vision.resolution`` es la única fuente de verdad (defaults
    mergeados de config.example.yaml; el YAML per-device puede overridear).
    Cae al sensor default_res solo cuando ``vision`` falta enteramente
    (fixtures legacy / mínimas de tests).
    """
    cfg_res = (config.get("vision") or {}).get("resolution")
    if cfg_res:
        return tuple(cfg_res)
    return tuple(config["sensor"]["default_res"])


def _runtime_fps(config: dict[str, Any]) -> int:
    cfg_fps = (config.get("vision") or {}).get("fps")
    if cfg_fps:
        return int(cfg_fps)
    return int(config["sensor"]["default_fps"])


def build_capture(config: dict[str, Any], replay_dir: str | None = None):
    """Arma la fuente de captura adecuada."""
    if replay_dir:
        logger.info("Usando file replay desde %s", replay_dir)
        cap = FileCapture(
            directory=replay_dir,
            loop=True,
            fps=_runtime_fps(config),
        )
    else:
        # camera_left / camera_right vienen de config.bracket — seteado en
        # el lab, determinado por qué puerto CSI conecta cada ribbon.
        # max_exposure_us es opcional: cap-ea shutter time vía FrameDurationLimits
        # para reducir motion blur en runtime. Con AE auto y luz interior típica,
        # el shutter sube hasta 30ms — produce blur OOD del training distribution
        # para personas en movimiento rápido. 16000us (16ms) achata el blur a
        # niveles cubiertos por los frames motion-trigger del dataset.
        max_exposure_us = config.get("vision", {}).get("max_exposure_us")
        cap = StereoCapture(
            cam_left_id=config["bracket"]["camera_left_csi"],
            cam_right_id=config["bracket"]["camera_right_csi"],
            resolution=_runtime_resolution(config),
            fps=_runtime_fps(config),
            max_exposure_us=max_exposure_us,
        )
    return cap


class _NullMQTTClient:
    """Cliente MQTT no-op para debug local.

    Se usa cuando se pasa --no-mqtt: skipea todo I/O de red y en vez de eso
    loggea cada publish a stdout. El pipeline corre end-to-end así el operador
    puede validar eventos detect / track / count, pero nada se transmite a
    AWS IoT — útil antes de provisionar la infra de AWS o mientras se itera
    el runtime.
    """

    def __init__(self) -> None:
        self.connected = False
        self.disconnect_count = 0
        self.on_connected = None

    def connect(self) -> None:
        logger.info(
            "MQTT deshabilitado (--no-mqtt) — los publishes se loggearán a stdout, "
            "no se transmite nada a AWS IoT.",
        )
        self.connected = True
        if self.on_connected is not None:
            try:
                self.on_connected()
            except Exception:
                logger.exception("el callback on_connected raiseó")

    def disconnect(self) -> None:
        self.connected = False

    def subscribe_shadow_delta(self, device_id, handler) -> None:  # noqa: ARG002
        pass

    def publish_shadow_reported(self, device_id, reported) -> None:  # noqa: ARG002
        logger.info("[no-mqtt][shadow.reported] %s", reported)

    def publish_event(self, topic_key: str, event: dict) -> None:
        logger.info("[no-mqtt][%s] %s", topic_key, event)


def build_mqtt(
    config: dict[str, Any],
    no_mqtt: bool = False,
) -> tuple[Any, MessageBuffer]:
    """Arma cliente MQTT y buffer a partir del config.

    Cuando no_mqtt=True, devuelve un cliente no-op que loggea los publishes
    a stdout en vez de conectarse a AWS IoT.
    """
    buf_cfg = config["buffer"]
    buffer = MessageBuffer(
        db_path=buf_cfg["db_path"],
        max_age_hours=buf_cfg["max_age_hours"],
    )

    if no_mqtt:
        return _NullMQTTClient(), buffer

    mqtt_cfg = config["mqtt"]
    store_id = config["device"]["store_id"]
    device_id = config["device"]["id"]

    # Los templates de topic se interpolan con {store_id} / {device_id}.
    topics = {}
    for key, template in mqtt_cfg["topics"].items():
        topics[key] = template.replace("{store_id}", store_id).replace(
            "{device_id}", device_id
        )

    client = MQTTClient(
        device_id=config["device"]["id"],
        endpoint=mqtt_cfg["endpoint"],
        port=mqtt_cfg["port"],
        cert_path=mqtt_cfg["cert_path"],
        key_path=mqtt_cfg["key_path"],
        ca_path=mqtt_cfg["ca_path"],
        buffer=buffer,
        topics=topics,
    )
    return client, buffer


def _build_telemetry_state(
    frame_latencies_ms: "deque[float]",
    detection_counts: "deque[int]",
    detection_window_start_ts: float,
    fps: float | int | None,
    tracker: EuclideanTracker,
    mqtt_client: MQTTClient,
    buffer: MessageBuffer,
) -> dict[str, Any]:
    """Toma snapshot del estado runtime del pipeline al dict que espera
    :func:`collect_telemetry`. Cada lookup está wrappeado así un solo probe
    fallido nunca bloquea la emisión de telemetría.
    """
    state: dict[str, Any] = {
        "frame_latencies_ms": list(frame_latencies_ms),
        "detection_counts": list(detection_counts),
        "detection_window_start_ts": detection_window_start_ts,
        "fps": fps,
    }

    try:
        track_counts = tracker.count_by_state()
        state["tracker_confirmed"] = track_counts.get("confirmed", 0)
        state["tracker_pending"] = track_counts.get("pending", 0)
    except Exception:
        logger.exception("tracker.count_by_state failed")
        state["tracker_confirmed"] = None
        state["tracker_pending"] = None

    try:
        state["mqtt_disconnect_count"] = mqtt_client.disconnect_count
    except Exception:
        state["mqtt_disconnect_count"] = None
    try:
        state["mqtt_reconnect_ts"] = mqtt_client.reconnect_ts
    except Exception:
        state["mqtt_reconnect_ts"] = None

    try:
        state["buffer_backlog"] = buffer.count_unsent()
    except Exception:
        logger.exception("buffer.count_unsent failed")
        state["buffer_backlog"] = None

    return state


def get_telemetry(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recolecta telemetría del dispositivo.

    Wrapper delgado sobre :func:`src.telemetry.collect_telemetry` mantenido
    por backwards compatibility. La implementación original devolvía
    ``uptime_s = 0`` cuando ``/proc/uptime`` era ilegible; el módulo nuevo
    devuelve ``None``. Los callers que requieran un uptime numérico (tests
    legacy corriendo en Windows) hacen la coerción acá.
    """
    telem = collect_telemetry(state)
    if telem.get("uptime_s") is None:
        telem["uptime_s"] = 0
    return telem


def _auto_num_disparities(
    config: dict[str, Any],
    logger: logging.Logger,
) -> int:
    """Deriva un num_disparities que matchea el rango de distancia de cabezas
    en este sitio. Usa el baseline de diseño + un focal length escalado a la
    resolución de runtime, más el mounting_height_m que setea el operador.
    Redondea hacia arriba al siguiente múltiplo de 16 (constraint de SGBM).
    """
    vision_cfg = config["vision"]
    mount_m = float(vision_cfg.get("mounting_height_m", 3.0) or 3.0)
    full_w = float(config["sensor"]["full_res"][0])
    nominal_focal_full_px = float(config["sensor"]["nominal_focal_full_px"])
    baseline_mm = float(config["bracket"]["baseline_mm"])
    runtime_w = float(
        tuple(vision_cfg.get("resolution") or config["sensor"]["default_res"])[0]
    )
    # f_px escala linealmente con el ancho (región centro pinhole-equiv). El
    # focal del .npz va a estar cerca una vez que la calibración cargue; este
    # estimado solo necesita ser suficiente para dimensionar el rango de
    # búsqueda del SGBM.
    f_px = nominal_focal_full_px * runtime_w / full_w
    head_max_m = 1.85  # cabeza adulta más alta
    floor_margin_m = 0.5  # un poco más allá del piso así SGBM cubre todo el bg

    z_min = max(0.3, mount_m - head_max_m)
    z_max = mount_m + floor_margin_m
    disp_max = f_px * baseline_mm / (z_min * 1000)
    disp_min = f_px * baseline_mm / (z_max * 1000)

    # SGBM busca [minDisparity, minDisparity + numDisparities). Mantenemos
    # minDisparity=0 (default de OpenCV) por simplicidad, así que
    # numDisparities tiene que cubrir 0..disp_max. Redondear hacia arriba al
    # próximo múltiplo de 16 + un bucket extra de margen.
    raw = int(math.ceil(disp_max / 16.0) + 1) * 16
    # Clamp a un envelope razonable.
    num_disparities = max(64, min(512, raw))
    logger.info(
        "num_disparities auto: %dpx wide, mount=%.2fm → z=[%.2f,%.2f]m"
        " → disp=[%.0f,%.0f] → num_disparities=%d",
        int(runtime_w),
        mount_m,
        z_min,
        z_max,
        disp_min,
        disp_max,
        num_disparities,
    )
    return num_disparities


def run_pipeline(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Corre el pipeline principal de procesamiento."""
    device_id = config["device"]["id"]
    store_id = config["device"]["store_id"]
    # Todos los settings ahora están en `config` — defaults de config.example.yaml
    # (bundled en el repo) mergeados con /etc/people-counter/config.yaml
    # (overrides per-device) al momento de load_config(). Los deltas del
    # cloud shadow aplican encima vía apply_shadow_delta() dentro del main loop.
    vision_cfg = config["vision"]
    detect_cfg = config["detection"]
    track_cfg = config["tracking"]
    telem_cfg = config.get("telemetry", {})
    status_cfg = config.get("status_led", {}) or {}

    # --- Status LED + health monitor (arrancados antes de cualquier init pesado
    # así puede surface las fallas de hardware durante load del modelo / open
    # de cámaras). El LED cae a no-op cuando gpiozero falta, así esto es seguro
    # fuera de la RPi.
    health_signals = HealthSignals(
        provisioned=True,
        calibration_path=vision_cfg.get("calibration_file"),
    )
    status_led: StatusLED | None = None
    health_monitor: HealthMonitor | None = None
    if bool(status_cfg.get("enabled", True)):
        status_led = StatusLED()
        health_monitor = HealthMonitor(led=status_led, signals=health_signals)
        health_monitor.start()

    # --- Web viewer en vivo --------------------------------------------------
    # Default ON, puerto 80 (--web-viewer-port 0 deshabilita). Falla de bind
    # loggea warning y deja ``viewer = None`` así el pipeline sigue sin él.
    # Puerto 80 necesita CAP_NET_BIND_SERVICE bajo systemd; el unit del servicio
    # se lo otorga. Las dev runs fuera de systemd necesitan root para bindear <1024.
    viewer: WebViewer | None = None
    web_port = int(getattr(args, "web_viewer_port", 80))
    if web_port > 0:
        viewer = WebViewer(port=web_port)
        if not viewer.start():
            viewer = None
    last_depth_panel: np.ndarray | None = None
    last_fps_estimate: float = 0.0
    # Rate-limita SGBM para el viewer en vivo cuando no hay detecciones.
    # Si no, el panel de profundidad fuerza SGBM en cada frame aunque el
    # local esté vacío (~95% de los frames en un deploy real), gastando
    # ~60ms por frame a sgbm_downscale=4 para nada.
    VIEWER_DEPTH_INTERVAL_S = 0.5  # refresh de panel a 2 fps cuando está idle
    last_viewer_depth_t: float = 0.0

    # --- Cargar calibración ---
    # Un ``null`` explícito para calibration_file en config deshabilita la
    # calibración entera (usado por tests / runs pre-calibración). El .npz
    # en disco DEBE matchear la resolución de runtime o la depth está mal
    # silenciosamente.
    cal_file = vision_cfg.get("calibration_file")
    if cal_file:
        logger.info("Cargando calibración desde %s", cal_file)
        calibration = load_calibration(cal_file)
    else:
        calibration = None
        logger.warning("No hay archivo de calibración configurado — salteando rectificación")

    # --- Cargar modelo de detección ---
    model_path = detect_cfg["model_path"]
    detection_backend = getattr(args, "detection_backend", "auto")
    architecture = detect_cfg["architecture"]
    logger.info(
        "Cargando modelo: %s (backend=%s, arch=%s)",
        model_path,
        detection_backend,
        architecture,
    )
    model = load_model(
        model_path,
        backend=detection_backend,
        architecture=architecture,
    )

    # --- Construir SGBM ---
    sgbm_cfg = vision_cfg["sgbm"]
    # num_disparities puede ser un int concreto o el literal "auto" — en cuyo
    # caso derivamos el rango de búsqueda desde mounting_height_m + resolución
    # de runtime así el solver cubre exactamente la banda de profundidad
    # donde van a aparecer las cabezas más un margen chico de piso.
    num_disp_cfg = sgbm_cfg["num_disparities"]
    if isinstance(num_disp_cfg, str) and num_disp_cfg.lower() == "auto":
        num_disp_resolved = _auto_num_disparities(config, logger)
    else:
        num_disp_resolved = int(num_disp_cfg)
    # Downscale de SGBM: procesa a menor resolución por velocidad.
    # depth.compute_disparity después upscalea el mapa de disparidad y rescalea
    # los valores así la matemática de profundidad sigue siendo correcta.
    sgbm_downscale = int(sgbm_cfg["downscale"])
    if sgbm_downscale not in (1, 2, 4, 8):
        logger.warning(
            "vision.sgbm.downscale=%d inválido (usar 1/2/4/8) — fallback a 4",
            sgbm_downscale,
        )
        sgbm_downscale = 4
    # Pasa el num_disparities escalado a create_sgbm así el matcher opera a la
    # resolución downscaleada; compute_disparity rescalea los valores hacia arriba.
    sgbm_num_disp = (
        max(16, (num_disp_resolved // sgbm_downscale // 16) * 16)
        if sgbm_downscale > 1
        else num_disp_resolved
    )
    sgbm = create_sgbm(
        num_disparities=sgbm_num_disp,
        block_size=int(sgbm_cfg["block_size"]),
    )
    logger.info(
        "SGBM: downscale=%d, num_disparities=%d (effective at downscaled res)",
        sgbm_downscale,
        sgbm_num_disp,
    )

    # --- Construir tracker + counter ---
    sm_cfg = track_cfg["state_machine"]
    counter_cfg = config["counter"]
    # Los overrides per-site del tracker pueden vivir bajo counter.tracker
    # (path cloud-safe para tuning en runtime). En caso contrario usamos los
    # defaults mergeados de config.tracking.
    tracker_cfg = counter_cfg.get("tracker", {})
    tracker = EuclideanTracker(
        max_disappeared=int(
            tracker_cfg.get("max_disappeared", track_cfg["max_disappeared"])
        ),
        max_distance=float(
            tracker_cfg.get("max_distance", track_cfg["max_distance"])
        ),
        max_depth_delta=float(
            tracker_cfg.get("depth_gate_m", sm_cfg["depth_gate_m"])
        ) * 1000.0,
        confirm_frames=int(
            tracker_cfg.get("confirm_frames", sm_cfg["confirm_frames"])
        ),
        pending_max_frames=int(
            tracker_cfg.get("pending_max_frames", sm_cfg["pending_max_frames"])
        ),
        reid_gate_px=float(
            tracker_cfg.get("reid_gate_px", sm_cfg["reid_gate_px"])
        ),
        # Default 0.5 = nueva semántica production-grade (track sin obs
        # converge a quieto en ~3 frames). 1.0 desactiva (back-compat).
        # tracker_cfg precedence permite override per-site vía shadow.
        pending_velocity_decay=float(
            tracker_cfg.get(
                "pending_velocity_decay",
                sm_cfg.get("pending_velocity_decay", 0.5),
            )
        ),
    )

    # Static FP suppressor: defense-in-depth contra detecciones que el
    # detector emite consistentemente sobre clutter del ambiente
    # (sombras, blobs estructurales). Inactivo durante warm-up; cuando
    # se acumulan ~3s de historia identifica celdas hot por hit rate y
    # las descarta. Configurable bajo `detection.static_suppressor`;
    # `enabled: false` lo deshabilita (default true).
    detect_cfg_init = config.get("detection", {})
    ss_cfg = detect_cfg_init.get("static_suppressor", {}) or {}
    static_suppressor: StaticSuppressor | None
    if ss_cfg.get("enabled", True):
        static_suppressor = StaticSuppressor(
            cell_size_px=int(ss_cfg.get("cell_size_px", 30)),
            window_seconds=float(ss_cfg.get("window_seconds", 3.0)),
            hit_rate_threshold=float(ss_cfg.get("hit_rate_threshold", 0.7)),
            approx_fps=int(ss_cfg.get("approx_fps", 17)),
        )
        logger.info(
            "static_suppressor_enabled cell=%dpx window=%.1fs threshold=%.2f",
            static_suppressor.cell_size_px,
            float(ss_cfg.get("window_seconds", 3.0)),
            static_suppressor.hit_rate_threshold,
        )
    else:
        static_suppressor = None

    line_y = vision_cfg.get("counting_line_y", 0.5)
    # Counter construido lazy una vez que se conoce la altura del frame
    # (necesario para los valores relativos legacy de line_y)
    counter: Counter | None = None

    # Height bounds (head_depth gate + sanity filter). Cualquier knob
    # acepta "auto" o un valor literal en metros; "auto" calcula
    # `min(anthropometric_default, mount - geometric_clearance)` así
    # los valores siguen siendo coherentes incluso para mounts
    # inusuales (muy bajos) donde el default haría que el gate
    # colapse. Para mounts retail típicos (>2.4m) "auto" siempre
    # resuelve al anthropometric default.
    def _resolve_height_bound_mm(
        value,
        anthropometric_default_m: float,
        mount_m: float,
        geometric_clearance_m: float,
    ) -> float:
        if isinstance(value, str) and value.strip().lower() == "auto":
            if mount_m <= 0:
                return anthropometric_default_m * 1000.0
            geometric = mount_m - geometric_clearance_m
            return min(anthropometric_default_m, geometric) * 1000.0
        return float(value) * 1000.0

    mount_m_init = float(vision_cfg.get("mounting_height_m", 0.0) or 0.0)

    head_depth_cfg = config.get("head_depth", {}) or {}
    head_depth_max_mm = _resolve_height_bound_mm(
        head_depth_cfg.get("max_head_height_m", "auto"),
        anthropometric_default_m=1.80,
        mount_m=mount_m_init,
        geometric_clearance_m=0.30,
    )
    head_depth_min_mm = _resolve_height_bound_mm(
        head_depth_cfg.get("min_head_above_floor_m", 0.50),
        anthropometric_default_m=0.50,
        mount_m=mount_m_init,
        geometric_clearance_m=0.0,
    )
    # Radio de la columna (mm) para el filtro espacial 3-D dentro de
    # head_depth_in_bbox. Knob plano float-en-metros — sin auto, sin
    # mount-dependence: el radio es un hecho antropométrico sobre el ancho
    # de columna en el que entra el cuerpo humano.
    head_depth_column_radius_mm = (
        float(head_depth_cfg.get("column_radius_m", 0.25)) * 1000.0
    )
    # Percentile sobre los depths del blob (default 75): captura la superficie
    # del cráneo en el slice nearest del histograma sin promediar con
    # speckle near-camera de SGBM. Ver head_depth_in_bbox docstring para
    # rationale completo.
    head_depth_blob_percentile = float(
        head_depth_cfg.get("blob_percentile", 75.0)
    )

    height_cfg = config.get("height", {}) or {}
    height_sanity_min_mm = _resolve_height_bound_mm(
        height_cfg.get("sanity_min_m", 0.80),
        anthropometric_default_m=0.80,
        mount_m=mount_m_init,
        geometric_clearance_m=0.0,
    )
    height_sanity_max_mm = _resolve_height_bound_mm(
        height_cfg.get("sanity_max_m", "auto"),
        anthropometric_default_m=2.10,
        mount_m=mount_m_init,
        geometric_clearance_m=0.20,
    )

    logger.info(
        "Height bounds (mount=%.2fm): "
        "head_depth=[%.2f, %.2f]m, sanity=[%.2f, %.2f]m",
        mount_m_init,
        head_depth_min_mm / 1000.0,
        head_depth_max_mm / 1000.0,
        height_sanity_min_mm / 1000.0,
        height_sanity_max_mm / 1000.0,
    )

    # --- Best-frame selector (default OFF — baseline privacy-safe) ---
    # Cuando está disabled en config el manager se queda None y los paths
    # per-frame de abajo cortocircuitan con un único check ``is None``, así
    # que el overhead en el deploy default es cero. Ver src/vision/best_frame.py
    # y docs/privacy.md para el diseño completo + contexto legal.
    best_frame_mgr: BestFrameManager | None = None
    bf_cfg = config.get("best_frame", {}) or {}
    if bool(bf_cfg.get("enabled", False)):
        try:
            best_frame_mgr = BestFrameManager(
                output_dir=bf_cfg["output_dir"],
                buffer_size=int(bf_cfg.get("buffer_size", 20)),
                jpeg_quality=int(bf_cfg.get("jpeg_quality", 85)),
                weights=bf_cfg.get("scoring") or None,
            )
            logger.warning(
                "best_frame.enabled=True — los JPGs se van a escribir a %s "
                "(retention=%dd). Confirmar que DPIA + signage + privacy policy "
                "estén en su lugar.",
                bf_cfg.get("output_dir"),
                int(bf_cfg.get("retention_days", 7)),
            )
        except Exception:
            logger.exception(
                "init del manager best_frame falló; feature deshabilitado en este run."
            )
            best_frame_mgr = None

    # --- Construir MQTT ---
    mqtt_client, buffer = build_mqtt(
        config,
        no_mqtt=getattr(args, "no_mqtt", False),
    )

    # --- Wiring de los shadow delta ------------------------------------------
    # Los deltas llegan en el thread de red de paho; la queue es el handoff
    # thread-safe al main loop, que es el único writer de `config`. La misma
    # queue también lleva un sentinel ``{"__reconcile__": True}`` posteado por
    # el hook on_connected de MQTT, así el publishing de reported-state corre
    # en el main thread (sin llamadas de red en el thread de paho).
    shadow_queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=32)
    _RECONCILE_SENTINEL = {"__reconcile__": True}

    def _shadow_delta_handler(state: dict[str, Any]) -> None:
        try:
            shadow_queue.put_nowait(state)
        except queue.Full:
            logger.warning("Queue de shadow delta llena — descartando delta")

    def _on_mqtt_connected() -> None:
        # Corre en el thread de paho — encolar, no bloquear.
        try:
            shadow_queue.put_nowait(_RECONCILE_SENTINEL)
        except queue.Full:
            logger.warning("Shadow queue llena — descartando request de reconcile")

    mqtt_client.on_connected = _on_mqtt_connected
    mqtt_client.connect()

    try:
        mqtt_client.subscribe_shadow_delta(device_id, _shadow_delta_handler)
    except Exception:
        logger.exception("Falló la subscripción a shadow delta")

    config_path_arg = getattr(args, "config", None)

    # --- Construir captura ---
    capture = build_capture(config, replay_dir=getattr(args, "replay_dir", None))
    capture.open()

    # --- Focal length + baseline para profundidad ---
    # El .npz de calibración overridea ambos al cargar (ver _bootstrap_optical_*).
    # Hasta entonces, fallback a la constante de diseño del bracket del config.
    focal_length_px = None
    baseline_mm = float(config["bracket"]["baseline_mm"])
    # Intrínsecos rectificados (cámara izquierda) necesarios para la back-proyección
    # 3-D en head_depth_in_bbox. Se populan junto con focal_length_px una vez
    # que la calibración está cargada; hasta entonces ``None`` deshabilita el
    # filtro de columna 3-D y head_depth_in_bbox es no-op (devuelve None) para height.
    cx_rect_px: float | None = None
    cy_rect_px: float | None = None

    # --- Timer de telemetría ---
    telem_interval = telem_cfg.get("interval_seconds", 300)
    last_telem = time.time()

    # --- Shutdown gracioso ---
    running = True

    def _signal_handler(sig: int, frame: Any) -> None:
        nonlocal running
        logger.info("Señal de shutdown recibida (%d)", sig)
        running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        "Pipeline arrancado: device=%s store=%s",
        device_id,
        store_id,
    )

    frame_count = 0
    fps_start = time.time()
    telem_frame_count = 0
    telem_fps_start = time.time()
    last_hours_check = 0.0
    last_purge = time.time()
    last_watchdog = 0.0
    within_hours = True  # asumimos abierto hasta el primer check
    profile_enabled = bool(getattr(args, "profile", False))
    profile_every_n = max(1, int(getattr(args, "profile_every_n", 30)))
    profile_frame_idx = 0

    # --- Estado de observability (ventanas sliding) ---
    # Buffers rolling de latencia per-frame y counts de detección. Se limpian
    # después de cada emisión de telemetría así cada sample representa el
    # intervalo anterior.
    frame_latencies_ms: deque[float] = deque(maxlen=TELEMETRY_WINDOW_SIZE)
    detection_counts: deque[int] = deque(maxlen=TELEMETRY_WINDOW_SIZE)
    detection_window_start_ts = time.time()

    # Modo de falla de operating-hours — solo loggea el warning de startup una vez.
    schedule_invalid = has_schedule_error(config)
    invalid_mode = get_invalid_schedule_mode(config)
    if schedule_invalid:
        err = config.get("_schedule_error", "unknown")
        if invalid_mode == "fail_closed":
            logger.critical(
                "Invalid operating_hours (%s) and on_invalid_schedule=fail_closed "
                "— counting paused until a valid schedule is pushed",
                err,
            )
        else:
            logger.warning(
                "Invalid operating_hours (%s) and on_invalid_schedule=fail_open "
                "— continuing to count (may produce false positives)",
                err,
            )

    # Señaliza a systemd que el startup terminó. Pareado con WatchdogSec en
    # el archivo de unit — pingueamos WATCHDOG=1 dentro del loop de abajo.
    sd_notify("READY=1")
    health_signals.boot_complete = True

    try:
        while running:
            # --- Drenar shadow deltas pendientes (lado main-thread) ---
            while True:
                try:
                    delta_state = shadow_queue.get_nowait()
                except queue.Empty:
                    break
                # Sentinel de reconcile (en (re)connect): publica el estado
                # efectivo actual como reported. No-fatal en error.
                if delta_state is _RECONCILE_SENTINEL or delta_state.get(
                    "__reconcile__"
                ):
                    try:
                        reported = build_reported_state(config, calibration)
                        mqtt_client.publish_shadow_reported(device_id, reported)
                        logger.info(
                            "shadow_reconciliation_published",
                            extra={"keys": sorted(reported.keys())},
                        )
                    except Exception:
                        logger.exception("Falló publicar la reconciliación del shadow")
                    continue
                try:
                    new_config, applied_keys = apply_shadow_delta(
                        config, delta_state, config_path=config_path_arg
                    )
                except Exception:
                    logger.exception("apply_shadow_delta falló")
                    continue
                if not applied_keys:
                    continue
                config = new_config
                # Re-evalúa el flag de schedule luego de un update del shadow.
                schedule_invalid = has_schedule_error(config)
                invalid_mode = get_invalid_schedule_mode(config)
                last_hours_check = 0.0  # forzar recheck en la próxima iteración

                # Re-init selectivo basado en qué keys cambiaron.
                # counter.tracker.* y counter.height_classifier.* se leen en
                # vivo cada frame — no se necesita rebuild.
                if any(
                    k.startswith("counter.")
                    and not k.startswith("counter.tracker.")
                    and not k.startswith("counter.height_classifier.")
                    for k in applied_keys
                ):
                    counter = None
                    logger.info("Counter se va a reconstruir luego del shadow delta")
                if any(k.startswith("counter.tracker.") for k in applied_keys):
                    logger.info(
                        "Params del tracker actualizados; los nuevos valores aplican a tracks futuros"
                    )
                if any(
                    k
                    in (
                        "vision.num_disparities",
                        "vision.block_size",
                        "vision.mounting_height_m",
                    )
                    for k in applied_keys
                ):
                    vision_live = config["vision"]
                    sgbm_live = vision_live["sgbm"]
                    nd_cfg = sgbm_live["num_disparities"]
                    if isinstance(nd_cfg, str) and nd_cfg.lower() == "auto":
                        nd_resolved = _auto_num_disparities(config, logger)
                    else:
                        nd_resolved = int(nd_cfg)
                    sgbm_num_disp = (
                        max(16, (nd_resolved // sgbm_downscale // 16) * 16)
                        if sgbm_downscale > 1
                        else nd_resolved
                    )
                    sgbm = create_sgbm(
                        num_disparities=sgbm_num_disp,
                        block_size=int(sgbm_live["block_size"]),
                    )
                    logger.info("SGBM reconstruido luego del shadow delta")
                if any(k == "telemetry.interval_seconds" for k in applied_keys):
                    telem_interval = config.get("telemetry", {}).get(
                        "interval_seconds", telem_interval
                    )

                try:
                    mqtt_client.publish_shadow_reported(
                        device_id, {k: True for k in applied_keys}
                    )
                except Exception:
                    logger.exception("Falló publicar shadow reported")

            # --- Chequear operating hours cada 60 segundos ---
            # --ignore-schedule bypasea el gate enteramente — útil para runs
            # de PoC y sesiones de debug donde el config de operating_hours
            # de otra forma pausaría el pipeline.
            ignore_schedule = getattr(args, "ignore_schedule", False)
            now = time.time()
            if now - last_hours_check >= 60.0:
                if ignore_schedule:
                    within_hours = True
                elif schedule_invalid:
                    # fail_closed: tratar como fuera de horario; fail_open: contar.
                    within_hours = invalid_mode != "fail_closed"
                else:
                    dt = datetime.now()
                    day_name = dt.strftime("%A").lower()
                    within_hours = is_within_operating_hours(
                        config, day_name, dt.hour, dt.minute
                    )
                    if not within_hours:
                        logger.debug(
                            "Fuera de operating hours (%s %02d:%02d) — paused",
                            day_name,
                            dt.hour,
                            dt.minute,
                        )
                last_hours_check = now

            # --- Chequear si counting está enabled (toggle del cloud) ---
            if not is_counting_enabled(config) or not within_hours:
                # Mantener telemetría + watchdog vivos así ops puede re-pushear config.
                if schedule_invalid and invalid_mode == "fail_closed":
                    telem_now = time.time()
                    if telem_now - last_telem >= telem_interval:
                        telem = collect_telemetry(
                            _build_telemetry_state(
                                frame_latencies_ms,
                                detection_counts,
                                detection_window_start_ts,
                                config.get("vision", {}).get("fps"),
                                tracker,
                                mqtt_client,
                                buffer,
                            )
                        )
                        telem["error"] = "invalid_schedule"
                        telem["schedule_error_detail"] = config.get(
                            "_schedule_error", ""
                        )
                        mqtt_client.publish_event("telemetry", telem)
                        last_telem = telem_now
                    if telem_now - last_watchdog >= 60.0:
                        sd_notify("WATCHDOG=1")
                        last_watchdog = telem_now
                time.sleep(1.0)
                continue

            t_iter_start = time.perf_counter()
            t_capture_start = t_iter_start
            try:
                frame_l, frame_r = capture.read()
                health_signals.capture_ok = True
            except StopIteration:
                logger.info("File replay agotado")
                break
            except RuntimeError as e:
                logger.error("Error de captura: %s", e)
                health_signals.capture_ok = False
                time.sleep(0.1)
                continue
            t_capture_end = time.perf_counter()

            # --- Rectificación ---
            if calibration is not None:
                rect_l, rect_r = rectify_pair(frame_l, frame_r, calibration)
            else:
                rect_l, rect_r = frame_l, frame_r
            t_rectify_end = time.perf_counter()

            # --- Inicializar counter con la altura real del frame ---
            # Footpoint projection: cuando la calibración está cargada Y
            # ``counter.foot_projection_enabled`` es true (default false en
            # ``config/config.example.yaml``), tomamos el principal point
            # de P1 (matriz de proyección izquierda rectificada) y lo
            # combinamos con vision.mounting_height_m así el counter usa
            # pixels de pies corregidos por parallax para los cruces de
            # línea en vez del centroide del bbox. Disabled por default:
            # en geometrías de puerta central la proyección comprime la
            # trayectoria del foot-point dentro del ROI y los exits nunca
            # disparan (solo se detectan OUTs, sin INs). Cuando el toggle
            # está off — o cuando calibración / mount no están disponibles —
            # el counter cae al path del centroide.
            if counter is None:
                foot_projection_enabled = bool(
                    config["counter"]["foot_projection_enabled"]
                )
                if foot_projection_enabled and calibration is not None:
                    p1 = calibration["P1"]
                    principal_pt: tuple[float, float] | None = (
                        float(p1[0, 2]),
                        float(p1[1, 2]),
                    )
                else:
                    principal_pt = None
                mount_mm_for_counter = (
                    float(vision_cfg.get("mounting_height_m", 0.0) or 0.0) * 1000.0
                ) or None
                counter = build_counter(
                    config,
                    frame_height=rect_l.shape[0],
                    mounting_height_mm=mount_mm_for_counter,
                    principal_point=principal_pt,
                )
                logger.info(
                    "Counter inicializado: %s (footpoint_projection=%s, "
                    "toggle=%s)",
                    type(counter).__name__,
                    bool(principal_pt is not None and mount_mm_for_counter),
                    foot_projection_enabled,
                )

            # --- Setear focal length + baseline desde la calibración ---
            if focal_length_px is None and calibration is not None:
                focal_length_px = calibration["P1"][0, 0]
                # Usa el baseline real de la calibración (magnitud del vector T)
                T = calibration["T"]
                baseline_mm = float(np.linalg.norm(T))
                # Principal point rectificado (cámara izquierda). load_calibration
                # los populates junto con fx_rect; mirroreamos el fallback seguro
                # (P1 columnas 0,2 / 1,2) para archivos .npz más viejos generados
                # antes de ese cambio.
                cx_rect_px = float(calibration.get("cx_rect", calibration["P1"][0, 2]))
                cy_rect_px = float(calibration.get("cy_rect", calibration["P1"][1, 2]))
                logger.info(
                    "Focal length: %.1f px, Baseline: %.1f mm, "
                    "Principal point: (%.1f, %.1f)",
                    focal_length_px,
                    baseline_mm,
                    cx_rect_px,
                    cy_rect_px,
                )

            # --- Detección PRIMERO (depth on demand abajo) ---
            # Reordering: detect → compute depth solo si hay detecciones.
            # SGBM domina el budget per-frame (~80ms). Saltearlo en frames
            # sin detecciones (la mayoría en deploys reales) libera ese
            # budget para mayor FPS general sin perder info de depth donde
            # realmente importa (sobre las personas detectadas).
            # Threshold de matching two-stage (estilo ByteTrack). Cuando
            # ``low_confidence_threshold`` está seteado en config, corremos el
            # detector con el floor más bajo así obtenemos todas las
            # detecciones >= low; main.py después separa high vs low por
            # confidence per-detección y pipea el bucket low al tracker como
            # candidatos match-only que nunca spawnean tracks nuevos. Cuando
            # es null/falta, el detector corre con el threshold high regular
            # y el comportamiento es idéntico a single-stage.
            #
            # ``new_track_threshold`` (opcional, >= confidence_threshold)
            # sube el floor de spawn independientemente. Las detecciones en
            # [confidence_threshold, new_track_threshold) se vuelven
            # match-only — misma banda que los candidatos low-conf pero del
            # otro lado de confidence_threshold. Separar
            # ``new_track_threshold`` de ``confidence_threshold`` evita IDs
            # fantasma cuando el detector titubea sobre clutter.
            high_conf_thr = float(detect_cfg["confidence_threshold"])
            low_conf_thr_raw = detect_cfg.get("low_confidence_threshold")
            new_track_thr_raw = detect_cfg.get("new_track_threshold")
            two_stage_active = (
                low_conf_thr_raw is not None
                and float(low_conf_thr_raw) > 0.0
                and float(low_conf_thr_raw) < high_conf_thr
            )
            # Spawn floor: las detecciones >= spawn_thr son elegibles para
            # crear tracks nuevos. Default a confidence_threshold cuando
            # new_track_threshold es null/falta (comportamiento legacy).
            spawn_thr = (
                max(high_conf_thr, float(new_track_thr_raw))
                if new_track_thr_raw is not None
                else high_conf_thr
            )
            # Detect floor: la conf más baja a la que el detector emite un
            # bbox. Por debajo, los pixels se descartan antes de llegar al
            # tracker. Igual a low_conf_thr cuando two-stage está activo,
            # sino al threshold high-conf regular.
            detect_floor = (
                float(low_conf_thr_raw) if two_stage_active else high_conf_thr
            )
            try:
                all_detections = detect_persons(
                    rect_l,
                    model,
                    confidence_threshold=detect_floor,
                    nms_threshold=float(detect_cfg["nms_threshold"]),
                    cluster_distance_px=float(detect_cfg["cluster_distance_px"]),
                )
                health_signals.detect_ok = True
            except Exception:
                logger.exception("Detección falló")
                health_signals.detect_ok = False
                time.sleep(0.1)
                continue

            # Static FP suppressor: cualquier celda hot acumulada en los
            # últimos ~3s se descarta acá, antes del split en buckets.
            # Durante warm-up devuelve la lista intacta. Si el feature
            # está disabled, suppressor=None y skip.
            if static_suppressor is not None:
                all_detections = static_suppressor.update_and_filter(
                    all_detections
                )

            # Separa las detecciones en buckets spawn-eligible vs match-only.
            # Spawn-eligible: conf >= spawn_thr (max de high_conf_thr y
            # new_track_threshold). Match-only: conf < spawn_thr pero >=
            # detect_floor (cualquier cosa debajo de detect_floor ya fue
            # cortada por el detector). Cuando ni low_confidence_threshold
            # ni new_track_threshold están seteados, spawn_thr == detect_floor
            # y el bucket match-only queda vacío (flow legacy single-stage).
            if spawn_thr > detect_floor:
                detections = [
                    d for d in all_detections if d.confidence >= spawn_thr
                ]
                low_conf_detections = [
                    d for d in all_detections if d.confidence < spawn_thr
                ]
            else:
                detections = all_detections
                low_conf_detections = []
            t_detect_end = time.perf_counter()

            # --- Mapa de profundidad (solo cuando hay detecciones que querear) ---
            # SGBM domina el costo per-frame (~60ms a downscale=4 en Pi 5).
            # Lo corremos solo cuando realmente necesitamos depth fresca:
            #   - hay detecciones → height + z del tracker la necesitan ahora,
            #   - viewer enabled y el último refresh fue hace >0.5s →
            #     mantener el panel de depth actualizándose a ~2 fps así un
            #     operador ve liveness incluso cuando no hay nadie en frame.
            # Cuando ninguna aplica dejamos depth_map=None y el viewer
            # reusa last_depth_panel del frame anterior.
            viewer_depth_due = (
                viewer is not None
                and (t_iter_start - last_viewer_depth_t) >= VIEWER_DEPTH_INTERVAL_S
            )
            if (
                calibration is not None
                and focal_length_px is not None
                and (detections or low_conf_detections or viewer_depth_due)
            ):
                # CLAHE off — el histograma indoor del IMX708 se comporta bien
                # para el matching SGBM y el costo de CLAHE de ~10ms no vale.
                disparity = compute_disparity(
                    rect_l,
                    rect_r,
                    sgbm=sgbm,
                    downscale=sgbm_downscale,
                    use_clahe=False,
                )
                depth_map = disparity_to_depth(disparity, focal_length_px, baseline_mm)
                if viewer_depth_due:
                    last_viewer_depth_t = t_iter_start
            else:
                depth_map = None
            t_depth_end = time.perf_counter()

            # --- Construir posiciones 3D + metadata per-detección ---
            vision_cfg = config.get("vision", {})
            counter_cfg_live = config.get("counter", {})
            hc_cfg = counter_cfg_live.get("height_classifier", {}) or {}
            hc_enabled = bool(hc_cfg.get("enabled", False))
            mount_height_mm = (
                float(vision_cfg.get("mounting_height_m", 0.0) or 0.0) * 1000.0
            )
            adult_min_mm = float(hc_cfg.get("adult_min_m", 1.55)) * 1000.0

            def _build_positions_metas(
                dets: list,
            ) -> tuple[list[np.ndarray], list[dict]]:
                """Arma posiciones del tracker + metas per-detección para una
                lista de objetos Detection. Compartida entre los buckets
                high-conf y low-conf así la semántica de depth/height queda
                idéntica entre ambas ramas del matching two-stage.
                """
                pos: list[np.ndarray] = []
                m: list[dict] = []
                for det in dets:
                    cx, cy = det.centroid
                    # Tracker z = mediana del crop central del bbox. Esa es
                    # la depth del "centro de masa" del cuerpo (torso para
                    # YOLO stock, cabeza para RAPiD) — lo que queremos para
                    # gating de re-id entre frames. Head depth necesita un
                    # estimate separado, específico-cabeza, porque en
                    # geometría cenital con el bbox YOLO-COCO el centroide
                    # cae sobre el torso, no la cabeza.
                    if depth_map is not None:
                        z = depth_at_bbox(depth_map, det.bbox)
                        # El filtro de columna 3-D requiere intrínsecos
                        # rectificados de la calibración. Si todavía no están
                        # cargados (sin calibración, o iteración pre-bootstrap)
                        # skipeamos head_depth — la altura de ese frame cae
                        # a "unknown" en vez de devolver un valor que no
                        # podemos confiar geométricamente.
                        head_depth_mm = (
                            head_depth_in_bbox(
                                depth_map,
                                det.bbox,
                                mount_height_mm,
                                fx_px=float(focal_length_px),
                                cx_px=float(cx_rect_px),
                                cy_px=float(cy_rect_px),
                                # RAPiD emite un rectángulo rotado body-aligned
                                # que colapsamos al ``bbox`` axis-aligned para
                                # los consumers tracker / NMS; acá queremos
                                # el polígono ajustado de vuelta así el sampler
                                # de depth ignora el piso + estructura vecina
                                # que el envelope axis-aligned barre. yolov8
                                # deja ``rotated`` como None y la función cae
                                # al modo bbox-only.
                                rotated_bbox=det.rotated,
                                max_head_height_mm=head_depth_max_mm,
                                min_head_above_floor_mm=head_depth_min_mm,
                                column_radius_mm=head_depth_column_radius_mm,
                                blob_percentile=head_depth_blob_percentile,
                                # Cuando --depth-debug está on, pasamos el
                                # frame rectificado + la confidence de la
                                # detección así el panel de dump puede mostrar
                                # la vista actual de la cámara junto al análisis
                                # de depth. Barato (None cuando el toggle está
                                # off, la función de dump cortocircuita antes
                                # de leer).
                                debug_frame=rect_l,
                                debug_confidence=float(det.confidence),
                            )
                            if (
                                hc_enabled
                                and mount_height_mm > 0
                                and focal_length_px is not None
                                and cx_rect_px is not None
                                and cy_rect_px is not None
                            )
                            else None
                        )
                    else:
                        z = 0.0
                        head_depth_mm = None
                    pos.append(np.array([cx, cy, z]))

                    if head_depth_mm is not None:
                        head_mm = head_height_above_floor(
                            head_depth_mm,
                            mount_height_mm,
                        )
                        height_class = classify_height(head_mm, adult_min_mm)
                    else:
                        head_mm = None
                        height_class = "unknown"

                    # Sanity gate: alturas fuera del rango anthropometric
                    # son físicamente imposibles (estructura overhead
                    # dominando el bbox, speckle SGBM cerca-cámara, drift
                    # de calibración). El evento de conteo igual se emite
                    # porque hay detección real cruzando la línea, pero los
                    # campos de altura caen a None / "unknown" para que el
                    # dashboard no surfacee el valor falso. head_depth_mm
                    # también se limpia: si el head pick fue absurdo, su
                    # depth no es confiable como near_depth_mm para el
                    # tracker — mejor usar el z (mediana del bbox central)
                    # como fallback estable.
                    if head_mm is not None and (
                        head_mm < height_sanity_min_mm or head_mm > height_sanity_max_mm
                    ):
                        head_mm = None
                        head_depth_mm = None
                        height_class = "unknown"
                    m.append(
                        {
                            "confidence": float(det.confidence),
                            "near_depth_mm": (
                                head_depth_mm if head_depth_mm is not None else z
                            ),
                            "head_height_mm": head_mm,
                            "height_class": height_class,
                            # bbox pasado así el counter puede tomar el
                            # bbox-top como "pixel de cabeza" para la
                            # proyección de footpoint. Tuple mantenido como
                            # ints para mirrorear Detection.bbox; el counter
                            # lo lee con casts a float así los tipos son
                            # robustos.
                            "bbox": tuple(int(v) for v in det.bbox),
                        }
                    )
                return pos, m

            positions, metas = _build_positions_metas(detections)

            # --- Tracking ---
            # Matching two-stage: high-conf puede spawnear + matchear,
            # low-conf solo re-asocia tracks sobrantes. Cuando
            # two_stage_active es False, low_conf_detections siempre es []
            # y los kwargs son no-ops — el tracker degenera a la signature
            # anterior single-bucket.
            if two_stage_active and low_conf_detections:
                low_positions, low_metas = _build_positions_metas(
                    low_conf_detections,
                )
                tracks = tracker.update(
                    positions,
                    detection_metas=metas,
                    candidate_positions=low_positions,
                    candidate_metadata=low_metas,
                )
            else:
                tracks = tracker.update(positions, detection_metas=metas)

            # --- Buffering best-frame (solo cuando el feature está enabled) ---
            # Para cada track CONFIRMED / PENDING actualmente vivo, encuentra la
            # detección cuyo centroide está más cerca de la última posición del
            # track y pushea ese (frame, bbox, conf) al buffer rolling. Los
            # tracks CANDIDATE se skipean intencionalmente — son el mismo noise
            # floor que el counter ignora, y bufferearlos desperdiciaría RAM en
            # fantasmas que nunca cuentan.
            #
            # El matching es greedy nearest-neighbour con un gate de 60 px (un
            # max_distance del tracker), barato como para ser costo no-op en el
            # hot path. Los misses (sin detección dentro del gate, ej: un track
            # PENDING en predict) se skipean silenciosamente — el buffer solo
            # mantiene frames donde realmente hubo un bbox de detección.
            if best_frame_mgr is not None and detections:
                for tid, trk in tracks.items():
                    if trk.state not in ("confirmed", "pending"):
                        continue
                    if not trk.positions:
                        continue
                    tx, ty = float(trk.positions[-1][0]), float(trk.positions[-1][1])
                    best_idx = -1
                    best_dist = 60.0  # gate; misma escala que tracker.max_distance
                    for di, det in enumerate(detections):
                        dx, dy = det.centroid
                        d = ((dx - tx) ** 2 + (dy - ty) ** 2) ** 0.5
                        if d < best_dist:
                            best_dist = d
                            best_idx = di
                    if best_idx >= 0:
                        det = detections[best_idx]
                        best_frame_mgr.observe(
                            track_id=tid,
                            frame=rect_l,
                            bbox=det.bbox,
                            confidence=float(det.confidence),
                        )
                # GC periódico: descarta buffers de tracks que ya no están vivos.
                # Barato (diff de set) así lo corremos cada frame.
                best_frame_mgr.gc(set(tracks.keys()))

            # --- Conteo ---
            events = counter.check_all(tracks)
            t_track_end = time.perf_counter()

            # --- Log de profiling ---
            # Cuando --profile está seteado, loggea un breakdown per-stage cada
            # PROFILE_EVERY_N frames. Ayuda a identificar el bottleneck real
            # (capture / rectify / depth / detect / track) en vez de adivinar
            # desde los FPS totales.
            if profile_enabled:
                profile_frame_idx += 1
                if profile_frame_idx % profile_every_n == 0:
                    cap_ms = (t_capture_end - t_capture_start) * 1000
                    rect_ms = (t_rectify_end - t_capture_end) * 1000
                    detect_ms = (t_detect_end - t_rectify_end) * 1000
                    depth_ms = (t_depth_end - t_detect_end) * 1000
                    track_ms = (t_track_end - t_depth_end) * 1000
                    total_ms = (t_track_end - t_iter_start) * 1000
                    has_dets = "Y" if depth_map is not None else "N"
                    logger.info(
                        "PROFILE frame=%d cap=%.0fms rect=%.0fms detect=%.0fms "
                        "depth=%.0fms (det=%s) track=%.0fms TOTAL=%.0fms (%.1f FPS)",
                        profile_frame_idx,
                        cap_ms,
                        rect_ms,
                        detect_ms,
                        depth_ms,
                        has_dets,
                        track_ms,
                        total_ms,
                        1000.0 / total_ms if total_ms > 0 else 0.0,
                    )

            # --- Publicar eventos de conteo ---
            scaling = get_scaling_factor(config)
            for event in events:
                # Best-frame: escribe el JPG elegido localmente y adjunta el
                # path al payload del evento. Solo el path viaja por MQTT —
                # los bytes de la imagen nunca dejan el dispositivo. Cuando
                # el feature está OFF (default) ``best_frame_path`` queda
                # None y se omite del payload por la construcción de dict
                # de abajo.
                best_frame_path: str | None = None
                if best_frame_mgr is not None:
                    best_frame_path = best_frame_mgr.commit(
                        event.track_id,
                        event.timestamp,
                    )

                payload = {
                    "direction": event.direction,
                    "track_id": event.track_id,
                    "position_y": event.position_y,
                    "event_time": event.timestamp,
                    "total_in": counter.total_in,
                    "total_out": counter.total_out,
                    "scaling_factor": scaling,
                    "scaled_in": round(counter.total_in * scaling),
                    "scaled_out": round(counter.total_out * scaling),
                    "height_class": event.height_class,
                    "height_m": (
                        round(event.height_m, 2) if event.height_m is not None else None
                    ),
                    "head_depth_m": (
                        round(event.head_depth_m, 2)
                        if event.head_depth_m is not None
                        else None
                    ),
                    "confidence": (
                        round(event.confidence, 3)
                        if event.confidence is not None
                        else None
                    ),
                }
                if best_frame_path is not None:
                    payload["best_frame_path"] = best_frame_path
                mqtt_client.publish_event("counting", payload)

            # --- Tracking de FPS ---
            frame_count += 1
            telem_frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 10.0:
                fps = frame_count / elapsed
                last_fps_estimate = fps
                logger.info(
                    "Pipeline: %.1f FPS, %d detections, in=%d out=%d",
                    fps,
                    len(detections),
                    counter.total_in,
                    counter.total_out,
                )
                frame_count = 0
                fps_start = time.time()

            # --- Push al web viewer ---
            # Composite de 3 paneles (L annotated | R raw | depth colormap).
            # El panel de depth está cacheado: SGBM se computa solo en frames
            # con detecciones (ver sección depth), así entre detecciones
            # mostramos el depth map más reciente en vez de un panel negro.
            # ``push`` es no-bloqueante; las fallas se loggean pero no rompen
            # el pipeline.
            if viewer is not None:
                try:
                    if depth_map is not None:
                        last_depth_panel = depth_to_colormap(depth_map)
                    elif last_depth_panel is None:
                        last_depth_panel = depth_to_colormap(None)
                    left_annot = annotate_left(
                        rect_l,
                        detections,
                        tracks,
                        counter,
                        fps=last_fps_estimate,
                    )
                    composite = compose_3panel(
                        left_annot,
                        rect_r,
                        last_depth_panel,
                    )
                    confirmed_or_pending = sum(
                        1
                        for t in tracks.values()
                        if getattr(t, "state", None) in ("confirmed", "pending")
                    )
                    viewer.push(
                        composite,
                        {
                            "total_in": counter.total_in,
                            "total_out": counter.total_out,
                            "fps": last_fps_estimate,
                            "tracks": confirmed_or_pending,
                            "dets": len(detections),
                        },
                    )
                except Exception:
                    logger.exception("Falló el push al web viewer")

            # --- Observability: registra latencia per-frame + count de detecciones ---
            frame_latencies_ms.append((time.perf_counter() - t_iter_start) * 1000.0)
            detection_counts.append(len(detections))

            # --- Telemetría ---
            now = time.time()
            if now - last_telem >= telem_interval:
                telem_elapsed = now - telem_fps_start
                telem = collect_telemetry(
                    _build_telemetry_state(
                        frame_latencies_ms,
                        detection_counts,
                        detection_window_start_ts,
                        config.get("vision", {}).get("fps"),
                        tracker,
                        mqtt_client,
                        buffer,
                    )
                )
                telem["fps"] = telem_frame_count / max(telem_elapsed, 1)
                telem["total_in"] = counter.total_in
                telem["total_out"] = counter.total_out
                mqtt_client.publish_event("telemetry", telem)
                last_telem = now
                telem_frame_count = 0
                telem_fps_start = now
                # Resetea las ventanas rolling así el próximo sample es independiente.
                frame_latencies_ms.clear()
                detection_counts.clear()
                detection_window_start_ts = now

            # --- Mantenimiento del buffer (cada 60s, no cada frame) ---
            if now - last_purge >= 60.0:
                buffer.purge_old()
                last_purge = now

            # --- Keepalive del watchdog de systemd (cada 60s; WatchdogSec=300) ---
            if now - last_watchdog >= 60.0:
                sd_notify("WATCHDOG=1")
                last_watchdog = now

            # --- Señales de health (las lee el thread del monitor del LED) ---
            health_signals.last_loop_ts = now
            health_signals.mqtt_connected = mqtt_client.connected

    finally:
        sd_notify("STOPPING=1")
        if viewer is not None:
            viewer.stop()
        if health_monitor is not None:
            health_monitor.stop()
        if status_led is not None:
            status_led.close()
        capture.close()
        mqtt_client.disconnect()
        # Libera recursos de Hailo si el backend lo soporta
        backend_impl = model.get("backend")
        if hasattr(backend_impl, "close"):
            backend_impl.close()
        logger.info(
            "Pipeline detenido. Counts finales: in=%d out=%d",
            counter.total_in if counter else 0,
            counter.total_out if counter else 0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="People Counter Edge Device")
    parser.add_argument(
        "--config",
        default=DEFAULT_DEVICE_CONFIG_PATH,
        help=f"Path al config YAML (default: {DEFAULT_DEVICE_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--replay-dir",
        help="Replay desde pares estéreo guardados en vez de cámaras en vivo",
    )
    parser.add_argument(
        "--detection-backend",
        choices=["auto", "hailo", "opencv"],
        default="auto",
        help="Backend de detección (default: auto-detecta desde la extensión del modelo)",
    )
    parser.add_argument(
        "--no-mqtt",
        action="store_true",
        help="Skipea MQTT enteramente (útil antes de provisionar la infra de "
        "AWS). Todos los publishes se loggean a stdout en vez de transmitirse. "
        "El pipeline corre end-to-end así los eventos detect / track / count "
        "son visibles en los logs.",
    )
    parser.add_argument(
        "--ignore-schedule",
        action="store_true",
        help="Bypasea el gate de operating_hours. El pipeline cuenta siempre, "
        "independiente del día de la semana / hora del día. Útil para runs de "
        "PoC y sesiones de debug fuera de las horas configuradas del local.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Loggea timing per-stage (capture / rectify / depth / detect / "
        "track) cada N frames así el bottleneck real es visible en vez de solo "
        "los FPS totales. Usar con --profile-every-n para tunear la frecuencia "
        "del log.",
    )
    parser.add_argument(
        "--profile-every-n",
        type=int,
        default=30,
        help="Cuando --profile está seteado, emite un log PROFILE cada N frames "
        "(default 30 — alrededor de cada 6 segundos a 5 FPS).",
    )
    parser.add_argument(
        "--web-viewer-port",
        type=int,
        default=80,
        help="Puerto HTTP para el viewer de debug en vivo. Streamea un composite "
        "de 3 paneles (left annotated | right raw | depth colormap) como MJPEG. "
        "Abrir la IP del dispositivo en un browser para ver en vivo. Default 80 "
        "(necesita CAP_NET_BIND_SERVICE bajo systemd; el unit del servicio se "
        "lo otorga). Pasar 0 para deshabilitar. Las fallas de bind se loggean "
        "pero no matan el pipeline.",
    )
    parser.add_argument(
        "--depth-debug",
        action="store_true",
        help="Dumpea hasta 5 PNGs a /tmp/depth_debug_*.png con el heatmap de "
        "depth + máscaras en capas (anthropometric / column / blob de cabeza "
        "elegido) para las primeras detecciones. Cada dump también loggea los "
        "counts del histograma así el diagnóstico sigue siendo útil sin el PNG. "
        "Auto-detiene tras 5 dumps; restart para re-armar.",
    )
    args = parser.parse_args()

    if args.depth_debug:
        enable_depth_debug(True)

    # ``load_config`` raisea FileNotFoundError o ValueError cuando el config
    # falta o está malformado (corrupto, sección requerida ausente, valor
    # inválido). En vez de dejar burbujear el stack trace ante el operador,
    # los pescamos acá y mostramos un mensaje claro con el path del archivo
    # y la causa, después salimos con código de error no-cero.
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(
            f"\nERROR: archivo de config no encontrado en '{args.config}'.\n"
            f"  Detalle: {e}\n"
            f"  Aprovisioná el config per-device en /etc/people-counter/config.yaml\n"
            f"  o pasá --config <path> apuntando al archivo correcto.\n",
            file=sys.stderr,
        )
        sys.exit(2)
    except ValueError as e:
        print(
            f"\nERROR: config inválido en '{args.config}'.\n"
            f"  Detalle: {e}\n"
            f"  El archivo existe pero le falta una sección/key requerida o\n"
            f"  tiene un valor mal-formado. Comparalo contra\n"
            f"  config/config.example.yaml en el repo para ver qué falta.\n"
            f"  Si fue corrupción de un edit reciente, restaurá desde\n"
            f"  /etc/people-counter/config.yaml.bak.* (los backups previos).\n",
            file=sys.stderr,
        )
        sys.exit(2)

    setup_logging(config)

    # --- Intenta mergear el cloud config desde el Shadow de IoT ---
    # En producción esto haría fetch de AWS IoT vía MQTT $aws/things/{id}/shadow/get.
    # Para el MVP, leemos un archivo local de shadow cache si existe (actualizado
    # por un proceso background o en el boot anterior). Si no hay, aplican los
    # defaults locales.
    from pathlib import Path
    import json

    config_path = Path(args.config)
    shadow_file = Path(str(config_path.with_suffix("")) + ".shadow.json")
    shadow_path = str(shadow_file)
    try:
        if shadow_file.exists():
            shadow_data = json.loads(shadow_file.read_text())
            desired = shadow_data.get("state", {}).get("desired", {})
            config = merge_cloud_config(config, desired)
            logger.info("Cloud shadow mergeado desde %s", shadow_path)
        else:
            logger.info("Sin shadow cache en %s — usando defaults locales", shadow_path)
    except Exception as e:
        logger.warning("Falló cargar shadow cache: %s — usando defaults locales", e)

    logger.info(
        "Arrancando people-counter",
        extra={"device_id": config["device"]["id"]},
    )

    run_pipeline(config, args)


if __name__ == "__main__":
    main()

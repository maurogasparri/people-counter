"""Loader de configuración strict + cloud shadow.

Estrategia:
    - PER-DEVICE: ``/etc/people-counter/config.yaml`` es la fuente única de
      verdad. Tiene que contener TODAS las keys que el runtime espera
      (validadas en ``_validate``); cualquier key faltante es error duro al
      arrancar.
    - CLOUD: el AWS IoT Device Shadow puede pushear overrides runtime-safe
      encima del config cargado (bloque ``cloud_defaults`` + una whitelist
      chica de keys operacionales — ver ``RUNTIME_SAFE_KEYS`` debajo).

``config/config.example.yaml`` queda como TEMPLATE documentado del repo —
útil para aprovisionar un device nuevo o para auditar qué keys son válidas,
pero NO se mergea en runtime. Cambios fleet-wide se shippean editando el
template + redeployando el config a cada device (no hay magic merge en el
hot path).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Defaults que se aplican cuando el bloque opcional ``best_frame`` falta
# enteramente. Mirrorea el schema en config.example.yaml — mantener ambos
# en sync. ``enabled: False`` es el default GDPR/LPDP-safe: cero storage,
# cero PII, cero side effects cuando el operador no hizo opt-in explícito.
BEST_FRAME_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "output_dir": "/var/lib/people-counter/best_frames",
    "retention_days": 7,
    "buffer_size": 20,
    "jpeg_quality": 85,
    "scoring": {
        "confidence_weight": 0.4,
        "bbox_area_weight": 0.2,
        "centrality_weight": 0.2,
        "sharpness_weight": 0.2,
    },
}


# Keys que pueden ser overrideadas por el cloud shadow bajo ``cloud_defaults``.
#
# Scope acotado a feature toggles end-user (no técnicos):
#   - operating_hours: el horario de la tienda cambia (feriados, eventos
#     especiales, extended hours) y el end user lo ajusta sin tocar device.
#   - counting_enabled: toggle on/off del conteo de personas (vision) para
#     mantenimiento, limpieza, evento privado, etc.
#   - external_traffic_enabled: toggle on/off del subsistema WiFi/BLE
#     (passersby / shoppers). Útil para apagar privacidad por sucursal
#     ante consulta legal / cliente que pide pausarlo, sin SSH.
#
# El resto de los knobs (thresholds, counting_zone, tracker tuning, vision params)
# requieren visualización o análisis técnico — se siguen modificando vía
# SSH a /etc/people-counter/config.yaml + restart.
CLOUD_OVERRIDABLE = {
    "operating_hours",
    "counting_enabled",
    "external_traffic_enabled",
}

# Whitelist dotted-path de keys del config que pueden aplicarse en runtime
# desde un delta del IoT Shadow sin restartear el servicio. Cualquier cosa
# fuera de este set (y keys desconocidas) se loggea como "requires restart"
# y se skipea.
RUNTIME_SAFE_KEYS = frozenset(
    {
        "cloud_defaults.operating_hours",
        "cloud_defaults.counting_enabled",
        "cloud_defaults.external_traffic_enabled",
    }
)

# Prefijos bajo los cuales cualquier child key es runtime-safe.
# Vacío: todos los knobs técnicos requieren restart (SSH + edit + systemctl
# restart). Documentado en infra/README.md.
RUNTIME_SAFE_PREFIXES: tuple[str, ...] = ()

DEFAULT_DEVICE_CONFIG_PATH = "/etc/people-counter/config.yaml"


def load_device_config(path: str | Path = DEFAULT_DEVICE_CONFIG_PATH) -> dict[str, Any]:
    """Carga el config per-device strict — solo lee ``path``, sin merge.

    Lo usan los setup tools (``calibrate.py``, ``focus_assist.py``,
    ``preview.py``, ``counting_zone_picker.py``, ``diagnose_depth.py``) que necesitan
    pullear runtime knobs como ``vision.resolution`` de la misma fuente que
    el pipeline. Strict por diseño: si el archivo no existe o le falta una
    key que el tool requiere, el tool falla rápido con un error claro en
    vez de caer a un default hardcoded que pueda diferir del runtime.

    NO mergea con ``config/config.example.yaml`` — la fuente única de verdad
    es ``/etc/people-counter/config.yaml`` del device. Cualquier device que
    corra setup tools tiene que tener un config completo (no minimal override
    contra la example).

    Args:
        path: Path al config per-device. Default ``/etc/people-counter/config.yaml``.

    Returns:
        Dict crudo del YAML. Sin validation — los callers chequean lo que
        necesitan ellos (ej. ``cfg["vision"]["resolution"]``).

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Device config not found: {config_path}. Los setup tools "
            "necesitan el config per-device completo del dispositivo."
        )
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str) -> dict[str, Any]:
    """Carga y valida la config del dispositivo desde un archivo YAML strict.

    NO mergea con ``config/config.example.yaml`` — el archivo del device es la
    fuente única de verdad y tiene que contener todas las keys requeridas
    (validadas por ``_validate``). Si falta una key, error duro.

    Args:
        path: Path al archivo YAML de config per-device.

    Returns:
        Dict de config validado. ``operating_hours`` se valida fail-fast
        (raise ValueError si está malformado) — desde que el shadow
        persiste al mismo ``config.yaml``, no hay separación local-válido
        vs cloud-inválido; los deltas inválidos se rechazan ANTES de
        persistir (en ``apply_shadow_delta``).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    _apply_compat_renames(config)
    _validate(config)
    _normalise_best_frame(config)

    # Validar operating_hours fail-fast. Si el operator SSH-editó un
    # schedule malformado, el startup falla con error claro (antes esto se
    # silenciaba con _schedule_error + fail_open). Para los deltas del
    # shadow, la validación equivalente se hace en `apply_shadow_delta`
    # (rechazo del delta sin escribir el config.yaml).
    schedule_error = validate_operating_hours(
        get_effective_value(config, "operating_hours", None)
    )
    if schedule_error is not None:
        raise ValueError(
            f"Invalid operating_hours en {config_path}: {schedule_error}"
        )

    return config


# ---------------------------------------------------------------------------
# Backward-compat: renombre de keys que cambiaron de nombre
# ---------------------------------------------------------------------------

# Mapa de keys con nombres viejos a los actuales. Cuando un config
# per-device todavía usa el nombre viejo, se renombra silenciosamente
# en memoria (con un warning) antes de validar. Permite migrar la flota
# sin sincronizar el push del código con el push del config.
#
# Cada entry: (path_tupla_old, key_new). La path_tupla apunta al dict
# parent y key_new es la nueva clave dentro de ese dict.
_COMPAT_RENAMES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    # tracking.max_disappeared → max_disappeared_frames
    (("tracking",), "max_disappeared", "max_disappeared_frames"),
    # tracking.max_distance → max_distance_px
    (("tracking",), "max_distance", "max_distance_px"),
    # vision.ae_lock.initial_settle_s → initial_settle_seconds
    (("vision", "ae_lock"), "initial_settle_s", "initial_settle_seconds"),
    # vision.ae_lock.resettle_s → resettle_seconds
    (("vision", "ae_lock"), "resettle_s", "resettle_seconds"),
)

# Keys obsoletas que deben ser eliminadas (no renombradas). Si están
# presentes, se loggea un warning y se las saca del dict.
_COMPAT_REMOVED: tuple[tuple[tuple[str, ...], str], ...] = (
    # vision.sensor_raw_size → ahora se usa sensor.default_res como
    # fuente única.
    (("vision",), "sensor_raw_size"),
    # detection.static_suppressor.approx_fps → el supresor mide la
    # ventana con timestamps reales, ya no necesita estimación de FPS.
    (("detection", "static_suppressor"), "approx_fps"),
)


def _walk_nested_dict(
    parent: dict[str, Any], path: tuple[str, ...]
) -> dict[str, Any] | None:
    """Camina ``path`` dentro de ``parent`` y devuelve el dict hoja, o None
    si algún tramo no existe o no es un dict. Compartido entre el loader
    runtime y ``scripts/migrate_config.py``."""
    node: Any = parent
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return node if isinstance(node, dict) else None


def _apply_compat_renames(config: dict[str, Any]) -> None:
    """Aplica renombres de keys con compat para configs viejos in-place.

    Loggea warnings así el operador puede migrar el config a los nombres
    canónicos cuando le convenga. Idempotente: si las keys ya tienen el
    nombre nuevo, no hace nada.
    """
    for path, old_key, new_key in _COMPAT_RENAMES:
        parent = _walk_nested_dict(config, path)
        if parent is None:
            continue
        if old_key in parent and new_key not in parent:
            parent[new_key] = parent.pop(old_key)
            logger.warning(
                "config_key_renamed",
                extra={
                    "path": ".".join(path) if path else "<root>",
                    "old": old_key,
                    "new": new_key,
                    "action": (
                        "Run scripts/migrate_config.py para limpiar el "
                        "config per-device a los nombres canónicos."
                    ),
                },
            )
        elif old_key in parent and new_key in parent:
            # El operador metió el nombre nuevo Y dejó el viejo —
            # honoramos el nuevo y descartamos el viejo con warning.
            parent.pop(old_key)
            logger.warning(
                "config_key_obsolete_and_new_both_present",
                extra={
                    "path": ".".join(path) if path else "<root>",
                    "old": old_key,
                    "new": new_key,
                },
            )

    for path, old_key in _COMPAT_REMOVED:
        parent = _walk_nested_dict(config, path)
        if parent is None:
            continue
        if old_key in parent:
            parent.pop(old_key)
            logger.warning(
                "config_key_obsolete",
                extra={
                    "path": ".".join(path) if path else "<root>",
                    "old": old_key,
                    "action": (
                        "Removida silenciosamente. Run "
                        "scripts/migrate_config.py para limpiar el "
                        "config per-device."
                    ),
                },
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(config: dict[str, Any]) -> None:
    """Valida keys requeridas + tipos + rangos en el config mergeado.

    Se divide en dos fases por readability:
      1. Secciones requeridas + leaf keys (check de presencia).
      2. Checks de tipo / rango / cross-field (semánticos).
    """
    # --- Fase 1: secciones + keys requeridas -------------------------------
    required: dict[str, tuple[str, ...]] = {
        "device": ("id", "store_id"),
        "bracket": ("baseline_mm", "camera_left_csi", "camera_right_csi"),
        "sensor": (
            "model",
            "full_res",
            "default_res",
            "default_fps",
            "nominal_focal_full_px",
        ),
        "lens": ("type", "hfov_deg"),
        "vision": ("resolution", "fps", "calibration_file", "sgbm"),
        "detection": (
            "model_path",
            "confidence_threshold",
            "nms_threshold",
            "cluster_distance_px",
        ),
        "tracking": (
            "max_disappeared_frames", "max_distance_px", "state_machine",
        ),
        "counter": (),
        "wifi_ble": (
            "wifi_interface",
            "summary_interval_seconds",
            "cross_protocol_window_seconds",
            "cross_protocol_rssi_delta",
        ),
        "mqtt": ("endpoint", "port", "topics"),
        "buffer": ("db_path", "max_age_hours"),
        "logging": ("format", "file"),
    }
    for section, keys in required.items():
        if section not in config:
            raise ValueError(f"config missing section: {section}")
        if not isinstance(config[section], dict):
            raise ValueError(
                f"config.{section} must be a mapping, got "
                f"{type(config[section]).__name__}"
            )
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"config missing key: {section}.{key}")

    # --- Fase 2: checks de tipo + rango ------------------------------------
    bracket = config["bracket"]
    if not isinstance(bracket["camera_left_csi"], int):
        raise ValueError("bracket.camera_left_csi must be int")
    if not isinstance(bracket["camera_right_csi"], int):
        raise ValueError("bracket.camera_right_csi must be int")
    if bracket["camera_left_csi"] == bracket["camera_right_csi"]:
        raise ValueError(
            "bracket.camera_left_csi and camera_right_csi must differ"
        )

    sgbm = config["vision"]["sgbm"]
    for k in ("num_disparities", "block_size", "downscale"):
        if k not in sgbm:
            raise ValueError(f"config missing key: vision.sgbm.{k}")

    sm = config["tracking"]["state_machine"]
    for k in ("confirm_frames", "pending_max_frames", "reid_gate_px", "depth_gate_m"):
        if k not in sm:
            raise ValueError(f"config missing key: tracking.state_machine.{k}")
    # El bloque Kalman es OPCIONAL — el tracker tiene defaults razonables si
    # falta. Cuando está presente, cada leaf tiene que ser un number; las
    # leaves que falten igual caen al default del tracker en vez de crashear.
    kalman = sm.get("kalman")
    if kalman is not None:
        if not isinstance(kalman, dict):
            raise ValueError(
                "config.tracking.state_machine.kalman must be a mapping"
            )
        for k in (
            "process_noise",
            "measurement_noise",
            "initial_velocity_uncertainty",
        ):
            if k in kalman and not isinstance(kalman[k], (int, float)):
                raise ValueError(
                    "config.tracking.state_machine.kalman."
                    f"{k} must be a number"
                )

    # Threshold de debounce opcional para line crossing. ``None`` /
    # ausente / 0 deshabilita el feature; un float >= 0 lo activa.
    min_move = config["counter"].get("min_crossing_movement_px")
    if min_move is not None:
        if not isinstance(min_move, (int, float)) or isinstance(min_move, bool):
            raise ValueError(
                "config.counter.min_crossing_movement_px must be null or a number"
            )
        if min_move < 0:
            raise ValueError(
                "config.counter.min_crossing_movement_px must be >= 0"
            )

    topics = config["mqtt"]["topics"]
    for k in ("counting", "wifi_ble", "telemetry", "shadow"):
        if k not in topics:
            raise ValueError(f"config missing key: mqtt.topics.{k}")

    # Threshold opcional para matching two-stage. Debe ser:
    #   - missing o null  → feature deshabilitado
    #   - 0 < value < confidence_threshold → feature enabled, el matching
    #     two-stage pipea detecciones en [low, high) solo a re-asociación
    # Valores >= confidence_threshold se coercen silenciosamente a "off" +
    # warning así una misconfiguración nunca *expande* silenciosamente el
    # pool de spawn.
    detection = config["detection"]
    low = detection.get("low_confidence_threshold")
    if low is not None:
        if not isinstance(low, (int, float)) or isinstance(low, bool):
            raise ValueError(
                "config.detection.low_confidence_threshold "
                "must be null or a number"
            )
        low_f = float(low)
        if low_f <= 0.0:
            raise ValueError(
                "config.detection.low_confidence_threshold "
                "must be > 0 (or null to disable)"
            )
        high_f = float(detection["confidence_threshold"])
        if low_f >= high_f:
            logger.warning(
                "detection.low_confidence_threshold (%.3f) >= "
                "confidence_threshold (%.3f) — disabling two-stage "
                "matching (treated as null)",
                low_f,
                high_f,
            )
            detection["low_confidence_threshold"] = None

    # Threshold opcional de spawn de tracks nuevos. Debe ser:
    #   - missing o null  → el floor de spawn cae a confidence_threshold
    #   - >= confidence_threshold → las detecciones en [confidence, new_track)
    #     pueden matchear tracks existentes pero nunca spawnear nuevos
    # Valores < confidence_threshold se coercen silenciosamente a "off" +
    # warning así una misconfiguración nunca *baja* silenciosamente el floor
    # de spawn.
    new_track = detection.get("new_track_threshold")
    if new_track is not None:
        if not isinstance(new_track, (int, float)) or isinstance(new_track, bool):
            raise ValueError(
                "config.detection.new_track_threshold "
                "must be null or a number"
            )
        nt_f = float(new_track)
        if nt_f <= 0.0 or nt_f > 1.0:
            raise ValueError(
                "config.detection.new_track_threshold "
                "must be in (0, 1] (or null to disable)"
            )
        high_f = float(detection["confidence_threshold"])
        if nt_f < high_f:
            logger.warning(
                "detection.new_track_threshold (%.3f) < "
                "confidence_threshold (%.3f) — disabling spawn gate "
                "(treated as null)",
                nt_f,
                high_f,
            )
            detection["new_track_threshold"] = None

    # Valida los thresholds de RSSI si wifi_ble está configurado.
    wifi_cfg = config.get("wifi_ble", {})
    if wifi_cfg.get("enabled"):
        passerby = wifi_cfg.get("rssi_passerby_threshold", -75)
        shopper = wifi_cfg.get("rssi_shopper_threshold", -55)
        if shopper <= passerby:
            raise ValueError(
                f"rssi_shopper_threshold ({shopper}) must be greater than "
                f"rssi_passerby_threshold ({passerby}) — "
                "shoppers are closer so their signal is stronger (less negative)"
            )


def _normalise_best_frame(config: dict[str, Any]) -> None:
    """Rellena los defaults de ``best_frame`` + valida cuando el operador hizo opt-in.

    El bloque es *opcional* — un config sin él recibe los defaults GDPR-safe
    inlineados (``enabled: False``, sin side effects). Cuando el bloque está
    presente validamos tipos y rangos así un typo en el YAML no se cuela al
    runtime.
    """
    raw = config.get("best_frame")
    if raw is None:
        # Inlinea una copia fresca así los callers pueden mutar sin polucionar
        # el dict default a nivel de módulo.
        config["best_frame"] = {
            **BEST_FRAME_DEFAULTS,
            "scoring": dict(BEST_FRAME_DEFAULTS["scoring"]),
        }
        return

    if not isinstance(raw, dict):
        raise ValueError("config: best_frame must be a mapping")

    enabled = raw.get("enabled", BEST_FRAME_DEFAULTS["enabled"])
    if not isinstance(enabled, bool):
        raise ValueError("config: best_frame.enabled must be a bool")

    retention_days = raw.get("retention_days", BEST_FRAME_DEFAULTS["retention_days"])
    if not isinstance(retention_days, int) or retention_days <= 0:
        raise ValueError(
            "config: best_frame.retention_days must be a positive int"
        )

    buffer_size = raw.get("buffer_size", BEST_FRAME_DEFAULTS["buffer_size"])
    if not isinstance(buffer_size, int) or buffer_size <= 0:
        raise ValueError("config: best_frame.buffer_size must be a positive int")

    jpeg_quality = raw.get("jpeg_quality", BEST_FRAME_DEFAULTS["jpeg_quality"])
    if not isinstance(jpeg_quality, int) or not (1 <= jpeg_quality <= 100):
        raise ValueError(
            "config: best_frame.jpeg_quality must be an int in [1, 100]"
        )

    output_dir = raw.get("output_dir", BEST_FRAME_DEFAULTS["output_dir"])
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError(
            "config: best_frame.output_dir must be a non-empty string"
        )

    scoring_default = BEST_FRAME_DEFAULTS["scoring"]
    scoring_in = raw.get("scoring") or {}
    if not isinstance(scoring_in, dict):
        raise ValueError("config: best_frame.scoring must be a mapping")
    scoring: dict[str, float] = {}
    for key in scoring_default:
        val = scoring_in.get(key, scoring_default[key])
        if not isinstance(val, (int, float)) or val < 0:
            raise ValueError(
                f"config: best_frame.scoring.{key} must be a non-negative number"
            )
        scoring[key] = float(val)

    # Soft-warn (no raise) cuando los weights se alejan mucho de 1.0 — la
    # función de scoring sigue funcionando con weights positivos arbitrarios,
    # pero una suma muy off es casi siempre un typo del operador.
    weight_sum = sum(scoring.values())
    if weight_sum > 0 and not (0.9 <= weight_sum <= 1.1):
        logger.warning(
            "best_frame.scoring weights sum to %.3f — expected ~1.0. "
            "Component contributions will be non-uniform; verify YAML is "
            "intentional.",
            weight_sum,
        )

    config["best_frame"] = {
        "enabled": enabled,
        "output_dir": output_dir,
        "retention_days": retention_days,
        "buffer_size": buffer_size,
        "jpeg_quality": jpeg_quality,
        "scoring": scoring,
    }


# ---------------------------------------------------------------------------
# Cloud shadow merge
# ---------------------------------------------------------------------------


def merge_cloud_config(config: dict[str, Any], shadow: dict[str, Any]) -> dict[str, Any]:
    """Mergea overrides del AWS IoT Device Shadow al config.

    El estado 'desired' del shadow puede contener keys que matchean
    CLOUD_OVERRIDABLE. Estas overridean los valores correspondientes en
    config['cloud_defaults'].

    Esta función NO muta el config de entrada; devuelve un dict nuevo.

    Args:
        config: Config local cargada del YAML.
        shadow: Documento del IoT Shadow (la porción 'state.desired').

    Returns:
        Dict de config nuevo con los overrides del cloud aplicados.
    """
    merged = copy.deepcopy(config)

    if not shadow:
        logger.debug("No shadow data provided; using local defaults")
        return merged

    cloud = merged.setdefault("cloud_defaults", {})
    applied = []

    for key in CLOUD_OVERRIDABLE:
        if key in shadow:
            old_val = cloud.get(key)
            cloud[key] = shadow[key]
            applied.append(f"{key}: {old_val!r} → {shadow[key]!r}")

    if applied:
        logger.info(
            "cloud_config_overrides_applied",
            extra={"overrides": applied, "count": len(applied)},
        )
    else:
        logger.debug("Shadow contained no overridable keys")

    return merged


def get_effective_value(config: dict[str, Any], key: str, fallback: Any = None) -> Any:
    """Devuelve el valor efectivo de una key de config overrideable por cloud.

    Busca primero la key en cloud_defaults (que puede haber sido overrideada
    por el merge del shadow), después cae al default provisto.
    """
    cloud = config.get("cloud_defaults", {})
    return cloud.get(key, fallback)


def validate_operating_hours(hours: Any) -> str | None:
    """Valida una estructura operating_hours.

    Devuelve None si es válida, o un string human-readable describiendo el
    primer problema encontrado. Los días desconocidos se toleran
    (extensiones), pero un día presente debe ser o bien null (cerrado) o un
    string "HH:MM-HH:MM" con fin estrictamente posterior al inicio.
    """
    if hours is None:
        return "operating_hours is missing"
    if not isinstance(hours, dict):
        return f"operating_hours must be a mapping, got {type(hours).__name__}"

    for day, schedule in hours.items():
        if schedule is None:
            continue  # cerrado
        if not isinstance(schedule, str):
            return f"{day}: schedule must be a string or null, got {type(schedule).__name__}"
        if "-" not in schedule:
            return f"{day}: schedule {schedule!r} missing '-' separator"

        parts = schedule.split("-")
        if len(parts) != 2:
            return f"{day}: schedule {schedule!r} must be 'HH:MM-HH:MM'"
        open_str, close_str = parts[0].strip(), parts[1].strip()

        open_parsed = _parse_hhmm(open_str)
        if open_parsed is None:
            return f"{day}: invalid start time {open_str!r}"
        close_parsed = _parse_hhmm(close_str)
        if close_parsed is None:
            return f"{day}: invalid end time {close_str!r}"

        open_minutes = open_parsed[0] * 60 + open_parsed[1]
        close_minutes = close_parsed[0] * 60 + close_parsed[1]
        if close_minutes <= open_minutes:
            return (
                f"{day}: end {close_str!r} must be after start {open_str!r}"
            )

    return None


def validate_bool_toggle(value: Any) -> str | None:
    """Valida un feature toggle booleano del Device Shadow.

    Devuelve None si es válido, o un string describiendo el problema. Rechaza
    explícitamente strings (``"false"`` es truthy en ``bool(...)`` → un delta
    malformado encendería el toggle que el operador quiso apagar) y enteros
    (``0``/``1`` se toleran en muchos sitios pero acá queremos un contrato
    estricto bool para que ``build_reported_state`` no eche basura al shadow).
    """
    if not isinstance(value, bool):
        return f"must be a boolean (true/false), got {type(value).__name__}"
    return None


def _parse_hhmm(value: str) -> tuple[int, int] | None:
    """Parsea un string HH:MM. Devuelve (hour, minute) o None si es inválido."""
    if ":" not in value:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def is_within_operating_hours(config: dict[str, Any], day_name: str, hour: int, minute: int) -> bool:
    """Chequea si la hora actual cae dentro de las operating hours del día dado."""
    hours = get_effective_value(config, "operating_hours", {})
    schedule = hours.get(day_name)

    if not schedule:
        return False

    try:
        open_str, close_str = schedule.split("-")
        open_h, open_m = map(int, open_str.strip().split(":"))
        close_h, close_m = map(int, close_str.strip().split(":"))
    except (ValueError, AttributeError):
        logger.warning(
            "invalid_operating_hours_format",
            extra={"day": day_name, "schedule": schedule},
        )
        return True  # Fail open — contar si el formato está roto

    current_minutes = hour * 60 + minute
    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m

    return open_minutes <= current_minutes < close_minutes


def is_counting_enabled(config: dict[str, Any]) -> bool:
    """Chequea si counting (visión) está enabled (toggleable desde el cloud)."""
    return bool(get_effective_value(config, "counting_enabled", True))


def is_external_traffic_enabled(config: dict[str, Any]) -> bool:
    """Chequea si la captura WiFi/BLE (tráfico exterior) está enabled.

    Combina dos toggles: el local (``wifi_ble.enabled`` del config del
    device, requiere SSH+restart) y el cloud (``cloud_defaults.
    external_traffic_enabled``, overridable desde el shadow sin restart).
    Ambos deben ser true para que el subsystema arranque/siga.
    """
    local_enabled = config.get("wifi_ble", {}).get("enabled", False)
    cloud_enabled = bool(get_effective_value(config, "external_traffic_enabled", True))
    return local_enabled and cloud_enabled


# ---------------------------------------------------------------------------
# Shadow delta application
# ---------------------------------------------------------------------------


def _flatten_delta(
    delta: dict[str, Any],
    prefix: str = "",
) -> list[tuple[str, Any]]:
    """Aplana un delta anidado a ``[(dotted_path, value), ...]``.

    Las keys top-level que están en CLOUD_OVERRIDABLE se tratan
    ATÓMICAMENTE — no se recursa adentro. Esto preserva ``operating_hours``
    como dict entero (el shadow lo pushea como bloque, no día por día) y
    permite validar/asignar el dict completo. Sin este short-circuit,
    ``{operating_hours: {monday: ...}}`` se aplanaba a
    ``operating_hours.monday`` que no matchea RUNTIME_SAFE_KEYS y se
    descartaba como requires_restart.
    """
    items: list[tuple[str, Any]] = []
    for key, value in delta.items():
        path = f"{prefix}{key}"
        if prefix == "" and key in CLOUD_OVERRIDABLE:
            items.append((path, value))
            continue
        if isinstance(value, dict) and not _is_runtime_safe(path):
            items.extend(_flatten_delta(value, prefix=f"{path}."))
        else:
            items.append((path, value))
    return items


def _is_runtime_safe(path: str) -> bool:
    """Devuelve True si una key de delta dotted-path es runtime-aplicable."""
    if path in RUNTIME_SAFE_KEYS:
        return True
    return any(path.startswith(p) for p in RUNTIME_SAFE_PREFIXES)


def _set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    """Asigna ``value`` en el ``path`` dotted dentro de ``config`` (muta)."""
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        nxt = target.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            target[part] = nxt
        target = nxt
    target[parts[-1]] = value


def apply_shadow_delta(
    current_config: dict[str, Any],
    delta: dict[str, Any],
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Aplica un shadow delta al config corriendo.

    Solo se aplican las keys whitelisted (runtime-safe); las demás se
    loggean como que requieren restart y se skipean. El config devuelto
    es una deep copy — el input no se muta.
    """
    new_config = copy.deepcopy(current_config)
    applied: list[str] = []
    ignored: list[str] = []
    rejected: list[tuple[str, str]] = []

    if not delta:
        logger.debug("apply_shadow_delta called with empty delta")
        return new_config, applied

    # Validators por path. Si la validación falla, el delta se RECHAZA
    # (no se aplica al config en memoria + no se persiste al config.yaml).
    # El último valor válido sigue activo. Sin esto, un delta malformado
    # del shadow se escribiría al config.yaml y rompería el próximo boot.
    validators: dict[str, Any] = {
        "cloud_defaults.operating_hours": validate_operating_hours,
        "cloud_defaults.counting_enabled": validate_bool_toggle,
        "cloud_defaults.external_traffic_enabled": validate_bool_toggle,
    }

    for path, value in _flatten_delta(delta):
        # AWS IoT publica el state.desired al ras del root (ej.
        # ``{counting_enabled: false}``). ``merge_cloud_config`` (startup)
        # asume esa convención y monta bajo ``cloud_defaults.<key>``. Para
        # que ``apply_shadow_delta`` (runtime) sea coherente, mapeamos las
        # keys bare del top-level que están en CLOUD_OVERRIDABLE a
        # ``cloud_defaults.<key>`` antes de validar contra RUNTIME_SAFE_KEYS
        # y de hacer el set. Sin este mapeo, el delta llegaba pero se
        # rechazaba como requires_restart porque el path no matcheaba la
        # whitelist (operator-friendly publishing vs internal namespace).
        if "." not in path and path in CLOUD_OVERRIDABLE:
            path = f"cloud_defaults.{path}"
        if not _is_runtime_safe(path):
            ignored.append(path)
            continue
        validator = validators.get(path)
        if validator is not None:
            err = validator(value)
            if err is not None:
                rejected.append((path, err))
                continue
        _set_dotted(new_config, path, value)
        applied.append(path)

    if applied:
        logger.info(
            "shadow_delta_applied",
            extra={"keys": sorted(applied), "count": len(applied)},
        )
    if ignored:
        logger.warning(
            "shadow_delta_requires_restart",
            extra={"keys": sorted(ignored), "count": len(ignored)},
        )
    if rejected:
        logger.warning(
            "shadow_delta_rejected_invalid",
            extra={"rejections": [{"key": p, "reason": r} for p, r in rejected]},
        )

    if applied and config_path is not None:
        # Persistir el config mergeado al MISMO ``config.yaml`` (no a un
        # sibling .shadow.json). Single source of truth: el operator que
        # SSH a /etc/people-counter/config.yaml ve lo que el device tiene
        # corriendo. Trade-off: yaml.safe_dump pierde los comentarios del
        # YAML original — la doc canónica con comentarios vive en
        # config/config.example.yaml del repo, el del device es runtime
        # state (lo dice la cabecera del archivo después del primer
        # apply). Si se necesita preservar comentarios en el futuro,
        # usar ``ruamel.yaml`` que mantiene round-trip fidelity.
        try:
            cp = Path(config_path)
            # Limpiamos las keys internas que el loader inyecta in-memory
            # (no son parte del YAML del operator) antes de serializar.
            serializable = {k: v for k, v in new_config.items() if not k.startswith("_")}
            cp.write_text(
                "# Auto-managed: este archivo fue actualizado por el último\n"
                "# Device Shadow apply. La doc canónica con comentarios vive\n"
                "# en `config/config.example.yaml` del repo. Para auditar\n"
                "# qué keys son shadow-overridables ver CLOUD_OVERRIDABLE\n"
                "# en src/config/loader.py.\n"
                + yaml.safe_dump(
                    serializable, default_flow_style=False, sort_keys=False
                ),
                encoding="utf-8",
            )
            logger.info(
                "shadow_applied_to_yaml",
                extra={"path": str(cp), "applied": sorted(applied)},
            )
        except OSError:
            logger.exception("Falló persistir el shadow al config.yaml")

    return new_config, applied


def _get_dotted(config: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Busca un ``path`` dotted en ``config``."""
    parts = path.split(".")
    cur: Any = config
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def build_reported_state(
    config: dict[str, Any],
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extrae el subset de config runtime-visible para el reporting al shadow."""
    import time as _time

    reported: dict[str, Any] = {}

    for path in RUNTIME_SAFE_KEYS:
        found, value = _get_dotted(config, path)
        if found:
            _set_dotted(reported, path, value)

    for prefix in RUNTIME_SAFE_PREFIXES:
        prefix_path = prefix.rstrip(".")
        found, value = _get_dotted(config, prefix_path)
        if found and isinstance(value, dict):
            _set_dotted(reported, prefix_path, copy.deepcopy(value))

    device_cfg = config.get("device", {}) if isinstance(config.get("device"), dict) else {}
    reported["firmware_version"] = device_cfg.get("firmware_version", "unknown")
    reported["boot_ts"] = int(_time.time())

    cal_path: str | None = None
    vision_cfg = config.get("vision", {})
    if isinstance(vision_cfg, dict):
        cal_path = vision_cfg.get("calibration_file")
    reported["calibration_file_path"] = cal_path

    baseline_mm: float | None = None
    if calibration is not None:
        try:
            t_vec = calibration.get("T") if isinstance(calibration, dict) else None
            if t_vec is not None:
                import numpy as _np

                baseline_mm = float(_np.linalg.norm(_np.asarray(t_vec)))
        except Exception:
            logger.exception("Falló derivar el baseline desde la calibración")
            baseline_mm = None
    reported["effective_baseline_mm"] = baseline_mm

    return reported

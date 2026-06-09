"""AWS Lambda: persiste eventos del device en Postgres (RDS).

Invocada por las 3 Topic Rules de IoT Core (counting / telemetry / wifi_ble).
El payload del device viene del envelope estandar de :class:`src.mqtt.client.MQTTClient`:

    {
        "device_id":  "store-pilot-01-cam-01",
        "timestamp":  1762963200.0,
        "type":       "counting" | "telemetry" | "wifi_ble",
        "data":       { ... shape dependiente del type ... }
    }

``store_id`` se infiere del ``device_id`` (prefijo antes del ultimo ``-cam-``).

Auth a RDS: IAM database authentication. La Lambda corre con un role que
tiene ``rds-db:connect`` sobre el DB user ``lambda_writer`` (configurado en
el CFN). Cada conexion genera un token corto (~15 min de validez) via
``rds.generate_db_auth_token`` y lo usa como password. Postgres valida el
token contra IAM, sin password almacenado.

Variables de entorno (seteadas por CFN):
    PG_HOST:    endpoint de RDS.
    PG_PORT:    5432 default.
    PG_DB:      people_counter default.
    PG_USER:    lambda_writer default (DB user con grant rds_iam).
    PG_REGION:  region para generar el token IAM (= region de RDS).
    SSL_ROOT_CERT_PATH: path al bundle de Amazon RDS CA (default empacado
                       con la Lambda en /opt/rds-ca-bundle.pem si usas
                       Layer; o /var/task/rds-ca-bundle.pem si lo zippeas
                       con el code). Si la key falta y el archivo default
                       no existe, conectamos con sslmode=require sin
                       verificacion de CA (RDS force_ssl=1 igual encripta).

Errores conocidos NO se re-raisean (sino IoT Core reintenta la regla):
    - JSON malformado del device --> loguear y descartar.
    - Constraint violation (duplicate, check failed) --> loguear y descartar.

Errores transitorios (conexion rota, timeout, token expirado) SI se
re-raisean para reintento via IoT (max 1 retry para topic rules sin DLQ).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Conexiones se cachean entre invocaciones del mismo container Lambda (warm
# start). Tokens IAM caducan a los 15min; cuando eso pasa, la query falla y
# nos reconectamos transparente.
_pg_conn = None


def _generate_iam_token() -> str:
    """Genera un token IAM de RDS para autenticacion (~15min de validez)."""
    import boto3

    region = os.environ.get("PG_REGION", "us-east-1")
    host = os.environ["PG_HOST"]
    port = int(os.environ.get("PG_PORT", "5432"))
    user = os.environ.get("PG_USER", "lambda_writer")

    rds = boto3.client("rds", region_name=region)
    return rds.generate_db_auth_token(
        DBHostname=host,
        Port=port,
        DBUsername=user,
        Region=region,
    )


def _ssl_root_cert() -> str | None:
    """Devuelve el path al bundle de CA si existe, sino None.

    Si esta seteado SSL_ROOT_CERT_PATH se usa eso. Sino, intenta paths
    convencionales (Layer /opt/ o code /var/task/). Si nada existe,
    None y conectamos con sslmode=require (encripta pero no verifica).
    """
    env_path = os.environ.get("SSL_ROOT_CERT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    for path in ("/opt/rds-ca-bundle.pem", "/var/task/rds-ca-bundle.pem"):
        if os.path.exists(path):
            return path
    return None


def _get_connection():
    """Devuelve la conexion Postgres cacheada, o la abre si no existe.

    Si la conexion esta rota (token expirado, network blip), se cierra y
    se re-abre en la siguiente invocacion.
    """
    global _pg_conn
    if _pg_conn is not None:
        try:
            with _pg_conn.cursor() as cur:
                cur.execute("SELECT 1")
            return _pg_conn
        except Exception:
            logger.warning("pg_connection_stale_reconnecting")
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None

    import psycopg

    sslmode = "verify-full" if _ssl_root_cert() else "require"
    conn_kwargs: dict[str, Any] = {
        "host": os.environ["PG_HOST"],
        "port": int(os.environ.get("PG_PORT", "5432")),
        "dbname": os.environ.get("PG_DB", "people_counter"),
        "user": os.environ.get("PG_USER", "lambda_writer"),
        "password": _generate_iam_token(),
        "sslmode": sslmode,
        # 15s para tolerar el cold start: import psycopg + boto3 + DNS + TCP +
        # SSL handshake suelen tomar 3-8s la primera vez; 5s era marginal.
        "connect_timeout": 15,
        "autocommit": True,
    }
    root_cert = _ssl_root_cert()
    if root_cert:
        conn_kwargs["sslrootcert"] = root_cert

    _pg_conn = psycopg.connect(**conn_kwargs)
    return _pg_conn


def _infer_store_id(device_id: str) -> str:
    """``store-001-cam-01`` --> ``store-001``.

    Convencion: device_id = ``<store_id>-cam-<n>``. Si no matchea, devuelve
    el device_id entero como fallback.
    """
    if "-cam-" in device_id:
        return device_id.rsplit("-cam-", 1)[0]
    return device_id


def _ts(epoch: float | int | None) -> datetime | None:
    """epoch seconds --> datetime UTC para Postgres TIMESTAMPTZ (None passthrough)."""
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc)


def _insert_counting(conn, event: dict[str, Any]) -> None:
    data = event["data"]
    device_id = event["device_id"]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO count_events
                (device_id, store_id, event_ts, direction,
                 track_id, confidence, height_m)
            VALUES (%s, %s, %s, %s,
                    %s, %s, %s)
            ON CONFLICT (device_id, event_ts, track_id, direction) DO NOTHING
            """,
            (
                device_id,
                _infer_store_id(device_id),
                _ts(data.get("event_time", event.get("timestamp"))),
                # bucket_15min se omite — la columna RDS es GENERATED ALWAYS AS
                # STORED desde event_ts (date_bin server-side). El device manda
                # event_time RAW y Postgres deriva el bucket sin acoplar device
                # ↔ schema. Devices con firmware viejo que aún mandan
                # `bucket_15min` en el payload: la key se ignora silenciosamente.
                data["direction"],
                data.get("track_id"),
                data.get("confidence"),
                # height_class del payload (firmware viejo lo mandaba): se
                # ignora silenciosamente. La categorización se aplica
                # server-side via la función SQL height_class(height_m) en
                # las vistas. Solo persistimos la medición cruda.
                data.get("height_m"),
            ),
        )


def _insert_telemetry(conn, event: dict[str, Any]) -> None:
    data = event["data"]
    device_id = event["device_id"]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO telemetry
                (device_id, store_id, event_ts,
                 uptime_s, cpu_temp_c, hailo_temp_c,
                 disk_free_mb, mem_available_mb,
                 fps, frame_latency_p50_ms, frame_latency_p95_ms,
                 detection_rate_per_min,
                 tracker_confirmed_count, tracker_pending_count,
                 total_in, total_out,
                 mqtt_connected, mqtt_disconnect_count,
                 seconds_since_last_reconnect, buffer_backlog_messages,
                 wifi_probe_ok, ble_scanner_ok,
                 wifi_ble_stitching_ratio,
                 track_stitching_ratio, death_emit_count,
                 ghost_adoption_count,
                 last_shadow_apply_ts,
                 throttled_flags, arm_clock_mhz, fan_rpm, power_w, ext5v_v,
                 fs_readonly, service_restarts, clock_synchronized,
                 cam_left_ok, cam_right_ok, wifi_probe_rate_per_min,
                 error, schedule_error_detail)
            VALUES (%s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s,
                    %s, %s,
                    %s,
                    %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s)
            ON CONFLICT (device_id, event_ts) DO NOTHING
            """,
            (
                device_id,
                _infer_store_id(device_id),
                _ts(event.get("timestamp")),
                data.get("uptime_s"),
                data.get("cpu_temp_c"),
                data.get("hailo_temp_c"),
                data.get("disk_free_mb"),
                data.get("mem_available_mb"),
                data.get("fps"),
                data.get("frame_latency_p50_ms"),
                data.get("frame_latency_p95_ms"),
                data.get("detection_rate_per_min"),
                data.get("tracker_confirmed_count"),
                data.get("tracker_pending_count"),
                data.get("total_in"),
                data.get("total_out"),
                data.get("mqtt_connected"),
                data.get("mqtt_disconnect_count"),
                data.get("seconds_since_last_reconnect"),
                data.get("buffer_backlog_messages"),
                data.get("wifi_probe_ok"),
                data.get("ble_scanner_ok"),
                data.get("wifi_ble_stitching_ratio"),
                data.get("track_stitching_ratio"),
                data.get("death_emit_count"),
                data.get("ghost_adoption_count"),
                (
                    _ts(data.get("last_shadow_apply_ts"))
                    if data.get("last_shadow_apply_ts") is not None
                    else None
                ),
                data.get("throttled_flags"),
                data.get("arm_clock_mhz"),
                data.get("fan_rpm"),
                data.get("power_w"),
                data.get("ext5v_v"),
                data.get("fs_readonly"),
                data.get("service_restarts"),
                data.get("clock_synchronized"),
                data.get("cam_left_ok"),
                data.get("cam_right_ok"),
                data.get("wifi_probe_rate_per_min"),
                data.get("error"),
                data.get("schedule_error_detail"),
            ),
        )


def _insert_wifi_ble(conn, event: dict[str, Any]) -> None:
    """Persiste un batch de eventos per-device WiFi/BLE.

    Shape del payload (post PR 2):

        data = {
            "period_start": <epoch>,
            "period_end":   <epoch>,
            "devices": [
                {
                    "visitor_hash":  "<32 hex chars = 16 bytes>",
                    "protocol":      "wifi" | "ble",
                    "rssi_max":      -55,
                    "first_seen_ts": <epoch float>,
                    "last_seen_ts":  <epoch float>,
                },
                ...
            ]
        }

    Cada dict del array se inserta como una fila de ``wifi_ble_events``.
    ON CONFLICT (device_id, visitor_hash, period_start) DO UPDATE refina:
    si la misma ventana llega 2 veces (retry del device), se queda con el
    rssi_max más alto y el last_seen_ts más reciente.
    """
    data = event["data"]
    device_id = event["device_id"]
    store_id = _infer_store_id(device_id)
    period_start = _ts(data["period_start"])
    period_end = _ts(data["period_end"])
    devices = data.get("devices") or []

    if not devices:
        # Ventana vacía explícita — el publisher debería evitar emitir, pero si
        # llega no es error, solo no inserta nada.
        logger.info("wifi_ble_empty_window device_id=%s", device_id)
        return

    rows = []
    for d in devices:
        # visitor_hash llega como hex string. Postgres BYTEA acepta bytes desde
        # bytes.fromhex() — psycopg lo mapea a el formato binario nativo.
        try:
            visitor_hash = bytes.fromhex(d["visitor_hash"])
        except (KeyError, ValueError, TypeError):
            logger.warning(
                "wifi_ble_bad_visitor_hash device_id=%s hash=%r",
                device_id,
                d.get("visitor_hash"),
            )
            continue
        rows.append(
            (
                device_id,
                store_id,
                visitor_hash,
                d["protocol"],
                int(d["rssi_max"]),
                _ts(d["first_seen_ts"]),
                _ts(d["last_seen_ts"]),
                period_start,
                period_end,
            )
        )

    if not rows:
        return

    # executemany con ON CONFLICT — refina rssi_max al máximo visto y
    # last_seen_ts al más reciente, por si la misma ventana llega 2 veces.
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO wifi_ble_events
                (device_id, store_id, visitor_hash, protocol, rssi_max,
                 first_seen_ts, last_seen_ts, period_start, period_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (device_id, visitor_hash, period_start) DO UPDATE
                SET rssi_max      = GREATEST(wifi_ble_events.rssi_max, EXCLUDED.rssi_max),
                    last_seen_ts  = GREATEST(wifi_ble_events.last_seen_ts, EXCLUDED.last_seen_ts),
                    first_seen_ts = LEAST(wifi_ble_events.first_seen_ts, EXCLUDED.first_seen_ts)
            """,
            rows,
        )
    logger.info(
        "wifi_ble_events_inserted device_id=%s n=%d",
        device_id,
        len(rows),
    )


_DISPATCH = {
    "counting": _insert_counting,
    "telemetry": _insert_telemetry,
    "wifi_ble": _insert_wifi_ble,
}


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handler de Lambda — invocado por las 3 IoT Topic Rules.

    Errores de validacion se descartan (return 200) para que IoT no reintente.
    Errores transitorios de DB se re-raisean para que IoT reintente.
    """
    try:
        event_type = event.get("type")
        if event_type not in _DISPATCH:
            logger.warning(
                "persist_event_unknown_type type=%s device_id=%s",
                event_type,
                event.get("device_id"),
            )
            return {"statusCode": 400, "body": "unknown event type"}

        if not event.get("device_id"):
            logger.warning("persist_event_missing_device_id")
            return {"statusCode": 400, "body": "missing device_id"}

        conn = _get_connection()
        _DISPATCH[event_type](conn, event)

        logger.info(
            "persist_event_ok type=%s device_id=%s",
            event_type,
            event["device_id"],
        )
        return {"statusCode": 200, "body": "ok"}

    except KeyError as e:
        logger.warning(
            "persist_event_malformed missing_key=%s event_type=%s",
            e,
            event.get("type"),
        )
        return {"statusCode": 400, "body": f"missing key: {e}"}
    except (ValueError, TypeError) as e:
        logger.warning("persist_event_value_error error=%s", e)
        return {"statusCode": 400, "body": str(e)}
    except Exception as e:
        # Errores de DATOS del device (CheckViolation, NotNullViolation,
        # valores inadaptables — subclases de IntegrityError/DataError) NO
        # son transitorios: el retry de IoT Core falla idéntico. Se loguean
        # y descartan (400) como promete el contrato del módulo, SIN tirar
        # la conexión warm — cerrarla forzaba una reconexión IAM completa
        # (token + TLS, 3-8s) en la próxima invocación, así un device con
        # firmware buggy degradaba la ingesta de todo el container. La
        # conexión sigue usable: autocommit=True → no queda transacción
        # abortada que resetear.
        try:
            import psycopg

            is_data_error = isinstance(e, (psycopg.IntegrityError, psycopg.DataError))
        except (ImportError, TypeError):
            is_data_error = False
        if is_data_error:
            logger.warning(
                "persist_event_data_rejected type=%s device_id=%s error=%s",
                event.get("type"),
                event.get("device_id"),
                e,
            )
            return {"statusCode": 400, "body": str(e)}

        # Errores transitorios (conn rota, timeout, deadlock, token expirado).
        # Cerramos la conexion para forzar reconexion + re-raise para retry.
        global _pg_conn
        if _pg_conn is not None:
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None
        logger.exception("persist_event_transient_error")
        raise

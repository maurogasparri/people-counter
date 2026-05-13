"""AWS Lambda: persiste eventos del device en PostgreSQL.

Invocada por las 3 Topic Rules de IoT Core (counting, telemetry, wifi_ble).
El payload del device incluye ``type`` que determina la tabla destino.

Schema del event esperado (envelope estándar de :class:`src.mqtt.client.MQTTClient`):

    {
        "device_id":  "store-pilot-01-cam-01",
        "timestamp":  1762963200.0,
        "type":       "counting" | "telemetry" | "wifi_ble",
        "data":       { ... shape dependiente del type ... }
    }

``store_id`` se infiere del ``device_id`` (prefijo antes del último ``-cam-``).

Variables de entorno:
    PG_HOST:     endpoint de Postgres (EC2 interna o RDS).
    PG_PORT:     5432 default.
    PG_DB:       people_counter default.
    PG_USER:     people_counter default.
    PG_PASSWORD_SSM_PATH: path en SSM Parameter Store (SecureString) con la pwd.
                          Default ``/people-counter/dev/pg_password``.

Errores conocidos NO se re-raisean (sino IoT Core reintenta la regla):
    - JSON malformado del device → loguear y descartar.
    - Constraint violation (duplicate, check failed) → loguear y descartar.

Errores transitorios (conexión rota, timeout) SÍ se re-raisean para reintento.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Conexiones se cachean entre invocaciones del mismo container Lambda (warm
# start). El SDK de boto3 también se cachea. cold start ~500ms, warm <50ms.
_pg_conn = None
_pg_password: str | None = None


def _get_password() -> str:
    """Lee el password de SSM Parameter Store (SecureString)."""
    global _pg_password
    if _pg_password is not None:
        return _pg_password
    import boto3

    ssm_path = os.environ.get(
        "PG_PASSWORD_SSM_PATH", "/people-counter/dev/pg_password"
    )
    client = boto3.client("ssm")
    resp = client.get_parameter(Name=ssm_path, WithDecryption=True)
    _pg_password = resp["Parameter"]["Value"]
    return _pg_password


def _get_connection():
    """Devuelve la conexión Postgres cacheada, o la abre si no existe.

    Si la conexión está rota (psycopg.OperationalError al hacer la query),
    se cierra y se re-abre en la siguiente invocación.
    """
    global _pg_conn
    if _pg_conn is not None:
        try:
            # Healthcheck barato: SELECT 1
            with _pg_conn.cursor() as cur:
                cur.execute("SELECT 1")
            return _pg_conn
        except Exception:
            logger.warning("pg_connection_stale — reabriendo")
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None

    import psycopg

    _pg_conn = psycopg.connect(
        host=os.environ["PG_HOST"],
        port=int(os.environ.get("PG_PORT", "5432")),
        dbname=os.environ.get("PG_DB", "people_counter"),
        user=os.environ.get("PG_USER", "people_counter"),
        password=_get_password(),
        connect_timeout=5,
        autocommit=True,
    )
    return _pg_conn


def _infer_store_id(device_id: str) -> str:
    """``store-001-cam-01`` → ``store-001``.

    Convención: device_id = ``<store_id>-cam-<n>``. Si no matchea, devuelve
    el device_id entero como fallback — la fila igual se persiste, solo que
    el ``store_id`` no agrupa con sus hermanos.
    """
    if "-cam-" in device_id:
        return device_id.rsplit("-cam-", 1)[0]
    return device_id


def _ts_to_pg(epoch: float | int) -> str:
    """epoch seconds → ISO 8601 UTC para Postgres TIMESTAMPTZ."""
    return time.strftime("%Y-%m-%d %H:%M:%S+00", time.gmtime(float(epoch)))


def _insert_counting(conn, event: dict[str, Any]) -> None:
    data = event["data"]
    device_id = event["device_id"]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO counting_events
                (device_id, store_id, event_time, direction,
                 track_id, confidence, height_class)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (device_id, event_time, track_id, direction) DO NOTHING
            """,
            (
                device_id,
                _infer_store_id(device_id),
                _ts_to_pg(event["timestamp"]),
                data["direction"],
                data.get("track_id"),
                data.get("confidence"),
                data.get("height_class"),
            ),
        )


def _insert_telemetry(conn, event: dict[str, Any]) -> None:
    data = event["data"]
    device_id = event["device_id"]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO telemetry
                (device_id, store_id, sample_time,
                 cpu_temp_c, hailo_temp_c, disk_used_pct, uptime_s, fps,
                 p50_latency_ms, p99_latency_ms, detection_rate_per_min,
                 active_tracks, buffer_pending, mqtt_connected,
                 total_in, total_out,
                 wifi_probe_ok, ble_scanner_ok, schedule_error)
            VALUES (%s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s)
            """,
            (
                device_id,
                _infer_store_id(device_id),
                _ts_to_pg(event["timestamp"]),
                data.get("cpu_temp_c"),
                data.get("hailo_temp_c"),
                data.get("disk_used_pct"),
                data.get("uptime_s"),
                data.get("fps"),
                data.get("frame_latency_p50_ms"),
                data.get("frame_latency_p95_ms"),
                data.get("detection_rate_per_min"),
                data.get("tracker_confirmed_count"),
                data.get("buffer_backlog_messages"),
                data.get("mqtt_connected"),
                data.get("total_in"),
                data.get("total_out"),
                data.get("wifi_probe_ok"),
                data.get("ble_scanner_ok"),
                data.get("error"),
            ),
        )


def _insert_wifi_ble(conn, event: dict[str, Any]) -> None:
    data = event["data"]
    device_id = event["device_id"]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO wifi_ble_summaries
                (device_id, store_id, period_start, period_end,
                 passersby, shoppers)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (device_id, period_start, period_end) DO NOTHING
            """,
            (
                device_id,
                _infer_store_id(device_id),
                _ts_to_pg(data["period_start"]),
                _ts_to_pg(data["period_end"]),
                data["passersby"],
                data["shoppers"],
            ),
        )


_DISPATCH = {
    "counting": _insert_counting,
    "telemetry": _insert_telemetry,
    "wifi_ble": _insert_wifi_ble,
}


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handler de Lambda — invocado por las 3 IoT Topic Rules.

    Errores de validación se descartan (return 200) para que IoT no reintente.
    Errores transitorios de DB se re-raisean para que IoT reintente vía DLQ.
    """
    try:
        event_type = event.get("type")
        if event_type not in _DISPATCH:
            logger.warning(
                "persist_event_unknown_type",
                extra={"type": event_type, "device_id": event.get("device_id")},
            )
            return {"statusCode": 400, "body": "unknown event type"}

        if not event.get("device_id"):
            logger.warning("persist_event_missing_device_id")
            return {"statusCode": 400, "body": "missing device_id"}

        conn = _get_connection()
        _DISPATCH[event_type](conn, event)

        logger.info(
            "persist_event_ok",
            extra={
                "type": event_type,
                "device_id": event["device_id"],
            },
        )
        return {"statusCode": 200, "body": "ok"}

    except KeyError as e:
        # Payload malformado — descartar sin retry.
        logger.warning(
            "persist_event_malformed",
            extra={"missing_key": str(e), "event_type": event.get("type")},
        )
        return {"statusCode": 400, "body": f"missing key: {e}"}
    except (ValueError, TypeError) as e:
        # Tipos incorrectos (ej. timestamp no parseable) — descartar.
        logger.warning("persist_event_value_error", extra={"error": str(e)})
        return {"statusCode": 400, "body": str(e)}
    except Exception:
        # Errores transitorios (conn rota, timeout, deadlock). Cerramos la
        # conexión para forzar reconexión + re-raise para IoT retry/DLQ.
        global _pg_conn
        if _pg_conn is not None:
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None
        logger.exception("persist_event_transient_error")
        raise

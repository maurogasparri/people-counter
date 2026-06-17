"""Tests del Lambda persist_event con conexión Postgres y SSM mockeadas."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Resetea conexiones cacheadas entre tests para que cada uno arranque limpio."""
    from src.cloud import persist_event

    persist_event._pg_conn = None
    yield
    persist_event._pg_conn = None


@pytest.fixture
def fake_pg(monkeypatch):
    """Mockea psycopg.connect + boto3 (RDS IAM token). Devuelve la conn fake para asserts."""
    monkeypatch.setenv("PG_HOST", "localhost")
    monkeypatch.setenv("PG_DB", "test_db")
    monkeypatch.setenv("PG_USER", "test_user")
    monkeypatch.setenv("PG_REGION", "us-east-1")

    # Mock boto3 — rds.generate_db_auth_token() devuelve un string truthy.
    fake_rds = MagicMock()
    fake_rds.generate_db_auth_token.return_value = "fake-iam-token"
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = fake_rds

    # Mock psycopg
    fake_cursor = MagicMock()
    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    fake_conn.cursor.return_value.__exit__.return_value = False

    fake_psycopg = MagicMock()
    fake_psycopg.connect.return_value = fake_conn
    # Clases de excepción REALES (no MagicMock attrs): el handler hace
    # isinstance contra psycopg.IntegrityError/DataError para discriminar
    # errores de datos del device de los transitorios.
    fake_psycopg.IntegrityError = type("IntegrityError", (Exception,), {})
    fake_psycopg.DataError = type("DataError", (Exception,), {})

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    return {
        "rds": fake_rds,
        "psycopg": fake_psycopg,
        "conn": fake_conn,
        "cursor": fake_cursor,
    }


def test_counting_event_inserts(fake_pg):
    """Schema del INSERT (7 columnas, sin bucket_15min — ahora GENERATED en RDS;
    sin height_class — categorización delegada a la función SQL height_class()):
        0=device_id, 1=store_id, 2=event_ts, 3=direction,
        4=track_id, 5=confidence, 6=height_m.
    """
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {
            "direction": "in",
            "track_id": 42,
            "confidence": 0.87,
            "height_class": "adult",  # legacy: si llega, se ignora silenciosamente
            "height_m": 1.75,
        },
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    assert fake_pg["cursor"].execute.called
    call_args = fake_pg["cursor"].execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    assert "INSERT INTO count_events" in sql
    # bucket_15min ya no aparece en el INSERT — es GENERATED server-side.
    assert "bucket_15min" not in sql
    # height_class ya no es columna — la categorización vive en la función SQL.
    assert "height_class" not in sql
    assert params[0] == "store-001-cam-01"
    assert params[1] == "store-001"  # store_id inferido
    # params[2] = event_ts (datetime UTC desde event_time o envelope timestamp).
    assert params[3] == "in"  # direction
    assert params[4] == 42  # track_id
    assert params[5] == 0.87  # confidence
    assert params[6] == 1.75  # height_m


def test_counting_event_legacy_bucket_key_ignored(fake_pg):
    """Devices con firmware viejo mandan ``bucket_15min`` o ``event_bucket`` en
    el payload. Como el schema RDS ahora deriva el bucket server-side via
    GENERATED, esas keys del payload se ignoran silenciosamente (no rompe el
    INSERT, no contamina el bucket — el server lo recalcula desde event_ts).
    """
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {
            "direction": "in",
            "event_bucket": 1762963200,  # key legacy del firmware viejo
            "bucket_15min": 1762963200,  # idem
        },
    }
    result = handler(event, None)
    assert result["statusCode"] == 200
    sql = fake_pg["cursor"].execute.call_args[0][0]
    # El INSERT NO incluye la columna bucket_15min — keys legacy ignoradas.
    assert "bucket_15min" not in sql


def test_telemetry_event_inserts(fake_pg):
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "telemetry",
        "data": {
            "cpu_temp_c": 56.4,
            "hailo_temp_c": 48.2,
            "wifi_probe_ok": True,
            "ble_scanner_ok": False,
            "last_shadow_apply_ts": 1762962800.0,
            "throttled_flags": 0,
            "arm_clock_mhz": 2400,
            "fan_rpm": 5900,
            "power_w": 6.1,
            "ext5v_v": 5.12,
            "fs_readonly": False,
            "service_restarts": 2,
            "clock_synchronized": True,
            "cam_left_ok": True,
            "cam_right_ok": False,
            "wifi_probe_rate_per_min": 42.0,
        },
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    call_args = fake_pg["cursor"].execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    assert "INSERT INTO telemetry" in sql
    # Canary del Device Shadow llega como columna explícita.
    assert "last_shadow_apply_ts" in sql
    # Columnas de salud de hardware (Tier 1) presentes y mapeadas.
    for col in ("throttled_flags", "arm_clock_mhz", "fan_rpm", "power_w", "ext5v_v"):
        assert col in sql
    assert 0 in params and 2400 in params and 5900 in params
    assert 6.1 in params and 5.12 in params
    # Columnas Tier 2/3 (corre pero roto) presentes y mapeadas.
    for col in ("fs_readonly", "service_restarts", "clock_synchronized"):
        assert col in sql
    assert 2 in params  # service_restarts
    # Health por cámara + rate de probes WiFi presentes y mapeados.
    for col in ("cam_left_ok", "cam_right_ok", "wifi_probe_rate_per_min"):
        assert col in sql
    assert 42.0 in params  # wifi_probe_rate_per_min


def test_telemetry_event_without_shadow_apply_ts(fake_pg):
    """Backward compat: devices con firmware viejo o sin pushes de shadow
    no mandan ``last_shadow_apply_ts``. La columna debe quedar NULL."""
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "telemetry",
        "data": {"cpu_temp_c": 50.0},
        # sin last_shadow_apply_ts
    }
    result = handler(event, None)
    assert result["statusCode"] == 200
    # Sanity: no rompió por el campo ausente.


def test_wifi_ble_events_batch_insert(fake_pg):
    """Schema del INSERT a wifi_ble_events (post PR 2: per-device, batched).

    Cada element del array ``devices`` se inserta como una fila.
    Columnas del INSERT (9, sin bucket_* — GENERATED server-side):
        0=device_id, 1=store_id, 2=visitor_hash (BYTEA), 3=protocol,
        4=rssi_max, 5=first_seen_ts, 6=last_seen_ts,
        7=period_start, 8=period_end.
    """
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "wifi_ble",
        "data": {
            "period_start": 1762962300,
            "period_end": 1762963200,
            "devices": [
                {
                    "visitor_hash": "aa" * 16,
                    "protocol": "wifi",
                    "rssi_max": -55,
                    "first_seen_ts": 1762962400.0,
                    "last_seen_ts": 1762963180.0,
                },
                {
                    "visitor_hash": "bb" * 16,
                    "protocol": "ble",
                    "rssi_max": -68,
                    "first_seen_ts": 1762962500.0,
                    "last_seen_ts": 1762963100.0,
                },
            ],
        },
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    # executemany: call_args[0][0] = sql, [0][1] = lista de tuplas (rows).
    call_args = fake_pg["cursor"].executemany.call_args
    sql = call_args[0][0]
    rows = call_args[0][1]
    assert "INSERT INTO wifi_ble_events" in sql
    assert "bucket_15min" not in sql  # GENERATED server-side
    assert "ON CONFLICT" in sql  # idempotencia con MAX rssi_max
    assert len(rows) == 2
    # Primera fila: visitor_hash es bytes (BYTEA), rssi_max es int.
    row0 = rows[0]
    assert row0[0] == "store-001-cam-01"
    assert row0[1] == "store-001"
    assert row0[2] == bytes.fromhex("aa" * 16)
    assert row0[3] == "wifi"
    assert row0[4] == -55
    # Segunda fila: protocolo ble + rssi distinto.
    assert rows[1][3] == "ble"
    assert rows[1][4] == -68


def test_wifi_ble_empty_devices_array_is_noop(fake_pg):
    """devices=[] llega → no se inserta nada, no se rompe."""
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "wifi_ble",
        "data": {
            "period_start": 1762962300,
            "period_end": 1762963200,
            "devices": [],
        },
    }
    result = handler(event, None)
    assert result["statusCode"] == 200
    # No se llamó a executemany (ni execute para este branch).
    assert not fake_pg["cursor"].executemany.called


def test_wifi_ble_bad_visitor_hash_skipped(fake_pg):
    """Una entry con visitor_hash inválido (no es hex) se skippea, el resto se inserta."""
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "wifi_ble",
        "data": {
            "period_start": 1762962300,
            "period_end": 1762963200,
            "devices": [
                {
                    "visitor_hash": "ZZ" * 16,
                    "protocol": "wifi",
                    "rssi_max": -55,
                    "first_seen_ts": 1.0,
                    "last_seen_ts": 2.0,
                },  # ZZ no es hex
                {
                    "visitor_hash": "bb" * 16,
                    "protocol": "ble",
                    "rssi_max": -68,
                    "first_seen_ts": 3.0,
                    "last_seen_ts": 4.0,
                },
            ],
        },
    }
    result = handler(event, None)
    assert result["statusCode"] == 200
    rows = fake_pg["cursor"].executemany.call_args[0][1]
    assert len(rows) == 1
    assert rows[0][2] == bytes.fromhex("bb" * 16)


def test_unknown_type_returns_400(fake_pg):
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "unknown_type",
        "data": {},
    }
    result = handler(event, None)

    assert result["statusCode"] == 400
    # No debe haber intentado conectar a Postgres.
    assert not fake_pg["psycopg"].connect.called


def test_missing_device_id_returns_400(fake_pg):
    from src.cloud.persist_event import handler

    event = {
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {"direction": "in"},
    }
    result = handler(event, None)

    assert result["statusCode"] == 400


def test_malformed_payload_returns_400(fake_pg):
    """Falta una key requerida del data dict → KeyError → 400 (no re-raise)."""
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {},  # falta direction
    }
    result = handler(event, None)

    assert result["statusCode"] == 400


def test_store_id_inferred_from_device_id(fake_pg):
    """store-pilot-01-cam-03 → store-pilot-01."""
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-pilot-01-cam-03",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {"direction": "out"},
    }
    handler(event, None)

    params = fake_pg["cursor"].execute.call_args[0][1]
    assert params[1] == "store-pilot-01"


def test_store_id_fallback_when_no_cam_suffix(fake_pg):
    """Device id sin '-cam-' → device_id entero como store_id."""
    from src.cloud.persist_event import handler

    event = {
        "device_id": "weird-device-name",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {"direction": "in"},
    }
    handler(event, None)

    params = fake_pg["cursor"].execute.call_args[0][1]
    assert params[1] == "weird-device-name"


def test_transient_db_error_reraised(fake_pg):
    """Errores no-conocidos se re-raisean para que IoT reintente."""
    from src.cloud.persist_event import handler

    # Simula error transitorio en el INSERT (primera llamada al execute).
    fake_pg["cursor"].execute.side_effect = RuntimeError("connection lost")

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {"direction": "in"},
    }
    with pytest.raises(RuntimeError, match="connection lost"):
        handler(event, None)


def test_constraint_violation_discarded_without_closing_connection(fake_pg):
    """Regresión: las violaciones de constraint (CheckViolation por una
    direction inválida, NotNullViolation por timestamps ausentes) son
    errores de DATOS del device — el contrato del módulo promete loguear y
    descartar, pero caían en el except genérico que cierra la conexión warm
    y re-raisea como transitorio: cada payload malformado forzaba una
    reconexión IAM completa (3-8s) en la próxima invocación."""
    from src.cloud import persist_event
    from src.cloud.persist_event import handler

    fake_pg["cursor"].execute.side_effect = fake_pg["psycopg"].IntegrityError(
        'new row violates check constraint "count_events_direction_check"'
    )

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {"direction": "sideways", "track_id": 1},
    }
    result = handler(event, None)

    # Descartado (400, sin retry), NO re-raiseado.
    assert result["statusCode"] == 400
    # La conexión warm sigue viva — no se cerró ni se reseteó el cache.
    assert persist_event._pg_conn is not None
    fake_pg["conn"].close.assert_not_called()


def test_data_error_discarded_as_400(fake_pg):
    """DataError (valor inadaptable / fuera de rango) = error de datos del
    device → 400 sin retry, misma política que IntegrityError."""
    from src.cloud.persist_event import handler

    fake_pg["cursor"].execute.side_effect = fake_pg["psycopg"].DataError(
        "invalid input syntax"
    )

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {"direction": "in", "track_id": 1},
    }
    assert handler(event, None)["statusCode"] == 400


# ---------------------------------------------------------------------------
# Resiliencia de la conexión (Lambda caliente) + helpers SSL/ts.
# ---------------------------------------------------------------------------


def test_get_connection_reuses_healthy_cached(fake_pg):
    """Conexión cacheada que responde el health-check (SELECT 1) se reusa sin
    reconectar — el caso normal del Lambda caliente."""
    from src.cloud import persist_event

    persist_event._pg_conn = fake_pg["conn"]
    conn = persist_event._get_connection()

    assert conn is fake_pg["conn"]
    fake_pg["psycopg"].connect.assert_not_called()  # no reabrió
    fake_pg["conn"].close.assert_not_called()


def test_get_connection_reconnects_on_stale(fake_pg):
    """Conexión cacheada cuyo SELECT 1 falla (token expirado / network blip) se
    cierra y se reabre. Cubre el path de reconexión del Lambda caliente."""
    from src.cloud import persist_event

    stale = MagicMock()
    stale.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError(
        "stale connection"
    )
    persist_event._pg_conn = stale

    conn = persist_event._get_connection()

    stale.close.assert_called_once()
    assert conn is fake_pg["conn"]  # reabrió con psycopg.connect
    fake_pg["psycopg"].connect.assert_called_once()


def test_get_connection_swallows_stale_close_failure(fake_pg):
    """Si el close() de la conexión stale también falla, se traga la excepción
    y reconecta igual (no debe propagar)."""
    from src.cloud import persist_event

    stale = MagicMock()
    stale.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError(
        "stale"
    )
    stale.close.side_effect = RuntimeError("close failed too")
    persist_event._pg_conn = stale

    conn = persist_event._get_connection()

    assert conn is fake_pg["conn"]


def test_get_connection_sets_sslrootcert_when_present(fake_pg, monkeypatch):
    """Con un CA bundle disponible, la conexión usa sslmode=verify-full +
    sslrootcert (verificación del cert del server, no solo cifrado)."""
    from src.cloud import persist_event

    monkeypatch.setattr(persist_event, "_ssl_root_cert", lambda: "/opt/ca.pem")
    persist_event._pg_conn = None
    persist_event._get_connection()

    kwargs = fake_pg["psycopg"].connect.call_args.kwargs
    assert kwargs["sslmode"] == "verify-full"
    assert kwargs["sslrootcert"] == "/opt/ca.pem"


def test_ssl_root_cert_from_env(monkeypatch):
    """SSL_ROOT_CERT_PATH explícito gana si el archivo existe."""
    from src.cloud import persist_event

    monkeypatch.setenv("SSL_ROOT_CERT_PATH", "/custom/ca.pem")
    monkeypatch.setattr(
        persist_event.os.path, "exists", lambda p: p == "/custom/ca.pem"
    )
    assert persist_event._ssl_root_cert() == "/custom/ca.pem"


def test_ssl_root_cert_from_convention_path(monkeypatch):
    """Sin env var, cae a los paths convencionales (Layer /opt o code /var)."""
    from src.cloud import persist_event

    monkeypatch.delenv("SSL_ROOT_CERT_PATH", raising=False)
    monkeypatch.setattr(
        persist_event.os.path, "exists", lambda p: p == "/opt/rds-ca-bundle.pem"
    )
    assert persist_event._ssl_root_cert() == "/opt/rds-ca-bundle.pem"


def test_ssl_root_cert_none_when_absent(monkeypatch):
    """Sin env var ni paths convencionales → None (conecta con sslmode=require)."""
    from src.cloud import persist_event

    monkeypatch.delenv("SSL_ROOT_CERT_PATH", raising=False)
    monkeypatch.setattr(persist_event.os.path, "exists", lambda _p: False)
    assert persist_event._ssl_root_cert() is None


def test_ts_none_passthrough():
    """``_ts(None)`` devuelve None (passthrough para timestamps ausentes)."""
    from src.cloud.persist_event import _ts

    assert _ts(None) is None


def test_wifi_ble_all_bad_hashes_inserts_nothing(fake_pg):
    """Si todos los devices traen visitor_hash inválido, se saltean y no queda
    ninguna fila → early return sin ejecutar el INSERT (no crashea la ventana)."""
    from src.cloud.persist_event import _insert_wifi_ble

    _insert_wifi_ble(
        fake_pg["conn"],
        {
            "device_id": "store-001-cam-01",
            "data": {
                "period_start": 1762963200.0,
                "period_end": 1762964100.0,
                "devices": [
                    {
                        "visitor_hash": "ZZZ",  # no es hex válido → se saltea
                        "protocol": "wifi",
                        "rssi_max": -50,
                        "first_seen_ts": 1.0,
                        "last_seen_ts": 2.0,
                    }
                ],
            },
        },
    )
    fake_pg["cursor"].execute.assert_not_called()


def test_value_error_in_insert_returns_400(fake_pg):
    """Un ValueError/TypeError durante el dispatch = dato inadaptable → 400 sin
    retry (no es transitorio)."""
    from src.cloud.persist_event import handler

    fake_pg["cursor"].execute.side_effect = ValueError("bad value")
    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {"direction": "in"},
    }
    assert handler(event, None)["statusCode"] == 400


def test_transient_error_swallows_close_failure(fake_pg):
    """En el path transitorio, si el close() de la conexión rota también falla,
    se traga la excepción, resetea el cache y re-raisea el error original."""
    from src.cloud import persist_event
    from src.cloud.persist_event import handler

    # _pg_conn arranca en None (fixture) → _get_connection abre fresh sin
    # health-check; el INSERT es el único execute y revienta.
    fake_pg["cursor"].execute.side_effect = RuntimeError("conn lost")
    fake_pg["conn"].close.side_effect = RuntimeError("close failed too")

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "counting",
        "data": {"direction": "in"},
    }
    with pytest.raises(RuntimeError, match="conn lost"):
        handler(event, None)
    assert persist_event._pg_conn is None  # reseteado pese al fallo del close

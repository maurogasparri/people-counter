"""Tests del Lambda persist_event con conexión Postgres y SSM mockeadas."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

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
    assert params[1] == "store-001"          # store_id inferido
    # params[2] = event_ts (datetime UTC desde event_time o envelope timestamp).
    assert params[3] == "in"                 # direction
    assert params[4] == 42                   # track_id
    assert params[5] == 0.87                 # confidence
    assert params[6] == 1.75                 # height_m


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
        },
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    call_args = fake_pg["cursor"].execute.call_args
    sql = call_args[0][0]
    assert "INSERT INTO telemetry" in sql
    # Canary del Device Shadow llega como columna explícita.
    assert "last_shadow_apply_ts" in sql


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
    assert "ON CONFLICT" in sql       # idempotencia con MAX rssi_max
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
                {"visitor_hash": "ZZ" * 16, "protocol": "wifi", "rssi_max": -55,
                 "first_seen_ts": 1.0, "last_seen_ts": 2.0},  # ZZ no es hex
                {"visitor_hash": "bb" * 16, "protocol": "ble", "rssi_max": -68,
                 "first_seen_ts": 3.0, "last_seen_ts": 4.0},
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

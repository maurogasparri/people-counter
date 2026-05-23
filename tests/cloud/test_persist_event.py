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
    """Schema del INSERT (8 columnas, sin bucket_15min — ahora GENERATED en RDS):
        0=device_id, 1=store_id, 2=event_ts, 3=direction,
        4=track_id, 5=confidence, 6=height_class, 7=height_m.
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
            "height_class": "adult",
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
    assert params[0] == "store-001-cam-01"
    assert params[1] == "store-001"          # store_id inferido
    # params[2] = event_ts (datetime UTC desde event_time o envelope timestamp).
    assert params[3] == "in"                 # direction
    assert params[4] == 42                   # track_id
    assert params[5] == 0.87                 # confidence
    assert params[6] == "adult"              # height_class
    assert params[7] == 1.75                 # height_m


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


def test_wifi_ble_event_inserts(fake_pg):
    """Schema del INSERT (7 columnas, sin bucket_15min — ahora GENERATED en RDS;
    con last_seen_ts opcional nullable):
        0=device_id, 1=store_id, 2=period_start, 3=period_end,
        4=passersby, 5=shoppers, 6=last_seen_ts.
    """
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "wifi_ble",
        "data": {
            "period_start": 1762962300,
            "period_end": 1762963200,
            "passersby": 160,
            "shoppers": 27,
            "last_seen_ts": 1762963180.0,  # 20s antes del period_end
        },
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    call_args = fake_pg["cursor"].execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    assert "INSERT INTO wifi_ble_summary" in sql
    assert "bucket_15min" not in sql  # GENERATED server-side
    assert "last_seen_ts" in sql
    assert params[0] == "store-001-cam-01"
    assert params[1] == "store-001"
    assert params[4] == 160  # passersby
    assert params[5] == 27   # shoppers
    # params[6] = last_seen_ts (datetime UTC). Si no viene en el payload, None.
    assert params[6] is not None


def test_wifi_ble_event_without_last_seen_ts(fake_pg):
    """Devices con firmware viejo no mandan ``last_seen_ts`` — el INSERT debe
    funcionar con NULL en esa columna (la col del schema es NULLABLE)."""
    from src.cloud.persist_event import handler

    event = {
        "device_id": "store-001-cam-01",
        "timestamp": 1762963200.0,
        "type": "wifi_ble",
        "data": {
            "period_start": 1762962300,
            "period_end": 1762963200,
            "passersby": 50,
            "shoppers": 8,
            # last_seen_ts ausente — device viejo
        },
    }
    result = handler(event, None)
    assert result["statusCode"] == 200
    params = fake_pg["cursor"].execute.call_args[0][1]
    assert params[6] is None  # last_seen_ts → NULL


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

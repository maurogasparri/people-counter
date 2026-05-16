"""Tests del Lambda persist_event con conexión Postgres y SSM mockeadas."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Resetea conexiones cacheadas entre tests para que cada uno arranque limpio."""
    # Import lazy así el módulo se re-evalúa con env vars limpios si hace falta.
    from src.cloud import persist_event

    persist_event._pg_conn = None
    persist_event._pg_password = None
    yield
    persist_event._pg_conn = None
    persist_event._pg_password = None


@pytest.fixture
def fake_pg(monkeypatch):
    """Mockea psycopg.connect + SSM. Devuelve la conn fake para asserts."""
    monkeypatch.setenv("PG_HOST", "localhost")
    monkeypatch.setenv("PG_DB", "test_db")
    monkeypatch.setenv("PG_USER", "test_user")

    # Mock SSM get_parameter
    fake_ssm = MagicMock()
    fake_ssm.get_parameter.return_value = {"Parameter": {"Value": "fake-pw"}}
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = fake_ssm

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
        "ssm": fake_ssm,
        "psycopg": fake_psycopg,
        "conn": fake_conn,
        "cursor": fake_cursor,
    }


def test_counting_event_inserts(fake_pg):
    """Schema actual del INSERT (17 columnas) — ver _insert_counting.
        0=device_id, 1=store_id, 2=event_ts, 3=event_bucket, 4=direction,
        5=track_id, 6=confidence, 7=position_y,
        8=height_class, 9=height_m, 10=head_depth_m,
        11=total_in, 12=total_out,
        13=scaling_factor, 14=scaled_in, 15=scaled_out, 16=best_frame_path.
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
        },
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    # Verifica que ejecutó un INSERT a count_events (tabla nueva post-bootstrap).
    assert fake_pg["cursor"].execute.called
    call_args = fake_pg["cursor"].execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    assert "INSERT INTO count_events" in sql
    assert params[0] == "store-001-cam-01"
    assert params[1] == "store-001"  # store_id inferido del device_id
    # params[2] = event_ts (datetime UTC desde event_time o envelope timestamp).
    # params[3] = event_bucket — None cuando el cliente no lo manda.
    assert params[3] is None
    assert params[4] == "in"          # direction
    assert params[5] == 42            # track_id
    assert params[6] == 0.87          # confidence
    # params[7] = position_y (None aca, no viene en el payload)
    assert params[7] is None
    assert params[8] == "adult"       # height_class


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
        },
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    call_args = fake_pg["cursor"].execute.call_args
    sql = call_args[0][0]
    assert "INSERT INTO telemetry" in sql


def test_wifi_ble_event_inserts(fake_pg):
    """Schema actual del INSERT (7 columnas):
        0=device_id, 1=store_id, 2=period_start, 3=period_end,
        4=period_bucket, 5=passersby, 6=shoppers.

    ``period_bucket`` se agregó después del schema original — falla del
    test legacy era esperar passersby/shoppers en posiciones 4/5 cuando
    ahora están en 5/6.
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
        },
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    call_args = fake_pg["cursor"].execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    assert "INSERT INTO wifi_ble_summary" in sql
    assert params[0] == "store-001-cam-01"
    assert params[1] == "store-001"
    # params[4] = period_bucket (default = period_start cuando no viene).
    assert params[5] == 160  # passersby
    assert params[6] == 27   # shoppers


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

"""Tests del Lambda ingest_pos_transaction con conexion Postgres + boto3 mockeadas."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_state():
    from src.cloud import ingest_pos_transaction

    ingest_pos_transaction._pg_conn = None
    yield
    ingest_pos_transaction._pg_conn = None


@pytest.fixture
def fake_pg(monkeypatch):
    """Mockea psycopg.connect + boto3 (RDS IAM token). Devuelve la conn fake."""
    monkeypatch.setenv("PG_HOST", "localhost")
    monkeypatch.setenv("PG_DB", "test_db")
    monkeypatch.setenv("PG_USER", "test_user")
    monkeypatch.setenv("PG_REGION", "us-east-1")

    fake_rds = MagicMock()
    fake_rds.generate_db_auth_token.return_value = "fake-iam-token"
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = fake_rds

    fake_cursor = MagicMock()
    fake_cursor.rowcount = 1  # por defecto, todos los INSERTs son nuevos
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


def _apigw_event(body):
    """Wrap body en event shape de API Gateway HTTP API v2."""
    return {
        "version": "2.0",
        "routeKey": "POST /pos/transactions",
        "body": json.dumps(body) if not isinstance(body, str) else body,
        "isBase64Encoded": False,
    }


def _single_tx(**overrides):
    base = {
        "transaction_id": "POS-RECOLETA-20260518-001234",
        "store_id": "ar-recoleta",
        "event_ts": "2026-05-18T14:32:15-03:00",
        "type": "sale",
        "items": 2,
        "amount_minor": 4500000,
        "currency": "ARS",
        "payment_method": "credit_card",
    }
    base.update(overrides)
    return base


# =============================================================================
# Happy paths
# =============================================================================


def test_single_transaction_inserts(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    result = handler(_apigw_event(_single_tx()), None)

    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body == {"total": 1, "inserted": 1, "conflicted": 0}

    # Verifica el SQL + parametros
    call_args = fake_pg["cursor"].execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    assert "INSERT INTO pos_transactions" in sql
    assert "ON CONFLICT (store_id, transaction_id) DO NOTHING" in sql
    assert params[0] == "POS-RECOLETA-20260518-001234"  # transaction_id
    assert params[1] == "ar-recoleta"                   # store_id
    assert params[3] == "sale"                          # type
    assert params[4] == 2                               # items
    assert params[5] == 4500000                         # amount_minor
    assert params[6] == "ARS"                           # currency
    assert params[7] == "credit_card"                   # payment_method


def test_bulk_array_inserts(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    txs = [
        _single_tx(transaction_id="POS-001"),
        _single_tx(transaction_id="POS-002", type="return", amount_minor=2000000),
        _single_tx(transaction_id="POS-003", items=5, amount_minor=8000000),
    ]
    result = handler(_apigw_event(txs), None)

    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body == {"total": 3, "inserted": 3, "conflicted": 0}
    assert fake_pg["cursor"].execute.call_count == 3


def test_idempotency_conflicted_counted(fake_pg):
    """Cuando ON CONFLICT dispara, rowcount=0 — debe contarse como 'conflicted'."""
    from src.cloud.ingest_pos_transaction import handler

    # Primera y tercera transaccion chocan con UNIQUE; la del medio es nueva.
    fake_pg["cursor"].rowcount = 0
    rowcounts = [0, 1, 0]

    def side_effect(*args, **kwargs):
        fake_pg["cursor"].rowcount = rowcounts.pop(0)

    fake_pg["cursor"].execute.side_effect = side_effect

    txs = [_single_tx(transaction_id=f"POS-{i:03d}") for i in range(3)]
    result = handler(_apigw_event(txs), None)

    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body == {"total": 3, "inserted": 1, "conflicted": 2}


def test_optional_payment_method_null(fake_pg):
    """payment_method es nullable (batch con metodos mezclados)."""
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx()
    tx.pop("payment_method")
    result = handler(_apigw_event(tx), None)

    assert result["statusCode"] == 200
    params = fake_pg["cursor"].execute.call_args[0][1]
    assert params[7] is None


def test_default_currency_ars(fake_pg):
    """Sin currency, default ARS."""
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx()
    tx.pop("currency")
    handler(_apigw_event(tx), None)

    params = fake_pg["cursor"].execute.call_args[0][1]
    assert params[6] == "ARS"


def test_default_items_one(fake_pg):
    """Sin items, default 1."""
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx()
    tx.pop("items")
    handler(_apigw_event(tx), None)

    params = fake_pg["cursor"].execute.call_args[0][1]
    assert params[4] == 1


def test_epoch_seconds_accepted(fake_pg):
    """event_ts puede ser epoch numerico en segundos."""
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx(event_ts=1779897135.0)  # 2026-05-18T17:32:15Z aprox
    result = handler(_apigw_event(tx), None)

    assert result["statusCode"] == 200


def test_epoch_milliseconds_accepted(fake_pg):
    """event_ts puede ser epoch en milisegundos (heuristic > 10^12)."""
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx(event_ts=1779897135000)
    result = handler(_apigw_event(tx), None)

    assert result["statusCode"] == 200


# =============================================================================
# Validation errors → 400
# =============================================================================


def test_empty_body_400(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    result = handler({"body": ""}, None)
    assert result["statusCode"] == 400
    assert not fake_pg["psycopg"].connect.called


def test_invalid_json_400(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    result = handler({"body": "not json {{"}, None)
    assert result["statusCode"] == 400


def test_body_not_object_or_array_400(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    result = handler(_apigw_event("just a string"), None)
    assert result["statusCode"] == 400


def test_empty_array_400(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    result = handler(_apigw_event([]), None)
    assert result["statusCode"] == 400


def test_missing_required_field_400(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx()
    tx.pop("store_id")
    result = handler(_apigw_event(tx), None)

    assert result["statusCode"] == 400
    body = json.loads(result["body"])
    assert "store_id" in body["error"]


def test_invalid_type_400(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx(type="invalid_type")
    result = handler(_apigw_event(tx), None)

    assert result["statusCode"] == 400


def test_negative_amount_400(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx(amount_minor=-100)
    result = handler(_apigw_event(tx), None)

    assert result["statusCode"] == 400


def test_negative_items_400(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx(items=-1)
    result = handler(_apigw_event(tx), None)

    assert result["statusCode"] == 400


def test_naive_datetime_400(fake_pg):
    """event_ts sin timezone explicita es rechazado."""
    from src.cloud.ingest_pos_transaction import handler

    tx = _single_tx(event_ts="2026-05-18T14:32:15")
    result = handler(_apigw_event(tx), None)

    assert result["statusCode"] == 400


def test_atomic_validation_on_bulk(fake_pg):
    """Si UNA transaccion del bulk es invalida, NINGUNA se inserta."""
    from src.cloud.ingest_pos_transaction import handler

    txs = [
        _single_tx(transaction_id="POS-OK"),
        _single_tx(transaction_id="POS-BAD", type="WAT"),  # invalid
        _single_tx(transaction_id="POS-ALSO-OK"),
    ]
    result = handler(_apigw_event(txs), None)

    assert result["statusCode"] == 400
    assert not fake_pg["cursor"].execute.called  # ningun INSERT corrio


# =============================================================================
# Transient errors → re-raise (Lambda retorna 5xx, POS reintenta)
# =============================================================================


def test_transient_db_error_reraised(fake_pg):
    from src.cloud.ingest_pos_transaction import handler

    fake_pg["cursor"].execute.side_effect = RuntimeError("connection lost")

    with pytest.raises(RuntimeError, match="connection lost"):
        handler(_apigw_event(_single_tx()), None)

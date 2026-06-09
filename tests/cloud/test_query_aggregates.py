"""Tests del Lambda query_aggregates con conexion Postgres + boto3 mockeadas."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_state():
    from src.cloud import query_aggregates

    query_aggregates._pg_conn = None
    yield
    query_aggregates._pg_conn = None


@pytest.fixture
def fake_pg(monkeypatch):
    """Mockea psycopg.connect + boto3 (RDS IAM token). Devuelve la conn fake."""
    monkeypatch.setenv("PG_HOST", "localhost")
    monkeypatch.setenv("PG_DB", "test_db")
    monkeypatch.setenv("PG_USER", "test_user")
    monkeypatch.setenv("PG_REGION", "us-east-1")
    monkeypatch.setenv("API_BASE_URL", "https://api.test.local")

    fake_rds = MagicMock()
    fake_rds.generate_db_auth_token.return_value = "fake-iam-token"
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = fake_rds

    fake_cursor = MagicMock()
    fake_cursor.description = []
    fake_cursor.fetchall.return_value = []
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


def _apigw_event(
    path: str = "/v1/aggregates",
    qs: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
    """API Gateway HTTP API v2 event shape (GET request)."""
    return {
        "version": "2.0",
        "routeKey": f"GET {path}",
        "rawPath": path,
        "queryStringParameters": qs or {},
        "headers": headers or {},
        "requestContext": {"http": {"method": "GET", "path": path}},
    }


def _sql_row_factory(cursor_mock, columns: list[str], rows: list[tuple]):
    """Setea fake_cursor.description + fetchall para devolver `rows` con `columns`."""
    cursor_mock.description = [MagicMock(name=c) for c in columns]
    for desc, name in zip(cursor_mock.description, columns):
        desc.name = name
    cursor_mock.fetchall.return_value = rows


# =============================================================================
# Input parsing — happy path
# =============================================================================


def test_parse_input_minimal_required(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00Z",
            "to": "2026-05-02T00:00:00Z",
        }
    )
    params = query_aggregates._parse_input(event)
    assert params["bucket"] == "15min"
    assert params["sites"] is None
    assert params["cursor"] is None
    assert params["limit"] == 1000
    assert params["from"].tzinfo == timezone.utc
    assert params["to"].tzinfo == timezone.utc


def test_parse_sites_csv(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00Z",
            "to": "2026-05-02T00:00:00Z",
            "sites": "site_a,site_b,site_a",  # dedup
        }
    )
    params = query_aggregates._parse_input(event)
    assert params["sites"] == ["site_a", "site_b"]


def test_parse_input_with_offset_timezone(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00-03:00",
            "to": "2026-05-02T00:00:00-03:00",
        }
    )
    params = query_aggregates._parse_input(event)
    # -03:00 = UTC+3, así que la conversión a UTC suma 3hs
    assert params["from"].hour == 3
    assert params["from"].tzinfo == timezone.utc


# =============================================================================
# Input validation — RFC 7807 errors
# =============================================================================


def test_missing_from_returns_400(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(qs={"to": "2026-05-02T00:00:00Z"})
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 400
    assert resp["headers"]["content-type"] == "application/problem+json"
    body = json.loads(resp["body"])
    assert body["type"].endswith("/missing-parameter")
    assert body["status"] == 400
    assert body["parameter"] == "from"


def test_naive_datetime_returns_400(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00",  # sin TZ
            "to": "2026-05-02T00:00:00Z",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 400
    body = json.loads(resp["body"])
    assert body["type"].endswith("/invalid-datetime")


def test_from_gte_to_returns_400(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-02T00:00:00Z",
            "to": "2026-05-01T00:00:00Z",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 400
    body = json.loads(resp["body"])
    assert body["type"].endswith("/invalid-range")


def test_range_too_large_for_15min_returns_400(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-01-01T00:00:00Z",
            "to": "2026-04-01T00:00:00Z",  # 90d con bucket=15min, cap=7d
            "bucket": "15min",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 400
    body = json.loads(resp["body"])
    assert body["type"].endswith("/range-too-large")
    assert body["max_days_for_bucket"] == 7


def test_invalid_bucket_returns_400(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00Z",
            "to": "2026-05-02T00:00:00Z",
            "bucket": "30min",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 400
    body = json.loads(resp["body"])
    assert body["type"].endswith("/invalid-bucket")
    assert "15min" in body["allowed"]


def test_invalid_site_id_returns_400(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00Z",
            "to": "2026-05-02T00:00:00Z",
            "sites": "site_a,DROP TABLE users;--",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 400
    body = json.loads(resp["body"])
    assert body["type"].endswith("/invalid-site-id")


def test_limit_out_of_range_returns_400(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00Z",
            "to": "2026-05-02T00:00:00Z",
            "limit": "10000",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 400
    body = json.loads(resp["body"])
    assert body["type"].endswith("/out-of-range")


def test_invalid_cursor_returns_400(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00Z",
            "to": "2026-05-02T00:00:00Z",
            "cursor": "not-base64-!!!",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 400
    body = json.loads(resp["body"])
    assert body["type"].endswith("/invalid-cursor")


# =============================================================================
# Cursor encode/decode round trip
# =============================================================================


def test_cursor_round_trip():
    from src.cloud import query_aggregates

    bucket_iso = "2026-05-25T10:00:00Z"
    site_id = "site_54_21"

    encoded = query_aggregates._encode_cursor(bucket_iso, site_id)
    decoded_bucket, decoded_site = query_aggregates._decode_cursor(encoded)

    assert decoded_site == site_id
    assert decoded_bucket.isoformat() == "2026-05-25T10:00:00+00:00"


def test_cursor_decode_none_returns_none():
    from src.cloud import query_aggregates

    assert query_aggregates._decode_cursor(None) is None
    assert query_aggregates._decode_cursor("") is None


# =============================================================================
# Happy path — full response shape
# =============================================================================


_AGG_COLS = [
    "bucket_start",
    "store_id",
    "in_adult",
    "in_child",
    "in_unknown",
    "in_total",
    "out_adult",
    "out_child",
    "out_unknown",
    "out_total",
    "passersby",
    "shoppers",
    "sales",
    "returns",
    "transactions",
    "items_sale",
    "items_return",
    "amount_minor_sale",
    "amount_minor_return",
    "currency",
]


def test_aggregates_happy_path_shape(fake_pg):
    from src.cloud import query_aggregates

    bucket = datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc)
    rows = [
        (
            bucket,
            "site_a",
            12,
            0,
            2,
            14,
            11,
            0,
            3,
            14,
            87,
            22,
            5,
            1,
            6,
            18,
            2,
            142350,
            12500,
            "ARS",
        )
    ]

    cursor_mock = fake_pg["cursor"]

    def fake_execute(sql, params=None):
        if "FROM sites" in sql:
            cursor_mock.fetchall.return_value = [("site_a",)]
            cursor_mock.description = [MagicMock()]
            cursor_mock.description[0].name = "store_id"
        elif "data_freshness_by_store" in sql:
            cursor_mock.fetchall.return_value = [
                ("site_a", datetime(2026, 5, 25, 13, 42, 18, tzinfo=timezone.utc))
            ]
        else:
            _sql_row_factory(cursor_mock, _AGG_COLS, rows)

    cursor_mock.execute.side_effect = fake_execute

    event = _apigw_event(
        qs={
            "from": "2026-05-25T00:00:00Z",
            "to": "2026-05-25T23:59:59Z",
            "bucket": "1h",
        }
    )
    resp = query_aggregates.handler(event, None)

    assert resp["statusCode"] == 200
    assert resp["headers"]["content-type"] == "application/json"
    body = json.loads(resp["body"])

    assert body["bucket"] == "1h"
    assert body["data_freshness"]["site_a"] == "2026-05-25T13:42:18Z"
    assert len(body["rows"]) == 1

    row = body["rows"][0]
    assert row["site_id"] == "site_a"
    assert row["bucket_start"] == "2026-05-25T10:00:00Z"
    assert row["counts"]["in"] == {"adult": 12, "child": 0, "unknown": 2, "total": 14}
    assert row["counts"]["out"] == {"adult": 11, "child": 0, "unknown": 3, "total": 14}
    assert row["external_traffic"] == {"passersby": 87, "shoppers": 22}
    assert row["pos"] == {
        "sales": 5,
        "returns": 1,
        "transactions": 6,
        "items_sale": 18,
        "items_return": 2,
        "amount_minor_sale": 142350,
        "amount_minor_return": 12500,
        "currency": "ARS",
    }


def test_response_has_etag_and_cache_control(fake_pg):
    from src.cloud import query_aggregates

    cursor_mock = fake_pg["cursor"]
    cursor_mock.execute.side_effect = lambda *a, **k: (
        _sql_row_factory(cursor_mock, ["store_id"], [("site_a",)])
        if "FROM sites" in a[0]
        else _sql_row_factory(cursor_mock, _AGG_COLS, [])
    )

    # Rango muy en el pasado → cache_control immutable
    event = _apigw_event(
        qs={
            "from": "2020-01-01T00:00:00Z",
            "to": "2020-01-02T00:00:00Z",
            "bucket": "1d",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 200
    assert resp["headers"]["etag"].startswith('"')
    assert "immutable" in resp["headers"]["cache-control"]
    assert "max-age=86400" in resp["headers"]["cache-control"]


def test_response_cache_control_no_cache_for_recent_data(fake_pg):
    from src.cloud import query_aggregates

    cursor_mock = fake_pg["cursor"]
    cursor_mock.execute.side_effect = lambda *a, **k: (
        _sql_row_factory(cursor_mock, ["store_id"], [("site_a",)])
        if "FROM sites" in a[0]
        else _sql_row_factory(cursor_mock, _AGG_COLS, [])
    )

    # `to` apunta a futuro cercano → no-cache (datos siendo escritos)
    now = datetime.now(tz=timezone.utc)
    event = _apigw_event(
        qs={
            "from": (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "to": (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "bucket": "15min",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 200
    assert resp["headers"]["cache-control"] == "no-cache"


def test_link_header_first_only_when_no_more(fake_pg):
    from src.cloud import query_aggregates

    cursor_mock = fake_pg["cursor"]
    cursor_mock.execute.side_effect = lambda *a, **k: (
        _sql_row_factory(cursor_mock, ["store_id"], [("site_a",)])
        if "FROM sites" in a[0]
        else _sql_row_factory(cursor_mock, _AGG_COLS, [])
    )

    event = _apigw_event(
        qs={
            "from": "2026-05-25T00:00:00Z",
            "to": "2026-05-25T01:00:00Z",
        }
    )
    resp = query_aggregates.handler(event, None)
    link = resp["headers"]["link"]
    assert 'rel="first"' in link
    assert 'rel="next"' not in link


def test_link_header_includes_next_when_has_more(fake_pg):
    from src.cloud import query_aggregates

    bucket = datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc)
    cursor_mock = fake_pg["cursor"]

    def fake_execute(sql, params=None):
        if "FROM sites" in sql:
            _sql_row_factory(cursor_mock, ["store_id"], [("site_a",)])
            return
        if "data_freshness_by_store" in sql:
            cursor_mock.fetchall.return_value = []
            return
        # Devolvemos limit+1 filas → has_more=True
        n = params["limit_plus_one"]
        rows = [
            (
                bucket + timedelta(minutes=15 * i),
                "site_a",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                "ARS",
            )
            for i in range(n)
        ]
        _sql_row_factory(cursor_mock, _AGG_COLS, rows)

    cursor_mock.execute.side_effect = fake_execute

    event = _apigw_event(
        qs={
            "from": "2026-05-25T00:00:00Z",
            "to": "2026-05-25T23:59:59Z",
            "limit": "2",
        }
    )
    resp = query_aggregates.handler(event, None)
    link = resp["headers"]["link"]
    assert 'rel="first"' in link
    assert 'rel="next"' in link
    assert "cursor=" in link


def test_if_none_match_returns_304(fake_pg):
    from src.cloud import query_aggregates

    cursor_mock = fake_pg["cursor"]
    cursor_mock.execute.side_effect = lambda *a, **k: (
        _sql_row_factory(cursor_mock, ["store_id"], [("site_a",)])
        if "FROM sites" in a[0]
        else _sql_row_factory(cursor_mock, _AGG_COLS, [])
    )

    qs = {
        "from": "2020-01-01T00:00:00Z",
        "to": "2020-01-02T00:00:00Z",
        "bucket": "1d",
    }
    # Primero pedimos sin If-None-Match para obtener el etag
    resp = query_aggregates.handler(_apigw_event(qs=qs), None)
    etag = resp["headers"]["etag"]

    # Segunda llamada con If-None-Match igual al etag → 304
    resp2 = query_aggregates.handler(
        _apigw_event(qs=qs, headers={"If-None-Match": etag}), None
    )
    assert resp2["statusCode"] == 304
    assert resp2["headers"]["etag"] == etag
    assert "body" not in resp2


def test_empty_sites_returns_empty_body(fake_pg):
    from src.cloud import query_aggregates

    cursor_mock = fake_pg["cursor"]
    cursor_mock.execute.side_effect = lambda *a, **k: _sql_row_factory(
        cursor_mock, ["store_id"], []
    )

    event = _apigw_event(
        qs={
            "from": "2026-05-01T00:00:00Z",
            "to": "2026-05-02T00:00:00Z",
        }
    )
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["rows"] == []
    assert body["data_freshness"] == {}


# =============================================================================
# OpenAPI endpoint
# =============================================================================


def test_openapi_endpoint_no_auth(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(path="/v1/openapi.json")
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 200
    spec = json.loads(resp["body"])
    assert spec["openapi"].startswith("3.1")
    assert "/v1/aggregates" in spec["paths"]
    assert "/v1/openapi.json" in spec["paths"]


def test_unknown_path_returns_404(fake_pg):
    from src.cloud import query_aggregates

    event = _apigw_event(path="/v1/nonexistent")
    resp = query_aggregates.handler(event, None)
    assert resp["statusCode"] == 404
    body = json.loads(resp["body"])
    assert body["type"].endswith("/not-found")
    assert resp["headers"]["content-type"] == "application/problem+json"


# =============================================================================
# Bucket alignment helper
# =============================================================================


def test_align_to_bucket_15min():
    from src.cloud.query_aggregates import _align_to_bucket

    assert _align_to_bucket(
        datetime(2026, 5, 25, 10, 7, 33, 456, tzinfo=timezone.utc), "15min"
    ) == datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc)

    assert _align_to_bucket(
        datetime(2026, 5, 25, 10, 47, 33, tzinfo=timezone.utc), "15min"
    ) == datetime(2026, 5, 25, 10, 45, tzinfo=timezone.utc)


def test_align_to_bucket_hour():
    from src.cloud.query_aggregates import _align_to_bucket

    assert _align_to_bucket(
        datetime(2026, 5, 25, 10, 47, 33, tzinfo=timezone.utc), "1h"
    ) == datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc)


def test_align_to_bucket_day():
    from src.cloud.query_aggregates import _align_to_bucket

    assert _align_to_bucket(
        datetime(2026, 5, 25, 22, 30, tzinfo=timezone.utc), "1d"
    ) == datetime(2026, 5, 25, 0, 0, tzinfo=timezone.utc)


# =============================================================================
# Link header building
# =============================================================================


def test_build_link_no_more_only_first(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://api.test.local")
    from src.cloud import query_aggregates

    link = query_aggregates._build_link_header(
        raw_qs={"from": "2026-05-01T00:00:00Z", "to": "2026-05-02T00:00:00Z"},
        has_more=False,
        next_cursor=None,
    )
    assert 'rel="first"' in link
    assert 'rel="next"' not in link
    assert link.startswith("<https://api.test.local/v1/aggregates?")


def test_build_link_with_next_strips_cursor_from_first(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://api.test.local")
    from src.cloud import query_aggregates

    link = query_aggregates._build_link_header(
        raw_qs={
            "from": "2026-05-01T00:00:00Z",
            "to": "2026-05-02T00:00:00Z",
            "cursor": "oldcursor",
        },
        has_more=True,
        next_cursor="newcursor",
    )
    # "first" no debe llevar cursor (el primer page nunca lo tiene)
    first_part = link.split(", ")[0]
    assert "cursor=" not in first_part
    # "next" sí lleva el cursor nuevo
    next_part = link.split(", ")[1]
    assert "cursor=newcursor" in next_part

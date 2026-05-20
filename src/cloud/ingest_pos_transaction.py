"""AWS Lambda: ingesta de transacciones POS via API Gateway HTTP API (T9.11, US-06).

Invocada por API Gateway v2 (HTTP API) con IAM auth. El POS del cliente firma
los requests con SigV4 contra un IAM role que tiene ``execute-api:Invoke`` sobre
el endpoint POST /pos/transactions.

Body acepta dos formas (a discrecion del POS):

  1) Single transaction (objeto JSON)::

        {
            "transaction_id": "POS-RECOLETA-20260518-001234",
            "store_id": "ar-recoleta",
            "event_ts": "2026-05-18T14:32:15-03:00",
            "type": "sale",
            "items": 2,
            "amount_minor": 4500000,
            "currency": "ARS",
            "payment_method": "credit_card"
        }

  2) Bulk (array de transacciones)::

        [ {tx1}, {tx2}, ... ]

Idempotencia via ``ON CONFLICT (store_id, transaction_id) DO NOTHING`` — el POS
puede reintentar sin duplicar. Si manda corrección con mismo ID, se ignora
silenciosamente (v1 simplification — para correcciones reales usar otro id).

Auth a RDS: IAM database authentication contra el user ``lambda_pos_writer``
(grants solo sobre ``pos_transactions``, least privilege separado del
``lambda_writer`` del persist_event).

Variables de entorno (seteadas por CFN):
    PG_HOST:    endpoint de RDS.
    PG_PORT:    5432 default.
    PG_DB:      people_counter default.
    PG_USER:    lambda_pos_writer default.
    PG_REGION:  region para generar el token IAM (= region de RDS).
    SSL_ROOT_CERT_PATH: idem persist_event.

Respuestas HTTP:
    200: insercion OK (con conteos: {inserted, conflicted, total}).
    400: payload malformado, fields invalidos, type incorrecto.
    500: error transitorio de DB — re-raise para que Lambda devuelva 5xx y el
         POS reintente.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Conexion cacheada entre invocaciones (warm start). Token IAM dura ~15min;
# cuando expira el query falla y reconectamos transparente.
_pg_conn = None

# Whitelist de fields requeridos y opcionales en cada transaccion.
_REQUIRED_FIELDS = ("transaction_id", "store_id", "event_ts", "type", "amount_minor")
_OPTIONAL_FIELDS = ("items", "currency", "payment_method")
_VALID_TYPES = ("sale", "return")
_DEFAULT_CURRENCY = "ARS"
_DEFAULT_ITEMS = 1


def _generate_iam_token() -> str:
    """Genera un token IAM de RDS para autenticacion (~15min de validez)."""
    import boto3

    region = os.environ.get("PG_REGION", "us-east-1")
    host = os.environ["PG_HOST"]
    port = int(os.environ.get("PG_PORT", "5432"))
    user = os.environ.get("PG_USER", "lambda_pos_writer")

    rds = boto3.client("rds", region_name=region)
    return rds.generate_db_auth_token(
        DBHostname=host,
        Port=port,
        DBUsername=user,
        Region=region,
    )


def _ssl_root_cert() -> str | None:
    env_path = os.environ.get("SSL_ROOT_CERT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    for path in ("/opt/rds-ca-bundle.pem", "/var/task/rds-ca-bundle.pem"):
        if os.path.exists(path):
            return path
    return None


def _get_connection():
    """Devuelve la conexion Postgres cacheada, o la abre si no existe."""
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
        "user": os.environ.get("PG_USER", "lambda_pos_writer"),
        "password": _generate_iam_token(),
        "sslmode": sslmode,
        "connect_timeout": 15,
        "autocommit": True,
    }
    root_cert = _ssl_root_cert()
    if root_cert:
        conn_kwargs["sslrootcert"] = root_cert

    _pg_conn = psycopg.connect(**conn_kwargs)
    return _pg_conn


def _parse_event_ts(value: Any) -> datetime:
    """Acepta epoch numerico (s o ms) o ISO-8601 string. Devuelve datetime aware.

    Rechaza naive datetimes (sin timezone) para evitar ambiguedad — el POS
    debe declarar la zona horaria explicitamente (offset o 'Z').
    """
    if isinstance(value, (int, float)):
        # Heuristic: > 10^12 implica milisegundos (timestamps post-2001 en ms);
        # menores son segundos.
        epoch_s = float(value) / 1000.0 if value > 10**12 else float(value)
        return datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    if isinstance(value, str):
        # fromisoformat acepta '2026-05-18T14:32:15-03:00' nativo en Python 3.11+.
        # Tambien acepta 'Z' (UTC) desde 3.11.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError(f"event_ts sin timezone explicita: {value!r}")
        return dt
    raise ValueError(f"event_ts debe ser numerico o ISO-8601 string, got {type(value).__name__}")


def _validate_transaction(tx: Any) -> dict[str, Any]:
    """Valida una transaccion y la normaliza. Raises ValueError on issue."""
    if not isinstance(tx, dict):
        raise ValueError(f"transaccion debe ser un objeto, got {type(tx).__name__}")

    missing = [f for f in _REQUIRED_FIELDS if f not in tx]
    if missing:
        raise ValueError(f"fields requeridos faltantes: {missing}")

    if tx["type"] not in _VALID_TYPES:
        raise ValueError(f"type debe ser uno de {_VALID_TYPES}, got {tx['type']!r}")

    amount = tx["amount_minor"]
    if not isinstance(amount, int) or amount < 0:
        raise ValueError(f"amount_minor debe ser int >= 0, got {amount!r}")

    items = tx.get("items", _DEFAULT_ITEMS)
    if not isinstance(items, int) or items < 0:
        raise ValueError(f"items debe ser int >= 0, got {items!r}")

    return {
        "transaction_id": str(tx["transaction_id"]),
        "store_id": str(tx["store_id"]),
        "event_ts": _parse_event_ts(tx["event_ts"]),
        "type": tx["type"],
        "items": items,
        "amount_minor": amount,
        "currency": str(tx.get("currency", _DEFAULT_CURRENCY))[:3].upper(),
        "payment_method": tx.get("payment_method"),  # nullable
    }


def _insert_transactions(conn, transactions: list[dict[str, Any]]) -> tuple[int, int]:
    """Inserta N transacciones validadas. Devuelve (inserted, conflicted)."""
    if not transactions:
        return (0, 0)

    inserted = 0
    with conn.cursor() as cur:
        for tx in transactions:
            cur.execute(
                """
                INSERT INTO pos_transactions
                    (transaction_id, store_id, event_ts, type,
                     items, amount_minor, currency, payment_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (store_id, transaction_id) DO NOTHING
                """,
                (
                    tx["transaction_id"],
                    tx["store_id"],
                    tx["event_ts"],
                    tx["type"],
                    tx["items"],
                    tx["amount_minor"],
                    tx["currency"],
                    tx["payment_method"],
                ),
            )
            # cur.rowcount = 1 si inserto, 0 si choco con UNIQUE.
            if cur.rowcount > 0:
                inserted += 1
    return (inserted, len(transactions) - inserted)


def _http_response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
    """Shape de respuesta para API Gateway HTTP API (v2) proxy integration."""
    return {
        "statusCode": status_code,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handler de Lambda — invocado por API Gateway HTTP API (POST /pos/transactions).

    Body del request (string JSON en ``event['body']``):
      - Single: ``{tx}``
      - Bulk: ``[{tx1}, {tx2}, ...]``

    Errores de validacion se devuelven como 400 (el POS no debe reintentar).
    Errores transitorios de DB se re-raisean para que Lambda devuelva 5xx y
    el POS pueda reintentar con backoff.
    """
    try:
        body_raw = event.get("body")
        if not body_raw:
            return _http_response(400, {"error": "body vacio"})

        try:
            body = json.loads(body_raw)
        except json.JSONDecodeError as e:
            return _http_response(400, {"error": f"JSON invalido: {e}"})

        # Normalizamos a lista de transacciones.
        if isinstance(body, list):
            raw_txs = body
        elif isinstance(body, dict):
            raw_txs = [body]
        else:
            return _http_response(
                400, {"error": f"body debe ser objeto o array, got {type(body).__name__}"}
            )

        if not raw_txs:
            return _http_response(400, {"error": "array vacio"})

        # Validamos TODAS antes de insertar — si una falla, ninguna entra (atomic).
        try:
            validated = [_validate_transaction(tx) for tx in raw_txs]
        except ValueError as e:
            return _http_response(400, {"error": str(e)})

        conn = _get_connection()
        inserted, conflicted = _insert_transactions(conn, validated)

        logger.info(
            "ingest_pos_ok stores=%s total=%d inserted=%d conflicted=%d",
            sorted({tx["store_id"] for tx in validated}),
            len(validated),
            inserted,
            conflicted,
        )

        return _http_response(
            200,
            {"total": len(validated), "inserted": inserted, "conflicted": conflicted},
        )

    except (ValueError, TypeError) as e:
        logger.warning("ingest_pos_value_error error=%s", e)
        return _http_response(400, {"error": str(e)})
    except Exception:
        global _pg_conn
        if _pg_conn is not None:
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None
        logger.exception("ingest_pos_transient_error")
        raise

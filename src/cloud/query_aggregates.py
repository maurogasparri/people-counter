"""AWS Lambda: API REST de consulta de agregados (T9.12, US-09, RF-13).

Invocada por API Gateway v2 (HTTP API) con IAM auth (SigV4) desde sistemas
externos (ERP, CRM, BI corporativo del retailer). Devuelve una sola respuesta
unificada con counts (con breakdown adult/child/unknown), tráfico externo
WiFi/BLE y POS transactions agrupados por bucket (15min / 1h / 1d) × site.

Endpoints:

    GET /v1/aggregates?from=...&to=...&sites=A,B&bucket=15min&cursor=...&limit=1000
    GET /v1/openapi.json

Diseño:
    * Bucket-flexible: el caller elige granularidad (default 15min).
    * Empty buckets devuelven ceros (generate_series × sites en SQL).
    * Cursor opaco base64({last_bucket, last_site}) — estable ante inserts.
    * RFC 8288 Link header con rel="next" y rel="first".
    * RFC 7807 application/problem+json para errores.
    * ETag fuerte + Cache-Control immutable para rangos cerrados (to < now-1h);
      no-cache si el rango incluye datos aún siendo escritos.
    * EMF metric block embebido en los logs → CloudWatch metric sin Lambda
      Insights.
    * data_freshness por site para diagnóstico desde el cliente.

Caps de rango por bucket (defense contra full table scans):
    * 15min: max 7 días
    * 1h:    max 90 días
    * 1d:    max 365 días

Auth a RDS: IAM database authentication contra el user ``lambda_query_reader``
(SELECT-only sobre count_events, wifi_ble_events, pos_transactions, sites,
devices). Separado del ``lambda_writer`` y ``lambda_pos_writer`` por least
privilege.

Variables de entorno (seteadas por CFN):
    PG_HOST:    endpoint de RDS.
    PG_PORT:    5432 default.
    PG_DB:      people_counter default.
    PG_USER:    lambda_query_reader default.
    PG_REGION:  region para generar el token IAM.
    SSL_ROOT_CERT_PATH: path opcional al cert bundle de RDS (verify-full).
    API_BASE_URL: prefijo público (ej. https://api.<tu-dominio>) usado
                  para construir los Link headers absolutos.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_pg_conn = None

_BUCKET_COLS = {
    "15min": "bucket_15min",
    "1h": "bucket_hour",
    "1d": "bucket_day",
}
# Mapeo bucket → vista unificada (producto cartesiano, definido en bootstrap.sql).
# La Lambda consume estas vistas en lugar de tablas raw, simétrico con
# el rol readonly_external. La vista hace los FULL OUTER JOINs internamente.
_BUCKET_UNIFIED_VIEWS = {
    "15min": "metrics_unified_by_bucket_15min",
    "1h": "metrics_unified_by_bucket_hour",
    "1d": "metrics_unified_by_bucket_day",
}
_BUCKET_INTERVALS = {
    "15min": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}
_BUCKET_MAX_RANGE_DAYS = {
    "15min": 7,
    "1h": 90,
    "1d": 365,
}
_DEFAULT_BUCKET = "15min"
_DEFAULT_LIMIT = 1000
_MAX_LIMIT = 5000
_SITE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
# Base de los tipos de problema RFC 7807. Se deriva de API_BASE_URL para no
# fijar ningun dominio en el codigo; la plantilla de infraestructura inyecta
# esa variable con el dominio real del despliegue.
_PROBLEM_PATH = "/errors"
_CACHE_FRESH_THRESHOLD = timedelta(hours=1)
_CACHE_MAX_AGE_SECONDS = 86400  # 24h


class _BadRequest(Exception):
    """Error de validación: el caller mandó input inválido. → 400."""

    def __init__(
        self,
        problem_slug: str,
        title: str,
        detail: str,
        **extra: Any,
    ) -> None:
        super().__init__(detail)
        self.problem_slug = problem_slug
        self.title = title
        self.detail = detail
        self.extra = extra


# =============================================================================
# Postgres connection (IAM auth, cached warm)
# =============================================================================


def _generate_iam_token() -> str:
    import boto3

    region = os.environ.get("PG_REGION", "us-east-1")
    host = os.environ["PG_HOST"]
    port = int(os.environ.get("PG_PORT", "5432"))
    user = os.environ.get("PG_USER", "lambda_query_reader")

    rds = boto3.client("rds", region_name=region)
    return rds.generate_db_auth_token(
        DBHostname=host, Port=port, DBUsername=user, Region=region
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

    import time

    import psycopg

    sslmode = "verify-full" if _ssl_root_cert() else "require"
    base_kwargs: dict[str, Any] = {
        "host": os.environ["PG_HOST"],
        "port": int(os.environ.get("PG_PORT", "5432")),
        "dbname": os.environ.get("PG_DB", "people_counter"),
        "user": os.environ.get("PG_USER", "lambda_query_reader"),
        "sslmode": sslmode,
        "connect_timeout": 15,
        "autocommit": True,
    }
    root_cert = _ssl_root_cert()
    if root_cert:
        base_kwargs["sslrootcert"] = root_cert

    # Retry de conexión: en cold-start la primera conexión a RDS con token IAM
    # puede fallar transitoriamente (generación/validación del token, handshake
    # SSL, RDS despertando) → el primer hit tras inactividad devolvía 500. Tres
    # intentos con backoff corto lo absorben. El token IAM se regenera en cada
    # intento (es la causa transitoria más común; expira a los 15 min).
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            _pg_conn = psycopg.connect(password=_generate_iam_token(), **base_kwargs)
            return _pg_conn
        except Exception as exc:  # OperationalError + fallos de token/SSL en cold-start
            last_err = exc
            logger.warning(
                "pg_connect_retry",
                extra={"attempt": attempt + 1, "max_attempts": 3, "error": str(exc)},
            )
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
    assert last_err is not None
    raise last_err


# =============================================================================
# Input parsing + validation
# =============================================================================


def _parse_iso_utc(name: str, value: str) -> datetime:
    """Acepta ISO 8601 con timezone explícita (Z u offset). Devuelve UTC aware."""
    if not value:
        raise _BadRequest(
            "missing-parameter",
            "Missing required parameter",
            f"'{name}' is required",
            parameter=name,
        )
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise _BadRequest(
            "invalid-datetime",
            "Invalid ISO 8601 datetime",
            f"'{name}' must be ISO 8601 with explicit timezone (got {value!r}): {e}",
            parameter=name,
        ) from None
    if dt.tzinfo is None:
        raise _BadRequest(
            "invalid-datetime",
            "Naive datetime not allowed",
            f"'{name}' must include timezone offset or 'Z' (got {value!r})",
            parameter=name,
        )
    return dt.astimezone(timezone.utc)


def _parse_sites(value: str | None) -> list[str] | None:
    """CSV de site IDs. None/'' = todos. Valida regex anti-injection."""
    if not value:
        return None
    sites = [s.strip() for s in value.split(",") if s.strip()]
    if not sites:
        return None
    bad = [s for s in sites if not _SITE_ID_RE.match(s)]
    if bad:
        raise _BadRequest(
            "invalid-site-id",
            "Invalid site_id format",
            f"sites must match [a-zA-Z0-9_-]+: invalid={bad}",
            parameter="sites",
            invalid_values=bad,
        )
    # Dedup preservando orden.
    seen: dict[str, None] = {}
    for s in sites:
        seen.setdefault(s, None)
    return list(seen)


def _parse_int(name: str, value: str | None, default: int, lo: int, hi: int) -> int:
    if value is None:
        return default
    try:
        n = int(value)
    except ValueError:
        raise _BadRequest(
            "invalid-integer",
            "Invalid integer parameter",
            f"'{name}' must be an integer (got {value!r})",
            parameter=name,
        ) from None
    if n < lo or n > hi:
        raise _BadRequest(
            "out-of-range",
            "Parameter out of range",
            f"'{name}' must be in [{lo}, {hi}] (got {n})",
            parameter=name,
            min=lo,
            max=hi,
        )
    return n


def _encode_cursor(bucket_iso: str, site_id: str) -> str:
    payload = json.dumps({"b": bucket_iso, "s": site_id}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None) -> tuple[datetime, str] | None:
    if not value:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        parsed = json.loads(raw)
        bucket = datetime.fromisoformat(parsed["b"].replace("Z", "+00:00"))
        if bucket.tzinfo is None:
            raise ValueError("naive datetime in cursor")
        return (bucket.astimezone(timezone.utc), str(parsed["s"]))
    except Exception as e:
        raise _BadRequest(
            "invalid-cursor",
            "Invalid pagination cursor",
            f"cursor is malformed or expired: {e}",
            parameter="cursor",
        ) from None


def _parse_input(event: dict[str, Any]) -> dict[str, Any]:
    qs = event.get("queryStringParameters") or {}

    from_dt = _parse_iso_utc("from", qs.get("from"))
    to_dt = _parse_iso_utc("to", qs.get("to"))
    if from_dt >= to_dt:
        raise _BadRequest(
            "invalid-range",
            "Invalid date range",
            f"'from' must be strictly before 'to' (from={from_dt.isoformat()}, to={to_dt.isoformat()})",
        )

    bucket = qs.get("bucket", _DEFAULT_BUCKET)
    if bucket not in _BUCKET_COLS:
        raise _BadRequest(
            "invalid-bucket",
            "Invalid bucket size",
            f"'bucket' must be one of {sorted(_BUCKET_COLS)} (got {bucket!r})",
            parameter="bucket",
            allowed=sorted(_BUCKET_COLS),
        )

    max_days = _BUCKET_MAX_RANGE_DAYS[bucket]
    range_days = (to_dt - from_dt).total_seconds() / 86400.0
    if range_days > max_days:
        raise _BadRequest(
            "range-too-large",
            "Date range exceeds maximum for bucket size",
            f"Requested {range_days:.1f} days with bucket={bucket}, max allowed is {max_days}d",
            parameter="bucket",
            requested_days=round(range_days, 1),
            max_days_for_bucket=max_days,
        )

    sites = _parse_sites(qs.get("sites"))
    cursor = _decode_cursor(qs.get("cursor"))
    limit = _parse_int("limit", qs.get("limit"), _DEFAULT_LIMIT, 1, _MAX_LIMIT)

    return {
        "from": from_dt,
        "to": to_dt,
        "bucket": bucket,
        "sites": sites,
        "cursor": cursor,
        "limit": limit,
        "raw_qs": qs,
    }


# =============================================================================
# SQL execution
# =============================================================================


def _align_to_bucket(dt: datetime, bucket: str) -> datetime:
    """Trunca un datetime al inicio del bucket que lo contiene."""
    if bucket == "15min":
        minutes = (dt.minute // 15) * 15
        return dt.replace(minute=minutes, second=0, microsecond=0)
    if bucket == "1h":
        return dt.replace(minute=0, second=0, microsecond=0)
    if bucket == "1d":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"unknown bucket: {bucket}")


def _query_sites_if_omitted(conn, sites: list[str] | None) -> list[str]:
    if sites is not None:
        return sites
    with conn.cursor() as cur:
        cur.execute("SELECT store_id FROM sites ORDER BY store_id")
        return [row[0] for row in cur.fetchall()]


def _query_aggregates(
    conn, params: dict[str, Any], resolved_sites: list[str]
) -> tuple[list[dict[str, Any]], bool]:
    """Ejecuta el query principal. Devuelve (rows, has_more).

    ``has_more`` = True si el LIMIT N+1 trajo N+1 filas (hay próxima página).
    Las filas en el output son el primer N (sin la N+1 sentinel).

    La query consume la vista ``metrics_unified_by_bucket_<grano>``
    (definida en bootstrap.sql) que ya hace el FULL OUTER
    JOIN de counting + wifi_ble + pos internamente. El Lambda mantiene el
    grid (generate_series × sites) para devolver filas con ceros donde la
    vista no tiene data y la paginación por cursor.

    Los aliases ``in_*`` / ``out_*`` se preservan en la SELECT para no
    cambiar el shape que espera ``_shape_row`` (la vista usa ``ins_*`` /
    ``outs_*``).
    """
    bucket_col = _BUCKET_COLS[params["bucket"]]
    unified_view = _BUCKET_UNIFIED_VIEWS[params["bucket"]]
    bucket_interval = {
        "15min": "15 minutes",
        "1h": "1 hour",
        "1d": "1 day",
    }[params["bucket"]]

    from_aligned = _align_to_bucket(params["from"], params["bucket"])
    # 'to' es exclusivo. generate_series con end inclusivo, así que restamos un
    # microsegundo para no incluir el bucket de 'to' exacto.
    to_excl = params["to"]
    cursor = params["cursor"]
    limit = params["limit"]

    # cursor_bucket / cursor_site usados en WHERE; si no hay cursor, valores
    # que nunca matchean (bucket < from_aligned, site='').
    if cursor is not None:
        cursor_bucket, cursor_site = cursor
    else:
        cursor_bucket = from_aligned - timedelta(seconds=1)
        cursor_site = ""

    sql = f"""
    WITH bucket_series AS (
        SELECT generate_series(
            %(from_aligned)s::timestamptz,
            %(to_excl)s::timestamptz - INTERVAL '1 microsecond',
            %(interval)s::interval
        ) AS bucket_start
    ),
    sites_filter AS (
        SELECT unnest(%(sites)s::text[]) AS store_id
    ),
    grid AS (
        SELECT b.bucket_start, s.store_id
        FROM bucket_series b
        CROSS JOIN sites_filter s
    )
    SELECT
        g.bucket_start,
        g.store_id,
        -- Mapeo de columnas de la vista (ins_*, outs_*) al shape histórico
        -- (in_*, out_*) que consume _shape_row sin cambios.
        COALESCE(m.ins_adult,   0) AS in_adult,
        COALESCE(m.ins_child,   0) AS in_child,
        COALESCE(m.ins_unknown, 0) AS in_unknown,
        COALESCE(m.ins,         0) AS in_total,
        COALESCE(m.outs_adult,  0) AS out_adult,
        COALESCE(m.outs_child,  0) AS out_child,
        COALESCE(m.outs_unknown,0) AS out_unknown,
        COALESCE(m.outs,        0) AS out_total,
        COALESCE(m.passersby,   0) AS passersby,
        COALESCE(m.shoppers,    0) AS shoppers,
        COALESCE(m.sales,        0) AS sales,
        COALESCE(m.returns,      0) AS returns,
        COALESCE(m.transactions, 0) AS transactions,
        COALESCE(m.items_sale,   0) AS items_sale,
        COALESCE(m.items_return, 0) AS items_return,
        COALESCE(m.amount_minor_sale,   0) AS amount_minor_sale,
        COALESCE(m.amount_minor_return, 0) AS amount_minor_return,
        COALESCE(m.currency, 'ARS') AS currency
    FROM grid g
    LEFT JOIN {unified_view} m
        ON m.store_id = g.store_id AND m.{bucket_col} = g.bucket_start
    WHERE g.store_id = ANY(%(sites)s)
      AND (g.bucket_start, g.store_id) > (%(cursor_bucket)s, %(cursor_site)s)
    ORDER BY g.bucket_start ASC, g.store_id ASC
    LIMIT %(limit_plus_one)s
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            {
                "from_aligned": from_aligned,
                "to_excl": to_excl,
                "interval": bucket_interval,
                "sites": resolved_sites,
                "cursor_bucket": cursor_bucket,
                "cursor_site": cursor_site,
                "limit_plus_one": limit + 1,
            },
        )
        cols = [d.name for d in cur.description]
        raw_rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    has_more = len(raw_rows) > limit
    return (raw_rows[:limit], has_more)


def _query_data_freshness(conn, sites: list[str]) -> dict[str, str]:
    """Último timestamp de ingesta por sucursal. Devuelve dict con ISO strings UTC.

    Consume la vista ``data_freshness_by_store`` (definida en bootstrap.sql)
    que ya hace el UNION ALL + MAX por sucursal
    sobre las 3 facts (count_events, wifi_ble_events, pos_transactions).
    """
    if not sites:
        return {}
    sql = """
    SELECT store_id, last_received_at
    FROM data_freshness_by_store
    WHERE store_id = ANY(%(sites)s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"sites": sites})
        out: dict[str, str] = {}
        for store_id, last_seen in cur.fetchall():
            if last_seen is not None:
                out[store_id] = (
                    last_seen.astimezone(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
        return out


# =============================================================================
# Response shaping
# =============================================================================


def _shape_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Transforma una fila SQL plana al shape público (nested)."""
    bucket_start_iso = (
        raw["bucket_start"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    return {
        "site_id": raw["store_id"],
        "bucket_start": bucket_start_iso,
        "counts": {
            "in": {
                "adult": int(raw["in_adult"]),
                "child": int(raw["in_child"]),
                "unknown": int(raw["in_unknown"]),
                "total": int(raw["in_total"]),
            },
            "out": {
                "adult": int(raw["out_adult"]),
                "child": int(raw["out_child"]),
                "unknown": int(raw["out_unknown"]),
                "total": int(raw["out_total"]),
            },
        },
        "external_traffic": {
            "passersby": int(raw["passersby"]),
            "shoppers": int(raw["shoppers"]),
        },
        "pos": {
            "sales": int(raw["sales"]),
            "returns": int(raw["returns"]),
            "transactions": int(raw["transactions"]),
            "items_sale": int(raw["items_sale"]),
            "items_return": int(raw["items_return"]),
            "amount_minor_sale": int(raw["amount_minor_sale"]),
            "amount_minor_return": int(raw["amount_minor_return"]),
            "currency": raw["currency"],
        },
    }


# =============================================================================
# HTTP headers (Link RFC 8288, ETag, Cache-Control)
# =============================================================================


def _api_base_url() -> str:
    # Sin API_BASE_URL no hay dominio que suponer: se usa un dominio reservado
    # por RFC 2606, que senala configuracion faltante en vez de inventar un
    # extremo. En el despliegue la variable la inyecta la plantilla.
    return os.environ.get("API_BASE_URL", "https://api.example.invalid").rstrip("/")


def _build_link_header(
    raw_qs: dict[str, str],
    has_more: bool,
    next_cursor: str | None,
) -> str | None:
    """Construye Link header RFC 8288 con rel="first" y rel="next" (si hay más)."""
    base = f"{_api_base_url()}/v1/aggregates"

    first_qs = {k: v for k, v in raw_qs.items() if k != "cursor"}
    first_url = f"{base}?{urlencode(first_qs, doseq=False)}"
    parts = [f'<{first_url}>; rel="first"']

    if has_more and next_cursor:
        next_qs = dict(first_qs)
        next_qs["cursor"] = next_cursor
        next_url = f"{base}?{urlencode(next_qs, doseq=False)}"
        parts.append(f'<{next_url}>; rel="next"')

    return ", ".join(parts)


def _compute_etag(body_dict: dict[str, Any]) -> str:
    """ETag fuerte derivado del body canónico. Quoting RFC 7232."""
    canonical = json.dumps(body_dict, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f'"{digest[:32]}"'


def _build_cache_headers(to_dt: datetime, etag: str, now: datetime) -> dict[str, str]:
    """Si el rango es totalmente histórico (to < now-1h), cachear 24h immutable.

    Otherwise, no-cache (la ventana incluye datos posiblemente aún siendo escritos).
    En ambos casos devolvemos ETag para que el cliente pueda usar If-None-Match.
    """
    headers = {"etag": etag}
    if to_dt < now - _CACHE_FRESH_THRESHOLD:
        headers["cache-control"] = (
            f"public, max-age={_CACHE_MAX_AGE_SECONDS}, immutable"
        )
    else:
        headers["cache-control"] = "no-cache"
    return headers


# =============================================================================
# Lambda response builders
# =============================================================================


def _http_response(
    status_code: int,
    body: dict[str, Any] | None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "statusCode": status_code,
        "headers": headers or {},
    }
    if body is not None:
        out["headers"].setdefault("content-type", "application/json")
        out["body"] = json.dumps(body)
    return out


def _problem_response(
    status_code: int,
    slug: str,
    title: str,
    detail: str,
    instance: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """RFC 7807 Problem Details."""
    body: dict[str, Any] = {
        "type": f"{_api_base_url()}{_PROBLEM_PATH}/{slug}",
        "title": title,
        "status": status_code,
        "detail": detail,
    }
    if instance is not None:
        body["instance"] = instance
    body.update(extra)
    return _http_response(
        status_code,
        body,
        headers={"content-type": "application/problem+json"},
    )


def _emf_log(
    metrics: list[tuple[str, float, str]],
    dimensions: dict[str, str],
    **extra: Any,
) -> None:
    """Emite un log JSON con bloque EMF (Embedded Metric Format).

    CloudWatch detecta el bloque ``_aws.CloudWatchMetrics`` y publica las
    métricas en el namespace indicado sin necesidad de PutMetricData.
    """
    payload: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": "PeopleCounter/QueryAPI",
                    "Dimensions": [list(dimensions.keys())] if dimensions else [],
                    "Metrics": [
                        {"Name": name, "Unit": unit} for name, _, unit in metrics
                    ],
                }
            ],
        },
    }
    payload.update(dimensions)
    for name, value, _ in metrics:
        payload[name] = value
    payload.update(extra)
    logger.info(json.dumps(payload, default=str))


# =============================================================================
# OpenAPI 3.1 spec
# =============================================================================


def _openapi_spec() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "People Counter — API REST",
            "version": "1.0.0",
            "summary": (
                "Consulta de agregados (counts demográficos + tráfico externo + "
                "POS) e ingesta de transacciones POS."
            ),
            "description": (
                "API REST del sistema People Counter para clientes B2B (ERP, BI, "
                "apps del retailer).\n\n"
                "Endpoints disponibles:\n"
                "* `GET /v1/aggregates` — consulta unificada de agregados por "
                "bucket (15min / 1h / 1d) × sitio.\n"
                "* `GET /v1/openapi.json` — esta especificación (público, sin auth).\n"
                "* `POST /pos/transactions` — ingesta de transacciones del POS "
                "del cliente.\n\n"
                "**Autenticación**: AWS Signature Version 4 (SigV4) con un IAM "
                "principal (usuario o rol) autorizado para "
                "`execute-api:Invoke` sobre el ARN del endpoint. El cliente "
                "firma cada request con sus credenciales AWS — no se usan "
                "API keys ni tokens custom.\n\n"
                "**Throttling**: 10 req/s steady + 20 req/s burst a nivel API "
                "Gateway (no por cliente). Devuelve `429` cuando se excede.\n\n"
                "**Formato de errores**: los errores 4xx/5xx devuelven "
                "`application/problem+json` siguiendo RFC 7807."
            ),
            "contact": {
                "name": "Mauro Gasparri",
                "email": "mauro@gasparri.com.ar",
            },
        },
        "servers": [
            {"url": _api_base_url(), "description": "Endpoint productivo"},
        ],
        "components": {
            "securitySchemes": {
                "sigv4": {
                    "type": "apiKey",
                    "name": "Authorization",
                    "in": "header",
                    "description": (
                        "AWS Signature Version 4. El cliente firma el request "
                        "con sus credenciales AWS y el servicio `execute-api`."
                    ),
                    "x-amazon-apigateway-authtype": "awsSigv4",
                }
            },
            "schemas": {
                "Problem": {
                    "type": "object",
                    "description": (
                        "Detalle de error siguiendo RFC 7807 (Problem Details "
                        "for HTTP APIs). Content-Type: `application/problem+json`."
                    ),
                    "required": ["type", "title", "status"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "format": "uri",
                            "description": (
                                "URI que identifica el tipo de error. Estable "
                                "entre versiones; los clientes pueden hacer "
                                "match sobre el último segmento (slug)."
                            ),
                            "example": "https://api.<tu-dominio>/errors/range-too-large",
                        },
                        "title": {
                            "type": "string",
                            "description": "Resumen corto del tipo de error (no varía entre instancias).",
                            "example": "Date range exceeds maximum for bucket size",
                        },
                        "status": {
                            "type": "integer",
                            "description": "Código HTTP (replicado del status line).",
                            "example": 400,
                        },
                        "detail": {
                            "type": "string",
                            "description": "Explicación específica de esta instancia del error.",
                            "example": "Requested 180.0 days with bucket=15min, max allowed is 7d",
                        },
                        "instance": {
                            "type": "string",
                            "description": "Path del request que generó el error.",
                            "example": "/v1/aggregates",
                        },
                    },
                    "additionalProperties": True,
                },
                "DirectionBreakdown": {
                    "type": "object",
                    "description": (
                        "Conteos discriminados por clasificación demográfica "
                        "derivada de la altura medida por estéreo del dispositivo. "
                        "`unknown` agrupa eventos sin medición confiable de altura."
                    ),
                    "required": ["adult", "child", "unknown", "total"],
                    "properties": {
                        "adult": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Personas clasificadas como adultos.",
                        },
                        "child": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Personas clasificadas como niños.",
                        },
                        "unknown": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Eventos sin clasificación demográfica confiable.",
                        },
                        "total": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Suma de adult + child + unknown.",
                        },
                    },
                },
                "AggregatesRow": {
                    "type": "object",
                    "description": (
                        "Una fila del response: agregados de un (sitio, bucket) "
                        "puntual. Si el bucket no tuvo datos, todos los "
                        "contadores son 0 y `pos.currency` defaultea a `ARS`."
                    ),
                    "required": [
                        "site_id",
                        "bucket_start",
                        "counts",
                        "external_traffic",
                        "pos",
                    ],
                    "properties": {
                        "site_id": {
                            "type": "string",
                            "description": "Identificador del sitio (sucursal).",
                            "example": "demo-01",
                        },
                        "bucket_start": {
                            "type": "string",
                            "format": "date-time",
                            "description": (
                                "Inicio del bucket en UTC, alineado al epoch. "
                                "Para `15min`: minutos en `00/15/30/45`. Para "
                                "`1h`: minuto `:00`. Para `1d`: medianoche UTC."
                            ),
                            "example": "2026-05-25T10:00:00Z",
                        },
                        "counts": {
                            "type": "object",
                            "description": "Conteos del detector de visión (entradas y salidas).",
                            "required": ["in", "out"],
                            "properties": {
                                "in": {
                                    "allOf": [
                                        {
                                            "$ref": "#/components/schemas/DirectionBreakdown"
                                        }
                                    ],
                                    "description": "Personas que entraron al local en el bucket.",
                                },
                                "out": {
                                    "allOf": [
                                        {
                                            "$ref": "#/components/schemas/DirectionBreakdown"
                                        }
                                    ],
                                    "description": "Personas que salieron del local en el bucket.",
                                },
                            },
                        },
                        "external_traffic": {
                            "type": "object",
                            "description": (
                                "Tráfico externo medido pasivamente por WiFi/BLE "
                                "probing en la vereda del local. Dispositivos "
                                "agrupados por stitching anti-MAC-rotation y "
                                "deduplicados cuando hay más de una cámara."
                            ),
                            "required": ["passersby", "shoppers"],
                            "properties": {
                                "passersby": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Dispositivos detectados cerca del local.",
                                },
                                "shoppers": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": (
                                        "Subconjunto de `passersby` con señal "
                                        "fuerte (proxy de cercanía a la entrada)."
                                    ),
                                },
                            },
                        },
                        "pos": {
                            "type": "object",
                            "description": (
                                "Transacciones POS ingestadas por el cliente "
                                "vía `POST /pos/transactions` para el bucket."
                            ),
                            "required": [
                                "sales",
                                "returns",
                                "transactions",
                                "items_sale",
                                "items_return",
                                "amount_minor_sale",
                                "amount_minor_return",
                                "currency",
                            ],
                            "properties": {
                                "sales": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Cantidad de ventas (`type='sale'`).",
                                },
                                "returns": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Cantidad de devoluciones (`type='return'`).",
                                },
                                "transactions": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Total de transacciones (sales + returns).",
                                },
                                "items_sale": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Suma de ítems vendidos en el bucket.",
                                },
                                "items_return": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Suma de ítems devueltos en el bucket.",
                                },
                                "amount_minor_sale": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": (
                                        "Monto vendido en la **unidad menor** de "
                                        "la moneda (centavos para ARS/USD, mils "
                                        "para BHD/KWD, sin división para JPY/KRW). "
                                        "Entero BIGINT — sin floats."
                                    ),
                                    "example": 142350,
                                },
                                "amount_minor_return": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Monto devuelto en unidad menor de la moneda.",
                                    "example": 12500,
                                },
                                "currency": {
                                    "type": "string",
                                    "description": "Código ISO 4217 (3 letras). `ARS` por defecto cuando el bucket no tuvo transacciones.",
                                    "example": "ARS",
                                },
                            },
                        },
                    },
                },
                "AggregatesResponse": {
                    "type": "object",
                    "description": (
                        "Página de resultados del endpoint de consulta. La "
                        "paginación se controla con el header `Link` (RFC 8288); "
                        "el body no contiene metadatos de paginación."
                    ),
                    "required": ["bucket", "data_freshness", "rows"],
                    "properties": {
                        "bucket": {
                            "type": "string",
                            "enum": ["15min", "1h", "1d"],
                            "description": "Eco del parámetro `bucket` del request.",
                        },
                        "data_freshness": {
                            "type": "object",
                            "description": (
                                "Timestamp del último evento ingestado por cada "
                                "sitio del response (máximo de `received_at` "
                                "entre las tres fuentes: counts, WiFi/BLE, POS). "
                                "Permite al cliente detectar dispositivos "
                                "offline sin pegar a otro endpoint."
                            ),
                            "additionalProperties": {
                                "type": "string",
                                "format": "date-time",
                                "description": "ISO 8601 UTC del último dato del sitio.",
                            },
                            "example": {"demo-01": "2026-05-25T13:42:18Z"},
                        },
                        "rows": {
                            "type": "array",
                            "description": "Filas ordenadas por (bucket_start asc, site_id asc).",
                            "items": {"$ref": "#/components/schemas/AggregatesRow"},
                        },
                    },
                },
                "PosTransaction": {
                    "type": "object",
                    "description": (
                        "Una transacción individual del POS del cliente. "
                        "Idempotente por `(store_id, transaction_id)` — el "
                        "reintento de una transacción ya ingestada no duplica "
                        "filas (se ignora silenciosamente vía `ON CONFLICT DO "
                        "NOTHING`). El monto va en la **unidad menor** de la "
                        "moneda (centavos para ARS/USD, mils para BHD), nunca "
                        "como float."
                    ),
                    "required": [
                        "transaction_id",
                        "store_id",
                        "event_ts",
                        "type",
                        "amount_minor",
                    ],
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": (
                                "ID único de factura o batch del POS. Debe ser "
                                "estable entre reintentos para que la "
                                "idempotencia funcione."
                            ),
                            "example": "POS-RECOLETA-20260518-001234",
                        },
                        "store_id": {
                            "type": "string",
                            "description": "Identificador del sitio donde ocurrió la transacción.",
                            "example": "demo-01",
                        },
                        "event_ts": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "format": "date-time",
                                    "description": "ISO 8601 con timezone explícita (`Z` u offset).",
                                },
                                {
                                    "type": "number",
                                    "description": "Epoch numérico. Valores > 10^12 se interpretan como milisegundos; el resto, como segundos.",
                                },
                            ],
                            "description": "Timestamp de la transacción. Datetimes naive (sin timezone) son rechazados.",
                            "example": "2026-05-18T14:32:15-03:00",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["sale", "return"],
                            "description": "Tipo de movimiento: venta o devolución.",
                        },
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 1,
                            "description": "Cantidad de ítems en la transacción. Default `1`.",
                        },
                        "amount_minor": {
                            "type": "integer",
                            "minimum": 0,
                            "description": (
                                "Monto en la unidad menor de la moneda. "
                                "**Entero**, sin decimales (centavos para "
                                "ARS/USD = $45.000 → `4500000`)."
                            ),
                            "example": 4500000,
                        },
                        "currency": {
                            "type": "string",
                            "description": "Código ISO 4217 (3 letras). Default `ARS`.",
                            "default": "ARS",
                            "example": "ARS",
                        },
                        "payment_method": {
                            "type": "string",
                            "nullable": True,
                            "description": (
                                "Método de pago. Opcional, puede ser `null`. "
                                "Usar `mixed` para batches con múltiples métodos."
                            ),
                            "example": "credit_card",
                        },
                    },
                },
                "PosIngestResponse": {
                    "type": "object",
                    "description": "Resumen de la operación de ingesta (single o bulk).",
                    "required": ["total", "inserted", "conflicted"],
                    "properties": {
                        "total": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Transacciones recibidas en el body del request.",
                        },
                        "inserted": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Transacciones nuevas (no existían) que se insertaron.",
                        },
                        "conflicted": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Transacciones que ya existían por su `(store_id, transaction_id)` y se ignoraron silenciosamente.",
                        },
                    },
                },
            },
        },
        "security": [{"sigv4": []}],
        "paths": {
            "/v1/aggregates": {
                "get": {
                    "summary": "Consultar agregados por bucket × sitio",
                    "description": (
                        "Devuelve una respuesta unificada con counts del "
                        "detector (entradas y salidas con breakdown "
                        "adult/child/unknown), tráfico externo medido por "
                        "WiFi/BLE y transacciones POS, todo agrupado por "
                        "bucket × sitio.\n\n"
                        "**Buckets vacíos**: si para un par `(sitio, bucket)` "
                        "no hubo datos, la fila se devuelve igual con todos los "
                        "contadores en 0.\n\n"
                        "**Paginación**: cursor opaco vía header `Link` "
                        '(RFC 8288). Iterar siguiendo el `rel="next"` hasta '
                        "que el header no lo incluya.\n\n"
                        "**Cacheo**: el response trae `ETag` siempre, y "
                        "`Cache-Control: public, max-age=86400, immutable` "
                        "cuando el rango es totalmente histórico (`to` está "
                        "más de 1 hora en el pasado). Los rangos que incluyen "
                        "datos siendo escritos devuelven `Cache-Control: "
                        "no-cache`. Usar `If-None-Match` con el ETag previo "
                        "para obtener `304 Not Modified` sin transferir body."
                    ),
                    "parameters": [
                        {
                            "name": "from",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "format": "date-time"},
                            "description": (
                                "Inicio del rango (inclusivo). ISO 8601 con "
                                "timezone explícita (`Z` para UTC u offset "
                                "como `-03:00`). Datetimes naive son rechazados."
                            ),
                            "example": "2026-05-24T00:00:00Z",
                        },
                        {
                            "name": "to",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "format": "date-time"},
                            "description": (
                                "Fin del rango (exclusivo). Debe ser estrictamente "
                                "mayor a `from`."
                            ),
                            "example": "2026-05-25T00:00:00Z",
                        },
                        {
                            "name": "sites",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": (
                                "Lista de IDs de sitios separados por coma "
                                "(regex `[a-zA-Z0-9_-]+`). Omitir = todos los "
                                "sitios del sistema."
                            ),
                            "example": "demo-01,demo-02",
                        },
                        {
                            "name": "bucket",
                            "in": "query",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["15min", "1h", "1d"],
                                "default": "15min",
                            },
                            "description": (
                                "Granularidad de agrupación. Caps de rango "
                                "por bucket (defensa contra full scans):\n"
                                "* `15min`: máximo 7 días\n"
                                "* `1h`: máximo 90 días\n"
                                "* `1d`: máximo 365 días"
                            ),
                        },
                        {
                            "name": "cursor",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": (
                                "Cursor opaco de paginación. Tomar del header "
                                '`Link; rel="next"` de la respuesta anterior '
                                "y reenviar sin modificar. No incluir en el "
                                "primer request."
                            ),
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "schema": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5000,
                                "default": 1000,
                            },
                            "description": "Máximo de filas por página. Rango: 1 a 5000.",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": (
                                "Página de agregados. El header `Link` "
                                'incluye `rel="first"` siempre y '
                                '`rel="next"` solo cuando hay más páginas.'
                            ),
                            "headers": {
                                "Link": {
                                    "schema": {"type": "string"},
                                    "description": "Links de paginación RFC 8288 (`rel=first`, `rel=next`).",
                                },
                                "ETag": {
                                    "schema": {"type": "string"},
                                    "description": "Hash fuerte del response. Usable en `If-None-Match`.",
                                },
                                "Cache-Control": {
                                    "schema": {"type": "string"},
                                    "description": (
                                        "`public, max-age=86400, immutable` para "
                                        "rangos cerrados; `no-cache` para rangos "
                                        "con datos siendo escritos."
                                    ),
                                },
                            },
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AggregatesResponse"
                                    }
                                }
                            },
                        },
                        "304": {
                            "description": (
                                "No modificado. Devuelto cuando el header "
                                "`If-None-Match` del request coincide con el "
                                "ETag actual del response (sin body)."
                            ),
                            "headers": {
                                "ETag": {"schema": {"type": "string"}},
                            },
                        },
                        "400": {
                            "description": (
                                "Request inválido. Causas posibles: parámetros "
                                "obligatorios faltantes, datetime mal formado "
                                "o naive, rango invertido, rango excede el cap "
                                "del bucket, bucket inválido, site_id con "
                                "caracteres no permitidos, cursor malformado, "
                                "`limit` fuera de rango."
                            ),
                            "content": {
                                "application/problem+json": {
                                    "schema": {"$ref": "#/components/schemas/Problem"}
                                }
                            },
                        },
                        "403": {
                            "description": (
                                "SigV4 inválida o el IAM principal no tiene "
                                "permiso `execute-api:Invoke` sobre este "
                                "endpoint."
                            )
                        },
                        "429": {
                            "description": (
                                "Throttled por API Gateway. El cliente "
                                "excedió 10 req/s steady o 20 req/s en burst. "
                                "Reintentar con backoff."
                            )
                        },
                        "500": {
                            "description": (
                                "Error transitorio del servidor (DB inaccesible, "
                                "query timeout, etc). Reintentar con backoff "
                                "exponencial."
                            ),
                            "content": {
                                "application/problem+json": {
                                    "schema": {"$ref": "#/components/schemas/Problem"}
                                }
                            },
                        },
                    },
                }
            },
            "/v1/openapi.json": {
                "get": {
                    "summary": "Obtener esta especificación OpenAPI",
                    "description": (
                        "Devuelve la especificación OpenAPI 3.1 completa de "
                        "este API. No requiere autenticación — los clientes "
                        "B2B la consumen para generar sus SDKs antes de "
                        "tener credenciales SigV4 configuradas."
                    ),
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "Especificación OpenAPI 3.1 en JSON.",
                            "content": {"application/json": {}},
                        }
                    },
                }
            },
            "/pos/transactions": {
                "post": {
                    "summary": "Ingestar transacciones POS (single o bulk)",
                    "description": (
                        "Endpoint usado por el POS del cliente para empujar "
                        "transacciones. El body acepta dos formas a "
                        "discreción del POS:\n\n"
                        "1. **Single**: un objeto `PosTransaction` con una "
                        "transacción.\n"
                        "2. **Bulk**: un array de objetos `PosTransaction` "
                        "con N transacciones (atómico — si una falla "
                        "validación, ninguna se inserta).\n\n"
                        "**Idempotencia**: cada transacción se identifica "
                        "por `(store_id, transaction_id)`. Reintentos con "
                        "los mismos IDs no duplican filas (se ignoran "
                        "silenciosamente, vuelven en `conflicted` del "
                        "response). Si el POS necesita corregir una "
                        "transacción ya ingestada, debe usar un "
                        "`transaction_id` diferente — esta versión del API "
                        "no soporta updates."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "oneOf": [
                                        {"$ref": "#/components/schemas/PosTransaction"},
                                        {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/PosTransaction"
                                            },
                                            "minItems": 1,
                                            "description": "Bulk: array de transacciones (mínimo 1).",
                                        },
                                    ]
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": (
                                "Operación procesada. `inserted + conflicted "
                                "== total`. `conflicted > 0` es esperable en "
                                "reintentos del POS y no es un error."
                            ),
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/PosIngestResponse"
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": (
                                "Payload inválido: JSON malformado, fields "
                                "obligatorios faltantes, `type` distinto de "
                                "`sale`/`return`, `amount_minor` negativo o no "
                                "entero, `event_ts` sin timezone. El POS NO "
                                "debe reintentar el request tal cual; debe "
                                "corregir el payload."
                            ),
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {
                                                "type": "string",
                                                "description": "Mensaje legible del problema de validación.",
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "403": {
                            "description": (
                                "SigV4 inválida o el IAM principal no tiene "
                                "permiso `execute-api:Invoke` sobre este "
                                "endpoint."
                            )
                        },
                        "429": {
                            "description": (
                                "Throttled por API Gateway (10 req/s steady, "
                                "20 burst). Reintentar con backoff."
                            )
                        },
                        "500": {
                            "description": (
                                "Error transitorio del servidor (DB "
                                "inaccesible). El POS DEBE reintentar el "
                                "request con backoff — la idempotencia por "
                                "`transaction_id` hace que el reintento sea "
                                "seguro."
                            )
                        },
                    },
                }
            },
        },
    }


# =============================================================================
# Handler
# =============================================================================


def _request_path(event: dict[str, Any]) -> str:
    return (
        event.get("rawPath")
        or event.get("requestContext", {}).get("http", {}).get("path")
        or "/"
    )


def _request_headers(event: dict[str, Any]) -> dict[str, str]:
    raw = event.get("headers") or {}
    return {k.lower(): v for k, v in raw.items()}


def _aggregates_handler(event: dict[str, Any], start_ts: datetime) -> dict[str, Any]:
    params = _parse_input(event)
    conn = _get_connection()

    resolved_sites = _query_sites_if_omitted(conn, params["sites"])
    if not resolved_sites:
        # No hay sites en el sistema (o el filtro descartó todos). Devolvemos
        # body vacío con shape válido en vez de problem details — no es error.
        body = {"bucket": params["bucket"], "data_freshness": {}, "rows": []}
        etag = _compute_etag(body)
        cache_headers = _build_cache_headers(
            params["to"], etag, datetime.now(tz=timezone.utc)
        )
        link = _build_link_header(params["raw_qs"], has_more=False, next_cursor=None)
        headers = {"link": link, **cache_headers} if link else cache_headers
        return _http_response(200, body, headers)

    raw_rows, has_more = _query_aggregates(conn, params, resolved_sites)
    freshness = _query_data_freshness(conn, resolved_sites)

    rows = [_shape_row(r) for r in raw_rows]
    body = {"bucket": params["bucket"], "data_freshness": freshness, "rows": rows}
    etag = _compute_etag(body)

    # If-None-Match short-circuit (RFC 7232) → 304 sin body.
    req_headers = _request_headers(event)
    inm = req_headers.get("if-none-match")
    if inm and inm.strip() == etag:
        return _http_response(304, None, {"etag": etag})

    next_cursor: str | None = None
    if has_more and raw_rows:
        last = raw_rows[-1]
        last_iso = (
            last["bucket_start"]
            .astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        next_cursor = _encode_cursor(last_iso, last["store_id"])

    link = _build_link_header(params["raw_qs"], has_more, next_cursor)
    cache_headers = _build_cache_headers(
        params["to"], etag, datetime.now(tz=timezone.utc)
    )
    out_headers = {**cache_headers}
    if link:
        out_headers["link"] = link

    latency_ms = (datetime.now(tz=timezone.utc) - start_ts).total_seconds() * 1000.0
    _emf_log(
        metrics=[
            ("LatencyMs", latency_ms, "Milliseconds"),
            ("RowsReturned", float(len(rows)), "Count"),
            ("HasMore", 1.0 if has_more else 0.0, "Count"),
        ],
        dimensions={"Endpoint": "aggregates", "Bucket": params["bucket"]},
        sites_count=len(resolved_sites),
        from_iso=params["from"].isoformat(),
        to_iso=params["to"].isoformat(),
        cursor_in=bool(params["cursor"]),
        cursor_out=bool(next_cursor),
    )

    return _http_response(200, body, out_headers)


def _openapi_handler() -> dict[str, Any]:
    return _http_response(200, _openapi_spec())


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    start_ts = datetime.now(tz=timezone.utc)
    path = _request_path(event)

    try:
        if path.endswith("/openapi.json"):
            return _openapi_handler()
        if path.endswith("/aggregates"):
            return _aggregates_handler(event, start_ts)
        return _problem_response(
            404,
            "not-found",
            "Not Found",
            f"path {path!r} is not handled by this Lambda",
            instance=path,
        )
    except _BadRequest as e:
        latency_ms = (datetime.now(tz=timezone.utc) - start_ts).total_seconds() * 1000.0
        _emf_log(
            metrics=[
                ("LatencyMs", latency_ms, "Milliseconds"),
                ("ValidationError", 1.0, "Count"),
            ],
            dimensions={"Endpoint": "aggregates", "ErrorSlug": e.problem_slug},
        )
        return _problem_response(
            400, e.problem_slug, e.title, e.detail, instance=path, **e.extra
        )
    except Exception as e:
        # Reset de la connection cacheada si murió el query.
        global _pg_conn
        if _pg_conn is not None:
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None
        latency_ms = (datetime.now(tz=timezone.utc) - start_ts).total_seconds() * 1000.0
        _emf_log(
            metrics=[
                ("LatencyMs", latency_ms, "Milliseconds"),
                ("InternalError", 1.0, "Count"),
            ],
            dimensions={"Endpoint": "aggregates"},
            error_type=type(e).__name__,
        )
        logger.exception("query_aggregates_unhandled")
        return _problem_response(
            500,
            "internal-error",
            "Internal Server Error",
            "Unexpected error processing the request; try again later",
            instance=path,
        )

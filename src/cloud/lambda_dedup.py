"""AWS Lambda: dedup inter-cámara (Capa 3).

Recibe resúmenes de hashes WiFi/BLE de múltiples cámaras del mismo local y
deduplica entre cámaras usando DynamoDB.

Schema de la tabla DynamoDB:
    Partition key: store_date (str) — "store-001#2026-03-30"
    Sort key: hash (str) — hex SHA-256 truncado
    ttl (number): epoch seconds al que DynamoDB auto-expira el item.

Cada invocación:
    1. Recibe un payload de regla de IoT Core con hashes de una cámara.
    2. Por cada hash, hace put condicional a DynamoDB (solo si no existe).
    3. Devuelve la cantidad de visitantes únicos genuinamente nuevos.

Variables de entorno:
    DEDUP_TABLE_NAME: Nombre de la tabla DynamoDB (default: "people-counter-dedup")
    DEDUP_TTL_DAYS: Días después de los cuales los hashes auto-expiran (default: 7).
"""

import logging
import os
import time
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import lazy de boto3 — disponible en el runtime de Lambda, no necesariamente local
_dynamodb_table = None


def _get_table():
    """Inicializa el resource de la tabla DynamoDB de manera lazy."""
    global _dynamodb_table
    if _dynamodb_table is None:
        import boto3

        table_name = os.environ.get("DEDUP_TABLE_NAME", "people-counter-dedup")
        dynamodb = boto3.resource("dynamodb")
        _dynamodb_table = dynamodb.Table(table_name)
    return _dynamodb_table


def deduplicate_hashes(
    store_id: str,
    date: str,
    hashes: list[str],
    source_device: str,
) -> dict[str, Any]:
    """Deduplica hashes contra la tabla DynamoDB a nivel de local.

    Args:
        store_id: Identificador del local (ej: "store-001").
        date: String de fecha (ej: "2026-03-30").
        hashes: Lista de hashes SHA-256 truncados en hex.
        source_device: Device ID que envió estos hashes.

    Returns:
        Dict con:
            new_count: Cantidad de visitantes únicos genuinamente nuevos.
            duplicate_count: Cantidad ya vista por otra cámara.
            total_unique: Total único para este store+date (aproximado).
    """
    table = _get_table()
    partition_key = f"{store_id}#{date}"
    ttl_days = int(os.environ.get("DEDUP_TTL_DAYS", "7"))
    now_epoch = int(time.time())
    ttl_epoch = now_epoch + ttl_days * 86400

    new_count = 0
    duplicate_count = 0

    for h in hashes:
        try:
            # Put condicional — solo tiene éxito si el item no existe
            table.put_item(
                Item={
                    "store_date": partition_key,
                    "hash": h,
                    "source_device": source_device,
                    "first_seen": now_epoch,
                    "ttl": ttl_epoch,
                },
                ConditionExpression="attribute_not_exists(#h)",
                ExpressionAttributeNames={"#h": "hash"},
            )
            new_count += 1
        except table.meta.client.exceptions.ConditionalCheckFailedException:
            duplicate_count += 1
        except Exception:
            logger.exception("DynamoDB put_item error for hash %s", h[:8])

    logger.info(
        "dedup_l3_complete",
        extra={
            "store_id": store_id,
            "date": date,
            "device_id": source_device,
            "new_count": new_count,
            "duplicate_count": duplicate_count,
        },
    )

    return {
        "new_count": new_count,
        "duplicate_count": duplicate_count,
    }


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handler de Lambda — invocado por la regla de IoT Core.

    Estructura esperada del event (de la regla SQL de IoT Core):
    {
        "device_id": "store-001-cam-01",
        "store_id": "store-001",
        "date": "2026-03-30",
        "type": "wifi_ble",
        "data": {
            "hashes": ["abc123...", "def456...", ...],
            "protocol": "wifi",
            "period_start": 1711800000,
            "period_end": 1711800900
        }
    }
    """
    try:
        device_id = event["device_id"]
        store_id = event.get("store_id", device_id.rsplit("-", 2)[0])
        date = event.get("date", time.strftime("%Y-%m-%d"))
        hashes = event.get("data", {}).get("hashes", [])

        if not hashes:
            return {"statusCode": 200, "body": {"new_count": 0, "duplicate_count": 0}}

        result = deduplicate_hashes(store_id, date, hashes, device_id)

        return {"statusCode": 200, "body": result}

    except Exception:
        logger.exception("Lambda handler error")
        return {"statusCode": 500, "body": {"error": "Internal error"}}

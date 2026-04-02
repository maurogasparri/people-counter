"""AWS Lambda: Inter-camera deduplication (Layer 3).

Receives WiFi/BLE hash summaries from multiple cameras in the same store
and deduplicates across cameras using DynamoDB.

DynamoDB table schema:
    Partition key: store_date (str) — "store-001#2026-03-30"
    Sort key: hash (str) — truncated SHA-256 hex

Each invocation:
    1. Receives an IoT Core rule payload with hashes from one camera.
    2. For each hash, conditionally puts to DynamoDB (only if not exists).
    3. Returns the count of genuinely new unique visitors.

Environment variables:
    DEDUP_TABLE_NAME: DynamoDB table name (default: "people-counter-dedup")
"""

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Lazy import boto3 — available in Lambda runtime, not necessarily locally
_dynamodb_table = None


def _get_table():
    """Lazily initialize DynamoDB table resource."""
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
    """Deduplicate hashes against store-level DynamoDB table.

    Args:
        store_id: Store identifier (e.g. "store-001").
        date: Date string (e.g. "2026-03-30").
        hashes: List of truncated SHA-256 hex hashes.
        source_device: Device ID that sent these hashes.

    Returns:
        Dict with:
            new_count: Number of genuinely new unique visitors.
            duplicate_count: Number already seen by another camera.
            total_unique: Total unique for this store+date (approximate).
    """
    table = _get_table()
    partition_key = f"{store_id}#{date}"

    new_count = 0
    duplicate_count = 0

    for h in hashes:
        try:
            # Conditional put — only succeeds if the item doesn't exist
            table.put_item(
                Item={
                    "store_date": partition_key,
                    "hash": h,
                    "source_device": source_device,
                    "first_seen": int(time.time()),
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
        "Dedup L3: store=%s date=%s device=%s new=%d dup=%d",
        store_id,
        date,
        source_device,
        new_count,
        duplicate_count,
    )

    return {
        "new_count": new_count,
        "duplicate_count": duplicate_count,
    }


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler — invoked by IoT Core rule.

    Expected event structure (from IoT Core SQL rule):
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

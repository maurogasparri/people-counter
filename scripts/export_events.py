"""Export counting events, telemetry and WiFi/BLE summaries to CSV.

Operators occasionally report that the device count does not match a manual
count. When that happens engineering needs the raw events for a given time
window so they can reconcile. This script queries AWS Timestream (and
optionally DynamoDB for WiFi/BLE dedup hashes) and dumps them as CSV.

Usage:
    python scripts/export_events.py \
        --store-id store-001 \
        --start 2026-04-18T14:00:00 \
        --end 2026-04-18T15:00:00 \
        [--device-id <id>] \
        [--output ./events.csv] \
        [--region us-east-1] \
        [--env prod] \
        [--include telemetry] \
        [--include wifi_ble] \
        [--preview] \
        [--verbose]

Notes on the schema (see ``infra/cloudformation/people-counter.yaml``):

- Timestream DB name is ``people-counter-${env}``.
- ``counting`` and ``telemetry`` are the two tables written by IoT rules.
- Dimensions on ``counting`` are ``device_id``, ``store_id``, ``direction``.
- Dimensions on ``telemetry`` are ``device_id``, ``store_id``.
- All other payload fields are written as separate ``measure_name`` rows by
  IoT Core's Timestream action (one row per field), so we pivot per
  (time, device_id) to reconstruct an event.

Assumptions (flagged in the report):

- ``track_id`` is published inside the counting payload by ``src/main.py`` but
  the IoT rule (``SELECT * FROM 'store/+/counting'``) writes *all* non-dimension
  payload fields as separate measures. This script picks up ``track_id`` as a
  measure — if the CloudFormation rule is later changed to promote it to a
  dimension the pivot still works (it just reads it from the dimension set).
- ``direction_labels`` in ``counter`` config controls the value of the
  ``direction`` dimension. Common values are ``ingress`` / ``egress`` but the
  summary aggregates whatever pair of labels appears, so custom labels still
  render correctly.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("export_events")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PivotedRow:
    """A single time/device observation with all measures merged in.

    Timestream stores one row per ``measure_name``. To produce a friendly CSV
    we bucket rows by ``(timestamp, device_id)`` and merge every measure into
    a single dict.
    """

    timestamp_utc: str
    store_id: str
    device_id: str
    dimensions: dict[str, str] = field(default_factory=dict)
    measures: dict[str, Any] = field(default_factory=dict)


# Column orders for the two CSV families. Kept small and documented so we can
# review what engineering sees in the spreadsheet.
COUNTING_COLUMNS = (
    "timestamp_utc",
    "store_id",
    "device_id",
    "direction",
    "track_id",
)

TELEMETRY_COLUMNS = (
    "timestamp_utc",
    "store_id",
    "device_id",
    "cpu_temp_c",
    "hailo_temp_c",
    "mem_available_mb",
    "disk_free_mb",
    "frame_latency_p50_ms",
    "frame_latency_p95_ms",
    "tracker_confirmed_count",
    "tracker_pending_count",
    "mqtt_disconnect_count",
    "buffer_backlog_messages",
)

# Mapping from Timestream measure names to CSV column names for telemetry.
# The device publishes slightly shorter names (no "_count" suffix) — keep both
# sides explicit so renames upstream are easy to spot.
TELEMETRY_MEASURE_MAP = {
    "cpu_temp_c": "cpu_temp_c",
    "hailo_temp_c": "hailo_temp_c",
    "mem_available_mb": "mem_available_mb",
    "disk_free_mb": "disk_free_mb",
    "frame_latency_p50_ms": "frame_latency_p50_ms",
    "frame_latency_p95_ms": "frame_latency_p95_ms",
    "tracker_confirmed": "tracker_confirmed_count",
    "tracker_pending": "tracker_pending_count",
    "mqtt_disconnect_count": "mqtt_disconnect_count",
    "buffer_backlog": "buffer_backlog_messages",
}

WIFI_BLE_COLUMNS = (
    "timestamp_utc",
    "store_id",
    "device_id",
    "protocol",
    "unique_count",
    "new_count",
    "duplicate_count",
    "period_start",
    "period_end",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export counting events (and optionally telemetry / WiFi+BLE "
            "summaries) from AWS Timestream for a given store and time "
            "window."
        )
    )
    parser.add_argument("--store-id", required=True, help="Store identifier, e.g. store-001")
    parser.add_argument(
        "--start",
        required=True,
        help="Window start, ISO format (e.g. 2026-04-18T14:00:00). Local tz assumed if no offset.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Window end, ISO format. Must be after --start.",
    )
    parser.add_argument("--device-id", help="Optional device filter")
    parser.add_argument(
        "--output",
        help=(
            "Base output path. Event-type suffix is added (e.g. _counting, "
            "_telemetry). Defaults to ./events_{store}_{start}_{end}.csv."
        ),
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default us-east-1, matches CloudFormation stack)",
    )
    parser.add_argument(
        "--env",
        default="prod",
        choices=("prod", "staging", "dev"),
        help="Resolves Timestream DB name as people-counter-<env>",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        choices=("telemetry", "wifi_ble"),
        help="Extra event types to include. 'counting' is always exported.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Do not write CSV files, just print the summary.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def parse_iso_datetime(value: str, field_name: str) -> datetime:
    """Parse an ISO datetime. Assumes local tz if no offset is given."""
    raw = value.strip()
    # Accept trailing 'Z' which fromisoformat refuses before 3.11 edge cases.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid {field_name}: {value!r}. "
            "Expected ISO format, e.g. 2026-04-18T14:00:00 or "
            "2026-04-18T14:00:00Z or 2026-04-18T14:00:00-03:00."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.astimezone()  # Attach local tz
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Timestream query layer
# ---------------------------------------------------------------------------


def _format_ts_literal(dt: datetime) -> str:
    """Timestream wants 'YYYY-MM-DD HH:MM:SS.fff' literals, UTC."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _build_where(
    store_id: str,
    device_id: str | None,
    start_utc: datetime,
    end_utc: datetime,
) -> str:
    """Produce a safe WHERE clause.

    store_id / device_id come from the CLI; they go through simple quoting.
    Timestream does not support parameterised queries in ``query`` the way
    RDBMS do, so we escape single quotes manually. Acceptable for an internal
    engineering tool — reject any value with control characters.
    """
    for label, value in (("store_id", store_id), ("device_id", device_id)):
        if value is None:
            continue
        if any(ch in value for ch in ("\n", "\r", "\0", ";")):
            raise SystemExit(f"Invalid {label}: {value!r}")

    def q(s: str) -> str:
        return s.replace("'", "''")

    clauses = [
        f"store_id = '{q(store_id)}'",
        f"time BETWEEN '{_format_ts_literal(start_utc)}' AND '{_format_ts_literal(end_utc)}'",
    ]
    if device_id:
        clauses.append(f"device_id = '{q(device_id)}'")
    return " AND ".join(clauses)


def _parse_datum(datum: dict[str, Any]) -> Any:
    """Flatten a Timestream Datum dict into a Python scalar/None."""
    if datum.get("NullValue"):
        return None
    if "ScalarValue" in datum:
        return datum["ScalarValue"]
    # Timestream also supports arrays/rows/timeseries; the counting + telemetry
    # tables are scalar-only so we don't expand those shapes.
    return None


def _rows_to_dicts(columns: list[dict[str, Any]], rows: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    col_names = [c["Name"] for c in columns]
    for row in rows:
        data = row.get("Data", [])
        yield {col_names[i]: _parse_datum(data[i]) for i in range(len(col_names))}


def query_timestream(
    client: Any,
    database: str,
    table: str,
    where: str,
) -> list[dict[str, Any]]:
    """Run a paginated Timestream query and return flattened records."""
    sql = (
        f'SELECT time, device_id, store_id, measure_name, '
        f'measure_value::varchar AS value_varchar, '
        f'measure_value::double AS value_double, '
        f'measure_value::bigint AS value_bigint '
        f'FROM "{database}"."{table}" '
        f"WHERE {where} ORDER BY time ASC"
    )
    logger.debug("Timestream query: %s", sql)

    flattened: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"QueryString": sql}
        if next_token:
            kwargs["NextToken"] = next_token
        resp = client.query(**kwargs)
        columns = resp.get("ColumnInfo", [])
        rows = resp.get("Rows", [])
        flattened.extend(_rows_to_dicts(columns, rows))
        next_token = resp.get("NextToken")
        if not next_token:
            break
    logger.debug("Fetched %d raw measure rows from %s", len(flattened), table)
    return flattened


def pivot_measures(
    raw_rows: Iterable[dict[str, Any]],
) -> list[PivotedRow]:
    """Collapse per-measure rows into per-(time, device) PivotedRow objects."""
    buckets: dict[tuple[str, str], PivotedRow] = {}
    for row in raw_rows:
        ts = row.get("time")
        device_id = row.get("device_id") or ""
        store_id = row.get("store_id") or ""
        measure_name = row.get("measure_name") or ""

        # Pick the most informative representation.
        value: Any = (
            row.get("value_varchar")
            if row.get("value_varchar") is not None
            else row.get("value_double")
            if row.get("value_double") is not None
            else row.get("value_bigint")
        )

        if ts is None:
            continue
        key = (ts, device_id)
        pivot = buckets.get(key)
        if pivot is None:
            pivot = PivotedRow(
                timestamp_utc=_normalise_ts(ts),
                store_id=store_id,
                device_id=device_id,
            )
            buckets[key] = pivot
        pivot.measures[measure_name] = value
    return list(buckets.values())


def _normalise_ts(ts: str) -> str:
    """Timestream returns 'YYYY-MM-DD HH:MM:SS.fffffffff'. Emit ISO UTC."""
    # Trim nanoseconds to microseconds for datetime parsing.
    main, _, frac = ts.partition(".")
    if frac:
        frac = frac[:6]
        normalised = f"{main}.{frac}"
    else:
        normalised = main
    try:
        dt = datetime.strptime(normalised, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(main, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Domain-specific collection
# ---------------------------------------------------------------------------


def fetch_counting(
    client: Any,
    database: str,
    where_for_counting: str,
    where_for_direction: str,
) -> list[dict[str, Any]]:
    """Fetch and flatten counting events.

    Counting rows have ``direction`` as a dimension, so we must query it
    separately (SELECT time, device_id, direction, measure_name, ...) and
    merge.
    """
    sql = (
        f'SELECT time, device_id, store_id, direction, measure_name, '
        f'measure_value::varchar AS value_varchar, '
        f'measure_value::double AS value_double, '
        f'measure_value::bigint AS value_bigint '
        f'FROM "{database}"."counting" '
        f"WHERE {where_for_counting} ORDER BY time ASC"
    )
    _ = where_for_direction  # kept for future expansion — same where today
    logger.debug("Timestream counting query: %s", sql)

    flattened: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"QueryString": sql}
        if next_token:
            kwargs["NextToken"] = next_token
        resp = client.query(**kwargs)
        columns = resp.get("ColumnInfo", [])
        rows = resp.get("Rows", [])
        flattened.extend(_rows_to_dicts(columns, rows))
        next_token = resp.get("NextToken")
        if not next_token:
            break

    # Pivot: one row per (time, device_id, direction).
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in flattened:
        ts = row.get("time")
        device_id = row.get("device_id") or ""
        store_id = row.get("store_id") or ""
        direction = row.get("direction") or ""
        if ts is None:
            continue
        key = (ts, device_id, direction)
        bucket = buckets.setdefault(
            key,
            {
                "timestamp_utc": _normalise_ts(ts),
                "store_id": store_id,
                "device_id": device_id,
                "direction": direction,
                "track_id": None,
            },
        )
        measure_name = row.get("measure_name") or ""
        value: Any = (
            row.get("value_varchar")
            if row.get("value_varchar") is not None
            else row.get("value_double")
            if row.get("value_double") is not None
            else row.get("value_bigint")
        )
        if measure_name == "track_id":
            bucket["track_id"] = value

    return sorted(buckets.values(), key=lambda r: r["timestamp_utc"])


def shape_telemetry(pivots: list[PivotedRow]) -> list[dict[str, Any]]:
    """Project telemetry PivotedRows onto the documented CSV columns."""
    out: list[dict[str, Any]] = []
    for p in pivots:
        row: dict[str, Any] = {
            "timestamp_utc": p.timestamp_utc,
            "store_id": p.store_id,
            "device_id": p.device_id,
        }
        for measure_name, col in TELEMETRY_MEASURE_MAP.items():
            row[col] = p.measures.get(measure_name)
        out.append(row)
    return sorted(out, key=lambda r: r["timestamp_utc"])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows -> %s", len(rows), path)


def default_output_base(store_id: str, start_utc: datetime, end_utc: datetime) -> Path:
    start_str = start_utc.strftime("%Y%m%d")
    end_str = end_utc.strftime("%Y%m%d")
    if start_str == end_str:
        stem = f"events_{store_id}_{start_str}.csv"
    else:
        stem = f"events_{store_id}_{start_str}_{end_str}.csv"
    return Path.cwd() / stem


def resolve_paths(base: Path, include_telemetry: bool, include_wifi_ble: bool) -> dict[str, Path]:
    """Produce one CSV path per event type, sharing the base stem."""
    # Drop the suffix and append a per-type tag.
    stem = base.stem
    suffix = base.suffix or ".csv"
    parent = base.parent
    paths = {"counting": parent / f"{stem}{suffix}"}
    if include_telemetry:
        # Remap 'events_' prefix to 'telemetry_' when present for readability.
        tel_stem = stem.replace("events_", "telemetry_", 1) if stem.startswith("events_") else f"{stem}_telemetry"
        paths["telemetry"] = parent / f"{tel_stem}{suffix}"
    if include_wifi_ble:
        wb_stem = stem.replace("events_", "wifi_ble_", 1) if stem.startswith("events_") else f"{stem}_wifi_ble"
        paths["wifi_ble"] = parent / f"{wb_stem}{suffix}"
    return paths


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    counting_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]] | None,
    wifi_ble_rows: list[dict[str, Any]] | None,
    output_paths: dict[str, Path] | None,
) -> None:
    direction_counter: Counter[str] = Counter(r["direction"] for r in counting_rows if r.get("direction"))
    total = sum(direction_counter.values())

    print(f"Counting events: {total}")
    for label, count in sorted(direction_counter.items()):
        print(f"  - {label.capitalize()}: {count}")

    if telemetry_rows is not None:
        print(f"Telemetry samples: {len(telemetry_rows)}")
    if wifi_ble_rows is not None:
        print(f"WiFi/BLE summaries: {len(wifi_ble_rows)}")

    if output_paths:
        print("Output:")
        for kind, path in output_paths.items():
            n = {
                "counting": len(counting_rows),
                "telemetry": len(telemetry_rows or []),
                "wifi_ble": len(wifi_ble_rows or []),
            }.get(kind, 0)
            print(f"  {path} ({n} rows)")

    # Top 3 hours by the "entry"-like label. Pick the label that looks like an
    # entrance: 'ingress' > 'in' > first alphabetically.
    entry_label = None
    for preferred in ("ingress", "in", "entry", "entered"):
        if preferred in direction_counter:
            entry_label = preferred
            break
    if entry_label is None and direction_counter:
        entry_label = sorted(direction_counter)[0]

    if entry_label:
        per_hour: dict[str, int] = defaultdict(int)
        for row in counting_rows:
            if row.get("direction") != entry_label:
                continue
            ts = row["timestamp_utc"]
            # ts looks like 2026-04-18T14:23:45.123000+00:00
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            hour_key = dt.strftime("%H:00")
            per_hour[hour_key] += 1
        top = sorted(per_hour.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if top:
            print(f"\nTop 3 hours by {entry_label}:")
            for hour, n in top:
                # Display hour-(hour+1) range.
                h = int(hour.split(":")[0])
                print(f"  {hour}-{(h + 1) % 24:02d}:00: {n}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _friendly_boto_error(exc: Exception, env: str, database: str) -> str:
    """Translate boto3 exceptions into a one-liner."""
    name = type(exc).__name__
    message = str(exc)
    if "NoCredentialsError" in name or "Unable to locate credentials" in message:
        return (
            "AWS credentials not found. Run `aws configure` or export "
            "AWS_PROFILE=... / AWS_ACCESS_KEY_ID etc."
        )
    if "ExpiredToken" in message or "InvalidClientTokenId" in message:
        return "AWS credentials expired or invalid. Refresh your session (aws sso login)."
    if "ResourceNotFoundException" in name or "not found" in message.lower():
        return (
            f"Timestream database/table not found for env={env} "
            f"(looked up '{database}'). Check --env matches your deployment."
        )
    if "AccessDeniedException" in name or "not authorized" in message:
        return f"AWS denied access: {message}. Your IAM principal needs timestream:Select."
    return f"AWS error ({name}): {message}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    start_utc = parse_iso_datetime(args.start, "--start")
    end_utc = parse_iso_datetime(args.end, "--end")
    if end_utc <= start_utc:
        print("Error: --end must be strictly after --start.", file=sys.stderr)
        return 2

    include_telemetry = "telemetry" in args.include
    include_wifi_ble = "wifi_ble" in args.include

    database = f"people-counter-{args.env}"

    try:
        import boto3  # type: ignore
        from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
    except ImportError:
        print(
            "boto3 is required. Install with `pip install boto3` or "
            "activate the project venv.",
            file=sys.stderr,
        )
        return 3

    try:
        ts_client = boto3.client("timestream-query", region_name=args.region)
    except Exception as exc:  # pragma: no cover - boto configuration errors
        print(
            f"Failed to create Timestream client: {_friendly_boto_error(exc, args.env, database)}",
            file=sys.stderr,
        )
        if args.verbose:
            raise
        return 4

    where = _build_where(args.store_id, args.device_id, start_utc, end_utc)

    try:
        counting_rows = fetch_counting(ts_client, database, where, where)

        telemetry_rows: list[dict[str, Any]] | None = None
        if include_telemetry:
            telemetry_raw = query_timestream(ts_client, database, "telemetry", where)
            telemetry_rows = shape_telemetry(pivot_measures(telemetry_raw))

        wifi_ble_rows: list[dict[str, Any]] | None = None
        if include_wifi_ble:
            # WiFi/BLE is routed to Lambda/DynamoDB, not Timestream. Without
            # per-event storage the best we can offer is a note — this slot is
            # reserved for a future DynamoDB scan if the Lambda stores
            # per-summary aggregates.
            logger.warning(
                "WiFi/BLE payloads are routed to Lambda+DynamoDB and are not "
                "currently persisted as per-event aggregates. Returning an "
                "empty list; extend DedupTable schema to expose summaries."
            )
            wifi_ble_rows = []

    except (ClientError, BotoCoreError) as exc:
        print(_friendly_boto_error(exc, args.env, database), file=sys.stderr)
        if args.verbose:
            raise
        return 5
    except Exception as exc:  # pragma: no cover - unexpected failures
        print(f"Unexpected error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    total_rows = len(counting_rows) + len(telemetry_rows or []) + len(wifi_ble_rows or [])
    if total_rows == 0:
        msg = f"No events found in given window for store_id={args.store_id}"
        if args.device_id:
            msg += f" device_id={args.device_id}"
        msg += f" window={start_utc.isoformat()}..{end_utc.isoformat()}"
        print(msg)
        return 0

    base = Path(args.output) if args.output else default_output_base(args.store_id, start_utc, end_utc)
    paths = resolve_paths(base, include_telemetry, include_wifi_ble)

    if args.preview:
        print_summary(counting_rows, telemetry_rows, wifi_ble_rows, None)
        print("\n(preview mode: no files written)")
        return 0

    written: dict[str, Path] = {}
    if counting_rows:
        write_csv(paths["counting"], COUNTING_COLUMNS, counting_rows)
        written["counting"] = paths["counting"]
    if include_telemetry and telemetry_rows:
        write_csv(paths["telemetry"], TELEMETRY_COLUMNS, telemetry_rows)
        written["telemetry"] = paths["telemetry"]
    if include_wifi_ble and wifi_ble_rows:
        write_csv(paths["wifi_ble"], WIFI_BLE_COLUMNS, wifi_ble_rows)
        written["wifi_ble"] = paths["wifi_ble"]

    print_summary(counting_rows, telemetry_rows, wifi_ble_rows, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())

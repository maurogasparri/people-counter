"""Aplica la migración wifi_ble_events (PR 2) a RDS.

Replica el patrón de ``apply_bootstrap.py``: psycopg + boto3, sin docker,
una sola llamada execute() para el bloque DDL completo.

Uso:
    py infra/sql/apply_wifi_ble_events.py [stack_name] [region]
Defaults: stack ``people-counter`` región ``us-east-1``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.provision import _rds_connect  # noqa: E402

MIGRATION_SQL = REPO / "infra" / "sql" / "migrations" / "2026-05-26-wifi-ble-events.sql"


def main() -> None:
    stack = sys.argv[1] if len(sys.argv) > 1 else "people-counter"
    region = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"

    sql = MIGRATION_SQL.read_text(encoding="utf-8")
    conn = _rds_connect(stack, region)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        print(f"  Applied {MIGRATION_SQL.name} ({len(sql)} bytes)")
        # Verificación rápida
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM wifi_ble_events")
            n = cur.fetchone()[0]
            print(f"  wifi_ble_events rows: {n}")
            cur.execute("SELECT rssi_class(-50), rssi_class(-65), rssi_class(-90), rssi_class(NULL)")
            r = cur.fetchone()
            print(f"  rssi_class probes: -50->{r[0]} -65->{r[1]} -90->{r[2]} NULL->{r[3]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

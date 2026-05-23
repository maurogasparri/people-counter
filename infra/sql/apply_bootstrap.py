"""Bootstrap del schema RDS sin docker — corre ``infra/sql/bootstrap.sql``
contra la instancia RDS via psycopg + boto3, igual que el resto del path de
provisioning (``scripts/provision.py``, ``debug/apply_telemetry_migration.py``).

Reemplaza el bloque ``docker run postgres:16 psql ...`` de ``deploy.ps1``
phase 2 — la única dependencia local pasa a ser ``boto3 + psycopg[binary]``,
cross-platform y ya instalado en cualquier workstation que provisione un
device (``pip install '.[provisioning]'``).

⚠ **NO es idempotente para data**: ``bootstrap.sql`` hace ``DROP TABLE IF
EXISTS`` de las tablas de hechos (``count_events``, ``wifi_ble_summary``,
``telemetry``, ``pos_transactions``). Re-correr este script BORRA esa data.
Las tablas de dimensiones (``sites``, ``devices``) usan ``CREATE TABLE IF NOT
EXISTS`` y se preservan. Para migraciones aditivas sobre prod, usar archivos
en ``infra/sql/migrations/`` (estilo ``ADD COLUMN IF NOT EXISTS``), no
re-correr este bootstrap.

Uso:
    py infra/sql/apply_bootstrap.py [stack_name] [region]
Defaults: stack ``people-counter`` región ``us-east-1``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.provision import _rds_connect  # noqa: E402

BOOTSTRAP_SQL = REPO / "infra" / "sql" / "bootstrap.sql"


def _ensure_grafana_db(stack: str, region: str) -> None:
    """Crea la DB ``grafana`` (storage de dashboards/usuarios de la app
    Grafana) si no existe. ``CREATE DATABASE`` NO puede ir dentro de una
    transacción → autocommit + conexión separada del bootstrap principal."""
    conn = _rds_connect(stack, region)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'grafana'")
            if cur.fetchone() is not None:
                print("  grafana database already exists")
                return
            cur.execute("CREATE DATABASE grafana OWNER people_counter")
            print("  grafana database created")
    finally:
        conn.close()


def _apply_bootstrap_sql(stack: str, region: str) -> None:
    """Ejecuta el contenido de ``bootstrap.sql`` contra la DB
    ``people_counter`` en una sola llamada ``execute()``. psycopg3 acepta
    multi-statement; los DDL no necesitan envolverse en transacción explícita
    porque corremos con ``autocommit=True`` (algunos statements como ``CREATE
    INDEX CONCURRENTLY`` o ``GRANT`` no pueden estar dentro de tx)."""
    sql = BOOTSTRAP_SQL.read_text(encoding="utf-8")
    conn = _rds_connect(stack, region)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        print(f"  Applied {BOOTSTRAP_SQL.name} ({len(sql)} bytes)")
    finally:
        conn.close()


def _summary(stack: str, region: str) -> None:
    """Resumen post-bootstrap: tablas, roles, databases creados."""
    conn = _rds_connect(stack, region)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema='public' ORDER BY table_name"""
            )
            tables = [r[0] for r in cur.fetchall()]
            cur.execute(
                """SELECT rolname FROM pg_roles
                   WHERE rolname IN ('people_counter','lambda_writer','lambda_pos_writer')
                   ORDER BY rolname"""
            )
            roles = [r[0] for r in cur.fetchall()]
            cur.execute(
                """SELECT datname FROM pg_database
                   WHERE datname IN ('people_counter','grafana')
                   ORDER BY datname"""
            )
            dbs = [r[0] for r in cur.fetchall()]
        print(f"  tables:    {', '.join(tables) or '(none)'}")
        print(f"  roles:     {', '.join(roles) or '(none)'}")
        print(f"  databases: {', '.join(dbs) or '(none)'}")
    finally:
        conn.close()


def main() -> None:
    stack = sys.argv[1] if len(sys.argv) > 1 else "people-counter"
    region = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"
    print(f"Bootstrap via psycopg -> stack={stack!r} region={region!r}")
    _ensure_grafana_db(stack, region)
    _apply_bootstrap_sql(stack, region)
    _summary(stack, region)
    print("OK")


if __name__ == "__main__":
    main()

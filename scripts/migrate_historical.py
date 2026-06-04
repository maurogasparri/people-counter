"""Migración de histórico AGREGADO (resúmenes por hora) → tablas base rollup_*.

Para cuando el sistema anterior exporta data YA AGREGADA (sin eventos crudos).
Carga un CSV a la tabla staging POR LOTES (commits incrementales → no infla la
transacción ni satura RDS) y luego corre el transform SQL que la mapea a los
rollups (infra/sql/migrate_historical_rollups.example.sql).

Si tu histórico viene a nivel EVENTO, NO uses esto: insertá en las tablas crudas
y dejá que refresh_rollups() derive los rollups.

CSV esperado: header con un subconjunto de estas columnas (las que tengas).
  Obligatorias: store_id, local_hour
  Counting:     ins, outs, ins_adult, ins_child
  POS:          sales, returns, transactions, items_sale, items_return,
                amount_minor_sale, amount_minor_return, currency
  WiFi/BLE:     passersby, shoppers, visitors

  ⚠️ local_hour = hora de pared LOCAL, grano horario (ej. '2025-11-03 18:00:00').
     El transform la guarda como local-as-UTC (ver el .sql). NO la pases en UTC.

Uso:
  py -3 -m scripts.migrate_historical --csv export_historico.csv
  py -3 -m scripts.migrate_historical --csv x.csv --truncate-staging --batch-size 5000
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_historical")

# Columnas de la staging (orden canónico). Mantener en sync con el .sql.
STAGING_COLS = [
    "store_id", "local_hour",
    "ins", "outs", "ins_adult", "ins_child",
    "sales", "returns", "transactions", "items_sale", "items_return",
    "amount_minor_sale", "amount_minor_return", "currency",
    "passersby", "shoppers", "visitors",
]

STAGING_DDL = """
CREATE TABLE IF NOT EXISTS stg_historical_hourly (
    store_id    TEXT      NOT NULL,
    local_hour  TIMESTAMP NOT NULL,
    ins BIGINT, outs BIGINT, ins_adult BIGINT, ins_child BIGINT,
    sales BIGINT, returns BIGINT, transactions BIGINT, items_sale BIGINT, items_return BIGINT,
    amount_minor_sale NUMERIC, amount_minor_return NUMERIC, currency CHAR(3),
    passersby BIGINT, shoppers BIGINT, visitors BIGINT,
    PRIMARY KEY (store_id, local_hour)
);
"""

SQL_TRANSFORM = Path(__file__).resolve().parent.parent / "infra" / "sql" / "migrate_historical_rollups.example.sql"


def _norm(v: str):
    """Celda CSV → valor o None (vacío = NULL)."""
    v = (v or "").strip()
    return v if v != "" else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="CSV agregado (header con columnas del staging).")
    ap.add_argument("--stack-name", default=os.environ.get("PC_STACK", "people-counter-dev"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--batch-size", type=int, default=5000, help="Filas por commit (default 5000).")
    ap.add_argument("--truncate-staging", action="store_true", help="Vaciar la staging antes de cargar.")
    ap.add_argument("--keep-staging", action="store_true", help="No dropear la staging al terminar.")
    ap.add_argument("--sql-file", default=str(SQL_TRANSFORM), help="Transform SQL a aplicar.")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    if not rows:
        logger.error("CSV vacío."); return
    present = [c for c in STAGING_COLS if c in rows[0]]
    for req in ("store_id", "local_hour"):
        if req not in present:
            logger.error("Falta la columna obligatoria %r en el CSV.", req); return
    logger.info("CSV: %d filas, columnas mapeadas: %s", len(rows), present)

    from scripts.provision import _rds_connect  # noqa: E402
    conn = _rds_connect(args.stack_name, args.region)
    try:
        with conn.cursor() as cur:
            cur.execute(STAGING_DDL)
            if args.truncate_staging:
                cur.execute("TRUNCATE stg_historical_hourly")
            conn.commit()

            # Carga POR LOTES con commit incremental (footprint chico → no OOM).
            cols_sql = ", ".join(present)
            ph = "(" + ", ".join(["%s"] * len(present)) + ")"
            insert = (f"INSERT INTO stg_historical_hourly ({cols_sql}) VALUES {ph} "
                      f"ON CONFLICT (store_id, local_hour) DO UPDATE SET "
                      + ", ".join(f"{c}=EXCLUDED.{c}" for c in present if c not in ("store_id", "local_hour")))
            loaded = 0
            for i in range(0, len(rows), args.batch_size):
                chunk = rows[i:i + args.batch_size]
                for r in chunk:
                    cur.execute(insert, [_norm(r[c]) for c in present])
                conn.commit()
                loaded += len(chunk)
                logger.info("  staging: %d/%d", loaded, len(rows))

            # Transform staging → rollups (idempotente, ON CONFLICT). El CREATE
            # IF NOT EXISTS del .sql es no-op acá.
            logger.info("Aplicando transform %s …", args.sql_file)
            cur.execute(Path(args.sql_file).read_text(encoding="utf-8"))
            conn.commit()

            cur.execute("SELECT count(*) FROM rollup_counting_hour")
            logger.info("rollup_counting_hour total ahora: %d filas", cur.fetchone()[0])
            if not args.keep_staging:
                cur.execute("DROP TABLE stg_historical_hourly")
                conn.commit()
                logger.info("staging dropeada.")
        logger.info("Migración OK. ⚠️ NO resetees rollup_state (el refresh del vivo no toca el histórico).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TC-14 (privacidad RDS) + TC-11 (POS idempotencia + conversión) en RDS real.
Reproducible: py docs/validacion/query_tc.py
"""
import json
import re
import boto3
import psycopg

EP = "people-counter-dev.cabqikscg4gm.us-east-1.rds.amazonaws.com"
SECRET = "people-counter/dev/rds-master"
DEV = "store-pilot-01-cam-01"
MAC_RE = re.compile(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", re.I)


def main():
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    s = json.loads(sm.get_secret_value(SecretId=SECRET)["SecretString"])
    cx = psycopg.connect(host=EP, port=5432, dbname="people_counter",
                         user=s["username"], password=s["password"],
                         sslmode="require", connect_timeout=15)
    cur = cx.cursor()

    print("=== TC-14 — Privacidad en RDS ===")
    # 1. visitor_hash debe ser hash opaco (hex, sin formato MAC).
    cur.execute("SELECT visitor_hash FROM wifi_ble_events ORDER BY received_at "
                "DESC LIMIT 500")
    sample = [r[0] for r in cur.fetchall()]
    mac_like = [h for h in sample if MAC_RE.search(str(h))]
    print(f"  wifi_ble_events.visitor_hash: {len(sample)} muestras, "
          f"con formato MAC crudo: {len(mac_like)} (debe ser 0)")
    if sample:
        print(f"    ej: {sample[0]!r} (hash opaco)")
    # 2. ninguna columna de imagen/blob/PII en el schema público.
    cur.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns"
        " WHERE table_schema='public' AND (data_type='bytea' OR column_name ~* "
        "'image|photo|face|frame|(^|_)mac($|_)|raw_|email|phone')"
    )
    pii_cols = cur.fetchall()
    print(f"  columnas imagen/PII/bytea en schema: {len(pii_cols)} {pii_cols if pii_cols else '(ninguna)'}")
    tc14 = len(mac_like) == 0 and len(pii_cols) == 0
    print(f"  VEREDICTO TC-14 (RDS): {'CUMPLE' if tc14 else 'REVISAR'}")

    print("\n=== TC-11 — POS idempotencia + conversión ===")
    cur.execute("SELECT count(*) FROM pos_transactions")
    n_pos = cur.fetchone()[0]
    print(f"  pos_transactions total: {n_pos}")
    if n_pos > 0:
        cur.execute("SELECT store_id, transaction_id, event_ts, type, items, "
                    "amount_minor, currency FROM pos_transactions LIMIT 1")
        ex = cur.fetchone()
        before = n_pos
        cur.execute("""INSERT INTO pos_transactions
            (store_id,transaction_id,event_ts,type,items,amount_minor,currency)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (store_id, transaction_id) DO NOTHING""", ex)
        inserted = cur.rowcount
        cx.commit()
        cur.execute("SELECT count(*) FROM pos_transactions")
        after = cur.fetchone()[0]
        idem = inserted == 0 and before == after
        print(f"  re-INSERT de transaction_id existente -> filas={inserted}, "
              f"count {before}->{after} | idempotente: {idem}")
    else:
        idem = None
        print("  sin pos_transactions para probar idempotencia (no ejecutado)")
    # conversión = transacciones de venta / visitantes (count in) por store/día
    cur.execute("""SELECT p.store_id, p.bucket_day,
                     count(*) FILTER (WHERE p.type='sale') sales
                   FROM pos_transactions p GROUP BY 1,2
                   ORDER BY 2 DESC LIMIT 3""")
    conv = cur.fetchall()
    print(f"  muestra ventas/día (para conversion_rate): {conv if conv else 'sin datos'}")
    cx.close()


if __name__ == "__main__":
    main()

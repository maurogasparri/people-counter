#!/usr/bin/env python3
"""TC-15 (live device only) — aisla el device del piloto de los datos demo/seed.

Lista device_ids con actividad reciente y mide e2e latency SOLO del device
real, ultimas 48h (sin replay de cortes). Reproducible:
    py docs/validacion/query_e2e_live.py
"""
import csv
import json

import boto3
import psycopg

ENDPOINT = "people-counter-dev.cabqikscg4gm.us-east-1.rds.amazonaws.com"
SECRET = "people-counter/dev/rds-master"
REGION = "us-east-1"
OUT_CSV = "docs/validacion/e2e_latency_live.csv"
TABLES = [("count_events", "event_ts"), ("telemetry", "event_ts"),
          ("wifi_ble_events", "period_end")]


def main():
    sm = boto3.client("secretsmanager", region_name=REGION)
    s = json.loads(sm.get_secret_value(SecretId=SECRET)["SecretString"])
    conn = psycopg.connect(host=ENDPOINT, port=5432, dbname="people_counter",
                           user=s["username"], password=s["password"],
                           sslmode="require", connect_timeout=15)
    with conn.cursor() as cur:
        print("=== device_ids con actividad ultimas 48h (telemetry) ===")
        cur.execute("""SELECT device_id, count(*), max(received_at)
                       FROM telemetry WHERE received_at >= now()-INTERVAL '48 h'
                       GROUP BY device_id ORDER BY 2 DESC""")
        live = None
        for did, c, mx in cur.fetchall():
            print(f"  {did:<28} n={c:<6} last={mx}")
            if live is None and not str(did).startswith("demo"):
                live = did
        print(f"\nDevice real elegido: {live}\n")
        print(f"{'table':<18} {'n':>6} {'p50_s':>8} {'p95_s':>8} "
              f"{'min_s':>8} {'max_s':>8} {'n_neg':>6}")
        rows = []
        for table, col in TABLES:
            cur.execute(f"""
              WITH lags AS (
                SELECT EXTRACT(EPOCH FROM (received_at-{col})) lag
                FROM {table}
                WHERE device_id=%s AND received_at>=now()-INTERVAL '48 h')
              SELECT count(*) FILTER (WHERE lag>=0 AND lag<120),
                     percentile_cont(0.5) WITHIN GROUP (ORDER BY lag)
                       FILTER (WHERE lag>=0 AND lag<120),
                     percentile_cont(0.95) WITHIN GROUP (ORDER BY lag)
                       FILTER (WHERE lag>=0 AND lag<120),
                     min(lag) FILTER (WHERE lag>=0 AND lag<120),
                     max(lag) FILTER (WHERE lag>=0 AND lag<120),
                     count(*) FILTER (WHERE lag<0),
                     min(lag)
              FROM lags""", (live,))
            n, p50, p95, mn, mx, nneg, gmin = cur.fetchone()

            def f(x):
                return f"{x:.3f}" if x is not None else "n/a"
            print(f"{table:<18} {n or 0:>6} {f(p50):>8} {f(p95):>8} "
                  f"{f(mn):>8} {f(mx):>8} {nneg or 0:>6}")
            rows.append({"table": table, "device_id": live, "n": n or 0,
                         "p50_s": p50, "p95_s": p95, "min_s": mn, "max_s": mx,
                         "n_negative": nneg or 0, "global_min_lag_s": gmin})
    conn.close()
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()

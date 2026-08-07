#!/usr/bin/env python3
"""TC-15 — Latencia extremo a extremo (device -> IoT Core -> Lambda -> RDS).

Mide ``received_at - event_ts`` (segundos) por tabla. p50/p95 sobre la ventana
con conexion estable (lag < CAP s); cuenta aparte los eventos de replay
(lag >= CAP) para no contaminar los percentiles. Verifica sincronizacion de
reloj (lag minimo > 0 => el device no va adelantado del server).

Reproducible:
    py docs/validacion/query_e2e.py
Requiere boto3 (lee secret people-counter/dev/rds-master) + psycopg.
"""
import csv
import json
import sys

import boto3
import psycopg

ENDPOINT = "people-counter-dev.cabqikscg4gm.us-east-1.rds.amazonaws.com"
PORT = 5432
DBNAME = "people_counter"
SECRET = "people-counter/dev/rds-master"
REGION = "us-east-1"
CAP_S = 120.0          # umbral conexion-estable vs replay
WINDOW = "30 days"     # ventana de busqueda
OUT_CSV = "docs/validacion/e2e_latency_samples.csv"

TABLES = [
    ("count_events", "event_ts"),
    ("telemetry", "event_ts"),
    ("wifi_ble_events", "period_end"),
]


def get_creds():
    sm = boto3.client("secretsmanager", region_name=REGION)
    s = json.loads(sm.get_secret_value(SecretId=SECRET)["SecretString"])
    return s["username"], s["password"]


def main():
    user, pwd = get_creds()
    conn = psycopg.connect(
        host=ENDPOINT, port=PORT, dbname=DBNAME, user=user, password=pwd,
        sslmode="require", connect_timeout=15,
    )
    rows_out = []
    print(f"{'table':<18} {'n_stable':>8} {'p50_s':>8} {'p95_s':>8} "
          f"{'min_s':>8} {'max_s':>9} {'n_replay':>8} {'replay_max_s':>12}")
    with conn.cursor() as cur:
        for table, col in TABLES:
            q = f"""
            WITH lags AS (
              SELECT EXTRACT(EPOCH FROM (received_at - {col})) AS lag
              FROM {table}
              WHERE received_at >= now() - INTERVAL '{WINDOW}'
            )
            SELECT
              (SELECT count(*) FROM lags WHERE lag >= 0 AND lag < {CAP_S}) AS n_stable,
              (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY lag)
                 FROM lags WHERE lag >= 0 AND lag < {CAP_S}) AS p50,
              (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY lag)
                 FROM lags WHERE lag >= 0 AND lag < {CAP_S}) AS p95,
              (SELECT min(lag) FROM lags WHERE lag >= 0 AND lag < {CAP_S}) AS minl,
              (SELECT max(lag) FROM lags WHERE lag >= 0 AND lag < {CAP_S}) AS maxl,
              (SELECT count(*) FROM lags WHERE lag >= {CAP_S}) AS n_replay,
              (SELECT max(lag) FROM lags WHERE lag >= {CAP_S}) AS replay_max,
              (SELECT count(*) FROM lags WHERE lag < 0) AS n_negative,
              (SELECT min(lag) FROM lags) AS global_min
            """
            cur.execute(q)
            (n_stable, p50, p95, minl, maxl, n_replay, replay_max,
             n_neg, gmin) = cur.fetchone()

            def f(x):
                return f"{x:.3f}" if x is not None else "n/a"

            print(f"{table:<18} {n_stable or 0:>8} {f(p50):>8} {f(p95):>8} "
                  f"{f(minl):>8} {f(maxl):>9} {n_replay or 0:>8} "
                  f"{f(replay_max):>12}")
            rows_out.append({
                "table": table, "event_col": col, "window": WINDOW,
                "cap_s": CAP_S, "n_stable": n_stable or 0,
                "p50_s": p50, "p95_s": p95, "min_s": minl, "max_s": maxl,
                "n_replay": n_replay or 0, "replay_max_s": replay_max,
                "n_negative_clock": n_neg or 0, "global_min_lag_s": gmin,
            })
    conn.close()
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nCSV -> {OUT_CSV}")
    print("Nota: lag minimo global > 0 => device NO adelantado (reloj sano).")


if __name__ == "__main__":
    sys.exit(main())

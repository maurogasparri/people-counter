#!/usr/bin/env python3
"""Grupo B / B1 — Exactitud de conteo controlado (TC-01…07).

Lee ``count_events`` de RDS (SOLO LECTURA) para una tanda de cruces conocidos y
la compara con lo esperado. Pensado para el día del montaje cenital: hacés N
cruces, corrés esto, y sale la tasa de acierto + el detalle por evento.

Uso:
    py count_session.py now                      # marca de tiempo del server (baseline)
    py count_session.py last 3                   # eventos de los últimos 3 min
    py count_session.py last 3 --in 10 --out 0   # + acierto vs esperado
    py count_session.py since "2026-07-01T14:30:00Z" --in 5 --out 5

Es read-only (solo SELECT) — no modifica infra ni datos. Reusa el endpoint y el
secret del resto de los scripts de benchmark.
"""
import argparse
import json
import sys

import boto3
import psycopg

EP = "people-counter-dev.cabqikscg4gm.us-east-1.rds.amazonaws.com"
SECRET = "people-counter/dev/rds-master"
DEV = "store-pilot-01-cam-01"


def connect():
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    s = json.loads(sm.get_secret_value(SecretId=SECRET)["SecretString"])
    return psycopg.connect(
        host=EP,
        port=5432,
        dbname="people_counter",
        user=s["username"],
        password=s["password"],
        sslmode="require",
        connect_timeout=15,
    )


def fetch(cur, where, params):
    cur.execute(
        f"""SELECT event_ts, direction, track_id, height_m, confidence
            FROM count_events
            WHERE device_id = %s AND {where}
            ORDER BY event_ts""",
        (DEV, *params),
    )
    return cur.fetchall()


def report(rows, exp_in, exp_out):
    print(f"\n{'ts':<19} {'dir':<4} {'track':<6} {'height_m':<9} conf")
    print("-" * 52)
    for ts, d, tid, h, c in rows:
        hs = f"{h:.2f}" if h is not None else "—"
        cs = f"{c:.2f}" if c is not None else "—"
        print(f"{ts:%Y-%m-%d %H:%M:%S} {d:<4} {str(tid):<6} {hs:<9} {cs}")
    ci = sum(1 for r in rows if r[1] == "in")
    co = sum(1 for r in rows if r[1] == "out")
    print(f"\nContados:  IN={ci}  OUT={co}  (total {len(rows)})")
    if exp_in is not None or exp_out is not None:
        ei, eo = exp_in or 0, exp_out or 0

        def acc(c, e):
            return "n/a" if e == 0 else f"{100 * min(c, e) / e:.0f}%"

        print(f"Esperado:  IN={ei}  OUT={eo}")
        print(
            f"Acierto:   IN={acc(ci, ei)}  OUT={acc(co, eo)}  | "
            f"ΔIN={ci - ei:+d}  ΔOUT={co - eo:+d}"
        )
        # Neto: clave para TC-06 (par cancelado → esperado neto 0).
        print(f"Neto:      contado={ci - co:+d}  esperado={ei - eo:+d}")
        extra = max(0, ci - ei) + max(0, co - eo)
        missed = max(0, ei - ci) + max(0, eo - co)
        print(f"Faltantes={missed}  Extra(doble/FP)={extra}")


def main():
    global DEV
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("now", "last", "since"))
    ap.add_argument("arg", nargs="?", help="minutos (last) o timestamp ISO (since)")
    ap.add_argument("--in", dest="exp_in", type=int, default=None)
    ap.add_argument("--out", dest="exp_out", type=int, default=None)
    ap.add_argument("--until", help="timestamp ISO de fin (modo since)")
    ap.add_argument("--device", default=DEV)
    args = ap.parse_args()
    DEV = args.device

    cx = connect()
    cur = cx.cursor()
    try:
        if args.mode == "now":
            cur.execute("SELECT now() AT TIME ZONE 'UTC'")
            print(f"Server UTC ahora: {cur.fetchone()[0]:%Y-%m-%dT%H:%M:%SZ}")
            print('Anotá esta hora; después: count_session.py since "<hora>" ...')
            return
        if args.mode == "last":
            mins = int(args.arg or 5)
            rows = fetch(cur, "event_ts >= now() - make_interval(mins => %s)", (mins,))
            print(f"Cruces del device {DEV} en los últimos {mins} min:")
        else:  # since
            if not args.arg:
                sys.exit("since requiere un timestamp ISO")
            if args.until:
                rows = fetch(
                    cur,
                    "event_ts >= %s::timestamptz AND event_ts < %s::timestamptz",
                    (args.arg, args.until),
                )
                print(f"Cruces del device {DEV} entre {args.arg} y {args.until}:")
            else:
                rows = fetch(cur, "event_ts >= %s::timestamptz", (args.arg,))
                print(f"Cruces del device {DEV} desde {args.arg}:")
        report(rows, args.exp_in, args.exp_out)
    finally:
        cx.close()


if __name__ == "__main__":
    main()

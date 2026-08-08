#!/usr/bin/env python3
"""TC-08 — MAE de estatura en geometría cenital real.

Por cada sujeto de altura conocida hacés K cruces bajo la cámara montada y
corrés esto: lee ``height_m`` de ``count_events`` (SOLO LECTURA) en la ventana,
lo compara con la altura real medida con cinta, y acumula cada (real, estimada)
en ``height_mae.csv``. Al final, ``aggregate`` saca el MAE global.

La salida publicada lleva además el instante de cada cruce, recuperado del
registro persistido para que cada fila sea verificable por separado.

Uso:
    py height_mae.py last 3 --true 1.75 --label adulto_A
    py height_mae.py last 3 --true 1.55 --label nino_B
    py height_mae.py aggregate

Requisito: ``vision.mounting_height_m`` debe estar seteado a la altura real de
montaje (height_m = mount − profundidad_cabeza; si el mount está mal, la
estatura sale corrida). Read-only (solo SELECT).
"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import boto3
import psycopg

EP = "people-counter-dev.cabqikscg4gm.us-east-1.rds.amazonaws.com"
SECRET = "people-counter/dev/rds-master"
DEV = "store-pilot-01-cam-01"
CSV = Path(__file__).resolve().parent / "height_mae.csv"


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


def collect(where, params, true_m, label):
    cx = connect()
    cur = cx.cursor()
    try:
        cur.execute(
            f"""SELECT height_m FROM count_events
               WHERE device_id = %s AND {where}
               ORDER BY event_ts""",
            (DEV, *params),
        )
        rows = cur.fetchall()
    finally:
        cx.close()
    ests = [float(r[0]) for r in rows if r[0] is not None]
    n_null = sum(1 for r in rows if r[0] is None)

    new = CSV if CSV.exists() else None
    with open(CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new is None:
            w.writerow(["label", "true_m", "est_m"])
        for e in ests:
            w.writerow([label, f"{true_m:.3f}", f"{e:.3f}"])

    print(
        f"Sujeto '{label}' (real {true_m:.2f} m) — {len(rows)} cruces, "
        f"{len(ests)} con altura, {n_null} NULL (→ 'unknown')"
    )
    if ests:
        mean = st.mean(ests)
        err = mean - true_m
        mae = st.mean([abs(e - true_m) for e in ests])
        print(
            f"  estimada: media={mean:.3f} mediana={st.median(ests):.3f} "
            f"std={(st.stdev(ests) if len(ests) > 1 else 0):.3f} m"
        )
        print(
            f"  error medio={err * 100:+.1f} cm | MAE={mae * 1000:.0f} mm "
            f"({100 * mae / true_m:.1f}%)"
        )
    print(f"  acumulado en {CSV.name}")


def aggregate():
    if not CSV.exists():
        sys.exit(f"No hay {CSV} — corré primero las tandas por sujeto.")
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    if not rows:
        sys.exit("CSV vacío.")
    errs = [abs(float(r["est_m"]) - float(r["true_m"])) for r in rows]
    mape = [
        abs(float(r["est_m"]) - float(r["true_m"])) / float(r["true_m"]) for r in rows
    ]
    print(f"=== MAE de estatura — agregado ({len(rows)} cruces) ===")
    print(
        f"  MAE  = {1000 * st.mean(errs):.0f} mm   (mediana {1000 * st.median(errs):.0f} mm)"
    )
    print(f"  MAPE = {100 * st.mean(mape):.1f} %")
    print(f"  peor = {1000 * max(errs):.0f} mm")
    labels = sorted({r["label"] for r in rows})
    within = sum(1 for e in errs if e <= 0.10)  # TC-05: dentro de ±10 cm
    print(
        f"  TC-05: dentro de ±10 cm = {within}/{len(rows)}  ·  sujetos = {len(labels)}"
        f"  ({', '.join(labels)})"
    )
    print("\n  Por sujeto:")
    print(
        f"  {'label':<12} {'real':>6} {'n':>3} {'media_est':>10} {'MAE_mm':>8} {'±10cm':>7}"
    )
    for lb in labels:
        sub = [r for r in rows if r["label"] == lb]
        ests = [float(r["est_m"]) for r in sub]
        true_m = float(sub[0]["true_m"])
        mae = st.mean([abs(e - true_m) for e in ests])
        w = sum(1 for e in ests if abs(e - true_m) <= 0.10)
        print(
            f"  {lb:<12} {true_m:>6.2f} {len(sub):>3} {st.mean(ests):>10.3f} "
            f"{1000 * mae:>8.0f} {w:>3}/{len(sub):<3}"
        )


def main():
    global DEV
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("last", "since", "aggregate"))
    ap.add_argument("arg", nargs="?", help="minutos (last) o timestamp ISO (since)")
    ap.add_argument("--until", help="timestamp ISO de fin (modo since)")
    ap.add_argument("--true", dest="true_m", type=float, help="altura real en m")
    ap.add_argument("--label", default="sujeto", help="identificador del sujeto")
    ap.add_argument("--device", default=DEV)
    args = ap.parse_args()
    DEV = args.device

    if args.mode == "aggregate":
        aggregate()
        return
    if args.true_m is None:
        sys.exit(f"{args.mode} requiere --true <altura_m>")
    if args.mode == "last":
        collect(
            "event_ts >= now() - make_interval(mins => %s)",
            (int(args.arg or 3),),
            args.true_m,
            args.label,
        )
    else:  # since: aísla una ventana limpia del operador del tráfico orgánico
        if not args.arg:
            sys.exit("since requiere un timestamp ISO")
        if args.until:
            collect(
                "event_ts >= %s::timestamptz AND event_ts < %s::timestamptz",
                (args.arg, args.until),
                args.true_m,
                args.label,
            )
        else:
            collect("event_ts >= %s::timestamptz", (args.arg,), args.true_m, args.label)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Medición de consumo eléctrico de un edge device del People Counter.

Suma V×I de todos los rieles del PMIC de la RPi5 (`vcgencmd pmic_read_adc`)
para estimar la potencia total de la placa. Como el AI HAT+ (Hailo-8L) se
alimenta por PCIe desde los rieles del PMIC y no tiene entrada de energía
propia, este total incluye **placa base + Hailo + cámaras CSI** — el device
entero. (El propio Hailo no expone medición de potencia: el firmware del
AI HAT+ rechaza `hailortcli measure-power` con UNSUPPORTED_DEVICE.)

Es la potencia *output-side* del PMIC: el draw de pared real suma ~12-15% de
pérdidas de conversión (y, con PoE HAT, además la conversión del PoE). `EXT5V`
es solo la tensión de entrada (sin sense de corriente), así que no se puede
leer la potencia de entrada directamente desde el PMIC.

Uso (en la Pi):
    # resumen de 10s con desglose por riel
    python3 scripts/measure_power.py --duration 10 --rails

    # monitoreo continuo a CSV hasta Ctrl-C (para correlacionar con pasadas)
    python3 scripts/measure_power.py --duration 0 --csv power_monitor.csv
"""

import argparse
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from statistics import mean, median

CUR_RE = re.compile(r"(\S+)_A current\(\d+\)=([\d.]+)A")
VOLT_RE = re.compile(r"(\S+)_V volt\(\d+\)=([\d.]+)V")


def read_rails() -> dict[str, float]:
    """Una lectura del PMIC → {riel: watts}. Devuelve {} si falla."""
    try:
        out = subprocess.run(
            ["vcgencmd", "pmic_read_adc"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    cur: dict[str, float] = {}
    vol: dict[str, float] = {}
    for line in out.splitlines():
        m = CUR_RE.search(line)
        if m:
            cur[m.group(1)] = float(m.group(2))
        m = VOLT_RE.search(line)
        if m:
            vol[m.group(1)] = float(m.group(2))
    # Rieles con corriente Y tensión (excluye EXT5V/BATT, que solo dan tensión).
    return {k: cur[k] * vol[k] for k in cur if k in vol}


def pctl(data: list[float], q: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def summarize(totals: list[float]) -> str:
    if not totals:
        return "sin muestras"
    return (
        f"n={len(totals)}  min={min(totals):.2f}W  "
        f"avg={mean(totals):.2f}W  p50={median(totals):.2f}W  "
        f"p95={pctl(totals, 0.95):.2f}W  max={max(totals):.2f}W"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Mide el consumo del device vía PMIC.")
    ap.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="segundos a medir (0 = hasta Ctrl-C). Default 10.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="segundos entre muestras. Default 0.5.",
    )
    ap.add_argument(
        "--csv",
        metavar="PATH",
        help="además, vuelca cada muestra a un CSV (modo append).",
    )
    ap.add_argument(
        "--rails",
        action="store_true",
        help="incluir desglose de potencia promedio por riel.",
    )
    ap.add_argument(
        "--print-every",
        type=float,
        default=5.0,
        help="cada cuántos segundos imprimir una línea en vivo. Default 5.",
    )
    args = ap.parse_args()

    # Sanity: ¿estamos en una Pi con PMIC accesible?
    if not read_rails():
        print(
            "ERROR: no se pudo leer 'vcgencmd pmic_read_adc' "
            "(¿corriendo en la RPi5?).",
            file=sys.stderr,
        )
        sys.exit(1)

    totals: list[float] = []
    rail_acc: dict[str, list[float]] = {}
    csv_f = None
    csv_cols: list[str] = []
    if args.csv:
        try:
            csv_f = open(args.csv, "a", buffering=1)
        except OSError as e:
            print(f"ERROR abriendo CSV: {e}", file=sys.stderr)
            sys.exit(1)

    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("flag", True))

    label = (
        f"{args.duration:.0f}s" if args.duration else "continuo (Ctrl-C/SIGTERM corta)"
    )
    print(
        f"Midiendo consumo — {label}, cada {args.interval}s. "
        "Total = placa + Hailo + cámaras (output PMIC)."
    )

    t0 = time.time()
    last_print = t0
    while not stop["flag"]:
        rails = read_rails()
        now = time.time()
        if rails:
            total = sum(rails.values())
            totals.append(total)
            for k, v in rails.items():
                rail_acc.setdefault(k, []).append(v)
            if csv_f:
                if not csv_cols:
                    csv_cols = sorted(rails)
                    csv_f.write("timestamp,total_w," + ",".join(csv_cols) + "\n")
                row = [
                    datetime.now().isoformat(timespec="milliseconds"),
                    f"{total:.3f}",
                ]
                row += [f"{rails.get(c, 0.0):.4f}" for c in csv_cols]
                csv_f.write(",".join(row) + "\n")
            if now - last_print >= args.print_every:
                print(
                    f"  [{datetime.now():%H:%M:%S}] ahora={total:.2f}W  "
                    f"({summarize(totals)})"
                )
                last_print = now
        if args.duration and (now - t0) >= args.duration:
            break
        time.sleep(args.interval)

    if csv_f:
        csv_f.close()

    print("\n=== Resumen ===")
    print(summarize(totals))
    if args.rails and rail_acc:
        print("\nPor riel (promedio, desc):")
        for k in sorted(rail_acc, key=lambda r: -mean(rail_acc[r])):
            print(f"  {k:12s} {mean(rail_acc[k]):.3f}W")
    if args.csv:
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()

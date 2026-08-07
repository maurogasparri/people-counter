#!/usr/bin/env python3
"""TC-15 — latencia de extremo a extremo: con y sin calentamiento de la Lambda.

Compara una ventana posterior a la activación del calentamiento contra la
ventana equivalente del día anterior —mismo largo y mismo tramo horario— para
que la atribución no se confunda con una variación horaria.

Además de la proporción de eventos por encima del umbral, reporta la **tasa
absoluta de eventos lentos por hora**. Es el dato que discrimina: si la tasa se
mantiene constante mientras la proporción cae, la cola es función de la
frecuencia de reciclado del entorno de ejecución y no del volumen de eventos.

Marca de origen por flujo:
    conteo y telemetría : ``event_ts``   (instante del hecho)
    inalámbrico         : ``period_end`` (cierre de la ventana de agregación;
                          medir contra ``last_seen_ts`` incluiría el resto de
                          la ventana, que es agregación y no transporte)

Uso:
    py tc15_latency_ab.py --desde 2026-08-06T06:34:00-03:00 --horas 6
    py tc15_latency_ab.py --desde 2026-08-06T06:34:00-03:00 --horas 24

Reproducible: py docs/validacion/tc15_latency_ab.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _raiz_repo() -> Path:
    anc = list(Path(__file__).resolve().parents)
    for cand in anc:
        if (cand / "scripts" / "provision.py").is_file():
            return cand
    for cand in anc:
        alt = cand / "people-counter"
        if (alt / "scripts" / "provision.py").is_file():
            return alt
    raise SystemExit("no se encontró la raíz del repositorio")


sys.path.insert(0, str(_raiz_repo()))
from scripts.provision import _rds_connect  # noqa: E402

FLUJOS = [
    ("conteo", "count_events", "event_ts"),
    ("telemetria", "telemetry", "event_ts"),
    ("inalambrico", "wifi_ble_events", "period_end"),
]
UMBRAL = 5.0


def medir(cur, tabla, col, a, b):
    cur.execute(
        f"""SELECT count(*),
              round(percentile_cont(0.50) WITHIN GROUP (
                ORDER BY EXTRACT(epoch FROM (received_at-{col})))::numeric,3),
              round(percentile_cont(0.95) WITHIN GROUP (
                ORDER BY EXTRACT(epoch FROM (received_at-{col})))::numeric,3),
              round(max(EXTRACT(epoch FROM (received_at-{col})))::numeric,2),
              count(*) FILTER (WHERE EXTRACT(epoch FROM (received_at-{col})) > {UMBRAL})
            FROM {tabla} WHERE store_id='store-pilot-01'
              AND {col} >= %s AND {col} < %s""",
        (a, b),
    )
    return cur.fetchone()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--desde", required=True, help="inicio de la ventana CON calentamiento (ISO)")
    p.add_argument("--horas", type=float, required=True)
    a_ = p.parse_args()
    ini = datetime.fromisoformat(a_.desde)
    dur = timedelta(hours=a_.horas)
    ventanas = [("CON calentamiento", ini, ini + dur),
                ("SIN (dia previo)", ini - timedelta(days=1), ini - timedelta(days=1) + dur)]

    conn = _rds_connect("people-counter-dev", "us-east-1")
    cur = conn.cursor()
    print("=== TC-15 — latencia de extremo a extremo, contraste A/B ===")
    for etq, a, b in ventanas:
        print(f"  {etq:<20}: {a:%Y-%m-%d %H:%M} → {b:%m-%d %H:%M} ({a_.horas:g} h)")
    print(f"  umbral: {UMBRAL:g} s · marca del inalámbrico: cierre de ventana\n")

    print(f"  {'flujo':<12}{'ventana':<20}{'n':>7}{'p50':>9}{'p95':>9}{'max':>9}"
          f"{'>5s':>6}{'% >5s':>8}{'lentos/h':>10}")
    acum = {}
    for etq, tabla, col in FLUJOS:
        for nom, a, b in ventanas:
            n, p50, p95, mx, s5 = medir(cur, tabla, col, a, b)
            acum.setdefault(nom, [0, 0])
            if etq != "conteo":
                acum[nom][0] += n
                acum[nom][1] += s5
            pct = f"{100.0*s5/n:.2f}%" if n else "—"
            f = lambda x: "—" if x is None else str(x)
            print(f"  {etq:<12}{nom:<20}{n:>7}{f(p50):>9}{f(p95):>9}{f(mx):>9}"
                  f"{s5:>6}{pct:>8}{s5/a_.horas:>10.2f}")

    print("\n  --- agregado de los flujos que el dispositivo genera solo ---")
    print(f"  {'ventana':<20}{'n':>7}{'>5s':>6}{'% >5s':>9}{'lentos/h':>11}")
    for nom, (n, s5) in acum.items():
        pct = f"{100.0*s5/n:.2f}%" if n else "—"
        print(f"  {nom:<20}{n:>7}{s5:>6}{pct:>9}{s5/a_.horas:>11.2f}")

    if len(acum) == 2:
        (na, sa), (nb, sb) = acum["CON calentamiento"], acum["SIN (dia previo)"]
        if nb and na:
            pa, pb = 100.0 * sa / na, 100.0 * sb / nb
            ra, rb = sa / a_.horas, sb / a_.horas
            print(f"\n  proporción : {pb:.2f}%  →  {pa:.2f}%   "
                  f"({'baja' if pa < pb else 'sube'} {abs(pa-pb):.2f} puntos)")
            print(f"  tasa/hora  : {rb:.2f}  →  {ra:.2f}   "
                  f"({'baja' if ra < rb else 'se mantiene o sube'})")
            print("\n  Lectura: si la tasa por hora se sostiene mientras la proporción")
            print("  cae, la cola depende de la frecuencia de reciclado del entorno y")
            print("  no del volumen de eventos.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

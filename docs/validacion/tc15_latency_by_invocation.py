#!/usr/bin/env python3
"""TC-15 — latencia de extremo a extremo contada POR INVOCACIÓN.

Las cifras que se venían manejando contaban **filas persistidas**. Para el
conteo y la telemetría eso coincide con la invocación —cada mensaje MQTT
inserta una fila—, pero el flujo inalámbrico publica **un array por ventana de
15 min** que inserta N filas en una sola invocación. Contar filas multiplica
cada invocación lenta por el número de dispositivos que llevaba dentro, y las
filas de un mismo mensaje comparten latencia por construcción: no son
observaciones independientes.

La unidad correcta es la invocación, que es lo que atraviesa la cadena
device → IoT Core → Lambda → RDS y lo que se retrasa.

Marca de origen por flujo:
    conteo, telemetría : ``event_ts``   (instante del hecho)
    inalámbrico        : ``period_end`` (cierre de la ventana de agregación)

Se reporta el registro completo y, en paralelo, la variante que excluye los
replays del buffer local —eventos encolados durante un corte de conectividad y
entregados después—, que son latencia de red del sitio y no de la cadena.

Reproducible: py docs/validacion/tc15_latency_by_invocation.py
"""

from __future__ import annotations

import argparse
import sys
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

TIENDA = "store-pilot-01"
UMBRAL = 5.0
REPLAY = 60.0  # por encima de esto se considera entrega diferida del buffer

# Cada flujo se reduce a una fila por INVOCACIÓN.
INVOCACIONES = {
    "conteo": f"""
        SELECT event_ts AS origen, received_at, 1 AS filas
        FROM count_events WHERE store_id = '{TIENDA}'""",
    "telemetria": f"""
        SELECT event_ts AS origen, received_at, 1 AS filas
        FROM telemetry WHERE store_id = '{TIENDA}'""",
    # una invocación = una ventana; N filas comparten latencia
    "inalambrico": f"""
        SELECT period_end AS origen, min(received_at) AS received_at, count(*) AS filas
        FROM wifi_ble_events WHERE store_id = '{TIENDA}'
        GROUP BY device_id, period_end""",
}

METRICAS = """
 SELECT count(*) n,
        round(percentile_cont(0.50) WITHIN GROUP (ORDER BY lat)::numeric, 3) p50,
        round(percentile_cont(0.95) WITHIN GROUP (ORDER BY lat)::numeric, 3) p95,
        round(max(lat)::numeric, 2) maximo,
        count(*) FILTER (WHERE lat > {u}) lentos,
        round(EXTRACT(epoch FROM (max(received_at) - min(received_at)))/3600.0) horas,
        sum(filas) filas
 FROM (SELECT *, EXTRACT(epoch FROM (received_at - origen)) lat FROM ({sub}) i) x
 {filtro}
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sin-replays", action="store_true",
                   help="excluye entregas diferidas del buffer (>60 s)")
    a = p.parse_args()

    conn = _rds_connect("people-counter-dev", "us-east-1")
    cur = conn.cursor()

    print("=== TC-15 — latencia de extremo a extremo POR INVOCACIÓN ===")
    print(f"  dispositivo : {TIENDA}   umbral: {UMBRAL:g} s")
    print("  unidad      : una invocación = un mensaje MQTT = una ejecución de la Lambda\n")

    # El 2026-06-08 la cadencia de telemetría pasó de 1/hora a 1/5 min
    # (commit 3d7a7d9). Eso multiplicó por 12 la tasa de invocación de la
    # Lambda y, con ella, la probabilidad de encontrar el entorno tibio. El
    # agregado histórico mezcla las dos configuraciones: se reportan aparte.
    CORTE = "2026-06-08"
    for etiqueta, filtro in (
        ("REGISTRO COMPLETO", ""),
        ("EXCLUYENDO REPLAYS DEL BUFFER (>60 s)", f"WHERE lat <= {REPLAY}"),
        (f"CONFIGURACIÓN ACTUAL (desde {CORTE}, sin replays)",
         f"WHERE lat <= {REPLAY} AND origen >= '{CORTE}'"),
        (f"CONFIGURACIÓN ANTERIOR (hasta {CORTE}, sin replays)",
         f"WHERE lat <= {REPLAY} AND origen < '{CORTE}'"),
    ):
        print(f"  --- {etiqueta} ---")
        print(f"  {'flujo':<13}{'invoc.':>8}{'filas':>8}{'p50':>9}{'p95':>10}"
              f"{'max':>11}{'>5s':>6}{'% >5s':>8}{'lentas/h':>10}")
        tot_n = tot_l = 0
        horas_max = 0
        for flujo, sub in INVOCACIONES.items():
            cur.execute(METRICAS.format(u=UMBRAL, sub=sub, filtro=filtro))
            n, p50, p95, mx, lentos, horas, filas = cur.fetchone()
            if not n:
                continue
            horas = float(horas or 1)
            horas_max = max(horas_max, horas)
            tot_n += n
            tot_l += lentos
            print(f"  {flujo:<13}{n:>8}{int(filas):>8}{p50:>9}{p95:>10}{mx:>11}"
                  f"{lentos:>6}{100.0*lentos/n:>7.2f}%{lentos/horas:>10.3f}")
        print(f"  {'AGREGADO':<13}{tot_n:>8}{'':>8}{'':>9}{'':>10}{'':>11}"
              f"{tot_l:>6}{100.0*tot_l/tot_n:>7.2f}%{tot_l/horas_max:>10.3f}")
        print()

    # --- hueco desde la invocación anterior, por flujo -----------------------
    print("  --- hueco desde la invocación anterior (mismo flujo) ---")
    for flujo, sub in INVOCACIONES.items():
        cur.execute(f"""
          WITH i AS ({sub}),
          g AS (SELECT received_at,
                       EXTRACT(epoch FROM (received_at - origen)) lat,
                       EXTRACT(epoch FROM (received_at -
                         lag(received_at) OVER (ORDER BY received_at))) hueco
                FROM i)
          SELECT CASE WHEN hueco IS NULL THEN 'z) primera'
                      WHEN hueco < 30   THEN 'a) < 30 s'
                      WHEN hueco < 120  THEN 'b) 30-120 s'
                      WHEN hueco < 240  THEN 'c) 2-4 min'
                      WHEN hueco < 360  THEN 'd) 4-6 min'
                      WHEN hueco < 1800 THEN 'e) 6-30 min'
                      ELSE 'f) > 30 min' END rango,
                 count(*) n,
                 round(percentile_cont(0.50) WITHIN GROUP (ORDER BY lat)::numeric,3) p50,
                 round(max(lat)::numeric,2) maximo,
                 count(*) FILTER (WHERE lat > {UMBRAL}) lentos
          FROM g GROUP BY 1 ORDER BY 1""")
        filas = cur.fetchall()
        print(f"\n    {flujo}")
        print(f"      {'hueco previo':<14}{'n':>7}{'p50':>9}{'max':>11}{'lentas':>8}{'% lentas':>10}")
        for r in filas:
            pct = f"{100.0*r[4]/r[1]:.1f}%" if r[1] else "—"
            print(f"      {r[0]:<14}{r[1]:>7}{r[2]:>9}{r[3]:>11}{r[4]:>8}{pct:>10}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

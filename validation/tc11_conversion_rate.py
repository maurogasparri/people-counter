#!/usr/bin/env python3
"""TC-11 — tasa de conversión de extremo a extremo contra un valor esperado.

La corrida del 21/06 dejó verificado que las ventas por sucursal y día son
computables, pero NO contrastó la tasa contra ningún valor de referencia. Este
guion cierra ese hueco.

Camino de extremo a extremo (el que consumen los tableros):

    count_events     -> rollup_counting_day -> counting_by_bucket_day  --.
                                                                          >-- conversion_by_bucket_day.conversion_rate
    pos_transactions -> rollup_pos_day      -> pos_by_bucket_day       --'

Valor esperado, calculado de forma independiente: se cuenta directamente sobre
las tablas base, sin tocar rollups ni vistas, aplicando el mismo criterio de
día local (``lday``) que usan las vistas.

Un caso CUMPLE cuando coinciden las tres magnitudes: visitantes, ventas y la
tasa (con tolerancia de 1e-9 por el redondeo del punto flotante).

Reproducible: py validation/tc11_conversion_rate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _raiz_repo() -> Path:
    """Ubica la raíz del repo (la que contiene scripts/provision.py).

    Desde ``validation/`` —ubicación final— la raíz es un ancestro. Desde
    el directorio de preparación no lo es: cuelga de un árbol hermano, así que
    también se prueba ``<ancestro>/people-counter``.
    """
    anc = list(Path(__file__).resolve().parents)
    for cand in anc:
        if (cand / "scripts" / "provision.py").is_file():
            return cand
    for cand in anc:
        alt = cand / "people-counter"
        if (alt / "scripts" / "provision.py").is_file():
            return alt
    raise SystemExit("no se encontró la raíz del repositorio (scripts/provision.py)")


sys.path.insert(0, str(_raiz_repo()))
from scripts.provision import _rds_connect  # noqa: E402

DESDE, HASTA = "2026-06-01", "2026-06-07"
EPS = 1e-9

# --- extremo a extremo: lo que la vista publica ----------------------------
E2E = """
 SELECT store_id, bucket_day, visits, sales, conversion_rate
 FROM conversion_by_bucket_day
 WHERE bucket_day BETWEEN %s AND %s AND store_id LIKE 'demo-%%'
 ORDER BY store_id, bucket_day
"""

# --- esperado: recuento directo sobre las tablas base ----------------------
ESPERADO = """
 WITH v AS (
   SELECT e.store_id, lday(e.event_ts, s.timezone) AS d, count(*) AS visits
   FROM count_events e JOIN sites s USING (store_id)
   WHERE e.direction = 'in' AND e.store_id LIKE 'demo-%%'
   GROUP BY 1, 2),
 t AS (
   SELECT p.store_id, lday(p.event_ts, s.timezone) AS d, count(*) AS sales
   FROM pos_transactions p JOIN sites s USING (store_id)
   WHERE p.type = 'sale' AND p.store_id LIKE 'demo-%%'
   GROUP BY 1, 2)
 SELECT COALESCE(v.store_id, t.store_id), COALESCE(v.d, t.d),
        COALESCE(v.visits, 0), COALESCE(t.sales, 0)
 FROM v FULL OUTER JOIN t ON v.store_id = t.store_id AND v.d = t.d
 WHERE COALESCE(v.d, t.d) BETWEEN %s AND %s
 ORDER BY 1, 2
"""


def main() -> int:
    conn = _rds_connect("people-counter-dev", "us-east-1")
    cur = conn.cursor()

    cur.execute(E2E, (DESDE, HASTA))
    e2e = {(r[0], r[1]): (r[2], r[3], r[4]) for r in cur.fetchall()}
    cur.execute(ESPERADO, (DESDE, HASTA))
    esp = {(r[0], r[1]): (r[2], r[3]) for r in cur.fetchall()}

    print("=== TC-11 — tasa de conversión: extremo a extremo vs valor esperado ===")
    print(f"  ventana: {DESDE} .. {HASTA}   sucursales: demo-*")
    print(f"  casos (sucursal x día): {len(set(e2e) | set(esp))}\n")
    print(f"  {'sucursal':<10}{'día':<12}"
          f"{'visit e2e':>10}{'visit esp':>10}"
          f"{'vta e2e':>9}{'vta esp':>9}"
          f"{'tasa e2e':>10}{'tasa esp':>10}  veredicto")

    ok = fallan = 0
    for k in sorted(set(e2e) | set(esp)):
        tienda, dia = k
        v_e, s_e, r_e = e2e.get(k, (None, None, None))
        v_x, s_x = esp.get(k, (None, None))
        r_x = (s_x / v_x) if (v_x and s_x is not None) else None
        coincide = (
            v_e == v_x
            and s_e == s_x
            and ((r_e is None and r_x is None)
                 or (r_e is not None and r_x is not None and abs(r_e - r_x) < EPS))
        )
        ok, fallan = (ok + 1, fallan) if coincide else (ok, fallan + 1)
        f = lambda x: "—" if x is None else (f"{x:.6f}" if isinstance(x, float) else str(x))
        print(f"  {tienda:<10}{str(dia):<12}{f(v_e):>10}{f(v_x):>10}"
              f"{f(s_e):>9}{f(s_x):>9}{f(r_e):>10}{f(r_x):>10}  "
              f"{'CUMPLE' if coincide else 'NO CUMPLE'}")

    total = ok + fallan
    print(f"\n  casos coincidentes: {ok}/{total} ({100*ok/total:.1f}%)")
    print(f"  VEREDICTO TC-11 (tasa de conversión): "
          f"{'CUMPLE' if fallan == 0 else 'NO CUMPLE'} "
          f"(criterio: coincidencia en el 100% de los casos)")
    conn.close()
    return 0 if fallan == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

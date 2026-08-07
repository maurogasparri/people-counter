#!/usr/bin/env python3
"""Cruces registrados durante la demostración funcional del 6 de agosto de 2026.

Extrae de RDS los eventos de conteo generados durante la grabación del video de
demostración, para que el registro del borde —el overlay que se ve en pantalla,
con sus cajas, alturas y confianzas— pueda cotejarse fila por fila contra lo que
efectivamente llegó a la plataforma.

Ese cotejo es el punto: el video demuestra que el dispositivo detecta; esta
extracción demuestra que la cadena completa —device → IoT Core → Lambda → RDS—
entregó los mismos ocho eventos, con los mismos valores.

La salida incluye la consulta SQL literal, para que la evidencia sea
reproducible: quien la lea puede ver qué se preguntó, no sólo qué se respondió.

Nota sobre saneamiento: ``count_events`` NO tiene columna de hash de visitante
—esa vive en ``wifi_ble_events``— de modo que acá no hay pseudónimo que
enmascarar. ``event_id`` es un UUID aleatorio sin relación con la persona, y
``device_id``/``store_id`` se conservan por decisión explícita: son lo que hace
verificable el registro y la sucursal ya está rotulada de forma genérica.

NO lleva código de caso a propósito. Con dos cruces por condición la muestra no
alcanza para verificar ningún criterio de aceptación —el de conteo individual
exige 9 de 10—, de modo que rotularla como verificación sería sobre-afirmar. Es
una demostración fechada del funcionamiento de la cadena, no una medición.

Reproducible: py docs/validacion/demo_crossings_20260806.py
"""

from __future__ import annotations

import statistics as st
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

SQL = """
SELECT
    event_id,
    device_id,
    store_id,
    event_ts AT TIME ZONE 'America/Argentina/Buenos_Aires'    AS emision_local,
    received_at AT TIME ZONE 'America/Argentina/Buenos_Aires' AS recepcion_local,
    EXTRACT(epoch FROM (received_at - event_ts))              AS latencia_s,
    direction,
    track_id,
    height_m,
    confidence
FROM count_events
WHERE store_id = 'store-pilot-01'
  AND event_ts >= TIMESTAMPTZ '2026-08-06 19:42:00-03:00'
  AND event_ts <= TIMESTAMPTZ '2026-08-06 19:43:30-03:00'
ORDER BY event_ts
"""

# Lo que el documento afirma, para contrastar.
AFIRMA = {
    "eventos": 8, "ingresos": 4, "egresos": 4, "alterna": True,
    "altura_min": 1.57, "altura_max": 1.68,
    "conf_min": 0.82, "conf_max": 0.90,
    "lat_mediana": 0.197, "lat_max": 0.225,
}


def main() -> int:
    conn = _rds_connect("people-counter-dev", "us-east-1")
    cur = conn.cursor()
    cur.execute(SQL)
    filas = cur.fetchall()
    cols = [d[0] for d in cur.description]

    print("=== Cruces de la demostración funcional del 2026-08-06 ===")
    print("  fecha        : 2026-08-06")
    print("  video         : demo_overlay_20260806_194126.mp4 (inicio 19:41:26 −03)")
    print("  ventana       : 19:42:00 a 19:43:30 hora local (−03)")
    print(f"  eventos       : {len(filas)}\n")

    print("--- CONSULTA SQL LITERAL ---")
    print(SQL.rstrip())
    print("\n--- SALIDA ---")
    print(f"  {'#':>2} {'event_id':<38}{'emisión':<13}{'recepción':<13}"
          f"{'lat_s':>8}{'dir':>5}{'track':>6}{'altura':>9}{'conf':>9}")
    lats, dirs = [], []
    t0 = filas[0][3] if filas else None
    for i, f in enumerate(filas, 1):
        d = dict(zip(cols, f))
        lats.append(float(d["latencia_s"]))
        dirs.append(d["direction"])
        print(f"  {i:>2} {str(d['event_id']):<38}"
              f"{d['emision_local']:%H:%M:%S}     "
              f"{d['recepcion_local']:%H:%M:%S}     "
              f"{float(d['latencia_s']):>8.3f}{d['direction']:>5}{d['track_id']:>6}"
              f"{d['height_m']:>9.4f}{d['confidence']:>9.4f}")

    print(f"\n  dispositivo: {filas[0][1]}   sucursal: {filas[0][2]}")
    print("  desplazamiento desde el primer evento y desde el inicio del video:")
    for i, f in enumerate(filas, 1):
        d = (f[3] - t0).total_seconds()
        print(f"    evento {i}: +{d:6.1f} s desde el 1º   ·   {d + 47.0:6.1f} s de video")

    # --- contraste con lo que afirma el documento -------------------------
    ins = dirs.count("in")
    outs = dirs.count("out")
    alterna = all(dirs[i] != dirs[i + 1] for i in range(len(dirs) - 1))
    alturas = [float(f[8]) for f in filas if f[8] is not None]
    confs = [float(f[9]) for f in filas if f[9] is not None]
    med, mx = st.median(lats), max(lats)

    print("\n--- CONTRASTE CON LO QUE AFIRMA EL DOCUMENTO ---")
    chequeos = [
        ("eventos", len(filas), AFIRMA["eventos"], len(filas) == AFIRMA["eventos"]),
        ("ingresos", ins, AFIRMA["ingresos"], ins == AFIRMA["ingresos"]),
        ("egresos", outs, AFIRMA["egresos"], outs == AFIRMA["egresos"]),
        ("alternan", alterna, AFIRMA["alterna"], alterna is AFIRMA["alterna"]),
        ("altura mínima", f"{min(alturas):.4f}", AFIRMA["altura_min"],
         round(min(alturas), 2) == AFIRMA["altura_min"]),
        ("altura máxima", f"{max(alturas):.4f}", AFIRMA["altura_max"],
         round(max(alturas), 2) == AFIRMA["altura_max"]),
        ("confianza mínima", f"{min(confs):.4f}", AFIRMA["conf_min"],
         round(min(confs), 2) == AFIRMA["conf_min"]),
        ("confianza máxima", f"{max(confs):.4f}", AFIRMA["conf_max"],
         round(max(confs), 2) == AFIRMA["conf_max"]),
        ("latencia mediana", f"{med:.4f}", AFIRMA["lat_mediana"],
         round(med, 3) == AFIRMA["lat_mediana"]),
        ("latencia máxima", f"{mx:.4f}", AFIRMA["lat_max"],
         round(mx, 3) == AFIRMA["lat_max"]),
    ]
    for etq, obs, dice, ok in chequeos:
        print(f"  {etq:<20} documento={str(dice):<10} observado={str(obs):<12}"
              f"{'coincide' if ok else '*** NO COINCIDE ***'}")

    print(f"\n  latencia — n={len(lats)}  mediana={med:.4f} s  máximo={mx:.4f} s  "
          f"mínimo={min(lats):.4f} s  >5 s={sum(1 for x in lats if x > 5)}")
    conn.close()
    return 0 if all(c[3] for c in chequeos) else 1


if __name__ == "__main__":
    raise SystemExit(main())

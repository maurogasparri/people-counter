#!/usr/bin/env python3
"""TC-19 — reconstrucción del estado de las alarmas sobre la ventana de la campaña.

El historial de transiciones de CloudWatch está vacío y sólo se conserva el
estado actual de cada alarma, de modo que no se puede consultar si alguna se
disparó durante la ventana. Pero las **métricas que alimentan a cada alarma**
sí se retienen 63 días, así que la condición se puede reevaluar.

Para cada alarma se trae su métrica subyacente en la ventana, con el período y
el estadístico que la propia alarma declara, y se aplica su condición completa:
operador y umbral, cantidad de períodos de evaluación, datapoints-to-alarm y
tratamiento de datos faltantes.

Un intervalo se marca EN ALARMA cuando, dentro de la ventana móvil de
``EvaluationPeriods``, al menos ``DatapointsToAlarm`` puntos incumplen el
umbral — que es la regla M-de-N que aplica CloudWatch.

CAVEAT DE RETENCIÓN: la resolución de 5 min vive 63 días; esta ventana deja de
ser reconstruible alrededor del 2026-08-23.

Reproducible: py docs/validacion/tc19_alarm_reconstruction.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import boto3

sys.stdout.reconfigure(encoding="utf-8")

OPS = {
    "GreaterThanThreshold": lambda v, t: v > t,
    "GreaterThanOrEqualToThreshold": lambda v, t: v >= t,
    "LessThanThreshold": lambda v, t: v < t,
    "LessThanOrEqualToThreshold": lambda v, t: v <= t,
}


def serie(cw, a, ini, fin) -> dict:
    stat = a.get("Statistic")
    kw = {"Statistics": [stat]} if stat else {"ExtendedStatistics": [a["ExtendedStatistic"]]}
    pts, cursor = {}, ini
    while cursor < fin:
        tope = min(cursor + timedelta(hours=24), fin)
        r = cw.get_metric_statistics(
            Namespace=a["Namespace"], MetricName=a["MetricName"],
            Dimensions=a.get("Dimensions", []), StartTime=cursor, EndTime=tope,
            Period=a["Period"], **kw)
        for d in r["Datapoints"]:
            pts[d["Timestamp"].replace(second=0, microsecond=0)] = d[stat] if stat else \
                d["ExtendedStatistics"][a["ExtendedStatistic"]]
        cursor = tope
    return pts


def evaluar(a, pts, ini, fin) -> tuple[bool, int, int, list]:
    """Devuelve (se_disparo, intervalos_en_alarma, periodos, momentos)."""
    per = a["Period"]
    n_per = int((fin - ini).total_seconds() // per)
    evalp = a.get("EvaluationPeriods", 1)
    m = a.get("DatapointsToAlarm") or evalp
    op = OPS[a["ComparisonOperator"]]
    thr = a["Threshold"]
    faltante = a.get("TreatMissingData", "missing")

    estados = []  # True incumple, False cumple, None faltante
    for i in range(n_per):
        t = ini + timedelta(seconds=i * per)
        v = pts.get(t.replace(second=0, microsecond=0))
        if v is None:
            estados.append({"breaching": True, "notBreaching": False,
                            "ignore": None, "missing": None}.get(faltante))
        else:
            estados.append(op(v, thr))

    momentos, en_alarma = [], 0
    for i in range(evalp - 1, n_per):
        ven = [e for e in estados[i - evalp + 1:i + 1] if e is not None]
        if len(ven) >= m and sum(1 for e in ven if e) >= m:
            en_alarma += 1
            if len(momentos) < 5:
                momentos.append(ini + timedelta(seconds=i * per))
    return en_alarma > 0, en_alarma, n_per, momentos


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--desde", default="2026-06-21T19:20:00Z")
    p.add_argument("--hasta", default="2026-06-22T19:20:00Z")
    a_ = p.parse_args()
    ini = datetime.fromisoformat(a_.desde.replace("Z", "+00:00"))
    fin = datetime.fromisoformat(a_.hasta.replace("Z", "+00:00"))

    cw = boto3.client("cloudwatch", region_name="us-east-1")
    alarmas = sorted(cw.describe_alarms(MaxRecords=100)["MetricAlarms"],
                     key=lambda x: x["AlarmName"])

    print("=== TC-19 — ¿se disparó alguna alarma en la ventana de la campaña? ===")
    print(f"  ventana UTC   : {ini:%Y-%m-%d %H:%M} a {fin:%Y-%m-%d %H:%M}")
    print(f"  ventana local : {ini - timedelta(hours=3):%Y-%m-%d %H:%M} a "
          f"{fin - timedelta(hours=3):%Y-%m-%d %H:%M} (−03)")
    print(f"  alarmas       : {len(alarmas)}\n")

    disparadas = []
    for a in alarmas:
        pts = serie(cw, a, ini, fin)
        fired, n_al, n_per, momentos = evaluar(a, pts, ini, fin)
        v = list(pts.values())
        print(f"--- {a['AlarmName']}")
        print(f"    condición : {a['MetricName']} {a.get('Statistic')} "
              f"{a['ComparisonOperator']} {a['Threshold']} · "
              f"{a['EvaluationPeriods']}×{a['Period']}s · faltantes={a.get('TreatMissingData')}")
        print(f"    datos     : {len(pts)}/{n_per} intervalos con métrica"
              + (f" · min={min(v):.4g} max={max(v):.4g}" if v else " · SIN DATOS"))
        print(f"    resultado : {'*** SE HABRÍA DISPARADO ***' if fired else 'nunca alcanzó la condición'}"
              + (f"  ({n_al} intervalos)" if fired else ""))
        for t in momentos:
            print(f"                {t:%Y-%m-%d %H:%M} UTC")
        print()
        if fired:
            disparadas.append(a["AlarmName"])

    print("=" * 70)
    if disparadas:
        print(f"  ALARMAS DISPARADAS EN LA VENTANA: {len(disparadas)}")
        for n in disparadas:
            print(f"    - {n}")
        print("  El criterio de TC-19 NO se satisface en esta condición.")
    else:
        print("  NINGUNA de las diez alarmas alcanzó su condición durante la ventana.")
        print("  La condición «sin alarmas disparadas» del criterio queda acreditada.")
    return 1 if disparadas else 0


if __name__ == "__main__":
    raise SystemExit(main())

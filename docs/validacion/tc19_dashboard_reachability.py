#!/usr/bin/env python3
"""TC-19 — accesibilidad de los tableros, reconstruida desde el balanceador.

Grafana corre en ECS Fargate detrás de un Application Load Balancer, que
publica por su cuenta el estado de sus destinos y el desglose de respuestas.
Eso permite MEDIR la accesibilidad en una ventana pasada en vez de inferirla:

    HealthyHostCount / UnHealthyHostCount   destinos que pasan el health check
    HTTPCode_Target_2XX/3XX/4XX/5XX_Count   respuestas del destino
    HTTPCode_ELB_5XX_Count                  errores del propio balanceador
    TargetResponseTime                      latencia de la respuesta
    RequestCount                            tráfico, para saber si hubo uso

Con al menos un destino sano en todos los intervalos y sin 5XX, la
accesibilidad queda acreditada. Un matiz de lectura importante: el ALB **no
publica** las métricas de conteo cuyo valor es cero, así que la ausencia de
puntos en 5XX significa «ninguna respuesta 5XX», no «falta de datos».

Reproducible: py docs/validacion/tc19_dashboard_reachability.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

import boto3

sys.stdout.reconfigure(encoding="utf-8")

LB = "app/pc-grafana-dev/2cbcb48da1b1ba0d"
TG = "targetgroup/pc-grafana-tg-dev/04fd54af9b650034"
PASO = 300

SONDAS = [
    ("HealthyHostCount", [{"Name": "TargetGroup", "Value": TG},
                          {"Name": "LoadBalancer", "Value": LB}], "Minimum",
     "destinos sanos (mínimo del intervalo)"),
    ("UnHealthyHostCount", [{"Name": "TargetGroup", "Value": TG},
                            {"Name": "LoadBalancer", "Value": LB}], "Maximum",
     "destinos NO sanos (máximo)"),
    ("RequestCount", [{"Name": "LoadBalancer", "Value": LB}], "Sum",
     "peticiones atendidas"),
    ("HTTPCode_Target_2XX_Count", [{"Name": "LoadBalancer", "Value": LB}], "Sum",
     "respuestas 2XX del destino"),
    ("HTTPCode_Target_3XX_Count", [{"Name": "LoadBalancer", "Value": LB}], "Sum",
     "respuestas 3XX del destino"),
    ("HTTPCode_Target_4XX_Count", [{"Name": "LoadBalancer", "Value": LB}], "Sum",
     "respuestas 4XX del destino"),
    ("HTTPCode_Target_5XX_Count", [{"Name": "LoadBalancer", "Value": LB}], "Sum",
     "respuestas 5XX del destino"),
    ("HTTPCode_ELB_5XX_Count", [{"Name": "LoadBalancer", "Value": LB}], "Sum",
     "errores del propio balanceador"),
    ("TargetResponseTime", [{"Name": "LoadBalancer", "Value": LB}], "Average",
     "latencia media de respuesta (s)"),
    ("TargetResponseTime", [{"Name": "LoadBalancer", "Value": LB}], "Maximum",
     "latencia máxima de respuesta (s)"),
]


def serie(cw, met, dims, stat, ini, fin) -> dict:
    pts, cursor = {}, ini
    while cursor < fin:
        tope = min(cursor + timedelta(hours=24), fin)
        r = cw.get_metric_statistics(
            Namespace="AWS/ApplicationELB", MetricName=met, Dimensions=dims,
            StartTime=cursor, EndTime=tope, Period=PASO, Statistics=[stat])
        for d in r["Datapoints"]:
            pts[d["Timestamp"].replace(second=0, microsecond=0)] = d[stat]
        cursor = tope
    return pts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--desde", default="2026-06-21T19:20:00Z")
    p.add_argument("--hasta", default="2026-06-22T19:20:00Z")
    a = p.parse_args()
    ini = datetime.fromisoformat(a.desde.replace("Z", "+00:00"))
    fin = datetime.fromisoformat(a.hasta.replace("Z", "+00:00"))
    esperados = int((fin - ini).total_seconds() // PASO)

    cw = boto3.client("cloudwatch", region_name="us-east-1")
    print("=== TC-19 — accesibilidad de los tableros (medida en el balanceador) ===")
    print(f"  ventana UTC   : {ini:%Y-%m-%d %H:%M} a {fin:%Y-%m-%d %H:%M}")
    print(f"  ventana local : {ini - timedelta(hours=3):%Y-%m-%d %H:%M} a "
          f"{fin - timedelta(hours=3):%Y-%m-%d %H:%M} (−03)")
    print(f"  intervalos de 5 min esperados: {esperados}\n")
    print(f"  {'métrica':<28}{'stat':<9}{'puntos':>7}{'suma/mín':>11}"
          f"{'máx':>11}  lectura")

    datos = {}
    for met, dims, stat, glosa in SONDAS:
        pts = serie(cw, met, dims, stat, ini, fin)
        datos[(met, stat)] = pts
        v = list(pts.values())
        izq = (sum(v) if stat == "Sum" else (min(v) if v else 0)) if v else 0
        der = max(v) if v else 0
        print(f"  {met:<28}{stat:<9}{len(pts):>7}{izq:>11.4g}{der:>11.4g}  {glosa}")

    sanos = datos[("HealthyHostCount", "Minimum")]
    insanos = datos[("UnHealthyHostCount", "Maximum")]
    c5 = datos[("HTTPCode_Target_5XX_Count", "Sum")]
    e5 = datos[("HTTPCode_ELB_5XX_Count", "Sum")]
    c2 = datos[("HTTPCode_Target_2XX_Count", "Sum")]
    req = datos[("RequestCount", "Sum")]

    sin_sano = [t for t, v in sanos.items() if v < 1]
    cob = 100.0 * len(sanos) / esperados if esperados else 0

    print("\n  --- lectura ---")
    print(f"  destinos sanos    : {len(sanos)}/{esperados} intervalos con métrica "
          f"({cob:.1f}%); intervalos con 0 destinos sanos: {len(sin_sano)}")
    print(f"  destinos NO sanos : máximo observado {max(insanos.values()) if insanos else 0:.0f}")
    print(f"  respuestas 5XX    : destino {sum(c5.values()):.0f} · "
          f"balanceador {sum(e5.values()):.0f}")
    print(f"  respuestas 2XX    : {sum(c2.values()):.0f} sobre "
          f"{sum(req.values()):.0f} peticiones")
    lat = datos[("TargetResponseTime", "Maximum")]
    if lat:
        print(f"  latencia          : media "
              f"{sum(datos[('TargetResponseTime','Average')].values())/max(1,len(datos[('TargetResponseTime','Average')])):.3f} s · "
              f"máx {max(lat.values()):.3f} s")

    ok = (cob >= 99.0 and not sin_sano
          and sum(c5.values()) == 0 and sum(e5.values()) == 0)
    print(f"\n  VEREDICTO accesibilidad: {'ACREDITADA' if ok else 'REVISAR'}")
    print("  (criterio: ≥1 destino sano en toda la ventana y ausencia de 5XX)")
    if not req:
        print("\n  Advertencia: sin peticiones registradas en la ventana. El destino")
        print("  estaba sano, pero nadie consultó los tableros: se acredita que el")
        print("  servicio estaba disponible, no que se lo haya usado.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

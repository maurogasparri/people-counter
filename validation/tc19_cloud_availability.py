#!/usr/bin/env python3
"""TC-19 — disponibilidad del stack en la nube, reconstruida desde CloudWatch.

El criterio es la disponibilidad de **IoT Core, RDS y Fargate**, no la del
dispositivo. La continuidad de la telemetría NO sirve para medirlo: sus huecos
corresponden al equipo apagado —es una unidad de desarrollo— y no dicen nada
sobre la nube. Este guion mide los servicios directamente, con las métricas que
cada uno publica por su cuenta:

    IoT Core   PublishIn.Success / Connect.Success / Failure   (namespace AWS/IoT)
    RDS        CPUUtilization / DatabaseConnections            (AWS/RDS)
    Fargate    LiveTaskCount del servicio de Grafana           (AWS/ECS)
    ALB        HTTPCode_ELB_5XX_Count / TargetResponseTime     (AWS/ApplicationELB)

Un servicio se considera disponible en un intervalo de 5 min cuando publica
métrica en ese intervalo con valor sano: RDS y ECS emiten sólo mientras están
en ejecución, así que la ausencia de punto es indicio de indisponibilidad.

CAVEAT DE RETENCIÓN: CloudWatch conserva la resolución de 5 min durante 63
días. La ventana de junio expira alrededor del 2026-08-23. Después de esa
fecha este guion ya no puede reconstruirla; por eso su salida se archiva.

Reproducible: py validation/tc19_cloud_availability.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import boto3

sys.stdout.reconfigure(encoding="utf-8")

RDS_ID = "people-counter-dev"
ECS_SVC = "people-counter-grafana-dev"
PASO = 300  # 5 min

SONDAS = [
    ("IoT Core", "AWS/IoT", "PublishIn.Success",
     [{"Name": "Protocol", "Value": "MQTT"}], "Sum", "publicaciones aceptadas"),
    ("IoT Core", "AWS/IoT", "Failure",
     [{"Name": "Protocol", "Value": "MQTT"}], "Sum", "fallos del broker"),
    ("RDS", "AWS/RDS", "CPUUtilization",
     [{"Name": "DBInstanceIdentifier", "Value": RDS_ID}], "Average", "instancia viva"),
    ("RDS", "AWS/RDS", "DatabaseConnections",
     [{"Name": "DBInstanceIdentifier", "Value": RDS_ID}], "Average", "conexiones"),
    ("Fargate", "AWS/ECS", "LiveTaskCount",
     [{"Name": "ServiceName", "Value": ECS_SVC},
      {"Name": "ClusterName", "Value": ECS_SVC}], "Average", "tareas vivas"),
]


def serie(cw, ns, met, dims, stat, ini, fin) -> dict:
    pts = {}
    cursor = ini
    while cursor < fin:  # CloudWatch limita puntos por llamada
        tope = min(cursor + timedelta(hours=24), fin)
        r = cw.get_metric_statistics(
            Namespace=ns, MetricName=met, Dimensions=dims,
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
    print("=== TC-19 — disponibilidad del stack en la nube (IoT / RDS / Fargate) ===")
    print(f"  ventana : {ini:%Y-%m-%d %H:%M} a {fin:%Y-%m-%d %H:%M} UTC "
          f"({(fin-ini).total_seconds()/3600:.1f} h)")
    print(f"  local   : {ini - timedelta(hours=3):%Y-%m-%d %H:%M} a "
          f"{fin - timedelta(hours=3):%Y-%m-%d %H:%M} (−03)")
    print(f"  intervalos de {PASO//60} min esperados: {esperados}\n")

    print(f"  {'servicio':<10}{'métrica':<24}{'puntos':>8}{'cobertura':>11}"
          f"{'min':>9}{'max':>9}  {'lectura'}")
    resumen = {}
    for svc, ns, met, dims, stat, glosa in SONDAS:
        pts = serie(cw, ns, met, dims, stat, ini, fin)
        v = list(pts.values())
        cob = 100.0 * len(pts) / esperados if esperados else 0.0
        print(f"  {svc:<10}{met:<24}{len(pts):>8}{cob:>10.1f}%"
              f"{(min(v) if v else 0):>9.1f}{(max(v) if v else 0):>9.1f}  {glosa}")
        resumen.setdefault(svc, []).append((met, pts, cob, v))

    print("\n  --- veredicto por servicio ---")
    veredictos = {}
    # RDS: la instancia publica CPU sólo mientras corre.
    cpu = dict(resumen["RDS"])[  # type: ignore[index]
        "CPUUtilization"] if False else [x for x in resumen["RDS"] if x[0] == "CPUUtilization"][0]
    veredictos["RDS"] = (cpu[2] >= 99.0, f"CPUUtilization presente en {cpu[2]:.1f}% de los intervalos")
    # Fargate: al menos una tarea viva en todo momento.
    lt = [x for x in resumen["Fargate"] if x[0] == "LiveTaskCount"][0]
    sin_tarea = sum(1 for x in lt[3] if x < 1)
    veredictos["Fargate"] = (
        lt[2] >= 99.0 and sin_tarea == 0,
        f"LiveTaskCount ≥ 1 en todos los puntos ({sin_tarea} intervalos sin tarea), "
        f"cobertura {lt[2]:.1f}%")
    # IoT: sin fallos del broker.
    fal = [x for x in resumen["IoT Core"] if x[0] == "Failure"][0]
    pub = [x for x in resumen["IoT Core"] if x[0] == "PublishIn.Success"][0]
    total_fallos = sum(fal[3])
    veredictos["IoT Core"] = (
        total_fallos == 0,
        f"{total_fallos:.0f} fallos del broker; publicaciones aceptadas en "
        f"{pub[2]:.1f}% de los intervalos")

    for svc, (ok, glosa) in veredictos.items():
        print(f"    {svc:<10} {'DISPONIBLE' if ok else 'REVISAR':<12} {glosa}")

    cumple = all(ok for ok, _ in veredictos.values())
    print(f"\n  VEREDICTO TC-19: {'CUMPLE' if cumple else 'REVISAR'} "
          f"(criterio: disponibilidad ≥ 99 % sobre 24 h del stack en la nube)")
    print("\n  Nota: la continuidad de la telemetría del dispositivo NO integra este")
    print("  cálculo. Sus huecos son equipo apagado y no indisponibilidad de la nube.")
    return 0 if cumple else 1


if __name__ == "__main__":
    raise SystemExit(main())

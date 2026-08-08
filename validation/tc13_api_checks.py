#!/usr/bin/env python3
"""TC-13 — control de acceso y validación de parámetros de la interfaz de consulta.

Cuatro invocaciones contra la interfaz de agregados:

  1. Petición bien formada                        -> 200 con agregados
  2. Falta el parámetro obligatorio ``from``      -> 400 RFC 7807
  3. Rango de 20 días con agrupamiento de 15 min  -> 400 RFC 7807 (máximo 7 d)
  4. Sin credenciales, por HTTP real              -> 403

Las tres primeras invocan la función directamente con el evento que le
entregaría la puerta de enlace, que es la forma de ejercitar la validación de
parámetros sin depender de credenciales firmadas. La cuarta es una petición
HTTP real contra el punto de entrada desplegado.

Escribe ``tc13_result.txt`` con las cuatro peticiones y sus respuestas.

Reproducible (requiere credenciales de AWS con permiso de invocación):

    py validation/tc13_api_checks.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import boto3  # noqa: E402

FUNCION = "people-counter-query-aggregates-dev"
REGION = "us-east-1"
URL_SIN_AUTH = "https://api.<tu-dominio>/pos/transactions"
DOMINIO_REAL = "tfg.gasparri.com.ar"
CORTE = 1800  # caracteres de cuerpo que se vuelcan al informe

BASE = {
    "version": "2.0",
    "rawPath": "/v1/aggregates",
    "routeKey": "GET /v1/aggregates",
    "requestContext": {"http": {"method": "GET", "path": "/v1/aggregates"}},
    "headers": {},
}

CASOS = [
    (
        "1 · Petición bien formada",
        "200 con agregados",
        {
            "from": "2026-06-15T00:00:00Z",
            "to": "2026-06-21T00:00:00Z",
            "store_id": "store-pilot-01",
        },
    ),
    (
        "2 · Falta el parámetro obligatorio `from`",
        "400 application/problem+json — missing-parameter",
        {"to": "2026-06-21T23:59:59Z"},
    ),
    (
        "3 · Rango de 20 días con agrupamiento de 15 min",
        "400 application/problem+json — range-too-large (máximo 7 d)",
        {
            "from": "2026-06-01T00:00:00Z",
            "to": "2026-06-21T00:00:00Z",
            "store_id": "store-pilot-01",
            "bucket": "15min",
        },
    ),
]


def _sanear(t: str) -> str:
    return t.replace(DOMINIO_REAL, "<tu-dominio>")


def _bloque(titulo: str, esperado: str, peticion: str, estado, cabeceras, cuerpo: str) -> str:
    cuerpo = _sanear(cuerpo)
    extra = ""
    if len(cuerpo) > CORTE:
        extra = (
            f"\n  [...] cuerpo completo: {len(cuerpo)} caracteres. "
            "Se vuelca el comienzo; el resto se reproduce corriendo este guion."
        )
        cuerpo = cuerpo[:CORTE]
    cab = json.dumps(cabeceras, ensure_ascii=False) if cabeceras else "(sin cabeceras)"
    return (
        f"\n{titulo}\n"
        + "-" * 74
        + f"\n  esperado : {esperado}\n"
        f"  petición : {_sanear(peticion)}\n"
        f"  estado   : {estado}\n"
        f"  cabeceras: {_sanear(cab)}\n"
        f"  cuerpo   : {cuerpo}{extra}\n"
    )


def main() -> None:
    partes = [
        "TC-13 — control de acceso y validación de parámetros",
        "Cuatro invocaciones contra la interfaz de consulta de agregados.",
        "Generado con validation/tc13_api_checks.py — la interpretación está en",
        "docs/benchmark_results.md §2.",
        "=" * 74,
    ]

    lam = boto3.client("lambda", region_name=REGION)
    for titulo, esperado, qs in CASOS:
        evento = dict(BASE, queryStringParameters=qs)
        r = lam.invoke(
            FunctionName=FUNCION,
            Payload=json.dumps(evento).encode("utf-8"),
        )
        resp = json.loads(r["Payload"].read().decode("utf-8"))
        partes.append(
            _bloque(
                titulo,
                esperado,
                json.dumps(qs, ensure_ascii=False),
                resp.get("statusCode"),
                resp.get("headers"),
                resp.get("body", ""),
            )
        )
        print(f"  {titulo}: {resp.get('statusCode')}")

    # 4 · sin credenciales, HTTP real
    req = urllib.request.Request(
        URL_SIN_AUTH.replace("<tu-dominio>", DOMINIO_REAL),
        data=b"{}",
        method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            estado, cuerpo = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        estado, cuerpo = e.code, e.read().decode("utf-8", "replace")
    partes.append(
        _bloque(
            "4 · Sin credenciales (petición HTTP real)",
            "403 Forbidden",
            f"POST {URL_SIN_AUTH} sin firma SigV4",
            estado,
            None,
            cuerpo,
        )
    )
    print(f"  4 · Sin credenciales: {estado}")

    destino = Path(__file__).resolve().parent / "tc13_result.txt"
    destino.write_text("\n".join(partes) + "\n", encoding="utf-8", newline="\n")
    print(f"\n  escrito: {destino.name}")


if __name__ == "__main__":
    main()

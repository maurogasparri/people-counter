#!/usr/bin/env python3
"""TC-01 a TC-08 — extrae de la base los eventos que sustentan cada caso.

Los casos TC-01 a TC-08 son cruces de personas bajo la cámara: su evidencia no
es la salida de un guion sino el **registro de eventos persistidos**. Este guion
lo extrae por ventana y escribe un archivo por caso, de modo que cada uno tenga
su respaldo localizable junto al resto de la evidencia.

No calcula veredictos ni tasas: sólo vuelca lo que la base contiene en la
ventana de cada ensayo, tal como se consultó para redactar el registro. La
interpretación está en `docs/benchmark_results.md` §2.

Reproducible (requiere credenciales de AWS con acceso a la base):

    py validation/tc01_08_field_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _raiz_repo() -> Path:
    for cand in list(Path(__file__).resolve().parents):
        if (cand / "scripts" / "provision.py").is_file():
            return cand
    raise SystemExit("no se encontró la raíz del repositorio (scripts/provision.py)")


RAIZ = _raiz_repo()
sys.path.insert(0, str(RAIZ))
from scripts.provision import _rds_connect  # noqa: E402

DEV = "store-pilot-01-cam-01"
TZ = "America/Argentina/Buenos_Aires"

# (archivo, título, [(subtítulo, desde_utc, hasta_utc)])
CASOS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        "tc01_02_result.txt",
        "TC-01 y TC-02 — conteo de ingreso y de egreso",
        [("Diez idas y vueltas del operador", "2026-06-25T11:11:27Z", "2026-06-25T11:14:15Z")],
    ),
    (
        "tc03_result.txt",
        "TC-03 — cruces simultáneos en direcciones opuestas",
        [
            ("Primera corrida", "2026-06-25T13:15:33Z", "2026-06-25T13:16:20Z"),
            ("Segunda corrida — con mayor separación lateral entre carriles (serie de cinco pares)", "2026-06-25T13:19:30Z", "2026-06-25T13:20:05Z"),
        ],
    ),
    (
        "tc04_result.txt",
        "TC-04 — ráfaga en el mismo sentido (tráfico real)",
        [
            ("Ráfaga de egreso", "2026-06-24T18:51:40Z", "2026-06-24T18:52:15Z"),
            ("Ráfaga de ingreso", "2026-06-24T18:57:54Z", "2026-06-24T18:58:12Z"),
        ],
    ),
    (
        "tc05_result.txt",
        "TC-05 — robustez a la variación de apariencia (con capucha)",
        [("Cinco idas y vueltas (serie completa)", "2026-06-25T11:23:56Z", "2026-06-25T11:25:16Z")],
    ),
    (
        "tc06_result.txt",
        "TC-06 — rechazo de objetos por debajo del umbral de altura",
        [("Ocho pasadas gateando", "2026-06-25T21:52:40Z", "2026-06-25T21:54:05Z")],
    ),
    (
        "tc07_result.txt",
        "TC-07 — hesitación: entrada a la zona sin cruzar la línea",
        [("Ocho aproximaciones", "2026-06-25T11:28:12Z", "2026-06-25T11:29:13Z")],
    ),
    (
        "tc08_result.txt",
        "TC-08 — estimación de estatura",
        [
            ("Tanda 1 — operador (reutiliza cruces de TC-01/TC-02)", "2026-06-25T11:11:50Z", "2026-06-25T11:12:20Z"),
            ("Tanda 2 — segundo sujeto", "2026-06-25T12:33:00Z", "2026-06-25T12:33:35Z"),
            ("Tanda 3 — operador", "2026-06-25T12:41:50Z", "2026-06-25T12:42:16Z"),
        ],
    ),
]

CAB = "  hora local   sentido  confianza  estatura"
SQL = (
    f"SELECT event_ts AT TIME ZONE '{TZ}', direction, confidence, height_m "
    "FROM count_events WHERE device_id = %s "
    "AND event_ts >= %s::timestamptz AND event_ts <= %s::timestamptz "
    "ORDER BY event_ts"
)


def main() -> None:
    cx = _rds_connect("people-counter-dev", "us-east-1")
    destino = Path(__file__).resolve().parent
    try:
        for archivo, titulo, ventanas in CASOS:
            lineas = [
                titulo,
                "Eventos persistidos en la base, por ventana de ensayo.",
                "Extraído con validation/tc01_08_field_events.py — la interpretación",
                "está en docs/benchmark_results.md §2.",
                "=" * 74,
            ]
            for sub, desde, hasta in ventanas:
                with cx.cursor() as cur:
                    cur.execute(SQL, (DEV, desde, hasta))
                    filas = cur.fetchall()
                ins = sum(1 for f in filas if f[1] == "in")
                lineas += [
                    "",
                    f"{sub}",
                    f"  ventana: {desde}  ->  {hasta}",
                    f"  eventos: {len(filas)}  ({ins} ingreso(s), {len(filas) - ins} egreso(s))",
                    "",
                    CAB,
                ]
                if not filas:
                    lineas.append("  (ninguno)")
                for ts, d, c, h in filas:
                    alt = f"{float(h):.2f}" if h is not None else "  -  "
                    lineas.append(
                        f"  {str(ts)[11:19]}   {d:<7}  {float(c):>8.3f}  {alt:>8}"
                    )
            texto = "\n".join(lineas) + "\n"
            (destino / archivo).write_text(texto, encoding="utf-8", newline="\n")
            n = sum(1 for line in lineas if line.startswith("  eventos:"))
            print(f"  {archivo:22} {len(texto.splitlines()):>4} lineas, {n} ventana(s)")
    finally:
        cx.close()


if __name__ == "__main__":
    main()

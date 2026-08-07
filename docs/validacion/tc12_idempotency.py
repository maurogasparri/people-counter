#!/usr/bin/env python3
"""TC-12 — idempotencia de la ingesta en la nube sobre 20 eventos duplicados.

El plan comprometía veinte eventos duplicados; la corrida del 21/06 ejecutó
UNA sola reinserción por tabla. Este guion corre las veinte, sobre filas
DISTINTAS, en las dos tablas con restricción de unicidad:

    count_events      UNIQUE (device_id, event_ts, track_id, direction)
    pos_transactions  UNIQUE (store_id, transaction_id)

Método: se eligen 20 filas existentes (determinista, por marca temporal) y se
reinsertan copiando todas sus columnas salvo la clave primaria subrogada —que
lleva un UUID nuevo, para que el rechazo lo produzca la restricción de
unicidad de negocio y no el choque de la primaria— y salvo las columnas
generadas, que la base recomputa.

Salvaguarda: todo corre dentro de una transacción. Se confirma SOLO si las 20
fueron descartadas. Si alguna se insertara —es decir, si la restricción no
actuara— se revierte, para no dejar duplicados en una base con datos.

Reproducible: py docs/validacion/tc12_idempotency.py
"""

from __future__ import annotations

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
    raise SystemExit("no se encontró la raíz del repositorio (scripts/provision.py)")


sys.path.insert(0, str(_raiz_repo()))
from scripts.provision import _rds_connect  # noqa: E402

N = 20
CASOS = [
    ("count_events", "event_id", "event_ts",
     ["device_id", "event_ts", "track_id", "direction"]),
    ("pos_transactions", "pos_id", "event_ts",
     ["store_id", "transaction_id"]),
]


def columnas(cur, tabla: str, pk: str) -> list[str]:
    """Columnas insertables: ni generadas ni la primaria subrogada."""
    cur.execute(
        """SELECT column_name FROM information_schema.columns
           WHERE table_schema='public' AND table_name=%s
             AND is_generated='NEVER' AND column_name <> %s
           ORDER BY ordinal_position""",
        (tabla, pk),
    )
    return [r[0] for r in cur.fetchall()]


def main() -> int:
    conn = _rds_connect("people-counter-dev", "us-east-1")
    conn.autocommit = False
    cur = conn.cursor()
    fallo_global = False

    print("=== TC-12 — idempotencia de ingesta sobre 20 duplicados por tabla ===\n")

    for tabla, pk, orden, unica in CASOS:
        cols = columnas(cur, tabla, pk)
        lista = ", ".join(cols)
        clave = ", ".join(unica)

        cur.execute(f"SELECT count(*) FROM {tabla}")
        antes = cur.fetchone()[0]

        cur.execute(f"SELECT {pk} FROM {tabla} ORDER BY {orden}, {pk} LIMIT {N}")
        ids = [r[0] for r in cur.fetchall()]

        print(f"--- {tabla}")
        print(f"    restricción de unicidad : ({clave})")
        print(f"    filas antes             : {antes}")
        print(f"    filas seleccionadas     : {len(ids)} (distintas, orden determinista)")

        descartadas = insertadas = 0
        for i, ident in enumerate(ids, 1):
            cur.execute(
                f"INSERT INTO {tabla} ({lista}) "
                f"SELECT {lista} FROM {tabla} WHERE {pk} = %s "
                f"ON CONFLICT ({clave}) DO NOTHING",
                (ident,),
            )
            if cur.rowcount == 0:
                descartadas += 1
            else:
                insertadas += 1
                print(f"      !! caso {i}: la restricción NO actuó "
                      f"(filas insertadas={cur.rowcount})")

        cur.execute(f"SELECT count(*) FROM {tabla}")
        despues = cur.fetchone()[0]

        ok = insertadas == 0 and antes == despues and descartadas == len(ids)
        print(f"    descartadas por unicidad: {descartadas}/{len(ids)}")
        print(f"    insertadas              : {insertadas}")
        print(f"    filas después           : {despues}  (delta {despues - antes})")
        print(f"    VEREDICTO {tabla}: {'CUMPLE' if ok else 'NO CUMPLE'}\n")
        fallo_global = fallo_global or not ok

    if fallo_global:
        conn.rollback()
        print("  TRANSACCIÓN REVERTIDA — alguna inserción prosperó; no se deja rastro.")
    else:
        conn.commit()
        print("  Transacción confirmada (no-op: no se insertó ninguna fila).")

    print(f"\n  VEREDICTO TC-12: {'NO CUMPLE' if fallo_global else 'CUMPLE'} "
          f"(criterio: los 20 duplicados descartados por la restricción)")
    conn.close()
    return 1 if fallo_global else 0


if __name__ == "__main__":
    raise SystemExit(main())

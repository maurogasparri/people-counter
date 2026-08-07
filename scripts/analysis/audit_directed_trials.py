#!/usr/bin/env python3
"""Auditoría documental de la traza TRACKDBG de una jornada de ensayos.

Reconstruye, a partir del log de aplicación del dispositivo, la secuencia de
visitas a la counting zone de un día concreto: entradas, cruces de línea con
su lado y balance neto, veredicto de salida, muertes de track, adopciones
desde el ghost pool y rechazos por guarda con su motivo.

**Naturaleza de la salida.** Esto es *evidencia documental* de ensayos ya
reportados, no una medición. La traza registra lo que el sistema decidió, no
lo que ocurrió frente a la cámara: no hay verdad de referencia asociada. Por
eso el script NO emite tasas ni porcentajes de acierto — solo reconstruye la
secuencia y marca qué visitas terminaron sin conteo, para que un humano las
interprete contra la bitácora del ensayo.

Uso:

    python scripts/analysis/audit_directed_trials.py \\
        --log debug/app.log --date 2026-06-25

Salida: reporte Markdown por consola (redirigible) y, opcionalmente, un CSV
por visita con ``--csv``.

Geometría: los valores por defecto (``--line-y``, ``--zone-*``) corresponden
al config vigente en el dispositivo el 2026-06-25 (respaldo
``config.yaml.bak.preL1tune2``, 09:58). El ROI se modificó después, así que
para auditar otra fecha hay que pasar la geometría de esa fecha.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --- Geometría del ensayo del 2026-06-25 (config.yaml.bak.preL1tune2) --------
DEFAULT_LINE_Y = 202
DEFAULT_ZONE = (344, 1004, 82, 322)  # x_min, x_max, y_min, y_max

# Momento a partir del cual el canary `ambiguous_reject_count` empezó a
# reportar ese día (primera muestra no nula en la tabla `telemetry` del
# backend). Antes de esa hora el rechazo por ratio-test NO queda registrado
# en ningún lado: ni la traza ni la telemetría lo capturan.
DEFAULT_ARC_AVAILABLE_FROM = "12:01:31"

RE_TIME = re.compile(r'"time":"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"')
RE_MSG = re.compile(r'"msg":"(.*?)"\}\s*$')

RE_ENTRY = re.compile(
    r"TRACKDBG entry tid=(\d+) pos=\((-?\d+),(-?\d+)\) "
    r"snap=\((-?\d+),(-?\d+)\) sides=\[([-\d, ]*)\] is_real=(\w+)"
)
RE_CROSS = re.compile(
    r"TRACKDBG cross tid=(\d+) line=(\d+) new_side=([+-]\d+) "
    r"net=([+-]?\d+) pos=\((-?\d+),(-?\d+)\) label=(\w+)"
)
RE_EXIT = re.compile(
    r"TRACKDBG exit tid=(\d+) is_real=(\w+) net=\[([^\]]*)\] verdict=(\S+) "
    r"exit_pos=\((-?\d+),(-?\d+)\)(?: real_inside_frames=(\d+))?"
)
RE_DEATH = re.compile(
    r"TRACKDBG death tid=(\d+) reason=(\S+) disappeared=(\d+) "
    r"inside_keepalive=(\w+) last_pos=\((-?\d+),(-?\d+)\)"
)
RE_GHOST = re.compile(
    r"TRACKDBG ghost_adopted tid=(\d+) dist=([\d.]+) iou=([\d.]+) age=(\d+)"
)
RE_SKIP = re.compile(r"TRACKDBG (\w*skipped) tid=(\d+) (.*)$")


@dataclass
class Visit:
    """Una estadía de un track dentro de la counting zone."""

    tid: int
    run: int
    t_entry: dt.datetime
    entry_pos: tuple[int, int]
    entry_side: int
    entry_is_real: bool
    crosses: list[dict[str, Any]] = field(default_factory=list)
    t_exit: Optional[dt.datetime] = None
    exit_pos: Optional[tuple[int, int]] = None
    exit_is_real: Optional[bool] = None
    exit_net: Optional[str] = None
    verdict: Optional[str] = None
    real_inside_frames: Optional[int] = None
    # Guardas que suprimieron la emisión (exit_*_skipped / death_emit_skipped).
    skips: list[tuple[str, str]] = field(default_factory=list)
    # `entry_kalman_skipped` NO suprime un conteo: marca frames en los que el
    # primer instante dentro de la zona fue predicción del Kalman y por eso no
    # se abrió ciclo de visita. Se contabiliza aparte para no confundirlo con
    # una guarda de emisión.
    entry_kalman_frames: int = 0
    adopted: bool = False

    @property
    def outcome(self) -> str:
        if self.skips:
            return "suprimida"
        if self.verdict and self.verdict != "None":
            return "contada"
        return "sin conteo"

    @property
    def reason(self) -> str:
        if self.skips:
            kinds: dict[str, int] = {}
            detail = ""
            for k, v in self.skips:
                kinds[k] = kinds.get(k, 0) + 1
                if not detail:
                    detail = v
            head = ", ".join(k if n == 1 else f"{k}×{n}" for k, n in kinds.items())
            return f"{head} ({detail})" if detail else head
        if self.verdict and self.verdict != "None":
            return ""
        if not self.crosses:
            return "no registró cruce de línea"
        return "balance neto cero (ida y vuelta)"

    def positions(self) -> list[tuple[int, int]]:
        pts = [self.entry_pos]
        pts += [(c["x"], c["y"]) for c in self.crosses]
        if self.exit_pos:
            pts.append(self.exit_pos)
        return pts


def parse_log(
    path: Path, date: str
) -> tuple[list[Visit], list[dt.datetime], dict[str, int]]:
    """Recorre el log y devuelve (visitas, reinicios, totales por tipo)."""
    visits: list[Visit] = []
    open_visits: dict[int, Visit] = {}
    restarts: list[dt.datetime] = []
    totals: dict[str, int] = {}
    adopted_pending: set[int] = set()
    run = 0
    needle = f'"time":"{date}'

    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if needle not in line:
                continue
            mt = RE_TIME.search(line)
            if not mt:
                continue
            ts = dt.datetime.strptime(mt.group(1), "%Y-%m-%d %H:%M:%S,%f")

            if "Arrancando people-counter" in line:
                restarts.append(ts)
                run += 1
                open_visits.clear()
                adopted_pending.clear()
                continue
            if "TRACKDBG" not in line:
                continue

            mm = RE_MSG.search(line)
            msg = mm.group(1) if mm else line
            kind = msg.split("TRACKDBG ", 1)[1].split(" ")[0]
            totals[kind] = totals.get(kind, 0) + 1

            m = RE_ENTRY.search(msg)
            if m:
                tid = int(m.group(1))
                prev = open_visits.pop(tid, None)
                if prev is not None:
                    visits.append(prev)
                sides = [int(s) for s in m.group(6).split(",") if s.strip()]
                v = Visit(
                    tid=tid,
                    run=run,
                    t_entry=ts,
                    entry_pos=(int(m.group(2)), int(m.group(3))),
                    entry_side=sides[0] if sides else 0,
                    entry_is_real=m.group(7) == "True",
                    adopted=tid in adopted_pending,
                )
                adopted_pending.discard(tid)
                open_visits[tid] = v
                continue

            m = RE_CROSS.search(msg)
            if m:
                tid = int(m.group(1))
                v = open_visits.get(tid)
                if v is not None:
                    v.crosses.append(
                        {
                            "t": ts,
                            "line": int(m.group(2)),
                            "new_side": int(m.group(3)),
                            "net": int(m.group(4)),
                            "x": int(m.group(5)),
                            "y": int(m.group(6)),
                            "label": m.group(7),
                        }
                    )
                continue

            m = RE_EXIT.search(msg)
            if m:
                tid = int(m.group(1))
                v = open_visits.get(tid)
                if v is not None:
                    v.t_exit = ts
                    v.exit_is_real = m.group(2) == "True"
                    v.exit_net = m.group(3)
                    v.verdict = m.group(4)
                    v.exit_pos = (int(m.group(5)), int(m.group(6)))
                    if m.group(7):
                        v.real_inside_frames = int(m.group(7))
                continue

            m = RE_SKIP.search(msg)
            if m:
                tid = int(m.group(2))
                v = open_visits.get(tid)
                if v is not None:
                    if m.group(1) == "entry_kalman_skipped":
                        v.entry_kalman_frames += 1
                    else:
                        v.skips.append((m.group(1), m.group(3).strip()))
                continue

            m = RE_GHOST.search(msg)
            if m:
                adopted_pending.add(int(m.group(1)))
                continue

            m = RE_DEATH.search(msg)
            if m:
                tid = int(m.group(1))
                v = open_visits.pop(tid, None)
                if v is not None:
                    visits.append(v)
                continue

    visits.extend(open_visits.values())
    visits.sort(key=lambda v: v.t_entry)
    return visits, restarts, totals


def _min_separation(a: Visit, b: Visit) -> Optional[float]:
    """Separación mínima observada entre dos visitas, en píxeles.

    Aproximada: la traza sólo registra posiciones en entrada, cruce y salida,
    no la trayectoria completa. Subestima el acercamiento real sólo si el
    cruce más próximo cayó entre dos puntos registrados.
    """
    pa, pb = a.positions(), b.positions()
    if not pa or not pb:
        return None
    return min(
        ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 for x1, y1 in pa for x2, y2 in pb
    )


def _overlaps(a: Visit, b: Visit) -> bool:
    ae = a.t_exit or a.t_entry
    be = b.t_exit or b.t_entry
    return a.t_entry <= be and b.t_entry <= ae


def find_bursts(visits: list[Visit], gap_s: float, min_size: int) -> list[list[Visit]]:
    """Agrupa visitas en ráfagas separadas por huecos > ``gap_s`` segundos."""
    bursts: list[list[Visit]] = []
    cur: list[Visit] = []
    for v in visits:
        if cur and (v.t_entry - cur[-1].t_entry).total_seconds() > gap_s:
            if len(cur) >= min_size:
                bursts.append(cur)
            cur = []
        cur.append(v)
    if len(cur) >= min_size:
        bursts.append(cur)
    return bursts


def find_converging_pairs(
    visits: list[Visit], max_sep_px: float
) -> list[tuple[Visit, Visit, float]]:
    """Pares solapados en el tiempo, de lados opuestos y espacialmente próximos.

    Ésta es la firma del escenario TC-03: dos personas que se cruzan en
    sentidos opuestos lo bastante cerca como para que el asociador tenga que
    desambiguarlas. Dos cruces simultáneos pero separados en el encuadre NO
    ejercen presión sobre el asociador.
    """
    out = []
    for i, a in enumerate(visits):
        for b in visits[i + 1 :]:
            if (b.t_entry - a.t_entry).total_seconds() > 5.0:
                break
            if a.tid == b.tid or not _overlaps(a, b):
                continue
            if a.entry_side == b.entry_side:
                continue
            sep = _min_separation(a, b)
            if sep is not None and sep <= max_sep_px:
                out.append((a, b, sep))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--log", type=Path, default=Path("debug/app.log"))
    p.add_argument("--date", default="2026-06-25")
    p.add_argument("--line-y", type=int, default=DEFAULT_LINE_Y)
    p.add_argument("--zone", type=int, nargs=4, default=list(DEFAULT_ZONE))
    p.add_argument("--burst-gap", type=float, default=10.0)
    p.add_argument("--burst-min", type=int, default=4)
    p.add_argument("--converge-px", type=float, default=250.0)
    p.add_argument("--arc-available-from", default=DEFAULT_ARC_AVAILABLE_FROM)
    p.add_argument("--csv", type=Path, help="Volcado por visita")
    args = p.parse_args()

    # El reporte lleva acentos y símbolos matemáticos; la consola de Windows
    # default-ea a cp1252 y rompe al redirigir. Forzamos UTF-8 en la salida.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.log.exists():
        sys.exit(f"No encuentro el log: {args.log}")

    visits, restarts, totals = parse_log(args.log, args.date)
    if not visits:
        sys.exit(f"Sin líneas TRACKDBG para {args.date} en {args.log}")

    counted = [v for v in visits if v.outcome == "contada"]
    nocount = [v for v in visits if v.outcome == "sin conteo"]
    supp = [v for v in visits if v.outcome == "suprimida"]

    print(f"# Auditoría de la traza — {args.date}\n")
    print(
        "> Evidencia documental de ensayos ya reportados. La traza registra lo\n"
        "> que el sistema decidió, no lo que ocurrió frente a la cámara: **no\n"
        "> tiene verdad de referencia** y de ella no se derivan tasas.\n"
    )
    print("## Cobertura y contexto\n")
    print(f"- Log: `{args.log}`")
    print(f"- Primera línea TRACKDBG: {visits[0].t_entry:%H:%M:%S}")
    print(
        f"- Última línea TRACKDBG: "
        f"{max(v.t_exit or v.t_entry for v in visits):%H:%M:%S}"
    )
    print(f"- Reinicios del servicio en la jornada: {len(restarts)}")
    if restarts:
        print("  (" + ", ".join(f"{r:%H:%M:%S}" for r in restarts) + ")")
    zx0, zx1, zy0, zy1 = args.zone
    print(f"- Counting zone: x {zx0}–{zx1}, y {zy0}–{zy1}; línea y={args.line_y}")
    print(f"- Visitas a la zona reconstruidas: {len(visits)}")
    print(
        f"  - con conteo: {len(counted)} · sin conteo: {len(nocount)}"
        f" · suprimidas por guarda: {len(supp)}"
    )
    print(
        "- Líneas TRACKDBG por tipo: "
        + ", ".join(f"{k}={v}" for k, v in totals.items())
    )
    print(
        f"\n- **Canary `ambiguous_reject_count`**: sólo reporta desde las "
        f"{args.arc_available_from} de esa jornada. Cualquier rechazo del "
        f"ratio-test anterior a esa hora no quedó registrado en ninguna "
        f"fuente.\n"
    )

    pairs = find_converging_pairs(visits, args.converge_px)
    print("## Pares convergentes en sentidos opuestos\n")
    print(
        "Visitas solapadas en el tiempo, de lados opuestos de la línea y con "
        f"separación mínima observada ≤ {args.converge_px:.0f} px — la "
        "condición que somete al asociador a desambiguar.\n"
    )
    if not pairs:
        print(
            "**Ninguno.** En toda la traza de la jornada no hay dos visitas "
            "que cumplan simultáneamente las tres condiciones.\n"
        )
    else:
        arc_from = dt.datetime.strptime(args.arc_available_from, "%H:%M:%S").time()
        print(
            "| Hora | tid A | tid B | separación mín. | resultado A | "
            "resultado B | canary de ambigüedad activo |"
        )
        print("|---|---:|---:|---:|---|---|---|")
        n_covered = 0
        for a, b, sep in pairs:
            covered = a.t_entry.time() >= arc_from
            n_covered += covered
            print(
                f"| {a.t_entry:%H:%M:%S} | {a.tid} | {b.tid} | {sep:.0f} px "
                f"| {a.outcome} | {b.outcome} | {'sí' if covered else 'NO'} |"
            )
        print(
            f"\nDe {len(pairs)} pares convergentes, {n_covered} ocurrieron con "
            f"el canary `ambiguous_reject_count` ya reportando y "
            f"{len(pairs) - n_covered} antes de que empezara a hacerlo.\n"
        )

    bursts = find_bursts(visits, args.burst_gap, args.burst_min)
    print(
        f"## Ráfagas de actividad (≥ {args.burst_min} visitas, hueco ≤ "
        f"{args.burst_gap:.0f} s)\n"
    )
    print(
        "Una ráfaga es candidata a ser un ensayo dirigido. **La traza no "
        "contiene marca de ensayo**: la correspondencia con un código TC "
        "concreto no puede establecerse desde el registro.\n"
    )
    for bi, burst in enumerate(bursts, 1):
        t0, t1 = burst[0].t_entry, max(v.t_exit or v.t_entry for v in burst)
        nb_c = sum(1 for v in burst if v.outcome == "contada")
        print(
            f"### Ráfaga {bi} — {t0:%H:%M:%S}–{t1:%H:%M:%S} "
            f"({len(burst)} visitas, {nb_c} con conteo)\n"
        )
        print(
            "| Hora | tid | lado entr. | entr. real | cruces | neto | "
            "veredicto | frames reales | resultado | observación |"
        )
        print("|---|---:|---:|---|---|---|---|---:|---|---|")
        for v in burst:
            cr = (
                " → ".join(f"{c['label']}({c['net']:+d})" for c in v.crosses)
                if v.crosses
                else "—"
            )
            print(
                f"| {v.t_entry:%H:%M:%S} | {v.tid} | {v.entry_side:+d} | "
                f"{'sí' if v.entry_is_real else 'no'} | {cr} | "
                f"{v.exit_net or '—'} | {v.verdict or '—'} | "
                f"{v.real_inside_frames if v.real_inside_frames is not None else '—'} | "
                f"{v.outcome} | {v.reason or ''} |"
            )
        print()

    print("## Visitas sin conteo\n")
    print(
        "Cada fila es una estadía en la counting zone que no produjo evento. "
        "Sin verdad de referencia no puede decidirse si corresponde a un "
        "cruce real perdido o a una aproximación que legítimamente no debía "
        "contar.\n"
    )
    print("| Hora | tid | entr. → salida | frames reales | motivo |")
    print("|---|---:|---|---:|---|")
    for v in nocount + supp:
        ex = f"({v.exit_pos[0]},{v.exit_pos[1]})" if v.exit_pos else "—"
        print(
            f"| {v.t_entry:%H:%M:%S} | {v.tid} | "
            f"({v.entry_pos[0]},{v.entry_pos[1]}) → {ex} | "
            f"{v.real_inside_frames if v.real_inside_frames is not None else '—'} | "
            f"{v.reason} |"
        )
    print()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "t_entry",
                    "run",
                    "tid",
                    "entry_x",
                    "entry_y",
                    "entry_side",
                    "entry_is_real",
                    "n_crosses",
                    "cross_labels",
                    "exit_net",
                    "verdict",
                    "real_inside_frames",
                    "outcome",
                    "reason",
                    "adopted",
                ]
            )
            for v in visits:
                w.writerow(
                    [
                        v.t_entry.isoformat(),
                        v.run,
                        v.tid,
                        v.entry_pos[0],
                        v.entry_pos[1],
                        v.entry_side,
                        v.entry_is_real,
                        len(v.crosses),
                        "|".join(c["label"] for c in v.crosses),
                        v.exit_net or "",
                        v.verdict or "",
                        (
                            v.real_inside_frames
                            if v.real_inside_frames is not None
                            else ""
                        ),
                        v.outcome,
                        v.reason,
                        v.adopted,
                    ]
                )
        print(f"CSV escrito en {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

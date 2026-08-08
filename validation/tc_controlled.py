#!/usr/bin/env python3
"""Orquestador de los casos de prueba CONTROLADOS TC-01…07 (Grupo B / B1).

Para cada TC toma la ventana temporal de la tanda (--last N | --since ISO
[--until ISO]), aplica el criterio de CONTEO ABSOLUTO k/n (Tabla 13 del TFG) y
emite PASS/FAIL + detalle por evento + k/n exacto. Reusa la conexión RDS de
``count_session.py`` (no la duplica). Es READ-ONLY (solo SELECT).

Criterios (Tabla 13, k/n porque a n≈10 el % no es medible):
  TC-01 ingreso        10 cruces   PASS ≥9/10 dirección+altura correctas
  TC-02 egreso         10 cruces   PASS ≥9/10 dirección correcta
  TC-03 multidirección 10 (5in+5out) PASS ≥9/10 dir. correcta y SIN fusión
  TC-04 rechazo <1m     8 reps     PASS 0/8 conteos (rechazado por gate altura)
  TC-05 estatura       12 medic.   PASS ≥11/12 dentro de ±10cm (≥2 sujetos)
  TC-06 hesitación      8 amagos   PASS 0/8 eventos netos espurios
  TC-07 apariencia     10 cruces   PASS ≥9/10 sin omisión ni doble conteo

Uso:
  py tc_controlled.py TC-01 --last 3
  py tc_controlled.py TC-01 --since 2026-06-25T13:30:00Z --until 2026-06-25T13:34:00Z
  py tc_controlled.py TC-03 --last 3 --in 5 --out 5
  py tc_controlled.py TC-04 --last 3 --pi-host <ip-del-dispositivo>   # + DEBUG on en la Pi
  py tc_controlled.py TC-05                                     # lee height_mae.csv
  py tc_controlled.py report                                    # tabla markdown Tabla 14

Cada corrida acumula un row en docs/benchmarks/<fecha>/tc_controlled_results.csv.
"""
import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_session import DEV, connect, fetch  # reusa la conexión RDS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HEIGHT_CSV = Path(__file__).resolve().parent / "height_mae.csv"

TCS = {
    "TC-01": dict(
        desc="ingreso",
        mode="count",
        exp_in=10,
        exp_out=0,
        n=10,
        pass_k=9,
        crit="≥9/10 dirección+altura correctas",
    ),
    "TC-02": dict(
        desc="egreso",
        mode="count",
        exp_in=0,
        exp_out=10,
        n=10,
        pass_k=9,
        crit="≥9/10 dirección correcta",
    ),
    "TC-03": dict(
        desc="multidirección",
        mode="count_nofuse",
        exp_in=5,
        exp_out=5,
        n=10,
        pass_k=9,
        crit="≥9/10 dir. correcta y SIN fusión de identidades",
    ),
    "TC-04": dict(
        desc="rechazo <1m",
        mode="reject_height",
        n=8,
        pass_k=0,
        crit="0/8 conteos (detección rechazada por gate de altura)",
    ),
    "TC-05": dict(
        desc="estatura",
        mode="height",
        n=12,
        pass_k=11,
        crit="≥11/12 dentro de ±10cm (≥2 sujetos)",
    ),
    "TC-06": dict(
        desc="hesitación",
        mode="reject",
        n=8,
        pass_k=0,
        crit="0/8 eventos netos espurios",
    ),
    "TC-07": dict(
        desc="apariencia",
        mode="count",
        exp_in=None,
        exp_out=None,
        n=10,
        pass_k=9,
        crit="≥9/10 sin omisión ni doble conteo",
    ),
}


def results_csv(date):
    d = REPO / "docs" / "benchmarks" / date
    d.mkdir(parents=True, exist_ok=True)
    return d / "tc_controlled_results.csv"


def save_result(date, tc, n, k, verdict, observado, window):
    path = results_csv(date)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                ["tc", "desc", "criterio", "n", "k", "verdict", "observado", "window"]
            )
        w.writerow(
            [tc, TCS[tc]["desc"], TCS[tc]["crit"], n, k, verdict, observado, window]
        )
    print(f"\n→ guardado en {path}")


def window_clause(args):
    """Devuelve (where_sql, params, label_humano) para fetch()."""
    if args.last is not None:
        return (
            "event_ts >= now() - make_interval(mins => %s)",
            (args.last,),
            f"últimos {args.last} min",
        )
    if args.since:
        if args.until:
            return (
                "event_ts >= %s::timestamptz AND event_ts < %s::timestamptz",
                (args.since, args.until),
                f"{args.since}..{args.until}",
            )
        return ("event_ts >= %s::timestamptz", (args.since,), f"desde {args.since}")
    sys.exit("Falta la ventana: --last N  o  --since ISO [--until ISO]")


def fetch_rows(args):
    where, params, label = window_clause(args)
    cx = connect()
    cur = cx.cursor()
    try:
        rows = fetch(cur, where, params)
    finally:
        cx.close()
    return rows, label


def print_events(rows):
    if not rows:
        print("  (sin eventos en la ventana)")
        return
    print(f"\n  {'ts':<19} {'dir':<4} {'track':<6} {'height_m':<9} conf")
    print("  " + "-" * 50)
    for ts, d, tid, h, c in rows:
        hs = f"{h:.2f}" if h is not None else "—"
        cs = f"{c:.2f}" if c is not None else "—"
        print(f"  {ts:%Y-%m-%d %H:%M:%S} {d:<4} {str(tid):<6} {hs:<9} {cs}")


# -------------------------- evaluadores por modo --------------------------


def eval_count(tc, rows, exp_in, exp_out, check_fusion=False, combined=False):
    ci = sum(1 for r in rows if r[1] == "in")
    co = sum(1 for r in rows if r[1] == "out")
    matched = min(ci, exp_in) + min(co, exp_out)
    missed = (exp_in - min(ci, exp_in)) + (exp_out - min(co, exp_out))
    # En sesión combinada (TC-01+TC-02 a la vez, round-trips) los eventos de la
    # dirección con expected=0 son del OTRO TC, no espurios → se ignoran acá.
    if combined:
        extra = (max(0, ci - exp_in) if exp_in else 0) + (
            max(0, co - exp_out) if exp_out else 0
        )
        ignored = (co if not exp_out else 0) + (ci if not exp_in else 0)
    else:
        extra = max(0, ci - exp_in) + max(0, co - exp_out)
        ignored = 0
    n = exp_in + exp_out
    spec = TCS[tc]
    print(f"\n  Esperado: IN={exp_in} OUT={exp_out}  ·  Contado: IN={ci} OUT={co}")
    if combined and ignored:
        print(
            f"  (sesión combinada: {ignored} eventos en la otra dirección → cuentan "
            f"para el otro TC, ignorados acá)"
        )
    print(
        f"  Correctos (k): {matched}/{n}  ·  omisiones: {missed}  ·  extra/dobles: {extra}"
    )
    ok = matched >= spec["pass_k"] and extra == 0
    obs = f"k={matched}/{n} (IN {ci}/{exp_in}, OUT {co}/{exp_out}, dobles={extra})"
    if combined:
        obs += " [combinada]"
    if check_fusion:
        tracks = [r[2] for r in rows if r[2] is not None]
        distinct = len(set(tracks))
        fused = distinct < len(rows)  # dos cruces compartiendo track = fusión
        print(
            f"  Identidades: {distinct} tracks distintos / {len(rows)} eventos"
            f"  → {'⚠ POSIBLE FUSIÓN' if fused else 'sin fusión'}"
        )
        ok = ok and not fused
        obs += f", tracks={distinct}/{len(rows)}{' FUSIÓN' if fused else ''}"
    return matched, n, ("PASS" if ok else "FAIL"), obs


def eval_reject(tc, rows):
    """TC-06: amagos sin cruce → 0 eventos. Un round-trip real (cruza línea
    ida y vuelta) SÍ cuenta 1 in + 1 out; un amago NO debe dejar ningún evento."""
    ci = sum(1 for r in rows if r[1] == "in")
    co = sum(1 for r in rows if r[1] == "out")
    spurious = len(rows)
    net = ci - co
    n = TCS[tc]["n"]
    print(f"\n  Eventos en la ventana: {spurious} (IN={ci} OUT={co}, neto={net:+d})")
    print(f"  Esperado para {n} amagos: 0 eventos (los amagos no cruzan la línea)")
    if spurious:
        print(
            "  ⚠ Hay eventos → o fueron round-trips reales (cruzaron) o conteos espurios."
            " Revisá el detalle: para TC-06 los amagos NO deben cruzar."
        )
    ok = spurious == 0
    obs = f"espurios={spurious}/{n} (neto={net:+d})"
    return spurious, n, ("PASS" if ok else "FAIL"), obs


def _ssh(host, cmd):
    """SSH no-interactivo (no cuelga en prompts de host-key/password). Devuelve
    el CompletedProcess, o None si falla/timeoutea (degrada graceful)."""
    try:
        return subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=8",
                host,
                cmd,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:  # noqa: BLE001
        return None


def eval_reject_height(tc, rows, args):
    """TC-04: 0 conteos + evidencia de que el detector vio algo <1m y el GATE de
    altura lo rechazó (no que simplemente no se detectó nada). La evidencia son
    los logs TRACKDBG ``exit_short_height_skipped`` / ``death_emit_skipped
    reason=short_height`` (DEBUG → la Pi debe correr con log_level DEBUG en TC-04).
    """
    spurious = len(rows)
    n = TCS[tc]["n"]
    print(
        f"\n  Conteos en la ventana: {spurious}  (esperado 0 para {n} reps de objeto <1m)"
    )
    gate_rej = None
    minh = None
    if args.pi_host:
        # min_count_height_m == 1.0 ?
        r = _ssh(
            args.pi_host, "grep -E 'min_count_height_m' /etc/people-counter/config.yaml"
        )
        minh = r.stdout.strip() if r else None
        if minh is None:
            print("  ⚠ SSH falló (no pude leer config/journal de la Pi).")
        print(f"  Config: {minh or '(no leído)'}")
        # Evidencia de rechazo por gate en la ventana
        if args.last is not None:
            tspec = f"--since '{args.last} min ago'"
        elif args.since:
            s = _iso_epoch(args.since)
            u = _iso_epoch(args.until) if args.until else None
            tspec = f"--since @{s}" + (f" --until @{u}" if u else "")
        else:
            tspec = ""
        grep = (
            "sudo journalctl -u people-counter " + tspec + " --no-pager | "
            "grep -cE 'exit_short_height_skipped|death_emit_skipped tid=[0-9]+ "
            "reason=short_height'"
        )
        rr = _ssh(args.pi_host, grep)
        try:
            gate_rej = int(rr.stdout.strip() or "0") if rr else None
        except ValueError:
            gate_rej = 0
        print(f"  Rechazos por gate de altura (TRACKDBG) en la ventana: {gate_rej}")
        if gate_rej == 0:
            print(
                "  ⚠ 0 rechazos logueados → ¿DEBUG apagado o nada detectó? "
                "Habilitá log_level: DEBUG en la Pi y reintentá TC-04."
            )
    else:
        print(
            "  ⚠ Sin --pi-host: no puedo confirmar el rechazo por gate (solo el 0/8 de RDS)."
        )
    ok = spurious == 0 and (gate_rej is not None and gate_rej >= 1)
    minh_ok = minh is not None and "1.0" in minh
    if args.pi_host and not minh_ok:
        print("  ⚠ min_count_height_m no parece 1.0 — TC-04 depende de ese gate.")
        ok = False
    obs = (
        f"conteos={spurious}/{n}, gate_rechazos={gate_rej if gate_rej is not None else 'n/v'}, "
        f"min_height={'1.0' if minh_ok else '?'}"
    )
    return spurious, n, ("PASS" if ok else "FAIL"), obs


def _iso_epoch(iso):
    return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def eval_height(tc):
    """TC-05: lee height_mae.csv (poblado con height_mae.py por sujeto), cuenta
    cuántas mediciones caen dentro de ±10cm → k/n, y exige ≥2 sujetos."""
    if not HEIGHT_CSV.exists():
        sys.exit(f"No hay {HEIGHT_CSV} — corré primero height_mae.py por cada sujeto.")
    rows = list(csv.DictReader(open(HEIGHT_CSV, encoding="utf-8")))
    if not rows:
        sys.exit("height_mae.csv vacío.")
    subjects = sorted({r["label"] for r in rows})
    within = sum(1 for r in rows if abs(float(r["est_m"]) - float(r["true_m"])) <= 0.10)
    nn = len(rows)
    print(f"\n  Mediciones: {nn}  ·  sujetos: {len(subjects)} ({', '.join(subjects)})")
    print(f"  Dentro de ±10cm (k): {within}/{nn}")
    for s in subjects:
        sub = [r for r in rows if r["label"] == s]
        w = sum(1 for r in sub if abs(float(r["est_m"]) - float(r["true_m"])) <= 0.10)
        print(
            f"    {s:<12} {w}/{len(sub)} dentro de ±10cm  (real {float(sub[0]['true_m']):.2f}m)"
        )
    ok = within >= TCS[tc]["pass_k"] and len(subjects) >= 2 and nn >= TCS[tc]["n"]
    if len(subjects) < 2:
        print("  ⚠ TC-05 requiere ≥2 sujetos de altura conocida.")
    obs = f"k={within}/{nn} dentro ±10cm, sujetos={len(subjects)}"
    return within, nn, ("PASS" if ok else "FAIL"), obs


def cmd_report(date):
    path = results_csv(date)
    if not path.exists():
        sys.exit(f"No hay {path} — corré primero los TC.")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    last = {r["tc"]: r for r in rows}  # última corrida por TC
    print(f"\n## Grupo B — controlado (TC-01…07) — {date}\n")
    print("| Caso | Criterio | Observado | Veredicto |")
    print("|------|----------|-----------|-----------|")
    for tc in sorted(TCS):
        r = last.get(tc)
        if r:
            v = {
                "PASS": "✅ PASS",
                "LIMITACION": "⚠️ LIMITACIÓN DOCUMENTADA",
            }.get(r["verdict"], "❌ FAIL")
            print(f"| {tc} {r['desc']} | {r['criterio']} | {r['observado']} | {v} |")
        else:
            print(f"| {tc} {TCS[tc]['desc']} | {TCS[tc]['crit']} | (sin correr) | ⏳ |")
    print("\nPegar en la Tabla 14 del TFG. Detalle crudo en tc_controlled_results.csv.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tc", help="TC-01..TC-07 o 'report'")
    ap.add_argument("--last", type=int, help="ventana: últimos N minutos")
    ap.add_argument("--since", help="ventana: timestamp ISO de inicio")
    ap.add_argument("--until", help="ventana: timestamp ISO de fin")
    ap.add_argument("--in", dest="exp_in", type=int, help="override de IN esperados")
    ap.add_argument("--out", dest="exp_out", type=int, help="override de OUT esperados")
    ap.add_argument("--pi-host", help="host SSH de la Pi (TC-04: evidencia de gate)")
    ap.add_argument(
        "--combined",
        action="store_true",
        help="sesión TC-01+TC-02 juntos (round-trips): evalúa solo la dirección del "
        "TC e ignora la otra (es del otro TC). Correr TC-01 y TC-02 sobre la misma ventana.",
    )
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    args = ap.parse_args()

    if args.tc == "report":
        cmd_report(args.date)
        return

    tc = args.tc.upper()
    if tc not in TCS:
        sys.exit(f"TC desconocido: {tc}. Opciones: {', '.join(sorted(TCS))} o report.")
    spec = TCS[tc]
    print(f"=== {tc} — {spec['desc']} ===\n  Criterio: {spec['crit']}")

    mode = spec["mode"]
    if mode == "height":
        k, n, verdict, obs = eval_height(tc)
        window = "height_mae.csv"
    else:
        rows, window = fetch_rows(args)
        print(f"  Ventana: {window} · device {DEV}")
        print_events(rows)
        if mode in ("count", "count_nofuse"):
            exp_in = args.exp_in if args.exp_in is not None else spec["exp_in"]
            exp_out = args.exp_out if args.exp_out is not None else spec["exp_out"]
            if exp_in is None or exp_out is None:
                sys.exit(
                    f"{tc} requiere --in y --out (mezcla de dirección la define el operador)."
                )
            k, n, verdict, obs = eval_count(
                tc,
                rows,
                exp_in,
                exp_out,
                check_fusion=(mode == "count_nofuse"),
                combined=args.combined,
            )
        elif mode == "reject":
            k, n, verdict, obs = eval_reject(tc, rows)
        elif mode == "reject_height":
            k, n, verdict, obs = eval_reject_height(tc, rows, args)
        else:
            sys.exit(f"modo desconocido: {mode}")

    print(f"\n  VEREDICTO {tc}: {verdict}")
    save_result(args.date, tc, n, k, verdict, obs, window)


if __name__ == "__main__":
    main()

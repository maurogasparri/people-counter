#!/usr/bin/env python3
"""Orquestador único de benchmarks / casos de prueba.

ORQUESTA los scripts existentes (no reimplementa lógica): corre cada bloque,
captura su crudo en ``docs/benchmarks/<fecha>/raw/`` y emite un reporte
consolidado. Acepta subconjuntos con ``--group`` y marca como *skipped* (con
razón) los que requieren intervención física (TC-13) o montaje cenital
(TC-01…07). Registra commit + fecha + entorno. Idempotente.

Uso:
    py scripts/run_benchmarks.py --list
    py scripts/run_benchmarks.py --group tests
    py scripts/run_benchmarks.py --group tests cloud
    py scripts/run_benchmarks.py            # = tests + cloud (los locales)
    py scripts/run_benchmarks.py --group sync hardware --pi-host people-counter.local

Grupos:
    tests     — suites + cobertura + TC sintéticos (stitching, buffer)   [local]
    cloud     — e2e RDS, privacidad/idempotencia, TC-18, costo, seguridad [local/AWS]
    sync      — re-test de sincronización estéreo                         [SSH a la Pi]
    hardware  — potencia / soak / perfil                                  [SSH a la Pi, largo]
    skipped   — TC-13 (físico) y TC-01…07 (cenital): sólo se listan
"""
import argparse
import datetime
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

# Consola en UTF-8 (Windows default cp1252 rompe con — / ▶ en los prints).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def bench_dir(date: str) -> Path:
    return REPO / "docs" / "benchmarks" / date


def blocks(date: str, pi_host: str):
    """Registro de bloques. mode: local | ssh | skip."""
    d = f"docs/benchmarks/{date}"
    return [
        # --- tests (local) ---
        (
            "coverage",
            "tests",
            "local",
            [
                PY,
                "-m",
                "pytest",
                "--cov=src",
                "--cov-report=term-missing:skip-covered",
                "-q",
            ],
            None,
        ),
        (
            "component-suites",
            "tests",
            "local",
            [
                PY,
                "-m",
                "pytest",
                "tests/wifi_ble/test_dedup.py",
                "tests/mqtt/test_buffer.py",
                "tests/cloud/test_ingest_pos_transaction.py",
                "tests/cloud/test_query_aggregates.py",
                "-q",
            ],
            None,
        ),
        (
            "TC-08_09-stitching",
            "tests",
            "local",
            [PY, f"{d}/tc08_09_stitching.py"],
            None,
        ),
        ("TC-11-buffer-breve", "tests", "local", [PY, f"{d}/tc11_brief.py"], None),
        ("TC-12-buffer-72h", "tests", "local", [PY, f"{d}/tc12_buffer_72h.py"], None),
        # --- cloud (local + AWS, solo lectura) ---
        ("A6-e2e-latency", "cloud", "local", [PY, f"{d}/query_e2e_live.py"], None),
        ("TC-14_17-privacidad-pos", "cloud", "local", [PY, f"{d}/query_tc.py"], None),
        ("TC-18-noauth", "cloud", "local", [PY, f"{d}/tc18_noauth.py"], None),
        ("cloud-cost", "cloud", "local", [PY, f"{d}/cloud_cost.py"], None),
        # --- sync (SSH a la Pi) ---
        (
            "sync-camsync-v2",
            "sync",
            "ssh",
            f"cd /usr/src/people-counter && python3 ~/bench/{date}/camsync_v2.py",
            None,
        ),
        # --- hardware (SSH a la Pi, largo/disruptivo) ---
        (
            "power-running",
            "hardware",
            "ssh",
            "python3 /usr/src/people-counter/scripts/measure_power.py "
            "--duration 60 --interval 0.5 --rails",
            None,
        ),
        # --- skipped (no automatizable) ---
        (
            "TC-13-corte-energia",
            "skipped",
            "skip",
            None,
            "corte físico real del riel PoE — requiere intervención humana",
        ),
        (
            "TC-01..07-conteo",
            "skipped",
            "skip",
            None,
            "conteo controlado — requiere montaje cenital + cruces de personas",
        ),
    ]


def run_block(name, mode, cmd, reason, raw_dir, pi_host):
    raw = raw_dir / f"{name}.txt"
    if mode == "skip":
        raw.write_text(f"SKIPPED: {reason}\n", encoding="utf-8")
        return "skipped", reason
    if mode == "ssh":
        full = ["ssh", pi_host, cmd]
    else:
        full = cmd
    try:
        p = subprocess.run(full, cwd=REPO, capture_output=True, text=True, timeout=900)
        out = (p.stdout or "") + (p.stderr or "")
        raw.write_text(out, encoding="utf-8", errors="replace")
        status = "ok" if p.returncode == 0 else f"fail(rc={p.returncode})"
        tail = "\n".join(out.strip().splitlines()[-3:])
        return status, tail
    except Exception as e:  # noqa: BLE001
        raw.write_text(f"ERROR: {e}\n", encoding="utf-8")
        return "error", str(e)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--group",
        nargs="+",
        choices=["tests", "cloud", "sync", "hardware", "skipped", "all"],
        default=["tests", "cloud"],
        help="grupos a correr (default: tests cloud)",
    )
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    ap.add_argument("--pi-host", default="people-counter.local")
    ap.add_argument("--list", action="store_true", help="lista bloques y sale")
    args = ap.parse_args()

    bd = bench_dir(args.date)
    bd.mkdir(parents=True, exist_ok=True)
    raw_dir = bd / "raw"
    raw_dir.mkdir(exist_ok=True)
    all_blocks = blocks(args.date, args.pi_host)

    if args.list:
        for name, group, mode, _, reason in all_blocks:
            print(
                f"  [{group:<8}] {name:<26} ({mode})"
                f"{'  -- ' + reason if reason else ''}"
            )
        return 0

    groups = set(args.group)
    if "all" in groups:
        groups = {"tests", "cloud", "sync", "hardware", "skipped"}
    # 'skipped' siempre se incluye en el reporte (se listan como pendientes)
    selected = [b for b in all_blocks if b[1] in groups or b[1] == "skipped"]

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    started = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"run_benchmarks — commit {commit} — {started} — grupos {sorted(groups)}")

    results = []
    for name, group, mode, cmd, reason in selected:
        print(f"  ▶ [{group}] {name} ...", flush=True)
        status, tail = run_block(name, mode, cmd, reason, raw_dir, args.pi_host)
        print(f"     -> {status}")
        results.append((name, group, mode, status, tail))

    # reporte consolidado
    rep = bd / f"run_benchmarks_{args.date}.md"
    lines = [
        f"# run_benchmarks — {args.date}",
        "",
        f"- commit: `{commit}`",
        f"- inicio: {started}",
        f"- entorno: {platform.platform()} · Python {platform.python_version()}",
        f"- grupos: {sorted(groups)}",
        f"- Pi host: `{args.pi_host}`",
        "",
        "| Bloque | Grupo | Modo | Estado | Resumen |",
        "|---|---|---|---|---|",
    ]
    for name, group, mode, status, tail in results:
        t = tail.replace("\n", " ").replace("|", "/")[:90]
        lines.append(f"| {name} | {group} | {mode} | {status} | {t} |")
    lines += ["", f"Crudos por bloque en `{raw_dir.relative_to(REPO)}/`.", ""]
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreporte -> {rep.relative_to(REPO)}")

    failed = [r for r in results if r[3] not in ("ok", "skipped")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

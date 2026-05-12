#!/usr/bin/env python3
"""Junta un sample diversificado de motion frames de las capturas MJPEG.

Uso típico:

    python scripts/training/sample_for_roboflow.py \\
        --sources training_data/captures \\
        --output training_data/roboflow_sample \\
        --per-site 40 \\
        --seed 42

Características:

- Solo toma archivos con ``_motion_`` en el nombre (descarta ``_bg_``).
- Sample por site con shuffle determinístico (``--seed``) — runs reproducibles.
- Copia con prefijo de site en el filename así es trazable post-upload.
- Imprime resumen al final con counts por site para chequear balance.

Pensado para drag-and-drop a Roboflow (UI web) o ``roboflow upload`` (CLI).
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sources", type=Path, nargs="+", required=True,
        help="Una o más carpetas raíz que contienen subdirs por site.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Carpeta destino. Se crea si no existe.",
    )
    parser.add_argument(
        "--per-site", type=int, default=40,
        help="Cantidad de motion frames a samplear por site (default 40).",
    )
    parser.add_argument(
        "--bg-per-site", type=int, default=0,
        help="Cantidad de bg frames a samplear por site (default 0). "
             "Background frames son negatives explícitos — ayudan al modelo "
             "a no alucinar persona en clutter.",
    )
    parser.add_argument(
        "--site-cap", action="append", default=[], metavar="SITE=N",
        help="Override el per-site para un site específico (ej. para sitios "
             "sobre-representados). Repetible: --site-cap <site_name>=12 "
             "--site-cap <other_site>=20",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed del random sampling (default 42, reproducible).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Imprime qué se copiaría sin tocar el filesystem.",
    )
    args = parser.parse_args()

    # Parsear los overrides de site-cap a dict
    site_caps: dict[str, int] = {}
    for entry in args.site_cap:
        if "=" not in entry:
            raise SystemExit(
                f"--site-cap mal formado: '{entry}'. Uso: SITE=N (ej. <site_name>=12)"
            )
        key, val = entry.split("=", 1)
        try:
            site_caps[key.strip()] = int(val)
        except ValueError:
            raise SystemExit(f"--site-cap '{entry}': N debe ser entero")

    rng = random.Random(args.seed)

    # Acumular frames agrupados por site name + tipo (motion/bg). El site name
    # es el último componente del path.
    def _collect(pattern: str) -> dict[str, list[Path]]:
        result: dict[str, list[Path]] = defaultdict(list)
        for source in args.sources:
            if not source.exists():
                print(f"WARN: source no existe: {source}")
                continue
            for jpg in source.rglob(pattern):
                site = jpg.parent.name
                result[site].append(jpg)
        return result

    motion_by_site = _collect("*_motion_*.jpg")
    bg_by_site = _collect("*_bg_*.jpg") if args.bg_per_site > 0 else {}

    if not motion_by_site:
        raise SystemExit("Ningún motion frame encontrado en las sources dadas.")

    print("Frames disponibles por site:")
    all_sites = sorted(set(motion_by_site) | set(bg_by_site))
    for site in all_sites:
        nm = len(motion_by_site.get(site, []))
        nb = len(bg_by_site.get(site, []))
        print(f"  {site}: motion={nm} bg={nb}")
    print()

    args.output.mkdir(parents=True, exist_ok=True)

    # Sample por site (motion + opcionalmente bg), después interleaving cross-site
    # con shuffle global. El prefijo numérico ``NNN_`` mata la agrupación
    # alfabética en Roboflow — sin él los sites se ven todos juntos en el upload
    # por orden alfabético del filename. Con shuffle global, el operador que
    # revisa va a ver frames de sites distintos uno tras otro, lo cual es
    # importante para detectar bias mientras se labelea.
    sample_summary: dict[str, dict[str, int]] = {}
    selected: list[tuple[str, Path]] = []

    def _sample_pool(pool: dict[str, list[Path]], default_n: int,
                     kind: str) -> None:
        for site, frames in sorted(pool.items()):
            # Site cap solo aplica a motion (los bg no se suelen sobre-representar
            # tanto, y el override sería confuso si afectara ambos pools).
            if kind == "motion":
                n = site_caps.get(site, default_n)
            else:
                n = default_n
            shuffled = sorted(frames)  # orden estable antes del shuffle
            rng_local = random.Random(args.seed + (1 if kind == "bg" else 0))
            rng_local.shuffle(shuffled)
            sampled = shuffled[:n]
            sample_summary.setdefault(site, {"motion": 0, "bg": 0})[kind] = len(sampled)
            selected.extend((site, src) for src in sampled)

    _sample_pool(motion_by_site, args.per_site, "motion")
    if args.bg_per_site > 0:
        _sample_pool(bg_by_site, args.bg_per_site, "bg")

    # Shuffle global de los seleccionados, después prefijar con índice
    # zero-padded. Roboflow ordena por nombre así NNN_ asegura el orden
    # interleaved en la UI de annotate.
    rng.shuffle(selected)
    width = max(3, len(str(len(selected) - 1)))

    total_copied = 0
    for idx, (site, src) in enumerate(selected):
        # Filename destino: ``NNN_<site>__<original-stem>.jpg``. NNN puro
        # sería ambiguo si dos pasadas distintas se mezclan; mantener el
        # site embebido permite trazar al origen sin un sidecar.
        dst_name = f"{idx:0{width}d}_{site}__{src.name}"
        dst = args.output / dst_name
        if args.dry_run:
            print(f"DRY: {src} → {dst}")
        else:
            shutil.copy2(src, dst)
        total_copied += 1

    print("\nSample por site:")
    total_motion = total_bg = 0
    for site, counts in sorted(sample_summary.items()):
        m = counts.get("motion", 0)
        b = counts.get("bg", 0)
        total_motion += m
        total_bg += b
        if args.bg_per_site > 0:
            print(f"  {site}: motion={m} bg={b}")
        else:
            print(f"  {site}: {m}")
    print(f"\nTotal: {total_copied} frames "
          f"({total_motion} motion + {total_bg} bg) en {args.output}/")
    if args.dry_run:
        print("(dry-run — no se copió nada)")


if __name__ == "__main__":
    main()

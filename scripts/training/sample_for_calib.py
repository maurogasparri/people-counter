#!/usr/bin/env python3
"""Junta un calib set balanceado para el QAT de Hailo, sin leak del train.

El compile a HEF en ``hailomz compile`` corre quantization-aware con un
sample de ~200 imágenes representativas. Para evitar leak entre el set
de quant y el de evaluación / training, sampleamos del pool original
(``debug/mjpeg_capture/``) excluyendo los frames que ya están en el
dataset de Roboflow.

Uso típico:

    python scripts/training/sample_for_calib.py \\
        --sources debug/mjpeg_capture \\
        --exclude-dir debug/people_counter_detector_sample \\
        --output models/training/people-counter-detector/calib \\
        --total 200 \\
        --motion-ratio 0.6 \\
        --seed 7

Características:

- Reconstruye el "original stem" desde los filenames renombrados del
  exclude-dir (formato del ``sample_for_roboflow.py``:
  ``NNN_<site>__<orig>.jpg``) y matchea contra (site, basename) en las
  sources, así no hay risk de reintroducir un frame del train al calib.
- Balanceado per-site con mix motion/bg configurable (default 60/40 —
  matchea aproximadamente la distribución de inferencia en producción).
- Determinístico vía ``--seed`` para que el calib set sea reproducible
  entre runs del compile.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


# Filenames renameados por sample_for_roboflow.py tienen la forma
# ``NNN_<site>__<original>.jpg`` (ej. ``000_site_57_27__20260509_122928_motion_L_9172c7.jpg``).
# Este regex extrae site + basename original para el matching contra mjpeg_capture/<site>/<basename>.
_RENAMED_RX = re.compile(r"^\d+_(?P<site>site_[^_]+(?:_[^_]+)*)__(?P<orig>.+\.jpg)$")


def _parse_exclude(exclude_dir: Path) -> set[tuple[str, str]]:
    """Devuelve un set de claves (site, basename) para los frames a excluir."""
    excluded: set[tuple[str, str]] = set()
    if not exclude_dir.exists():
        print(f"WARN: exclude-dir no existe: {exclude_dir}")
        return excluded
    for f in exclude_dir.glob("*.jpg"):
        m = _RENAMED_RX.match(f.name)
        if not m:
            print(f"WARN: no parsea como renamed: {f.name}")
            continue
        excluded.add((m.group("site"), m.group("orig")))
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sources", type=Path, nargs="+", required=True,
        help="Una o más carpetas raíz que contienen subdirs por site.",
    )
    parser.add_argument(
        "--exclude-dir", type=Path, action="append", default=[],
        help="Dir con frames renombrados (formato sample_for_roboflow.py) "
             "que deben EXCLUIRSE del calib. Repetible.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Carpeta destino. Se crea si no existe.",
    )
    parser.add_argument(
        "--total", type=int, default=200,
        help="Total target de imágenes en el calib (default 200).",
    )
    parser.add_argument(
        "--motion-ratio", type=float, default=0.6,
        help="Fracción de motion (vs bg) en el calib. Default 0.6 — "
             "matchea aproximadamente la distribución de inferencia.",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Seed del random sampling (default 7, reproducible).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Imprime qué se copiaría sin tocar el filesystem.",
    )
    args = parser.parse_args()

    # Acumular exclusion set desde TODOS los exclude-dirs.
    excluded: set[tuple[str, str]] = set()
    for d in args.exclude_dir:
        excluded |= _parse_exclude(d)
    print(f"Excluyendo {len(excluded)} frames del calib (train/eval overlap).")

    rng = random.Random(args.seed)

    def _collect(pattern: str) -> dict[str, list[Path]]:
        result: dict[str, list[Path]] = defaultdict(list)
        for source in args.sources:
            if not source.exists():
                print(f"WARN: source no existe: {source}")
                continue
            for jpg in source.rglob(pattern):
                site = jpg.parent.name
                if (site, jpg.name) in excluded:
                    continue
                result[site].append(jpg)
        return result

    motion_by_site = _collect("*_motion_*.jpg")
    bg_by_site = _collect("*_bg_*.jpg")

    sites = sorted(set(motion_by_site) | set(bg_by_site))
    if not sites:
        raise SystemExit("Ningún frame elegible encontrado en las sources.")

    n_motion_total = int(round(args.total * args.motion_ratio))
    n_bg_total = args.total - n_motion_total
    # Per-site quotas (round nearest; el ajuste fino se hace después).
    motion_per_site = max(1, round(n_motion_total / len(sites)))
    bg_per_site = max(0, round(n_bg_total / len(sites)))

    print(
        f"Target: {args.total} total ({n_motion_total} motion + "
        f"{n_bg_total} bg) sobre {len(sites)} sites -> "
        f"~{motion_per_site} motion + {bg_per_site} bg por site."
    )
    print()
    print("Pool elegible por site (post-exclude):")
    for site in sites:
        nm = len(motion_by_site.get(site, []))
        nb = len(bg_by_site.get(site, []))
        print(f"  {site}: motion={nm} bg={nb}")
    print()

    args.output.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, Path]] = []
    summary: dict[str, dict[str, int]] = {}

    def _sample(pool: dict[str, list[Path]], per_site: int, kind: str) -> None:
        for site in sites:
            frames = pool.get(site, [])
            ordered = sorted(frames)
            local_rng = random.Random(args.seed + (1 if kind == "bg" else 0))
            local_rng.shuffle(ordered)
            sampled = ordered[:per_site]
            summary.setdefault(site, {"motion": 0, "bg": 0})[kind] = len(sampled)
            selected.extend((site, p) for p in sampled)

    _sample(motion_by_site, motion_per_site, "motion")
    _sample(bg_by_site, bg_per_site, "bg")

    # Shuffle global así el orden alfabético del filesystem no agrupa
    # frames de un solo site (irrelevante para QAT pero ayuda al
    # operador a hacer spot-check).
    rng.shuffle(selected)
    width = max(3, len(str(len(selected) - 1)))

    total_copied = 0
    for idx, (site, src) in enumerate(selected):
        dst_name = f"{idx:0{width}d}_{site}__{src.name}"
        dst = args.output / dst_name
        if args.dry_run:
            print(f"DRY: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        total_copied += 1

    print()
    print("Sample final por site:")
    total_motion = total_bg = 0
    for site, counts in sorted(summary.items()):
        m = counts.get("motion", 0)
        b = counts.get("bg", 0)
        total_motion += m
        total_bg += b
        print(f"  {site}: motion={m} bg={b}")
    print()
    print(
        f"Total: {total_copied} frames ({total_motion} motion + "
        f"{total_bg} bg) en {args.output}/"
    )
    if args.dry_run:
        print("(dry-run — no se copió nada)")


if __name__ == "__main__":
    main()

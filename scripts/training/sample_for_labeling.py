#!/usr/bin/env python3
"""Muestreo estratificado de captures para labeling local (X-AnyLabeling).

Copia un sample diverso de ``training_data/captures/<site>/*.jpg`` a una
carpeta plana lista para abrir en X-AnyLabeling. Estratifica por site y por
motion/bg para que el set cubra todos los locales y condiciones, en vez de
sesgarse al site con mas frames.

Cada imagen copiada se renombra con prefijo ``<site>__`` para evitar
colisiones de nombre en la carpeta plana y poder rastrear el origen. Escribe
un ``manifest.txt`` con el mapeo, util para reincorporar los labels al
dataset despues — y para excluir esas imagenes al armar OTRO batch.

Exclusion para evitar leakage train/val: ``--exclude-manifest`` saltea las
imagenes ya usadas en otro batch (ej. el validation set). ``--exclude-window-
seconds`` extiende la exclusion a frames temporalmente cercanos del mismo site
(las ramos del motion-trigger son casi-duplicados; si una cae en val y otra en
train hay contaminacion). El timestamp se parsea del filename ``YYYYMMDD_HHMMSS``.

Uso:
    # validation set (motion-heavy)
    python scripts/training/sample_for_labeling.py \\
        --captures training_data/captures --output training_data/label_val_01 \\
        --n-total 250 --motion-ratio 0.85

    # training set (mas bg para capturar estaticos), sin solapar el val
    python scripts/training/sample_for_labeling.py \\
        --captures training_data/captures --output training_data/label_train_01 \\
        --n-total 300 --motion-ratio 0.5 --seed 123 \\
        --exclude-manifest training_data/label_val_01/manifest.txt \\
        --exclude-window-seconds 60
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path


def parse_ts(stem: str) -> float | None:
    """Epoch (s) desde el prefijo YYYYMMDD_HHMMSS del filename, o None."""
    try:
        return datetime.strptime(stem[:15], "%Y%m%d_%H%M%S").timestamp()
    except (ValueError, IndexError):
        return None


def load_exclusions(manifests: list[Path]) -> tuple[set[str], dict[str, list[float]]]:
    """Devuelve (relpaths exactos excluidos, {site: [timestamps excluidos]})."""
    exact: set[str] = set()
    by_site: dict[str, list[float]] = {}
    for mf in manifests:
        if not mf.exists():
            sys.exit(f"ERROR: exclude-manifest no existe: {mf}")
        for line in mf.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "\t" not in line:
                continue
            origin = line.split("\t", 1)[1].strip().replace("\\", "/")
            exact.add(origin)
            parts = origin.split("/")
            site = parts[0]
            ts = parse_ts(Path(origin).stem)
            if ts is not None:
                by_site.setdefault(site, []).append(ts)
    return exact, by_site


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-total", type=int, default=250,
                   help="Total de imgs a muestrear (repartidas entre los sites).")
    p.add_argument("--motion-ratio", type=float, default=0.85,
                   help="Fraccion del sample de frames _motion_ (resto _bg_). "
                        "Bajar (~0.5) para training: mas bg = mas estaticos.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                   help="Manifest(s) de batches previos a excluir (evita overlap).")
    p.add_argument("--exclude-window-seconds", type=float, default=0.0,
                   help="Excluir tambien frames del mismo site dentro de esta "
                        "ventana de un frame excluido (anti-leakage de rafagas).")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not args.captures.exists():
        sys.exit(f"ERROR: captures no existe: {args.captures}")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        sys.exit(
            f"ERROR: {args.output} ya existe y no esta vacia. Usa --force "
            "(cuidado: podes pisar un batch a medio labelar)."
        )

    exact, excl_by_site = load_exclusions(args.exclude_manifest)
    win = args.exclude_window_seconds
    if exact:
        print(f"excluyendo {len(exact)} imgs de {len(args.exclude_manifest)} "
              f"manifest(s)" + (f" + ventana +-{win:.0f}s" if win > 0 else ""))

    def is_excluded(rel: str, site: str, ts: float | None) -> bool:
        if rel in exact:
            return True
        if win > 0 and ts is not None:
            return any(abs(ts - t) <= win for t in excl_by_site.get(site, []))
        return False

    sites = sorted(d for d in args.captures.glob("site_*") if d.is_dir())
    if not sites:
        sys.exit(f"ERROR: no hay carpetas site_* en {args.captures}")

    per_site = max(1, args.n_total // len(sites))
    n_motion = int(round(per_site * args.motion_ratio))
    n_bg = per_site - n_motion

    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    total = 0
    excluded_count = 0

    for site_dir in sites:
        site = site_dir.name
        avail = []
        for jpg in site_dir.glob("*.jpg"):
            rel = f"{site}/{jpg.name}"
            if is_excluded(rel, site, parse_ts(jpg.stem)):
                excluded_count += 1
                continue
            avail.append(jpg)
        motion = [p for p in avail if "_motion_" in p.name]
        bg = [p for p in avail if "_bg_" in p.name]
        picked = rng.sample(motion, min(n_motion, len(motion)))
        picked += rng.sample(bg, min(n_bg, len(bg)))
        for src in picked:
            dst_name = f"{site}__{src.name}"
            shutil.copy2(src, args.output / dst_name)
            manifest.append(f"{dst_name}\t{src.relative_to(args.captures).as_posix()}")
            total += 1
        print(f"  {site}: {len(picked)} imgs ({len(motion)} motion / {len(bg)} bg disponibles)")

    (args.output / "manifest.txt").write_text(
        "# copia\torigen (relativo a captures/)\n" + "\n".join(manifest) + "\n",
        encoding="utf-8",
    )
    if excluded_count:
        print(f"\nsalteadas por exclusion: {excluded_count}")
    print(f"{total} imgs copiadas a {args.output}")
    print(f"manifest: {args.output / 'manifest.txt'}")
    print("\nAbri esa carpeta en X-AnyLabeling para empezar a labelar.")


if __name__ == "__main__":
    main()

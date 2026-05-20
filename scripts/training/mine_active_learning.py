#!/usr/bin/env python3
"""Active learning: minar el pool de captures por frames informativos para v_next.

Usa el modelo actual (v3) + el v1 (high-recall) para encontrar los frames que
mas le ensenan al modelo, en vez de labelar al azar. Tres senales:

  1. Disagreement v1<->v3 (senal de RECALL, la mas valiosa): v1 detecta con
     alta confianza algo que v3 NO cubre -> probable persona que v3 se perdio.
     Como v1 es silueta y v3 es cabeza+hombros, el match es "centro de una
     deteccion v3 cae dentro de la caja (silueta) de v1" -> si ninguna lo
     cubre, v3 se la perdio.
  2. Uncertainty: detecciones de v3 con confianza media (0.2-0.5) -> el modelo
     duda.
  3. (implicito) frames con muchas de las anteriores rankean alto.

Score = w_dis * #disagreements + w_unc * #uncertain. Selecciona top-N y los
copia a una carpeta plana para X-AnyLabeling, con un manifest que explica por
que se eligio cada uno.

Excluye train + val (+ ventana temporal) via --exclude-manifest para no
contaminar el split held-out.

Uso:
    python scripts/training/mine_active_learning.py \\
        --captures training_data/captures \\
        --output training_data/label_al_01 \\
        --v3 debug/kaggle_kernel/output/people-counter-v3/weights/best.pt \\
        --v1 models/training/people-counter-detector/people-counter-detector.pt \\
        --pool-sample 2000 --n-total 250 \\
        --exclude-manifest training_data/label_val_01/manifest.txt \\
        --exclude-manifest training_data/label_train_01/manifest.txt \\
        --exclude-window-seconds 60
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def parse_ts(stem: str):
    try:
        return datetime.strptime(stem[:15], "%Y%m%d_%H%M%S").timestamp()
    except (ValueError, IndexError):
        return None


def load_excl(manifests, win):
    exact = set()
    by_site = {}
    for mf in manifests:
        if not mf.exists():
            sys.exit(f"ERROR: manifest no existe: {mf}")
        for line in mf.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "\t" not in line:
                continue
            origin = line.split("\t", 1)[1].strip().replace("\\", "/")
            exact.add(origin)
            site = origin.split("/")[0]
            ts = parse_ts(Path(origin).stem)
            if ts is not None:
                by_site.setdefault(site, []).append(ts)

    def excluded(rel, site, ts):
        if rel in exact:
            return True
        if win > 0 and ts is not None:
            return any(abs(ts - t) <= win for t in by_site.get(site, []))
        return False

    return excluded


def boxes_conf(model, jpg, conf):
    res = model.predict(str(jpg), conf=conf, verbose=False)[0]
    out = []
    if res.boxes is not None and len(res.boxes):
        xy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        out = [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c))
               for b, c in zip(xy, cf)]
    return out


def center_in(box, px, py):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--v3", type=Path, required=True, help="Modelo actual (head+shoulders).")
    p.add_argument("--v1", type=Path, required=True, help="Modelo high-recall (oraculo).")
    p.add_argument("--pool-sample", type=int, default=2000,
                   help="Cuantos frames del pool escanear (subset, por costo de inferencia).")
    p.add_argument("--n-total", type=int, default=250, help="Cuantos seleccionar para labelar.")
    p.add_argument("--v1-conf", type=float, default=0.5, help="Conf alta de v1 (oraculo confiable).")
    p.add_argument("--v3-conf", type=float, default=0.2, help="Conf baja de v3 (para 'cubrio o no').")
    p.add_argument("--unc-lo", type=float, default=0.2)
    p.add_argument("--unc-hi", type=float, default=0.5)
    p.add_argument("--w-disagree", type=float, default=3.0)
    p.add_argument("--w-uncertain", type=float, default=1.0)
    p.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    p.add_argument("--exclude-window-seconds", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        sys.exit(f"ERROR: {args.output} ya existe y no esta vacia. Usa --force.")

    excluded = load_excl(args.exclude_manifest, args.exclude_window_seconds)

    # Pool de candidatos: captures que no son train/val (+ ventana).
    sites = sorted(d for d in args.captures.glob("site_*") if d.is_dir())
    pool = []
    for site_dir in sites:
        site = site_dir.name
        for jpg in site_dir.glob("*.jpg"):
            rel = f"{site}/{jpg.name}"
            if not excluded(rel, site, parse_ts(jpg.stem)):
                pool.append(jpg)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    pool = pool[: args.pool_sample]
    print(f"pool de candidatos a escanear: {len(pool)}")

    print(f"cargando v3: {args.v3}")
    v3 = YOLO(str(args.v3))
    print(f"cargando v1: {args.v1}")
    v1 = YOLO(str(args.v1))

    scored = []  # (score, dis, unc, jpg)
    for i, jpg in enumerate(pool, 1):
        d3 = boxes_conf(v3, jpg, args.v3_conf)
        d1 = boxes_conf(v1, jpg, args.v1_conf)
        # disagreement: dets de v1 (alta conf) que ninguna det de v3 cubre.
        dis = 0
        for (b1, _c1) in d1:
            cx = (b1[0] + b1[2]) / 2.0
            cy = (b1[1] + b1[3]) / 2.0
            # "cubierta por v3" si algun centro de v3 cae dentro de la silueta v1
            covered = any(center_in(b1, (b3[0] + b3[2]) / 2.0, (b3[1] + b3[3]) / 2.0)
                          for (b3, _c3) in d3)
            if not covered:
                dis += 1
        # uncertainty: dets de v3 con conf media
        unc = sum(1 for (_b, c) in d3 if args.unc_lo <= c < args.unc_hi)
        score = args.w_disagree * dis + args.w_uncertain * unc
        if score > 0:
            scored.append((score, dis, unc, jpg))
        if i % 200 == 0:
            print(f"  escaneados {i}/{len(pool)} | candidatos con score>0: {len(scored)}")

    scored.sort(key=lambda t: t[0], reverse=True)
    pick = scored[: args.n_total]
    print(f"\nseleccionados {len(pick)} de {len(scored)} con senal (top por score)")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = ["# copia\torigen\tscore\tdisagreement(v3_miss)\tuncertain"]
    for score, dis, unc, jpg in pick:
        site = jpg.parent.name
        dst = f"{site}__{jpg.name}"
        shutil.copy2(jpg, args.output / dst)
        rel = jpg.relative_to(args.captures).as_posix()
        manifest.append(f"{dst}\t{rel}\t{score:.1f}\t{dis}\t{unc}")
    (args.output / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    tot_dis = sum(t[1] for t in pick)
    tot_unc = sum(t[2] for t in pick)
    print(f"\n{len(pick)} imgs -> {args.output}")
    print(f"  total disagreements (v3 misses que v1 vio): {tot_dis}")
    print(f"  total detecciones inciertas de v3: {tot_unc}")
    print(f"manifest: {args.output / 'manifest.txt'}")
    print("\nLabelalos en X-AnyLabeling (misma convencion cabeza+hombros).")
    print("Foco: las personas que v3 NO marco -> agregalas (son el recall que falta).")


if __name__ == "__main__":
    main()

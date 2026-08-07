#!/usr/bin/env python3
"""Caracterización del detector sobre el conjunto de validación reservado.

Corre el detector desplegado sobre ``training_data/val_set`` (imágenes con
anotación de referencia, nunca usadas en entrenamiento) y reporta precisión,
exhaustividad y AP@0.5 globales y estratificados por sucursal, franja horaria
y origen (disparo por movimiento / muestreo de fondo), con intervalos de
confianza por bootstrap sobre imágenes. Aparte, la tasa de falsos positivos
sobre las imágenes de fondo sin ninguna caja anotada.

Es una **medición**: hay anotación de referencia por imagen. Su alcance está
acotado por tres hechos que conviene declarar junto al resultado:

  1. El conjunto proviene de las cámaras ya instaladas en las sucursales, no
     del prototipo. Caracteriza al detector sobre ese dominio; su
     transferencia al montaje del prototipo es un supuesto, no un resultado.
  2. Por defecto se ejecuta el modelo ONNX sobre CPU. El runtime del
     dispositivo usa el mismo modelo compilado y cuantizado a HEF sobre el
     acelerador: las cajas no son idénticas. Lo que se mide acá es el
     detector, no la versión cuantizada que corre en producción.
  3. El reparto de cajas por sucursal es muy desigual, de modo que los
     intervalos por sitio son anchos por construcción. Se reportan igual: la
     falta de potencia estadística por sitio es un dato del conjunto.

Uso:

    python scripts/analysis/eval_detector_valset.py

    python scripts/analysis/eval_detector_valset.py \\
        --model models/training/people-counter-detector/people-counter-detector.onnx \\
        --val-set training_data/val_set --bootstrap 2000
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vision.detect import detect_persons, load_model  # noqa: E402


DEFAULT_MODEL = "models/training/people-counter-detector/people-counter-detector.onnx"
IOU_THRESHOLD = 0.5
# Umbrales del runtime desplegado (``detection`` del config del dispositivo).
OPERATING_POINTS = (0.20, 0.35)
RE_NAME = re.compile(r"^(site_\d+_\d+)__(\d{8})_(\d{2})(\d{2})(\d{2})_(motion|bg)_")


@dataclass
class ImageResult:
    name: str
    site: str
    hour: int
    kind: str
    n_gt: int
    # (confianza, es_verdadero_positivo) por detección, ordenable globalmente
    dets: list[tuple[float, bool]]
    n_det: int


def parse_name(stem: str) -> Optional[tuple[str, int, str]]:
    m = RE_NAME.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(3)), m.group(6)


def load_gt(
    label_path: Path, w: int, h: int
) -> list[tuple[float, float, float, float]]:
    """Lee un .txt YOLO normalizado y devuelve cajas en píxeles (x1,y1,x2,y2)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, bw, bh = (float(v) for v in parts[:5])
        boxes.append(
            (
                (cx - bw / 2.0) * w,
                (cy - bh / 2.0) * h,
                (cx + bw / 2.0) * w,
                (cy + bh / 2.0) * h,
            )
        )
    return boxes


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_greedy(
    dets: list[tuple[float, tuple[float, float, float, float]]],
    gts: list[tuple[float, float, float, float]],
    thr: float,
) -> list[tuple[float, bool]]:
    """Asigna cada detección (de mayor a menor confianza) a lo sumo un GT."""
    used = [False] * len(gts)
    out = []
    for conf, box in sorted(dets, key=lambda t: -t[0]):
        best_i, best_v = -1, thr
        for i, g in enumerate(gts):
            if used[i]:
                continue
            v = iou(box, g)
            if v >= best_v:
                best_i, best_v = i, v
        if best_i >= 0:
            used[best_i] = True
            out.append((conf, True))
        else:
            out.append((conf, False))
    return out


def average_precision(results: list[ImageResult]) -> float:
    """AP@0.5 por interpolación de todos los puntos (estilo VOC 2010+)."""
    total_gt = sum(r.n_gt for r in results)
    if total_gt == 0:
        return float("nan")
    flat = [d for r in results for d in r.dets]
    if not flat:
        return 0.0
    flat.sort(key=lambda t: -t[0])
    tp = np.cumsum([1 if d[1] else 0 for d in flat], dtype=float)
    fp = np.cumsum([0 if d[1] else 1 for d in flat], dtype=float)
    rec = tp / total_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    # Envolvente monótona decreciente de la precisión.
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap, prev_r = 0.0, 0.0
    for r_, p_ in zip(rec, prec):
        ap += (r_ - prev_r) * p_
        prev_r = r_
    return float(ap)


def pr_at(results: list[ImageResult], thr: float) -> tuple[float, float, int, int, int]:
    tp = sum(1 for r in results for c, ok in r.dets if c >= thr and ok)
    fp = sum(1 for r in results for c, ok in r.dets if c >= thr and not ok)
    gt = sum(r.n_gt for r in results)
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    rc = tp / gt if gt else float("nan")
    return p, rc, tp, fp, gt


def bootstrap_ci(
    results: list[ImageResult],
    fn: Any,
    rng: np.random.Generator,
    reps: int,
) -> tuple[float, float]:
    """IC percentil 95 % remuestreando IMÁGENES (la unidad independiente)."""
    n = len(results)
    if n == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        v = fn([results[i] for i in idx])
        if v == v:  # descarta NaN
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def fmt_ci(v: float, ci: tuple[float, float]) -> str:
    if v != v:
        return "—"
    return f"{v:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"


def report_stratum(
    label: str, res: list[ImageResult], rng: np.random.Generator, reps: int, thr: float
) -> None:
    p, rc, tp, fp, gt = pr_at(res, thr)
    ap = average_precision(res)
    ci_p = bootstrap_ci(res, lambda r: pr_at(r, thr)[0], rng, reps)
    ci_r = bootstrap_ci(res, lambda r: pr_at(r, thr)[1], rng, reps)
    ci_ap = bootstrap_ci(res, average_precision, rng, reps)
    print(
        f"| {label} | {len(res)} | {gt} | {fmt_ci(p, ci_p)} | "
        f"{fmt_ci(rc, ci_r)} | {fmt_ci(ap, ci_ap)} |"
    )


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap_.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    ap_.add_argument("--val-set", type=Path, default=Path("training_data/val_set"))
    ap_.add_argument("--conf-floor", type=float, default=0.05)
    ap_.add_argument("--nms", type=float, default=0.45)
    ap_.add_argument("--cluster-distance-px", type=float, default=150.0)
    ap_.add_argument("--bootstrap", type=int, default=2000)
    ap_.add_argument("--seed", type=int, default=20260803)
    ap_.add_argument(
        "--use-cuda",
        action="store_true",
        help="No forzar CPU en el backend ONNX (requiere OpenCV con CUDA).",
    )
    args = ap_.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    img_dir = args.val_set / "images"
    lbl_dir = args.val_set / "labels"
    if not img_dir.is_dir():
        sys.exit(f"No encuentro {img_dir}")
    if not args.model.exists():
        sys.exit(f"No encuentro el modelo {args.model}")

    model = load_model(str(args.model))
    # ``OpenCVBackend`` intenta CUDA y cae a CPU en un ``except`` — pero
    # ``setPreferableTarget`` no falla al invocarse, sino recién en el primer
    # ``forward()``, así que en un build de OpenCV sin CUDA el fallback no se
    # dispara. Forzamos CPU acá para que el script corra en cualquier
    # workstation. (Es el camino de desarrollo; el runtime del dispositivo usa
    # el backend Hailo y no pasa por acá.)
    if not args.use_cuda and model["type"] == "opencv":
        net = model["backend"]._net
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    images = sorted(img_dir.glob("*.jpg"))
    results: list[ImageResult] = []

    for i, ip in enumerate(images, 1):
        frame = cv2.imread(str(ip))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gts = load_gt(lbl_dir / (ip.stem + ".txt"), w, h)
        dets = detect_persons(
            frame,
            model,
            confidence_threshold=args.conf_floor,
            nms_threshold=args.nms,
            cluster_distance_px=args.cluster_distance_px,
        )
        boxed = [(float(d.confidence), tuple(float(v) for v in d.bbox)) for d in dets]
        matched = match_greedy(boxed, gts, IOU_THRESHOLD)
        meta = parse_name(ip.stem)
        site, hour, kind = meta if meta else ("desconocido", -1, "desconocido")
        results.append(
            ImageResult(
                name=ip.name,
                site=site,
                hour=hour,
                kind=kind,
                n_gt=len(gts),
                dets=matched,
                n_det=len(dets),
            )
        )
        if i % 50 == 0:
            print(f"… {i}/{len(images)} imágenes", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    reps = args.bootstrap
    total_gt = sum(r.n_gt for r in results)
    empties = [r for r in results if r.n_gt == 0]
    nonempty = [r for r in results if r.n_gt > 0]

    print("# Caracterización del detector — conjunto de validación reservado\n")
    print("## Alcance\n")
    print(f"- Modelo: `{args.model}` (backend {model['type']})")
    print(f"- Conjunto: `{args.val_set}` — {len(results)} imágenes, {total_gt} cajas")
    print(
        f"- Imágenes con al menos una caja: {len(nonempty)} · "
        f"sin ninguna caja: {len(empties)}"
    )
    print(
        f"- Emparejamiento a IoU ≥ {IOU_THRESHOLD}; postproceso de producción "
        f"(NMS {args.nms} + fusión por centroide {args.cluster_distance_px:.0f} px)"
    )
    print(f"- IC 95 % por bootstrap sobre imágenes ({reps} remuestreos)\n")
    print(
        "> Conjunto proveniente de las cámaras instaladas en las sucursales, "
        "no del prototipo; modelo ONNX sobre CPU, no la versión cuantizada "
        "que corre en el acelerador. Ver el encabezado del script.\n"
    )

    for thr in OPERATING_POINTS:
        print(f"## Punto de operación: confianza ≥ {thr:.2f}\n")
        print("| estrato | imágenes | cajas | precisión | exhaustividad | AP@0.5 |")
        print("|---|---:|---:|---|---|---|")
        report_stratum("**global**", results, rng, reps, thr)

        by_site: dict[str, list[ImageResult]] = collections.defaultdict(list)
        for r in results:
            by_site[r.site].append(r)
        for site in sorted(by_site):
            report_stratum(site, by_site[site], rng, reps, thr)

        bands = [
            ("mañana (≤11 h)", 0, 11),
            ("mediodía (12–15 h)", 12, 15),
            ("tarde (16–19 h)", 16, 19),
            ("noche (≥20 h)", 20, 23),
        ]
        for lbl, lo, hi in bands:
            sub = [r for r in results if lo <= r.hour <= hi]
            if sub:
                report_stratum(lbl, sub, rng, reps, thr)

        for kind in ("motion", "bg"):
            sub = [r for r in results if r.kind == kind]
            if sub:
                report_stratum(
                    (
                        "disparo por movimiento"
                        if kind == "motion"
                        else "muestreo de fondo"
                    ),
                    sub,
                    rng,
                    reps,
                    thr,
                )
        print()

    # --- Falsos positivos sobre imágenes sin cajas anotadas -----------------
    print("## Falsos positivos sobre imágenes sin ninguna caja anotada\n")
    print(
        f"Sobre las {len(empties)} imágenes cuya anotación de referencia está "
        "vacía, toda detección es un falso positivo. No hay ambigüedad de "
        "emparejamiento en este subconjunto.\n"
    )
    print("| umbral | imágenes | con ≥1 detección | tasa | FP totales | FP/imagen |")
    print("|---:|---:|---:|---|---:|---:|")
    for thr in OPERATING_POINTS:
        n = len(empties)
        with_det = sum(1 for r in empties if any(c >= thr for c, _ in r.dets))
        fp_tot = sum(1 for r in empties for c, _ in r.dets if c >= thr)
        ci = bootstrap_ci(
            empties,
            lambda rr, t=thr: sum(1 for r in rr if any(c >= t for c, _ in r.dets))
            / max(len(rr), 1),
            rng,
            reps,
        )
        rate = with_det / n if n else float("nan")
        print(
            f"| ≥ {thr:.2f} | {n} | {with_det} | {fmt_ci(rate, ci)} | "
            f"{fp_tot} | {fp_tot/n if n else float('nan'):.3f} |"
        )
    print()

    print("## Reparto de cajas por sucursal\n")
    print("| sucursal | imágenes | cajas |")
    print("|---|---:|---:|")
    by_site2: dict[str, list[ImageResult]] = collections.defaultdict(list)
    for r in results:
        by_site2[r.site].append(r)
    for site in sorted(by_site2):
        sub = by_site2[site]
        print(f"| {site} | {len(sub)} | {sum(x.n_gt for x in sub)} |")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

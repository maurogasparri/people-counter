#!/usr/bin/env python3
"""Triage de modelos publicados en Roboflow Universe via inferencia API.

Sin descargar dataset, sin entrenar — apuntás cada modelo contra los
mismos frames locales y comparás detection rate / mean conf / FP rate /
zonas. La idea es responder "¿cuál de estos modelos detecta nuestras
cabezas cenitales reales?" antes de invertir tiempo en fine-tune.

Cada llamada usa el endpoint serverless ``detect.roboflow.com``:
    POST https://detect.roboflow.com/<workspace>/<project>/<version>
         ?api_key=<key>&confidence=<thr>&overlap=<iou>
    Body: imagen base64 (form-encoded)

Usage:
    export ROBOFLOW_API_KEY=...
    python scripts/training/bench_roboflow_api.py \\
        --frames debug/baseline_frames \\
        --report debug/bench_roboflow.json \\
        --annotated-dir debug/annotated_roboflow \\
        --models \\
            "abhay-c-mkdjq/overhead-head-detection/<v>" \\
            "coding-compass-nmjfb/overhead-head-detection-cwetj/<v>" \\
            "chelkatun-nauka/overhead-view/<v>"

Identificás <v> en el sidebar derecho de cada Universe page (sección
"Versions" → la versión más nueva con estado "Trained").
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import cv2

logger = logging.getLogger("bench_roboflow_api")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
ENDPOINT_DETECT = "https://detect.roboflow.com"
ENDPOINT_SERVERLESS = "https://serverless.roboflow.com"


def _zone_of(xc: float, yc: float, w: int, h: int) -> str:
    cx = xc / max(w, 1)
    cy = yc / max(h, 1)
    if 0.33 <= cx <= 0.67 and 0.33 <= cy <= 0.67:
        return "center"
    top = cy < 0.5
    left = cx < 0.5
    if top and left: return "tl"
    if top and not left: return "tr"
    if not top and left: return "bl"
    return "br"


def _infer(
    model_id: str, img_bytes: bytes, api_key: str,
    conf: float, iou: float, endpoint: str = ENDPOINT_DETECT,
) -> dict[str, Any]:
    """Llama a la inference API de Roboflow para una imagen. Devuelve JSON parseado.

    detect.roboflow.com espera ``<project>/<version>`` SIN el workspace.
    Si el caller pasa ``workspace/project/version`` (formato canónico
    universal), el workspace se descarta.
    """
    if model_id.count("/") == 2:
        _, project, version = model_id.split("/")
        endpoint_path = f"{project}/{version}"
    else:
        endpoint_path = model_id
    b64 = base64.b64encode(img_bytes)
    qs = urlencode({
        "api_key": api_key,
        "confidence": str(int(conf * 100)),  # API takes percentage
        "overlap": str(int(iou * 100)),
    })
    url = f"{endpoint}/{endpoint_path}?{qs}"
    req = Request(
        url,
        data=b64,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            # serverless.roboflow.com bloquea con Cloudflare 1010 si no
            # mandás User-Agent reconocible
            "User-Agent": "Mozilla/5.0 people-counter/bench",
        },
    )
    with urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _bench_one_model(
    model_id: str, images: list[Path], api_key: str,
    conf: float, iou: float, annotated_dir: Path | None,
    endpoint: str = ENDPOINT_DETECT,
) -> dict[str, Any]:
    """Run all frames against one model and aggregate stats."""
    short = model_id.split("/")[1]  # project name only, for filenames
    if annotated_dir:
        out_subdir = annotated_dir / short
        out_subdir.mkdir(parents=True, exist_ok=True)
    else:
        out_subdir = None

    per_frame: list[dict[str, Any]] = []
    confidences: list[float] = []
    class_counter: defaultdict[str, int] = defaultdict(int)
    zone_counts = {z: 0 for z in ("center", "tl", "tr", "bl", "br")}
    frames_with_any = 0
    failed_calls = 0
    latencies_ms: list[float] = []

    for img_path in images:
        img_bytes = img_path.read_bytes()
        try:
            t0 = time.time()
            result = _infer(model_id, img_bytes, api_key, conf, iou, endpoint)
            latencies_ms.append((time.time() - t0) * 1000)
        except (HTTPError, URLError) as e:
            failed_calls += 1
            logger.warning("[%s] %s -> %s", short, img_path.name, e)
            per_frame.append({"path": img_path.name, "error": str(e)})
            continue

        preds = result.get("predictions", [])
        n = len(preds)
        if n > 0:
            frames_with_any += 1
        frame_confs: list[float] = []

        # Load image to overlay (optional)
        img_np = None
        if out_subdir is not None:
            img_np = cv2.imread(str(img_path))

        for p in preds:
            c = float(p.get("confidence", 0.0))
            cls = str(p.get("class", "?"))
            confidences.append(c)
            frame_confs.append(c)
            class_counter[cls] += 1

            # Roboflow returns center-xy + w/h in pixel coordinates of input
            xc = float(p["x"])
            yc = float(p["y"])
            w = float(p["width"])
            h = float(p["height"])
            # Frame size from API response (post any auto-resize)
            fw = result.get("image", {}).get("width", 0)
            fh = result.get("image", {}).get("height", 0)
            if fw and fh:
                zone_counts[_zone_of(xc, yc, fw, fh)] += 1

            if img_np is not None and fw and fh:
                # Scale bbox back to original image dims
                oh, ow = img_np.shape[:2]
                sx, sy = ow / fw, oh / fh
                x1 = int((xc - w / 2) * sx)
                y1 = int((yc - h / 2) * sy)
                x2 = int((xc + w / 2) * sx)
                y2 = int((yc + h / 2) * sy)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_np, f"{cls} {c:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if img_np is not None:
            cv2.imwrite(str(out_subdir / img_path.name), img_np)

        per_frame.append({
            "path": img_path.name,
            "n_detections": n,
            "confidences": frame_confs,
            "classes": [str(p.get("class", "?")) for p in preds],
        })

    n_total = len(images)
    summary = {
        "model_id": model_id,
        "n_frames": n_total,
        "n_failed_calls": failed_calls,
        "frames_with_detections": frames_with_any,
        "detection_rate": frames_with_any / max(n_total - failed_calls, 1),
        "total_detections": sum(class_counter.values()),
        "class_counts": dict(class_counter),
        "zone_counts": zone_counts,
        "latency_ms": (
            {
                "mean": statistics.mean(latencies_ms),
                "median": statistics.median(latencies_ms),
                "max": max(latencies_ms),
            } if latencies_ms else None
        ),
    }
    if confidences:
        summary["confidence"] = {
            "mean": statistics.mean(confidences),
            "median": statistics.median(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "stdev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
        }
    else:
        summary["confidence"] = None

    return {"summary": summary, "per_frame": per_frame}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--annotated-dir", type=Path, default=None)
    p.add_argument(
        "--models", nargs="+", required=True,
        help="Lista de IDs <workspace>/<project>/<version>",
    )
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument(
        "--endpoint", choices=["detect", "serverless"], default="detect",
        help="detect.roboflow.com (legacy) o serverless.roboflow.com (Roboflow 3.0)",
    )
    p.add_argument(
        "--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Default: $ROBOFLOW_API_KEY",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.api_key:
        sys.exit("Falta ROBOFLOW_API_KEY (env var o --api-key)")
    if not args.frames.is_dir():
        sys.exit(f"Frames dir no encontrado: {args.frames}")

    images = sorted(p for p in args.frames.iterdir()
                    if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"No hay imágenes en {args.frames}")
    logger.info("Frames a probar: %d", len(images))

    if args.annotated_dir:
        args.annotated_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}
    for m in args.models:
        if m.count("/") not in (1, 2):
            logger.error("model id mal formado: %s (esperado proj/ver o ws/proj/ver)", m)
            continue
        logger.info("=== Benchmarking %s ===", m)
        endpoint = ENDPOINT_SERVERLESS if args.endpoint == "serverless" else ENDPOINT_DETECT
        all_results[m] = _bench_one_model(
            m, images, args.api_key, args.conf, args.iou, args.annotated_dir,
            endpoint=endpoint,
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(all_results, f, indent=2)

    # Tabla resumen
    print("\n" + "=" * 90)
    print(f"{'Model':<55}  {'detect%':>8}  {'mean_conf':>10}  {'total_dets':>10}")
    print("=" * 90)
    for m, r in all_results.items():
        s = r["summary"]
        det = s["detection_rate"] * 100 if s["detection_rate"] else 0
        mc = s["confidence"]["mean"] if s["confidence"] else 0.0
        td = s["total_detections"]
        print(f"{m:<55}  {det:>7.1f}%  {mc:>10.3f}  {td:>10}")
    print()
    for m, r in all_results.items():
        s = r["summary"]
        print(f"{m}")
        print(f"  classes: {s['class_counts']}")
        print(f"  zones:   {s['zone_counts']}")
        if s["latency_ms"]:
            print(f"  latency: {s['latency_ms']['mean']:.0f}ms mean / "
                  f"{s['latency_ms']['median']:.0f}ms median")
        if s["n_failed_calls"]:
            print(f"  failed:  {s['n_failed_calls']} calls")
        print()
    print(f"Report: {args.report}")
    if args.annotated_dir:
        print(f"Annotated frames: {args.annotated_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

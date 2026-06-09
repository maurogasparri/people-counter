#!/usr/bin/env python3
"""Benchea un detector YOLOv8 sobre una carpeta de frames.

Harness de bench Phase A para el detector cenital de cabezas. Dos casos de uso:

1. **Medición de baseline** antes del training: correr el modelo YOLOv8n COCO
   off-the-shelf (o el modelo de cabezas Owen718) sobre los frames que ya
   capturaste de la Pi, ver qué fracción detecta, dónde se sienta la confidence
   y qué zonas del frame son débiles.

2. **Comparación post-training** luego de que termina Phase A: mismos flags,
   weights nuevos. Diff de los dos reportes para ver si el fine-tune realmente
   ayudó en TU escena.

El bench es intencionalmente simple: no requiere ground truth (todavía no
tenemos anotaciones de nuestros propios frames). Reporta counts de detecciones
+ distribución de confidence por zona (grilla de 5: centro + 4 esquinas), lo
cual alcanza para comparar dos modelos rápido.

Uso:
    # Baseline con YOLOv8n stock
    python scripts/training/bench_detector.py \\
        --weights yolov8n.pt \\
        --frames /path/to/captured_frames/ \\
        --conf 0.25 \\
        --report debug/bench_baseline.json

    # Fine-tuned model
    python scripts/training/bench_detector.py \\
        --weights runs/detect/train/weights/best.pt \\
        --frames /path/to/captured_frames/ \\
        --conf 0.25 \\
        --report debug/bench_finetuned.json

    # Diff
    python scripts/training/bench_detector.py --diff \\
        debug/bench_baseline.json debug/bench_finetuned.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("bench_detector")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _zone_of(x_center: float, y_center: float, width: int, height: int) -> str:
    """5-grid zone label for a detection center.

    The frame is split into a 3×3 grid; we collapse the 9 cells into 5
    zones so reports stay readable: center, plus the 4 corner clusters
    (top-left/right and bottom-left/right edges fold in).
    """
    cx = x_center / max(width, 1)
    cy = y_center / max(height, 1)
    if 0.33 <= cx <= 0.67 and 0.33 <= cy <= 0.67:
        return "center"
    top = cy < 0.5
    left = cx < 0.5
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def run_bench(
    weights: Path, frames_dir: Path, conf: float, imgsz: int = 640
) -> dict[str, Any]:
    """Run inference over every image in ``frames_dir`` and aggregate.

    Returns a dict with frame-level + zone-level stats. Doesn't write to
    disk — caller decides where to dump.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "The 'ultralytics' package is not installed. Run:\n"
            "    pip install ultralytics\n"
            "(workstation-only dep — the Pi runs the HEF, not the .pt)."
        ) from e

    if not frames_dir.is_dir():
        raise SystemExit(f"Frames dir not found: {frames_dir}")

    images = sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise SystemExit(f"No images found in {frames_dir}")

    logger.info("Loading weights: %s", weights)
    model = YOLO(str(weights))

    logger.info("Benching %d frames (conf=%.2f, imgsz=%d)", len(images), conf, imgsz)

    per_frame = []
    confidences: list[float] = []
    zone_counts: dict[str, int] = {z: 0 for z in ("center", "tl", "tr", "bl", "br")}
    frames_with_any = 0

    for img_path in images:
        results = model.predict(
            source=str(img_path), conf=conf, imgsz=imgsz, verbose=False
        )
        if not results:
            continue
        r = results[0]
        boxes = r.boxes
        n = 0 if boxes is None else int(len(boxes))
        if n > 0:
            frames_with_any += 1
        h, w = r.orig_shape if hasattr(r, "orig_shape") else (1, 1)

        frame_confs: list[float] = []
        if boxes is not None and n > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            cnf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, score in zip(xyxy, cls, cnf):
                xc = float((x1 + x2) / 2)
                yc = float((y1 + y2) / 2)
                zone_counts[_zone_of(xc, yc, w, h)] += 1
                frame_confs.append(float(score))
                confidences.append(float(score))

        per_frame.append(
            {
                "path": img_path.name,
                "n_detections": n,
                "confidences": frame_confs,
            }
        )

    summary = {
        "weights": str(weights),
        "frames_dir": str(frames_dir),
        "conf_threshold": conf,
        "imgsz": imgsz,
        "n_frames": len(images),
        "frames_with_detections": frames_with_any,
        "detection_rate": frames_with_any / len(images),
        "total_detections": sum(zone_counts.values()),
        "zone_counts": zone_counts,
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


def diff_reports(a_path: Path, b_path: Path) -> dict[str, Any]:
    """Print a human-readable comparison of two bench reports."""
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)

    sa = a["summary"]
    sb = b["summary"]

    def _delta(name: str, va: Any, vb: Any) -> str:
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            sign = "+" if d > 0 else ""
            return f"  {name:30s}  {va!s:>10}  ->  {vb!s:>10}  ({sign}{d:.3f})"
        return f"  {name:30s}  {va!s:>10}  ->  {vb!s:>10}"

    out = []
    out.append(f"\nA: {sa['weights']}")
    out.append(f"B: {sb['weights']}")
    out.append(f"Frames: {sa['n_frames']} (A) vs {sb['n_frames']} (B)")
    out.append("")
    out.append(
        _delta(
            "frames_with_detections",
            sa["frames_with_detections"],
            sb["frames_with_detections"],
        )
    )
    out.append(
        _delta(
            "detection_rate",
            round(sa["detection_rate"], 3),
            round(sb["detection_rate"], 3),
        )
    )
    out.append(
        _delta("total_detections", sa["total_detections"], sb["total_detections"])
    )
    if sa.get("confidence") and sb.get("confidence"):
        out.append(
            _delta(
                "confidence.mean",
                round(sa["confidence"]["mean"], 3),
                round(sb["confidence"]["mean"], 3),
            )
        )
        out.append(
            _delta(
                "confidence.median",
                round(sa["confidence"]["median"], 3),
                round(sb["confidence"]["median"], 3),
            )
        )
    out.append("")
    out.append("Per-zone detections (A -> B):")
    for z in ("center", "tl", "tr", "bl", "br"):
        out.append(
            _delta(
                f"  zone.{z}", sa["zone_counts"].get(z, 0), sb["zone_counts"].get(z, 0)
            )
        )

    text = "\n".join(out)
    print(text)
    return {"a": sa, "b": sb}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd")

    bench_p = sub.add_parser("bench", help="Run inference + write report")
    bench_p.add_argument(
        "--weights", type=Path, required=True, help="Path to .pt weights (or .onnx)"
    )
    bench_p.add_argument(
        "--frames",
        type=Path,
        required=True,
        help="Folder of frames to run inference on",
    )
    bench_p.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    bench_p.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (default: 640)"
    )
    bench_p.add_argument(
        "--report", type=Path, required=True, help="Where to write the JSON report"
    )

    diff_p = sub.add_parser("diff", help="Diff two reports")
    diff_p.add_argument("a", type=Path, help="Baseline report (JSON)")
    diff_p.add_argument("b", type=Path, help="Comparison report (JSON)")

    # Back-compat: if the user runs without a subcommand, default to bench.
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "bench":
        report = run_bench(args.weights, args.frames, args.conf, args.imgsz)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        s = report["summary"]
        logger.info(
            "Done. %d/%d frames had detections (%.1f%%); %d total dets",
            s["frames_with_detections"],
            s["n_frames"],
            100 * s["detection_rate"],
            s["total_detections"],
        )
        if s.get("confidence"):
            logger.info(
                "Mean conf: %.3f, median: %.3f",
                s["confidence"]["mean"],
                s["confidence"]["median"],
            )
        print(f"\nReport written to: {args.report}\n")
    elif args.cmd == "diff":
        diff_reports(args.a, args.b)

    return 0


if __name__ == "__main__":
    sys.exit(main())

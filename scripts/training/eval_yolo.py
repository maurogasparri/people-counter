#!/usr/bin/env python3
"""Corre un modelo YOLO sobre una carpeta de frames y guarda anotaciones.

Uso:
    python scripts/training/eval_yolo.py \\
        --weights models/training/people-counter-detector/best.pt \\
        --source debug/eval_frames \\
        --output debug/eval_output \\
        --conf 0.25

Salida:
    <output>/<image>.jpg     - frame con bboxes dibujados
    <output>/labels/<image>.txt - bboxes en formato YOLO
    <output>/summary.txt     - count de detects por imagen
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path,
                        help="Carpeta con jpgs a evaluar")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    if args.output.exists():
        for f in args.output.rglob("*"):
            if f.is_file():
                f.unlink()
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        verbose=False,
    )

    # Reporte resumen
    summary_path = args.output / "summary.txt"
    total_dets = 0
    with open(summary_path, "w") as f:
        f.write(f"Modelo: {args.weights}\n")
        f.write(f"Conf threshold: {args.conf}\n")
        f.write(f"Imgsz: {args.imgsz}\n")
        f.write(f"Frames: {len(results)}\n\n")
        f.write("Detects por frame:\n")
        for r in results:
            n = len(r.boxes) if r.boxes is not None else 0
            total_dets += n
            stem = Path(r.path).name
            confs = []
            if r.boxes is not None and n > 0:
                confs = [f"{c:.2f}" for c in r.boxes.conf.tolist()]
            f.write(f"  {stem}: {n} detects [{','.join(confs)}]\n")
        f.write(f"\nTotal detects: {total_dets}\n")
        avg = total_dets / max(1, len(results))
        f.write(f"Avg per frame: {avg:.2f}\n")

    print(f"Total detects: {total_dets} en {len(results)} frames "
          f"(avg {total_dets/max(1,len(results)):.2f}/frame)")
    print(f"Output: {args.output}/")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()

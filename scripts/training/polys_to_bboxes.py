#!/usr/bin/env python3
"""Convierte labels YOLOv8-seg (polígonos) a YOLOv8-detection (bboxes).

Cada línea de los .txt en formato seg es:
    class x1 y1 x2 y2 ... xN yN
La conversión a detection es trivial — el bbox es el min/max del polígono:
    class cx cy w h    (todo normalizado [0,1])

Uso típico tras descargar un dataset de Roboflow exportado como YOLOv8 desde
un project de Instance Segmentation:

    python scripts/training/polys_to_bboxes.py \\
        --dataset dataset/roboflow_<workspace>_<project>_v<N>

Modifica los .txt en place. Hace backup .seg.txt por las dudas.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def poly_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """Min/max de los pares (x, y) → (cx, cy, w, h) normalizado."""
    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def convert_file(path: Path) -> tuple[int, int]:
    """Devuelve (líneas convertidas, líneas ya bbox que se dejaron como están)."""
    lines = path.read_text().strip().splitlines()
    out_lines: list[str] = []
    converted = skipped = 0
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = parts[0]
        coords = [float(x) for x in parts[1:]]
        if len(coords) == 4:
            # Ya es bbox (cx, cy, w, h) — no tocar
            out_lines.append(line.strip())
            skipped += 1
            continue
        if len(coords) < 6 or len(coords) % 2 != 0:
            raise ValueError(
                f"Formato inesperado en {path}: {len(coords)} coords "
                f"(se esperaba 4 para bbox o ≥6 par para polígono)"
            )
        cx, cy, w, h = poly_to_bbox(coords)
        out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        converted += 1

    # Backup del original (idempotente: si ya hay backup, no lo pisa)
    backup = path.with_suffix(".seg.txt")
    if not backup.exists() and converted > 0:
        backup.write_text(path.read_text())

    path.write_text("\n".join(out_lines) + "\n")
    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Raíz del dataset YOLOv8 (la carpeta con data.yaml + train/valid/test)",
    )
    args = parser.parse_args()

    if not (args.dataset / "data.yaml").exists():
        raise SystemExit(f"No se encuentra data.yaml en {args.dataset}")

    total_files = 0
    total_converted = 0
    total_skipped = 0
    for split in ("train", "valid", "test"):
        labels_dir = args.dataset / split / "labels"
        if not labels_dir.is_dir():
            continue
        for txt in labels_dir.glob("*.txt"):
            converted, skipped = convert_file(txt)
            total_files += 1
            total_converted += converted
            total_skipped += skipped

    print(f"Procesados {total_files} archivos:")
    print(f"  poligonos -> bbox: {total_converted}")
    print(f"  ya eran bbox (no tocados): {total_skipped}")
    print(f"  backup .seg.txt creado para los convertidos")


if __name__ == "__main__":
    main()

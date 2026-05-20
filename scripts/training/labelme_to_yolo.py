#!/usr/bin/env python3
"""Convierte una carpeta labelada en X-AnyLabeling/labelme a un dataset YOLO.

Lee los ``.json`` (formato labelme: shapes con points) que X-AnyLabeling
deja junto a cada imagen y produce la estructura canonica de Ultralytics::

    output/
      images/<name>.jpg
      labels/<name>.txt        (class cx cy w h normalizado, una linea por caja)
      data.yaml

Convencion de background: una imagen SIN ``.json`` se trata como **negativo
revisado** y se le escribe un ``.txt`` VACIO (no es lo mismo que "no revisada"
— el caller garantiza que todas las imagenes fueron miradas). Ultralytics
trata el .txt vacio como imagen sin objetos (background), util para bajar FPs.

Solo soporta shapes ``rectangle`` (2 puntos = esquinas opuestas). Una clase
unica por default (``person``).

Uso::

    python scripts/training/labelme_to_yolo.py \\
        --input training_data/label_val_01 \\
        --output training_data/val_set \\
        --class-name person
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def shape_to_yolo(points: list, W: int, H: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return cx, cy, w, h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, required=True,
                   help="Carpeta con <name>.jpg + <name>.json (labelme).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--class-name", default="person")
    p.add_argument("--split", default="val",
                   help="Nombre del split en data.yaml (val/train). Default val.")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: input no existe: {args.input}")

    img_out = args.output / "images"
    lbl_out = args.output / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(args.input.glob("*.jpg"))
    if not jpgs:
        sys.exit(f"ERROR: ningun .jpg en {args.input}")

    n_with_boxes = 0
    n_background = 0
    n_boxes = 0
    n_skipped_shape = 0

    for jpg in jpgs:
        shutil.copy2(jpg, img_out / jpg.name)
        jf = jpg.with_suffix(".json")
        lines: list[str] = []
        if jf.exists():
            d = json.loads(jf.read_text(encoding="utf-8"))
            W = d.get("imageWidth")
            H = d.get("imageHeight")
            if not W or not H:
                from PIL import Image
                with Image.open(jpg) as im:
                    W, H = im.size
            for sh in d.get("shapes", []):
                pts = sh.get("points", [])
                if sh.get("shape_type") not in (None, "rectangle") or len(pts) < 2:
                    n_skipped_shape += 1
                    continue
                cx, cy, w, h = shape_to_yolo(pts, W, H)
                # clamp defensivo a [0,1]
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        # .txt siempre se escribe (vacio = background revisado)
        (lbl_out / f"{jpg.stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )
        if lines:
            n_with_boxes += 1
            n_boxes += len(lines)
        else:
            n_background += 1

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        f"# generado por labelme_to_yolo.py\n"
        f"path: {args.output.resolve().as_posix()}\n"
        f"{args.split}: images\n"
        f"nc: 1\n"
        f"names: ['{args.class_name}']\n",
        encoding="utf-8",
    )

    print(f"imgs: {len(jpgs)}  con cajas: {n_with_boxes}  background: {n_background}")
    print(f"cajas totales: {n_boxes}")
    if n_skipped_shape:
        print(f"shapes no-rectangulo salteadas: {n_skipped_shape}")
    print(f"dataset YOLO en: {args.output}")
    print(f"data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()

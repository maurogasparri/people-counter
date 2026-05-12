#!/usr/bin/env python3
"""One-shot migration: empotra las matrices de calibración por site en el
YAML de configuración.

Lee la lista de sites desde un YAML viejo (formato con
``calibration: path/to/calib.npz``), abre cada ``.npz`` y extrae solo el
subset que ``capture_mjpeg.py`` consume — el suffix ``_4`` para K, D,
R_rect y P_rect por lente. Escribe un YAML nuevo con las matrices inline
así no necesitamos arrastrar 32M de dumps externos.

Uso:
    python scripts/training/_embed_calib_into_sites.py \\
        --in  debug/mjpeg_sites.yaml \\
        --out training_data/sites.yaml

Este script se corre una sola vez para la migración inicial. No tiene
sentido tenerlo en el flujo regular — si en el futuro hay que re-extraer
matrices, se vuelve a correr puntualmente.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


class _FlowList(list):
    """List marker for flow-style serialization (single-line)."""


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True,
    )


yaml.SafeDumper.add_representer(_FlowList, _flow_list_representer)

# Solo extraemos el set de matrices a la resolución de calibración (160×120)
# que es lo que capture_mjpeg.py escala on-the-fly. Los demás suffixes
# (_5..._9 y sin suffix) son redundantes para nuestro pipeline.
_KEYS_PER_SIDE = ("intrinsic_4", "distortion_4", "R_rect_4", "intrinsic_rect_4")


def _mat_to_flow(arr: np.ndarray) -> _FlowList:
    """Matrix 2-D → lista de filas flow-style (cada fila en una línea)."""
    return _FlowList(
        _FlowList(float(x) for x in row) for row in arr.astype(float)
    )


def _vec_to_flow(arr: np.ndarray) -> _FlowList:
    """Vector → flow-style en una sola línea."""
    return _FlowList(float(x) for x in arr.astype(float).reshape(-1))


def _extract_site_calib(npz_path: Path) -> dict:
    cal = np.load(npz_path)
    out: dict = {
        # Resolución a la que fueron escritas las matrices — capture_mjpeg
        # las re-escala al half_size real del stream linealmente.
        "ref_size": _FlowList([160, 120]),
    }
    for side in ("left", "right"):
        out[side] = {
            "K": _mat_to_flow(cal[f"scaled_{side}_intrinsic_4"]),
            "D": _vec_to_flow(cal[f"{side}_distortion_4"]),
            "R_rect": _mat_to_flow(cal[f"scaled_{side}_R_rect_4"]),
            "P_rect": _mat_to_flow(cal[f"scaled_{side}_intrinsic_rect_4"]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--in", dest="src", type=Path, required=True,
                        help="YAML viejo con paths a calib.npz")
    parser.add_argument("--out", type=Path, required=True,
                        help="YAML nuevo con matrices inline")
    args = parser.parse_args()

    with args.src.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sites_in = cfg.get("sites", [])
    if not sites_in:
        sys.exit("No hay sites en el YAML de entrada")

    sites_out = []
    for s in sites_in:
        name = s["name"]
        new_site = {"name": name, "url": s["url"], "sbs": bool(s.get("sbs"))}
        npz_path = s.get("calibration")
        if npz_path:
            p = Path(npz_path)
            if not p.exists():
                print(f"[{name}] WARN: calib no existe: {p} — site sin matrices",
                      file=sys.stderr)
            else:
                new_site["calibration"] = _extract_site_calib(p)
        sites_out.append(new_site)
        print(f"[{name}] embebido ({'con' if 'calibration' in new_site else 'sin'} calib)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"sites": sites_out}, f, sort_keys=False, allow_unicode=True)
    print(f"\nEscrito {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

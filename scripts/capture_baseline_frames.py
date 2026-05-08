#!/usr/bin/env python3
"""Capturar frames rectificados de la Pi para el bench de validación.

Script standalone — abre el mismo path de captura estéreo + rectificación
que usa main.py, pero en lugar de detectar guarda el frame izquierdo
rectificado a disco cada ``--interval`` segundos. Sirve para armar un
corpus de bench que matchea exactamente lo que YOLOv8 ve en producción
(1152×648, rectificado Kannala-Brandt, cámara izquierda).

Uso (correr en la Pi):

    python3 scripts/capture_baseline_frames.py \\
        --config /etc/people-counter/config.yaml \\
        --num-frames 30 \\
        --interval 2.0 \\
        --output /tmp/baseline_frames

Después SCP de la carpeta al workstation:

    scp -r pi@<host>:/tmp/baseline_frames C:/.../debug/baseline_frames

Tip: durante la captura, caminá frente a la cámara de vez en cuando.
Mezclar frames vacíos con frames con personas es exactamente lo que
queremos — el bench mide detection rate, así que necesita ambos casos
para ser informativo.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.vision.calibration import load_calibration, rectify_pair
from src.vision.capture import StereoCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("capture_baseline_frames")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", required=True,
        help="Path al config.yaml (e.g. /etc/people-counter/config.yaml)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=30,
        help="Cantidad de frames a capturar (default 30)",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Segundos entre capturas (default 2.0)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Carpeta de salida (se crea si no existe)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    vision_cfg = config["vision"]

    cal_file = vision_cfg.get("calibration_file")
    if not cal_file or not Path(cal_file).exists():
        raise SystemExit(
            f"Archivo de calibración no encontrado: {cal_file}. "
            "Configurá vision.calibration_file en config.yaml."
        )

    logger.info("Cargando calibración desde %s", cal_file)
    calibration = load_calibration(cal_file)

    cap = StereoCapture(
        cam_left_id=config["bracket"]["camera_left_csi"],
        cam_right_id=config["bracket"]["camera_right_csi"],
        resolution=tuple(vision_cfg["resolution"]),
        fps=int(vision_cfg["fps"]),
    )
    cap.open()
    logger.info(
        "Capturando %d frames cada %.1fs hacia %s",
        args.num_frames, args.interval, args.output,
    )

    try:
        for i in range(args.num_frames):
            frame_l, frame_r = cap.read()
            rect_l, _ = rectify_pair(frame_l, frame_r, calibration)
            out_path = args.output / f"frame_{i:03d}.jpg"
            ok = cv2.imwrite(str(out_path), rect_l)
            if not ok:
                logger.warning("imwrite falló para %s", out_path)
                continue
            logger.info("[%d/%d] %s", i + 1, args.num_frames, out_path)
            if i < args.num_frames - 1:
                time.sleep(args.interval)
    finally:
        cap.close()

    logger.info("Listo. %d frames en %s", args.num_frames, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

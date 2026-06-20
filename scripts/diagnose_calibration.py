#!/usr/bin/env python3
"""Monitor de salud de calibración estéreo — diagnóstico post-calibración.

Responde una sola pregunta: **¿la calibración guardada sigue siendo válida,
o hay que recalibrar?** Pensado como check de campo / periódico: detecta si
el bracket se movió, un lente se corrió en su holder, o hubo drift térmico
desde la última calibración.

Cómo funciona
-------------
A diferencia de un QC pre-calibración (que no puede medir bien la geometría
de un fisheye porque todavía no tiene los coeficientes de distorsión), esta
tool **aplica la calibración existente del device**:

  1. Carga la calibración (.npz): mapas de rectificación K-B + P1/P2.
  2. Captura ``--frames`` pares del board ChArUco.
  3. **Rectifica** ambas imágenes con la calibración guardada.
  4. Detecta esquinas ChArUco en cada imagen rectificada y matchea L↔R por ID.
  5. Mide el **error epipolar** = disparidad vertical residual ``|y_L - y_R|``
     de las esquinas correspondientes. En un par bien rectificado las
     correspondencias caen en la MISMA fila (y_L ≈ y_R); si el bracket se
     movió desde la calibración, los mapas (derivados de la calib vieja) ya
     no alinean y el error epipolar crece.

El error epipolar **no necesita ground-truth** (ni cinta métrica) — solo el
board — así que es un check rápido. Es complementario a ``diagnose_depth``,
que mide exactitud ABSOLUTA en metros (y sí necesita distancia conocida):
uno dice "la geometría estéreo sigue siendo la calibrada", el otro "el depth
da bien en metros".

Uso
---
    PYTHONPATH=. python3 scripts/diagnose_calibration.py
    # Poné el board ChArUco frontal a ~1-1.5m, visible en ambas cámaras.

Veredicto: RMS epipolar ≤ umbral → "calibración OK"; arriba → "recalibrar".
Un par sano da sub-píxel; el default de 1.0px tolera ruido de detección y
dispara solo ante drift geométrico real (que mueve las esquinas varios px).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config.hardware import load_hardware_params
from src.config.loader import DEFAULT_DEVICE_CONFIG_PATH, load_device_config
from src.vision.calibration import (
    create_charuco_board,
    detect_charuco_dual_pass,
    load_calibration,
    rectify_pair,
)
from src.vision.capture import StereoCapture

# Umbral default del RMS epipolar (px). Un par bien rectificado da sub-píxel
# (<0.5px); el ruido de detección sub-pixel aporta ~0.1-0.3px. 1.0px deja
# margen al ruido y dispara solo ante drift geométrico real — medio grado de
# yaw ya corre las esquinas varios px.
DEFAULT_EPIPOLAR_THRESHOLD_PX = 1.0

# Esquinas matcheadas mínimas (sumadas sobre todos los frames) para emitir un
# veredicto confiable. Por debajo, el board no se vió bien y el resultado no
# es concluyente (no es ni PASS ni FAIL — es "repetir la medición").
DEFAULT_MIN_MATCHED_CORNERS = 20


def match_corners(
    corners_l: Optional[np.ndarray],
    ids_l: Optional[np.ndarray],
    corners_r: Optional[np.ndarray],
    ids_r: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Matchea esquinas ChArUco entre L y R por ID compartido.

    Devuelve ``(pts_l, pts_r, ids)`` donde ``pts_l``/``pts_r`` son arrays
    ``(N, 2)`` float de las esquinas correspondientes (mismo orden), e
    ``ids`` la lista de IDs matcheados. Arrays vacíos si no hay match o
    alguna detección es None.
    """
    empty = (np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64), [])
    if corners_l is None or ids_l is None or corners_r is None or ids_r is None:
        return empty

    pts_l_by_id = {
        int(i): c.reshape(2) for i, c in zip(ids_l.flatten(), corners_l.reshape(-1, 2))
    }
    pts_r_by_id = {
        int(i): c.reshape(2) for i, c in zip(ids_r.flatten(), corners_r.reshape(-1, 2))
    }
    shared = sorted(set(pts_l_by_id) & set(pts_r_by_id))
    if not shared:
        return empty
    pts_l = np.array([pts_l_by_id[i] for i in shared], dtype=np.float64)
    pts_r = np.array([pts_r_by_id[i] for i in shared], dtype=np.float64)
    return pts_l, pts_r, shared


def epipolar_stats(pts_l: np.ndarray, pts_r: np.ndarray) -> dict[str, float]:
    """Estadísticas del residual epipolar de esquinas rectificadas matcheadas.

    En un par rectificado, las correspondencias caen en la misma fila →
    ``y_L - y_R ≈ 0``. Reporta RMS y máximo del residual vertical (px), y la
    disparidad horizontal mediana (px, ``x_L - x_R``, informativa: positiva y
    coherente con la distancia del board).
    """
    n = int(pts_l.shape[0])
    if n == 0:
        return {
            "n": 0,
            "rms_epipolar_px": float("nan"),
            "max_epipolar_px": float("nan"),
            "median_disparity_px": float("nan"),
        }
    dy = pts_l[:, 1] - pts_r[:, 1]
    dx = pts_l[:, 0] - pts_r[:, 0]
    return {
        "n": n,
        "rms_epipolar_px": float(np.sqrt(np.mean(dy**2))),
        "max_epipolar_px": float(np.max(np.abs(dy))),
        "median_disparity_px": float(np.median(dx)),
    }


def calibration_verdict(
    rms_epipolar_px: float,
    n_matched: int,
    threshold_px: float,
    min_matched: int,
) -> tuple[str, str]:
    """Veredicto de salud de calibración.

    Devuelve ``(verdict, detail)`` donde verdict ∈
    {``"OK"``, ``"RECALIBRAR"``, ``"INCONCLUSO"``}.
    """
    if n_matched < min_matched or not np.isfinite(rms_epipolar_px):
        return (
            "INCONCLUSO",
            f"solo {n_matched} esquinas matcheadas (mín {min_matched}) — "
            "reposicioná el board (más cerca/centrado) y repetí",
        )
    if rms_epipolar_px <= threshold_px:
        return (
            "OK",
            f"RMS epipolar {rms_epipolar_px:.2f}px ≤ {threshold_px:.2f}px — "
            "la calibración sigue siendo válida",
        )
    return (
        "RECALIBRAR",
        f"RMS epipolar {rms_epipolar_px:.2f}px > {threshold_px:.2f}px — "
        "la geometría estéreo cambió respecto a la calibración guardada",
    )


def _draw_annotated(
    rect_l: np.ndarray,
    pts_l: np.ndarray,
    pts_r: np.ndarray,
    threshold_px: float,
) -> np.ndarray:
    """Dibuja las esquinas matcheadas sobre la L rectificada, coloreadas por
    residual epipolar (verde ok / amarillo límite / rojo fuera)."""
    vis = rect_l.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for (xl, yl), (_, yr) in zip(pts_l, pts_r):
        err = abs(yl - yr)
        if err <= threshold_px:
            color = (0, 200, 0)
        elif err <= 2 * threshold_px:
            color = (0, 200, 200)
        else:
            color = (0, 0, 255)
        cv2.circle(vis, (int(round(xl)), int(round(yl))), 5, color, 2)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor de salud de calibración estéreo (error epipolar)"
    )
    parser.add_argument("--calibration", default="/etc/people-counter/calibration.npz")
    parser.add_argument(
        "--frames",
        type=int,
        default=15,
        help="Pares a capturar y agregar. Default 15.",
    )
    parser.add_argument(
        "--epipolar-threshold-px",
        type=float,
        default=DEFAULT_EPIPOLAR_THRESHOLD_PX,
        help=f"Umbral del RMS epipolar para PASS. Default "
        f"{DEFAULT_EPIPOLAR_THRESHOLD_PX}px.",
    )
    parser.add_argument(
        "--min-matched-corners",
        type=int,
        default=DEFAULT_MIN_MATCHED_CORNERS,
        help="Esquinas matcheadas mínimas (sumadas) para un veredicto "
        f"confiable. Default {DEFAULT_MIN_MATCHED_CORNERS}.",
    )
    parser.add_argument("--board-cols", type=int, default=None)
    parser.add_argument("--board-rows", type=int, default=None)
    parser.add_argument("--square-mm", type=float, default=None)
    parser.add_argument("--marker-mm", type=float, default=None)
    parser.add_argument("--dict", dest="aruco_dict", default=None)
    parser.add_argument(
        "--legacy-pattern", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--meter",
        choices=("matrix", "centre", "spot"),
        default="matrix",
        help="Modo de metering AE.",
    )
    parser.add_argument("--lock-ae", action="store_true")
    parser.add_argument(
        "--max-exposure-us",
        type=int,
        default=16000,
        help="Cap de exposure via FrameDurationLimits. Default 16000 "
        "(mismo que el runtime). 0 deshabilita.",
    )
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--output-dir", default="/tmp")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    def _emit(*a, **kw) -> None:
        if not args.quiet:
            print(*a, **kw)

    # CSI de cámara desde el config per-device strict (mismo mapping que el
    # runtime) — sin fallback al example.
    try:
        device_cfg = load_device_config(DEFAULT_DEVICE_CONFIG_PATH)
    except FileNotFoundError:
        parser.error(
            f"{DEFAULT_DEVICE_CONFIG_PATH} no existe. Aprovisioná el config "
            "per-device antes de correr diagnose_calibration."
        )
    bracket = device_cfg.get("bracket") or {}
    if "camera_left_csi" not in bracket or "camera_right_csi" not in bracket:
        parser.error(
            f"bracket.camera_left_csi / camera_right_csi no definidos en "
            f"{DEFAULT_DEVICE_CONFIG_PATH}."
        )
    cam_left = bracket["camera_left_csi"]
    cam_right = bracket["camera_right_csi"]

    cal_path = Path(args.calibration)
    if not cal_path.exists():
        parser.error(
            f"Calibración no encontrada: {cal_path}. Esta tool es "
            "POST-calibración — corré el wizard de calibración primero."
        )
    cal = load_calibration(str(cal_path))

    # La resolución de captura tiene que matchear la de la calibración (los
    # rectify maps están dimensionados a ella). Ver diagnose_depth para el
    # detalle del clipping silencioso si no matchea.
    if "image_size" in cal:
        cal_w, cal_h = (int(v) for v in cal["image_size"])
        resolution = (cal_w, cal_h)
    else:
        map_h, map_w = cal["map_l_x"].shape[:2]
        resolution = (map_w, map_h)

    # Params del board: flags CLI > config (HardwareParams) > defaults de fleet.
    hw = load_hardware_params()
    board = create_charuco_board(
        board_size=(
            args.board_cols or hw.board_cols,
            args.board_rows or hw.board_rows,
        ),
        square_length=args.square_mm or hw.square_mm,
        marker_length=args.marker_mm or hw.marker_mm,
        dict_id=getattr(
            cv2.aruco,
            (
                (args.aruco_dict or hw.aruco_dict)
                if (args.aruco_dict or hw.aruco_dict).startswith("DICT_")
                else f"DICT_{args.aruco_dict or hw.aruco_dict}"
            ),
        ),
        legacy_pattern=(
            hw.legacy_pattern if args.legacy_pattern is None else args.legacy_pattern
        ),
    )

    _emit("=" * 70)
    _emit("MONITOR DE SALUD DE CALIBRACIÓN")
    _emit("=" * 70)
    _emit(f"Calibración: {cal_path}  (rectify maps {resolution[0]}×{resolution[1]})")
    baseline_mm = float(np.linalg.norm(cal["T"]))
    fx = float(cal["P1"][0, 0])
    _emit(f"Baseline ||T||={baseline_mm:.1f}mm  fx(P1)={fx:.1f}px")
    _emit(f"Capturando {args.frames} frames del board...")

    cap = StereoCapture(
        cam_left_id=cam_left,
        cam_right_id=cam_right,
        resolution=resolution,
        fps=5,
        meter_mode=args.meter,
        lock_ae=args.lock_ae,
        max_exposure_us=(args.max_exposure_us if args.max_exposure_us > 0 else None),
        sensor_raw_size=hw.default_res,
        initial_settle_seconds=hw.ae_initial_settle_seconds,
        resettle_seconds=hw.ae_resettle_seconds,
    )
    cap.open()

    all_dy: list[float] = []
    all_dx: list[float] = []
    frames_detected = 0
    last_annotated: Optional[np.ndarray] = None
    last_rect_l: Optional[np.ndarray] = None
    last_rect_r: Optional[np.ndarray] = None
    try:
        for _ in range(max(1, args.frames)):
            left, right = cap.read()
            rect_l, rect_r = rectify_pair(left, right, cal)
            last_rect_l, last_rect_r = rect_l, rect_r
            c_l, i_l = detect_charuco_dual_pass(rect_l, board, min_corners=4)
            c_r, i_r = detect_charuco_dual_pass(rect_r, board, min_corners=4)
            pts_l, pts_r, ids = match_corners(c_l, i_l, c_r, i_r)
            if pts_l.shape[0] == 0:
                continue
            frames_detected += 1
            all_dy.extend((pts_l[:, 1] - pts_r[:, 1]).tolist())
            all_dx.extend((pts_l[:, 0] - pts_r[:, 0]).tolist())
            last_annotated = _draw_annotated(
                rect_l, pts_l, pts_r, args.epipolar_threshold_px
            )
    finally:
        cap.close()

    n = len(all_dy)
    dy_arr = np.array(all_dy, dtype=np.float64)
    dx_arr = np.array(all_dx, dtype=np.float64)
    rms_epipolar = float(np.sqrt(np.mean(dy_arr**2))) if n else float("nan")
    max_epipolar = float(np.max(np.abs(dy_arr))) if n else float("nan")
    median_disp = float(np.median(dx_arr)) if n else float("nan")
    approx_dist_mm = (
        fx * baseline_mm / median_disp if (n and median_disp > 0.1) else float("nan")
    )

    verdict, detail = calibration_verdict(
        rms_epipolar, n, args.epipolar_threshold_px, args.min_matched_corners
    )

    _emit(f"\n{'=' * 70}")
    _emit("RESULTADO")
    _emit("=" * 70)
    _emit(f"Frames con board detectado: {frames_detected}/{args.frames}")
    _emit(f"Esquinas matcheadas (total): {n}")
    _emit(
        f"RMS epipolar:   {rms_epipolar:.3f} px   (umbral {args.epipolar_threshold_px:.2f})"
    )
    _emit(f"Máx epipolar:   {max_epipolar:.3f} px")
    _emit(f"Disparidad mediana: {median_disp:.1f} px  (≈ {approx_dist_mm/1000:.2f} m)")
    _emit(f"\nVEREDICTO: {verdict} — {detail}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if last_annotated is not None:
        cv2.imwrite(str(out_dir / "calib_health_corners.jpg"), last_annotated)
    if last_rect_l is not None:
        cv2.imwrite(str(out_dir / "calib_health_rect_l.jpg"), last_rect_l)
    if last_rect_r is not None:
        cv2.imwrite(str(out_dir / "calib_health_rect_r.jpg"), last_rect_r)

    if args.json_path:
        payload = {
            "calibration_path": str(cal_path),
            "frames_requested": int(args.frames),
            "frames_detected": int(frames_detected),
            "matched_corners": int(n),
            "rms_epipolar_px": (
                None if not np.isfinite(rms_epipolar) else rms_epipolar
            ),
            "max_epipolar_px": (
                None if not np.isfinite(max_epipolar) else max_epipolar
            ),
            "median_disparity_px": (
                None if not np.isfinite(median_disp) else median_disp
            ),
            "approx_distance_mm": (
                None if not np.isfinite(approx_dist_mm) else approx_dist_mm
            ),
            "threshold_px": float(args.epipolar_threshold_px),
            "verdict": verdict,
            "detail": detail,
        }
        json_out = Path(args.json_path)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _emit(f"\nJSON guardado: {json_out}")


if __name__ == "__main__":
    main()

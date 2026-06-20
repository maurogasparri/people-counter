#!/usr/bin/env python3
"""Tool de focus assist para las cámaras estéreo Arducam IMX708.

Tool de foco guiado que valida la nitidez en 9 zonas por cámara, chequea
la simetría L/R, detecta el board ChArUco in-scene para validar la
distancia al target, y da un veredicto visual claro PASS/FAIL.

Uso:
    PYTHONPATH=. python3 scripts/focus_assist.py

    Después abrir: http://people-counter.local:8080

    Poner el board de calibración ChArUco a 1.5m ±20cm de las cámaras y
    ajustar el ring de cada lens mirando las barras en vivo. Click
    FINALIZAR cuando ambas cámaras pasen. Protocolo universal de lab:
    foco a 1.5m maximiza el DoF sobre el rango de profundidad operativa
    de la flota (cabezas + piso para mount 2.0–3.5m), con margen a ambos
    extremos. Ver docs/lab_calibration_guide.md para el protocolo completo.
"""

import argparse
import datetime as _dt
import os
import sys
import threading
import time
import unicodedata
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Agregar la raíz del proyecto al path para los imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.hardware import HardwareParams, load_hardware_params
from src.vision.calibration import (
    create_charuco_board,
    detect_charuco_dual_pass,
    live_lighting_warnings,
)

# Hardware params canónicos del device — leídos al main(). Inicializado
# con fleet defaults para que las funciones top-level (estimate_charuco_distance_mm)
# tengan algo razonable durante import / tests.
from src.config.hardware import FLEET_DEFAULTS as _FALLBACK_HW

HW: HardwareParams = _FALLBACK_HW

latest_jpeg: bytes = b""
jpeg_lock = threading.Lock()
shutting_down = False
finish_requested = False
# Se setea True cuando el operador apreta "Comenzar" en el browser.
# Bloquea el loop principal de captura hasta que eso pase para que (a)
# el operador tenga tiempo de posicionarse y (b) el click unlockee el
# AudioContext del browser para los beeps.
capture_started = False
capture_started_lock = threading.Lock()

MIN_SCORE = 200
MIN_CORNER_SCORE = 100.0  # Variance Laplaciana absoluta para
# las zonas de corner. Reemplaza al
# ratio bordes/centro viejo: en
# cuartos chicos de test el board
# llena el centro y arrastra el ratio
# cerca de cero incluso con lentes
# buenos. Medir corners en su propia
# escala es honesto entre escenas.
MAX_LR_DIFF_PCT = 15.0
MAX_LR_ZONE_DIFF_PCT = 30.0
TARGET_DISTANCE_MIN_MM = 1300.0  # Protocolo lab: foco a 1.5m ±20cm
TARGET_DISTANCE_MAX_MM = 1700.0  # Target universal para toda la flota —
# IMX708 + M12 f/2.0 con foco a 1.5m
# peakea el DoF sobre el rango
# operativo cabezas+piso 1.0–3.5m
# (mount 2.0–3.5m), con margen
# simétrico en ambos extremos. Ver
# docs/lab_calibration_guide.md.
DEFAULT_MOUNT_HEIGHT_M = 3.0  # Moda de nuestra distribución de altura de puerta
HEAD_HEIGHT_MAX_M = 1.85  # adulto alto
HEAD_HEIGHT_MIN_M = 1.20  # niño bajo
COMPACT_BBOX_THRESHOLD = 0.25  # Área bbox del board/frame > esto ->
# escena compacta (el board llena la
# vista, los corners ven paredes a
# profundidades muy distintas, así
# que la corner sharpness no es
# comparable al centro y salteamos
# ese check).


# Presets relajados para corridas PoC en luz baja / cuartos chicos. NO
# es una pasada productiva de foco — el "PASS" resultante solo confirma
# que el wizard corre, no que el lens esté realmente lo suficientemente
# enfocado para depth.
_LOW_LIGHT_DEFAULTS = {
    "min_score": 80.0,
    "min_corner_score": 30.0,
    "max_lr_diff_pct": 50.0,
    "max_lr_zone_diff_pct": 100.0,
    "target_distance_min_mm": 500.0,
    "target_distance_max_mm": 5000.0,
    "scene": "compact",
    "meter": "centre",
}


def _apply_threshold_overrides(args: argparse.Namespace) -> None:
    """Overridea los thresholds de calidad de foco desde flags CLI.
    Permite a los operadores aflojar checks para ambientes complicados
    (vidrio frontal, luz baja) sin editar código.

    Orden de resolución por setting: flag CLI explícito > preset
    --low-light > derivación geométrica desde mount_height > default
    productivo. Los defaults sentinela ``None`` en los args identifican
    "el operador no pasó este flag explícitamente".
    """
    global MIN_SCORE, MIN_CORNER_SCORE, MAX_LR_DIFF_PCT, MAX_LR_ZONE_DIFF_PCT
    global TARGET_DISTANCE_MIN_MM, TARGET_DISTANCE_MAX_MM

    low_light = getattr(args, "low_light", False)

    def _resolve(name: str, prod_default: float) -> float:
        explicit = getattr(args, name)
        if explicit is not None:
            return explicit
        if low_light:
            return _LOW_LIGHT_DEFAULTS[name]
        return prod_default

    MIN_SCORE = _resolve("min_score", MIN_SCORE)
    MIN_CORNER_SCORE = _resolve("min_corner_score", MIN_CORNER_SCORE)
    MAX_LR_DIFF_PCT = _resolve("max_lr_diff_pct", MAX_LR_DIFF_PCT)
    MAX_LR_ZONE_DIFF_PCT = _resolve("max_lr_zone_diff_pct", MAX_LR_ZONE_DIFF_PCT)

    # Distancia target: si no se pasa explícito y no estamos en --low-light,
    # deriva geométricamente desde mount_height. distance = mount - head_height
    # cubre el rango de cabezas (HEAD_MIN..HEAD_MAX), que es lo que el lens
    # tiene que enfocar para que la detección sea nítida en producción. Para
    # mount=3m da 1.15-1.80m, casi idéntico al universal 1.3-1.7m. Para mount
    # más bajo (testing, sites con techos bajos) se adapta automático.
    explicit_min = args.target_distance_min_mm
    explicit_max = args.target_distance_max_mm
    if explicit_min is not None or explicit_max is not None or low_light:
        TARGET_DISTANCE_MIN_MM = _resolve(
            "target_distance_min_mm",
            TARGET_DISTANCE_MIN_MM,
        )
        TARGET_DISTANCE_MAX_MM = _resolve(
            "target_distance_max_mm",
            TARGET_DISTANCE_MAX_MM,
        )
    else:
        derived_min_mm = max(
            200.0,
            (args.mount_height_m - HEAD_HEIGHT_MAX_M) * 1000,
        )
        derived_max_mm = max(
            derived_min_mm + 200.0,
            (args.mount_height_m - HEAD_HEIGHT_MIN_M) * 1000,
        )
        TARGET_DISTANCE_MIN_MM = derived_min_mm
        TARGET_DISTANCE_MAX_MM = derived_max_mm
        print(
            f"[focus] Target distance derivado de mount_height "
            f"{args.mount_height_m:.2f}m + head {HEAD_HEIGHT_MIN_M:.2f}-"
            f"{HEAD_HEIGHT_MAX_M:.2f}m → "
            f"{TARGET_DISTANCE_MIN_MM/1000:.2f}-"
            f"{TARGET_DISTANCE_MAX_MM/1000:.2f}m. Override con "
            f"--target-distance-min/max-mm.",
            flush=True,
        )

    if low_light and args.scene == "auto":
        args.scene = _LOW_LIGHT_DEFAULTS["scene"]
    if low_light and args.meter == "matrix":
        args.meter = _LOW_LIGHT_DEFAULTS["meter"]
    if low_light:
        print(
            "[low-light] Umbrales aflojados (centro/corners/L-R/distancia) y "
            "scene=compact. PoC only — no confiar en este PASS para producción.",
            flush=True,
        )


def focus_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


MIN_ZONE_STD = 8.0  # Bajo este std la zona es casi uniforme (pared
# plana, superficie lisa) — la varianza Laplaciana
# ahí no significa nada como indicador de foco, así
# que la flageamos como "sin contenido".

# Modo mapa de foco (default): varianza Laplaciana mínima para contar una zona
# "cubierta" (el board pasó por ahí). Bajo a propósito — registra el board aun
# medio blando; el check de calidad (corner >= MIN_CORNER_SCORE) flagea las
# zonas blandas aparte. Tuneable con --map-coverage-min.
FOCUS_MAP_COVERAGE_MIN = 60.0
ZONE_NAMES = (
    ("arriba-izq", "arriba-centro", "arriba-der"),
    ("centro-izq", "centro", "centro-der"),
    ("abajo-izq", "abajo-centro", "abajo-der"),
)


def focus_grid(
    frame: np.ndarray,
    rows: int = 3,
    cols: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Computa scores de foco per-zona + una máscara de validez.

    La máscara es False donde la zona tiene muy poco contraste (std <
    MIN_ZONE_STD en escala 0-255) — no hay contenido de borde para
    medir, así que la varianza Laplaciana ahí no refleja la calidad
    de foco. Los cálculos downstream (uniformidad, simetría) saltean
    zonas inválidas.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores = np.zeros((rows, cols), dtype=np.float64)
    valid = np.ones((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h // rows, (r + 1) * h // rows
            x1, x2 = c * w // cols, (c + 1) * w // cols
            patch = gray[y1:y2, x1:x2]
            scores[r, c] = cv2.Laplacian(patch, cv2.CV_64F).var()
            if float(patch.std()) < MIN_ZONE_STD:
                valid[r, c] = False
    return scores, valid


def estimate_charuco_distance_mm(
    frame: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    focal_px_override: float | None = None,
) -> tuple[float | None, int, float, float | None]:
    """Detecta ChArUco en el frame, estima la distancia vía solvePnP con K nominal.

    Devuelve (distance_mm o None, n_corners_detected, bbox_ratio,
              centroid_x o None). bbox_ratio es la fracción del área
    del frame cubierta por el rectángulo bounding de los corners
    detectados. centroid_x es la coordenada x media de los corners
    detectados (en píxeles del frame) — usado por el check de
    parity L/R upstream. Usa intrínsecos nominales de IMX708 — ±10%
    está bien para validar "¿el board está a 2.5-3m?".
    """
    corners, ids = detect_charuco_dual_pass(frame, board, min_corners=4)
    if corners is None or ids is None or len(corners) < 4:
        return None, 0, 0.0, None
    h, w = frame.shape[:2]
    pts = corners.reshape(-1, 2)
    bbox_w = float(pts[:, 0].max() - pts[:, 0].min())
    bbox_h = float(pts[:, 1].max() - pts[:, 1].min())
    frame_area = max(w * h, 1)
    bbox_ratio = (bbox_w * bbox_h) / frame_area
    centroid_x = float(pts[:, 0].mean())
    if focal_px_override is not None:
        fx = fy = focal_px_override
    else:
        scale_x = w / HW.full_res[0]
        scale_y = h / HW.full_res[1]
        fx = HW.nominal_focal_full_px * scale_x
        fy = HW.nominal_focal_full_px * scale_y
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    obj_pts = board.getChessboardCorners()[ids.flatten()].astype(np.float32)
    img_pts = corners.reshape(-1, 2).astype(np.float32)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error:
        return None, len(corners), bbox_ratio, centroid_x
    if not ok:
        return None, len(corners), bbox_ratio, centroid_x
    return float(tvec[2, 0]), len(corners), bbox_ratio, centroid_x


def evaluate_focus(
    grid_l: np.ndarray,
    grid_r: np.ndarray,
    distance_l_mm: float | None,
    distance_r_mm: float | None,
    valid_l: np.ndarray | None = None,
    valid_r: np.ndarray | None = None,
    compact_scene: bool = False,
) -> dict:
    if valid_l is None:
        valid_l = np.ones_like(grid_l, dtype=bool)
    if valid_r is None:
        valid_r = np.ones_like(grid_r, dtype=bool)

    center_l, center_r = grid_l[1, 1], grid_r[1, 1]
    corners_idx = [(0, 0), (0, 2), (2, 0), (2, 2)]

    # Solo incluir corners con suficiente contraste para llevar una
    # señal de sharpness significativa. Una zona de pared lisa con
    # std~0 tiene varianza Laplaciana cerca de 0 sin importar el foco —
    # incluirla distorsiona el promedio de los bordes.
    corners_l_vals = [grid_l[r, c] for r, c in corners_idx if valid_l[r, c]]
    corners_r_vals = [grid_r[r, c] for r, c in corners_idx if valid_r[r, c]]
    n_valid_corners_l = len(corners_l_vals)
    n_valid_corners_r = len(corners_r_vals)
    # Sharpness de corners absoluta (var Laplaciana media entre los
    # corners válidos). Comparable directamente con MIN_CORNER_SCORE —
    # un lens que realmente resuelve detalle en los corners va a leer
    # >= threshold sin importar lo que muestra el centro. El ratio
    # viejo (bordes/centro) fallaba en cuartos chicos porque un
    # ChArUco brillante centrado en el frame infl aba el denominador.
    corner_l = float(np.mean(corners_l_vals)) if corners_l_vals else 0.0
    corner_r = float(np.mean(corners_r_vals)) if corners_r_vals else 0.0

    # Promedio global y zone diffs solo sobre zonas válidas en AMBOS
    # lados, así comparar cámaras no penaliza una zona donde solo una
    # tiene contenido.
    both_valid = valid_l & valid_r
    if both_valid.any():
        global_l = float(grid_l[both_valid].mean())
        global_r = float(grid_r[both_valid].mean())
        zone_diffs_full = (
            np.abs(grid_l - grid_r) / np.maximum(grid_l, grid_r).clip(1) * 100
        )
        # Solo considerar zonas donde al menos un lado lleva sharpness
        # significativo. Dos zonas cerca de cero (sombra, parche de
        # bajo contraste) pueden producir diffs relativos enormes para
        # una diferencia absoluta minúscula — no una asimetría real
        # de lens.
        SIGNIFICANCE_THRESHOLD = 100.0
        significant = (
            np.maximum(grid_l, grid_r) >= SIGNIFICANCE_THRESHOLD
        ) & both_valid
        if significant.any():
            max_zone_diff = float(zone_diffs_full[significant].max())
        else:
            max_zone_diff = 0.0
    else:
        global_l = float(grid_l.mean())
        global_r = float(grid_r.mean())
        max_zone_diff = 0.0
    lr_diff = abs(global_l - global_r) / max(global_l, global_r, 1) * 100

    distance_avg = None
    if distance_l_mm is not None and distance_r_mm is not None:
        distance_avg = (distance_l_mm + distance_r_mm) / 2
    elif distance_l_mm is not None:
        distance_avg = distance_l_mm
    elif distance_r_mm is not None:
        distance_avg = distance_r_mm

    # Ambas cámaras tienen que detectar Y estar en rango. Caer a la
    # que detectó (el comportamiento previo) dejaba que LISTO se
    # disparara cuando una cámara estaba ciega al board —
    # enmascarando lentes sucios, sensores muertos o problemas de
    # exposición asimétrica detrás de un verdict PASS verde.
    distance_ok = (
        distance_l_mm is not None
        and distance_r_mm is not None
        and TARGET_DISTANCE_MIN_MM <= distance_l_mm <= TARGET_DISTANCE_MAX_MM
        and TARGET_DISTANCE_MIN_MM <= distance_r_mm <= TARGET_DISTANCE_MAX_MM
    )

    # El check de uniformidad pasa a ser "pass por default" cuando no
    # tuvimos suficientes corners válidos O estamos en una escena
    # compacta donde los corners ven paredes a profundidades no
    # relacionadas con el plano del board (el check fallaría
    # estructuralmente, no porque el lens sea malo). Flageado vía
    # uniformity_measurable=False.
    uniformity_l_measurable = n_valid_corners_l >= 2 and not compact_scene
    uniformity_r_measurable = n_valid_corners_r >= 2 and not compact_scene

    checks = {
        "center_l": center_l >= MIN_SCORE,
        "center_r": center_r >= MIN_SCORE,
        "uniformity_l": (not uniformity_l_measurable) or corner_l >= MIN_CORNER_SCORE,
        "uniformity_r": (not uniformity_r_measurable) or corner_r >= MIN_CORNER_SCORE,
        "lr_global": lr_diff <= MAX_LR_DIFF_PCT,
        "lr_zones": max_zone_diff <= MAX_LR_ZONE_DIFF_PCT,
        "distance": distance_ok,
    }
    all_pass = all(v for k, v in checks.items() if k != "distance")
    all_pass_with_distance = all_pass and distance_ok

    # Hints accionables
    hints: list[str] = []
    target_lo = TARGET_DISTANCE_MIN_MM / 1000
    target_hi = TARGET_DISTANCE_MAX_MM / 1000
    if distance_l_mm is None and distance_r_mm is None:
        hints.append("Poné el board ChArUco en la escena para validar la distancia")
    elif distance_l_mm is None:
        hints.append(
            "IZQ no detecta el board — limpiá el lente, chequeá foco "
            "o exposición (probá --meter centre / --low-light)"
        )
    elif distance_r_mm is None:
        hints.append(
            "DER no detecta el board — limpiá el lente, chequeá foco "
            "o exposición (probá --meter centre / --low-light)"
        )
    elif not distance_ok:
        # Ambos detectaron pero al menos uno está fuera de rango.
        # Elegir el lado que es peor offender así el operador sabe a
        # dónde mover el board.
        offender = (
            distance_l_mm
            if abs(
                distance_l_mm - (TARGET_DISTANCE_MIN_MM + TARGET_DISTANCE_MAX_MM) / 2
            )
            > abs(distance_r_mm - (TARGET_DISTANCE_MIN_MM + TARGET_DISTANCE_MAX_MM) / 2)
            else distance_r_mm
        )
        if offender < TARGET_DISTANCE_MIN_MM:
            hints.append(
                f"Board muy cerca ({distance_avg/1000:.2f}m). Objetivo: {target_lo:.2f}-{target_hi:.2f}m"
            )
        else:
            hints.append(
                f"Board muy lejos ({distance_avg/1000:.2f}m). Objetivo: {target_lo:.2f}-{target_hi:.2f}m"
            )
    if not checks["center_l"]:
        hints.append(
            f"IZQ: centro débil ({center_l:.0f}<{MIN_SCORE}) — girá el lente izquierdo"
        )
    if not checks["center_r"]:
        hints.append(
            f"DER: centro débil ({center_r:.0f}<{MIN_SCORE}) — girá el lente derecho"
        )
    if checks["center_l"] and uniformity_l_measurable and not checks["uniformity_l"]:
        hints.append(
            f"IZQ: corners débiles ({corner_l:.0f}<{MIN_CORNER_SCORE:.0f}) — revisá foco / agregá textura en los bordes"
        )
    if checks["center_r"] and uniformity_r_measurable and not checks["uniformity_r"]:
        hints.append(
            f"DER: corners débiles ({corner_r:.0f}<{MIN_CORNER_SCORE:.0f}) — revisá foco / agregá textura en los bordes"
        )
    # Solo molestar con corners de baja textura en modo "full" — en
    # modo compact la UI muestra un banner dedicado explicando que
    # los corners se saltean intencionalmente.
    if not compact_scene and (
        not uniformity_l_measurable or not uniformity_r_measurable
    ):
        hints.append(
            "Escena con poca textura en los bordes — agregá detalle (poster/empapelado) "
            "detrás de la zona de medición para validar uniformidad"
        )
    if not checks["lr_global"]:
        side = "IZQ" if global_l < global_r else "DER"
        hints.append(
            f"Cámara {side} más blanda que la otra ({lr_diff:.0f}% de diferencia)"
        )
    if not checks["lr_zones"]:
        hints.append(
            f"Asimetría por zona L/R: hasta {max_zone_diff:.0f}% entre zonas "
            f"correspondientes (máx permitido {MAX_LR_ZONE_DIFF_PCT:.0f}%). "
            f"Revisá alineación mecánica del par."
        )
    if all_pass_with_distance:
        hints = ["LISTO — fijá los lentes y pasá a calibración"]

    return {
        "center_l": center_l,
        "center_r": center_r,
        "corner_l": corner_l,
        "corner_r": corner_r,
        "lr_diff": lr_diff,
        "max_zone_diff": max_zone_diff,
        "distance_l_mm": distance_l_mm,
        "distance_r_mm": distance_r_mm,
        "distance_avg_mm": distance_avg,
        "distance_ok": distance_ok,
        "checks": checks,
        "all_pass": all_pass,
        "all_pass_with_distance": all_pass_with_distance,
        "hints": hints,
        "global_l": global_l,
        "global_r": global_r,
        "uniformity_l_measurable": uniformity_l_measurable,
        "uniformity_r_measurable": uniformity_r_measurable,
        "n_valid_corners_l": n_valid_corners_l,
        "n_valid_corners_r": n_valid_corners_r,
        "compact_scene": compact_scene,
    }


def focus_map_update(
    map_grid: np.ndarray,
    covered: np.ndarray,
    grid: np.ndarray,
    valid: np.ndarray,
    cov_min: float,
) -> None:
    """Acumula la nitidez máxima por zona (mapa de foco) y marca cobertura.

    Muta ``map_grid`` (max-por-zona donde la zona tiene contenido válido) y
    ``covered`` (True cuando el max acumulado supera ``cov_min`` — el board
    pasó por esa zona). Grillas 3×3.
    """
    for r in range(3):
        for c in range(3):
            if valid[r, c] and grid[r, c] > map_grid[r, c]:
                map_grid[r, c] = float(grid[r, c])
            if map_grid[r, c] >= cov_min:
                covered[r, c] = True


def focus_map_missing(covered_l: np.ndarray, covered_r: np.ndarray) -> list:
    """Nombres de las zonas aún sin cubrir en L o en R (mové el board ahí)."""
    miss = []
    for r in range(3):
        for c in range(3):
            if not (covered_l[r, c] and covered_r[r, c]):
                miss.append(ZONE_NAMES[r][c])
    return miss


def _focus_map_grid_html(
    covered_l: np.ndarray,
    covered_r: np.ndarray,
    map_l: np.ndarray,
    map_r: np.ndarray,
) -> str:
    """Mini-grilla 3×3 de cobertura del mapa de foco (verde=cubierta) por cámara."""

    def _one(covered: np.ndarray, mp: np.ndarray, label: str) -> str:
        cells = ""
        for r in range(3):
            row = "".join(
                '<span style="display:inline-block;width:30px;height:30px;'
                "margin:1px;border-radius:4px;font-size:10px;color:#fff;"
                "text-align:center;line-height:30px;background:"
                + ("#27ae60" if covered[r, c] else "#3a3a42")
                + f'">{int(mp[r, c]) if covered[r, c] else ""}</span>'
                for c in range(3)
            )
            cells += f"<div>{row}</div>"
        return (
            '<div style="display:inline-block;margin-right:16px;vertical-align:top">'
            f'<div style="color:#888;font-size:11px">{label}</div>{cells}</div>'
        )

    return (
        '<div style="margin-top:8px">'
        + _one(covered_l, map_l, "IZQ")
        + _one(covered_r, map_r, "DER")
        + "</div>"
    )


def _ascii(text: str) -> str:
    """Pliega a ASCII para cv2.putText (las fuentes Hershey no soportan unicode)."""
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return (
        stripped.replace("—", "-").replace("–", "-").replace("¿", "?").replace("¡", "!")
    )


LR_DISPARITY_OK_MIN_PX = 8.0  # Por debajo de esto no confiamos en el signo —
# podría ser ruido de parallax en un board chico.


def _expected_disparity_px(distance_mm: float, frame_width_px: int) -> float:
    """Predice la disparity L/R del centroide para un objeto a `distance_mm`.

    disparity_px = baseline * focal_px / depth, donde focal_px escala
    con la resolución de captura (HW.nominal_focal_full_px está referenciada
    a HW.full_res).
    """
    if distance_mm <= 0:
        return 0.0
    scale = frame_width_px / HW.full_res[0]
    f_px = HW.nominal_focal_full_px * scale
    return HW.baseline_mm * f_px / distance_mm


def _classify_lr(
    buffer: list[float],
    expected_px: Optional[float] = None,
) -> dict[str, object]:
    """Decide la parity L/R a partir de los samples recientes de disparity.

    Devuelve un dict con keys:
        state: "ok" | "swapped" | "unknown" | "magnitude_off"
        median_px: mediana de disparity (o None si el buffer está vacío)
        n: count de samples
        expected_px: disparity predicha para la escena actual (o None)

    "magnitude_off" dispara cuando el signo matchea pero la magnitud
    está muy lejos de la predicción — podría ser un baseline
    equivocado en código, un board mal-identificado por la detección,
    o una estimación de depth que drifteó. La marcamos ambigua en
    lugar de verde así el operador recibe el hint de que algo más
    profundo anda mal aunque el wiring L/R en sí esté fine.
    """
    if not buffer:
        return {
            "state": "unknown",
            "median_px": None,
            "n": 0,
            "expected_px": expected_px,
        }
    arr = sorted(buffer)
    median = arr[len(arr) // 2]
    if median > LR_DISPARITY_OK_MIN_PX:
        state = "ok"
    elif median < -LR_DISPARITY_OK_MIN_PX:
        state = "swapped"
    else:
        state = "unknown"
    # Check de magnitud — solo cuando tenemos un valor esperado Y el
    # signo es OK
    if state == "ok" and expected_px is not None and expected_px > 20:
        ratio = median / expected_px
        if ratio < 0.4 or ratio > 2.5:
            state = "magnitude_off"
    return {
        "state": state,
        "median_px": median,
        "n": len(arr),
        "expected_px": expected_px,
    }


class PeakTracker:
    """Tracker de pico con ventana rolling para scores de foco per camera.

    Mantiene el max valor visto en los últimos `window_frames`
    samples. Se usa para decirle al operador si el ajuste actual se
    está moviendo hacia o alejándose del mejor foco logrado
    recientemente — muy útil cuando el aro del M12 es sensible y
    fácil de overshootear.

    Thresholds de estado:
        state == "at_peak"   → current dentro del 5% del pico reciente
        state == "past_peak" → current < 85% del pico reciente
        state == "climbing"  → en el medio (mejorando activamente o todavía lejos)
    """

    def __init__(self, window_frames: int = 40):
        self.samples_l: deque[float] = deque(maxlen=window_frames)
        self.samples_r: deque[float] = deque(maxlen=window_frames)

    def update(self, value_l: float, value_r: float) -> None:
        self.samples_l.append(value_l)
        self.samples_r.append(value_r)

    @property
    def peak_l(self) -> float:
        return max(self.samples_l) if self.samples_l else 0.0

    @property
    def peak_r(self) -> float:
        return max(self.samples_r) if self.samples_r else 0.0

    @staticmethod
    def _classify(current: float, peak: float) -> str:
        if peak <= 0 or len(str(peak)) == 0:
            return "climbing"
        ratio = current / peak if peak > 0 else 0.0
        if ratio >= 0.95:
            return "at_peak"
        if ratio < 0.85:
            return "past_peak"
        return "climbing"

    def state(self, current_l: float, current_r: float) -> tuple[str, str]:
        return (
            self._classify(current_l, self.peak_l),
            self._classify(current_r, self.peak_r),
        )


def _draw_bar(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    target: float,
    label: str,
    max_value: float | None = None,
    peak: float | None = None,
    passing: bool | None = None,
) -> None:
    """Dibuja una barra horizontal: el valor actual rellena; la zona
    verde marca el target.

    El opcional `peak` renderiza un marker vertical fino en el pico
    reciente así el operador puede ver si el ajuste actual está
    subiendo hacia o alejándose del mejor valor visto recientemente.

    El opcional `passing` overridea el color de pass/fail cuando la
    barra depende de múltiples checks (ej. simetría L/R fallando en
    max-zone-diff aunque el diff global esté fine).
    """
    if max_value is None:
        candidate = max(target * 2.5, value * 1.2, 1.0)
        if peak is not None:
            candidate = max(candidate, peak * 1.1)
        max_value = candidate
    # Track
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    # Zona target: de target a max
    target_x = int(x + (target / max_value) * w)
    cv2.rectangle(img, (target_x, y), (x + w, y + h), (40, 90, 40), -1)
    # Fill actual
    fill = max(0.0, min(1.0, value / max_value))
    fill_x = int(x + fill * w)
    is_passing = passing if passing is not None else value >= target
    color = (0, 220, 60) if is_passing else (0, 90, 220)
    cv2.rectangle(img, (x, y), (fill_x, y + h), color, -1)
    # Marker de pico (línea vertical ámbar)
    if peak is not None and peak > 0:
        peak_ratio = max(0.0, min(1.0, peak / max_value))
        peak_x = int(x + peak_ratio * w)
        cv2.line(img, (peak_x, y), (peak_x, y + h), (60, 200, 240), 2)
    # Borde
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    # Label + valor (+ valor de pico si se provee)
    if peak is not None and peak > 0:
        txt = f"{label}: {value:.2f} (pico {peak:.2f})"
    else:
        txt = f"{label}: {value:.2f}"
    cv2.putText(
        img,
        _ascii(txt),
        (x + 4, y + h - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _compose_preview(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    ev: dict,
    grid_l: np.ndarray,
    grid_r: np.ndarray,
    peaks: PeakTracker | None = None,
) -> np.ndarray:
    """Arma la imagen del preview HTTP con barras visuales +
    distancia + banner grande."""
    h = 360  # mitad de altura del preview
    w = 640  # mitad de ancho del preview (16:9 para matchear el aspect nativo del sensor)
    vis_l = cv2.resize(frame_l, (w, h))
    vis_r = cv2.resize(frame_r, (w, h))

    # Overlay del grid de foco como rectángulos semi-transparentes
    # sobre la imagen misma
    for side_vis, grid in [(vis_l, grid_l), (vis_r, grid_r)]:
        rows, cols = grid.shape
        max_g = max(grid.max(), 1.0)
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * h // rows, (r + 1) * h // rows
                x1, x2 = c * w // cols, (c + 1) * w // cols
                ratio = grid[r, c] / max_g
                col = (0, int(ratio * 220), int((1 - ratio) * 220))
                cv2.rectangle(side_vis, (x1, y1), (x2, y2), col, 1)

    cv2.putText(vis_l, "L", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(vis_r, "R", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Panel de barras
    panel_h = 180
    total_w = w * 2
    panel = np.full((panel_h, total_w, 3), 18, dtype=np.uint8)

    # Fila 1: barras de sharpness del centro (con markers de pico)
    bar_w = w - 40
    peak_l_scaled = peaks.peak_l / 10.0 if peaks is not None else None
    peak_r_scaled = peaks.peak_r / 10.0 if peaks is not None else None
    _draw_bar(
        panel,
        20,
        14,
        bar_w,
        22,
        ev["center_l"] / 10.0,
        MIN_SCORE / 10.0,
        f"IZQ nitidez centro: {ev['center_l']:.0f}",
        max_value=max(MIN_SCORE / 10 * 2.5, ev["center_l"] / 10 * 1.2, 1.0),
        peak=peak_l_scaled,
    )
    _draw_bar(
        panel,
        w + 20,
        14,
        bar_w,
        22,
        ev["center_r"] / 10.0,
        MIN_SCORE / 10.0,
        f"DER nitidez centro: {ev['center_r']:.0f}",
        max_value=max(MIN_SCORE / 10 * 2.5, ev["center_r"] / 10 * 1.2, 1.0),
        peak=peak_r_scaled,
    )

    # Fila 2: sharpness de corners (absoluta). En escenas compactas
    # este check se saltea, así que poneselo gris al label de la
    # barra y forzamos pass a su color.
    compact = bool(ev.get("compact_scene"))
    corner_l_label = (
        "IZQ corners (omitido)" if compact else f"IZQ corners: {ev['corner_l']:.0f}"
    )
    corner_r_label = (
        "DER corners (omitido)" if compact else f"DER corners: {ev['corner_r']:.0f}"
    )
    corner_max = max(
        MIN_CORNER_SCORE * 2.5, ev["corner_l"] * 1.2, ev["corner_r"] * 1.2, 100.0
    )
    _draw_bar(
        panel,
        20,
        50,
        bar_w,
        22,
        ev["corner_l"],
        MIN_CORNER_SCORE,
        corner_l_label,
        max_value=corner_max,
        passing=ev["checks"]["uniformity_l"],
    )
    _draw_bar(
        panel,
        w + 20,
        50,
        bar_w,
        22,
        ev["corner_r"],
        MIN_CORNER_SCORE,
        corner_r_label,
        max_value=corner_max,
        passing=ev["checks"]["uniformity_r"],
    )

    # Fila 3: simetría L/R — la barra trackea el diff global, pero
    # también falla en rojo cuando el diff per-zone excede el threshold
    # (indicador visual único para los checks lr_global y lr_zones).
    sym_val = max(0.0, 100.0 - ev["lr_diff"])  # 100 = perfecto, 0 = muy asimétrico
    sym_passing = ev["checks"]["lr_global"] and ev["checks"]["lr_zones"]
    _draw_bar(
        panel,
        20,
        86,
        total_w - 40,
        22,
        sym_val,
        100 - MAX_LR_DIFF_PCT,
        f"Simetría L/R: {100-ev['lr_diff']:.0f}% (max zona diff {ev['max_zone_diff']:.0f}%)",
        max_value=100.0,
        passing=sym_passing,
    )

    # Fila 4: distancia — mostrar L, R, y avg así el operador puede ver divergencia
    if ev["distance_avg_mm"] is not None:
        dist_m = ev["distance_avg_mm"] / 1000
        color = (0, 220, 60) if ev["distance_ok"] else (50, 180, 240)
        dl = ev["distance_l_mm"]
        dr = ev["distance_r_mm"]
        dl_str = f"{dl/1000:.2f}m" if dl is not None else "—"
        dr_str = f"{dr/1000:.2f}m" if dr is not None else "—"
        # Warning de divergencia: >10% de diff entre L y R indica un
        # problema de distorsión en un lens (mal foco o misalignment
        # físico).
        divergence_pct = 0.0
        if dl is not None and dr is not None:
            divergence_pct = abs(dl - dr) / max(dl, dr) * 100
        cv2.putText(
            panel,
            _ascii(f"Distancia board: {dist_m:.2f} m  (IZQ {dl_str} / DER {dr_str})"),
            (20, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        if divergence_pct > 10:
            cv2.putText(
                panel,
                _ascii(
                    f"! IZQ/DER difieren {divergence_pct:.0f}% — revisar lente asimétrico"
                ),
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (50, 180, 240),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                panel,
                _ascii(
                    f"(objetivo {TARGET_DISTANCE_MIN_MM/1000:.2f}-{TARGET_DISTANCE_MAX_MM/1000:.2f} m)"
                ),
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
            )
    else:
        cv2.putText(
            panel,
            _ascii("Board ChArUco no detectado — ponelo frente al par"),
            (20, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 180, 240),
            2,
            cv2.LINE_AA,
        )

    # Banner
    if ev["all_pass_with_distance"]:
        banner_color = (0, 180, 0)
        banner_text = "LISTO — fijá los lentes y pasá a calibración"
    elif ev["all_pass"] and not ev["distance_ok"]:
        banner_color = (50, 150, 200)
        banner_text = ev["hints"][0] if ev["hints"] else "Ajustá distancia"
    else:
        banner_color = (0, 0, 200)
        banner_text = ev["hints"][0] if ev["hints"] else "Ajustando..."

    images_row = np.hstack([vis_l, vis_r])
    frame = np.vstack([images_row, panel])
    fh = frame.shape[0]
    fw = frame.shape[1]
    cv2.rectangle(frame, (0, fh - 36), (fw, fh), banner_color, -1)
    cv2.putText(
        frame,
        _ascii(banner_text),
        (12, fh - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


_status_html = ""
_status_lock = threading.Lock()
_last_eval: dict | None = None
_eval_lock = threading.Lock()
_report_path_global: Path | None = None


def _save_report(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    grid_l: np.ndarray,
    grid_r: np.ndarray,
    ev: dict,
) -> Path:
    """Guarda un reporte HTML con los frames finales + métricas."""
    import base64

    def _b64(img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    ts = _dt.datetime.now()
    out_dir = Path("/tmp") if os.name != "nt" else Path.cwd() / "focus_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"focus_report_{ts:%Y%m%d_%H%M%S}.html"

    def _pill(ok: bool, text: str) -> str:
        bg = "#2ecc71" if ok else "#e74c3c"
        return f'<span style="background:{bg};color:#fff;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600">{"PASS" if ok else "FAIL"}</span> {text}'

    def _grid_html(g: np.ndarray) -> str:
        rows = []
        for r in range(3):
            cells = "".join(f"<td>{g[r, c]:.0f}</td>" for c in range(3))
            rows.append(f"<tr>{cells}</tr>")
        return f'<table border=1 cellpadding=4>{"".join(rows)}</table>'

    dist_text = (
        f"{ev['distance_avg_mm']/1000:.2f} m"
        if ev["distance_avg_mm"] is not None
        else "no detectado"
    )
    compact_note = ""
    if ev.get("compact_scene"):
        compact_note = (
            '<p style="background:#fff3cd;border:1px solid #ffe39a;'
            'padding:8px 12px;border-radius:6px;color:#7a5d00">'
            "Escena compacta detectada — el check de corners fue omitido "
            "porque los bordes del frame ven superficies a distancia "
            "distinta del board. Validación basada en centro + simetría + "
            "distancia.</p>"
        )
    corner_l_label = (
        "IZQ corners: omitido (escena compacta)"
        if ev.get("compact_scene")
        else f"IZQ corners: {ev['corner_l']:.0f} (≥{MIN_CORNER_SCORE:.0f})"
    )
    corner_r_label = (
        "DER corners: omitido (escena compacta)"
        if ev.get("compact_scene")
        else f"DER corners: {ev['corner_r']:.0f} (≥{MIN_CORNER_SCORE:.0f})"
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Reporte de foco — {ts:%Y-%m-%d %H:%M}</title>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:20px auto;color:#222;padding:0 16px}}
h2{{border-bottom:2px solid #2c3e50;padding-bottom:4px;margin-top:24px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:center}}
img{{max-width:48%;border-radius:6px;margin:4px}}</style></head><body>
<h1>Asistente de foco — reporte</h1>
<p>Fecha/hora: {ts:%Y-%m-%d %H:%M:%S}</p>
{compact_note}
<h2>Resultado</h2>
<p>{_pill(bool(ev["all_pass_with_distance"]), "FOCO GLOBAL")}</p>
<ul>
<li>{_pill(ev["checks"]["center_l"], f"IZQ nitidez centro: {ev['center_l']:.0f} (≥{MIN_SCORE})")}</li>
<li>{_pill(ev["checks"]["center_r"], f"DER nitidez centro: {ev['center_r']:.0f} (≥{MIN_SCORE})")}</li>
<li>{_pill(ev["checks"]["uniformity_l"], corner_l_label)}</li>
<li>{_pill(ev["checks"]["uniformity_r"], corner_r_label)}</li>
<li>{_pill(ev["checks"]["lr_global"], f"Simetría L/R: diff {ev['lr_diff']:.1f}% (≤{MAX_LR_DIFF_PCT}%)")}</li>
<li>{_pill(ev["checks"]["lr_zones"], f"Max zona diff: {ev['max_zone_diff']:.1f}% (≤{MAX_LR_ZONE_DIFF_PCT}%)")}</li>
<li>{_pill(bool(ev["distance_ok"]), f"Distancia board: {dist_text} (objetivo {TARGET_DISTANCE_MIN_MM/1000:.2f}-{TARGET_DISTANCE_MAX_MM/1000:.2f} m)")}</li>
</ul>
<h2>Nitidez por zona</h2>
<table><tr><th>IZQ</th><th>DER</th></tr>
<tr><td>{_grid_html(grid_l)}</td><td>{_grid_html(grid_r)}</td></tr></table>
<h2>Frames finales</h2>
<img src="data:image/jpeg;base64,{_b64(frame_l)}"/>
<img src="data:image/jpeg;base64,{_b64(frame_r)}"/>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        global finish_requested, capture_started
        if self.path == "/finish":
            finish_requested = True
            self.send_response(204)
            self.end_headers()
        elif self.path == "/start":
            with capture_started_lock:
                capture_started = True
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Asistente de foco</title>
<style>
*{box-sizing:border-box}
body{background:#0b0b0d;margin:0;color:#eee;
     font-family:-apple-system,Segoe UI,Roboto,sans-serif;
     display:flex;flex-direction:column;min-height:100vh}
header{display:flex;align-items:center;justify-content:space-between;
       padding:10px 20px;background:#141418;border-bottom:1px solid #26262c}
header h1{margin:0;font-size:16px;font-weight:600;letter-spacing:.3px}
header .sub{color:#888;font-size:13px}
main{display:flex;flex:1;min-height:0}
#stage{flex:1;display:flex;align-items:center;justify-content:center;
       background:#000;padding:12px;min-width:0}
#stream{max-width:100%;max-height:calc(100vh - 60px);display:block;
        border-radius:6px;box-shadow:0 2px 16px rgba(0,0,0,0.6)}
#side{width:360px;background:#141418;border-left:1px solid #26262c;
      display:flex;flex-direction:column;padding:18px}
#status{font-size:14px;line-height:1.6;flex:1}
#status > div:first-child{font-size:17px!important;margin-bottom:10px}
.btn{padding:12px 20px;font-size:15px;border:none;border-radius:8px;
     cursor:pointer;font-weight:600;width:100%;margin-top:14px;
     transition:background .15s}
.btn-finish{background:#c0392b;color:#fff}
.btn-finish:hover{background:#a83224}
@media (max-width:900px){
  main{flex-direction:column}
  #side{width:100%;border-left:none;border-top:1px solid #26262c}
  #stream{max-height:70vh}
}
</style></head><body>
<header>
  <h1>Asistente de foco</h1>
  <span class="sub">Stereo IMX708 - ajuste guiado</span>
</header>
<div id="start-overlay" style="position:fixed;inset:0;background:rgba(11,11,13,0.95);
     z-index:9999;display:flex;align-items:center;justify-content:center;
     flex-direction:column;gap:18px">
  <div style="color:#eee;font-size:26px;font-weight:700">
    Asistente de foco listo
  </div>
  <div style="color:#aaa;font-size:15px;max-width:480px;text-align:center;line-height:1.6">
    Posicioná el board ChArUco o un target texturado frente a las cámaras.
    Cuando esté listo, presioná <b>Comenzar</b>. Esto también activa el audio
    del navegador para las notificaciones sonoras.
  </div>
  <button id="btn-start" style="padding:14px 36px;font-size:18px;
       background:#27ae60;color:#fff;border:none;border-radius:10px;
       cursor:pointer;font-weight:700" onclick="startCapture()">
    Comenzar
  </button>
</div>
<main>
  <div id="stage"><img id="stream" src="/stream"/></div>
  <aside id="side">
    <div id="status"></div>
    <button id="btn-audio" class="btn" onclick="toggleAudio()"
            style="background:#555">Audio OFF</button>
    <button id="finbtn" class="btn btn-finish" onclick="finish()">Finalizar y guardar reporte</button>
  </aside>
</main>
<script>
let finalized=false;
// Default ON — el operador lo puede apagar y la preferencia persiste.
let audioOn = localStorage.getItem('focus.audio') !== '0';
let audioCtx = null;
function ensureAudioCtx(){
  if(!audioCtx && window.AudioContext){
    try{ audioCtx = new AudioContext(); }catch(e){}
  }
  return audioCtx;
}
function beep(freq, durMs, gain){
  const ctx = ensureAudioCtx();
  if(!ctx) return;
  try{
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = 'sine'; osc.frequency.value = freq;
    g.gain.value = gain || 0.15;
    osc.connect(g); g.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + durMs/1000);
  }catch(e){}
}
// Patrones diferenciados para los eventos clave de la sesión.
function beepStart(){       // par ascendente — arranque
  beep(600, 80); setTimeout(() => beep(900, 80), 100);
}
function beepFinish(){      // triple descendente — fin de sesión
  beep(800, 100);
  setTimeout(() => beep(600, 100), 130);
  setTimeout(() => beep(400, 150), 260);
}
function beepActivated(){   // tap simple — toggle de audio ON
  beep(900, 60);
}
// Pulso tipo "detector": tap corto que se acelera a medida que el foco
// mejora. El intervalo se deriva del score normalizado (1.0 = umbral
// MIN_SCORE de paso). El operador puede ajustar el lente sin mirar la
// pantalla — la cadencia le dice qué tan cerca está del foco óptimo.
let pulseTimerId = null;
let pulseInterval = 0;
function updatePulse(score){
  let interval;
  if(score < 0.25){
    interval = 0;          // board fuera del frame o muy desenfocado
  }else if(score >= 1.5){
    interval = 130;        // lock holgado — pulso rápido sostenido
  }else{
    // [0.25 .. 1.5] → [1200 .. 130] ms (slow → fast)
    const t = (score - 0.25) / 1.25;
    interval = Math.round(1200 - t * 1070);
  }
  pulseInterval = interval;
  if(pulseTimerId === null && interval > 0 && audioOn && !finalized){
    schedulePulse();
  }
}
function schedulePulse(){
  pulseTimerId = null;
  if(!audioOn || finalized || pulseInterval <= 0) return;
  beep(700, 35, 0.06);
  pulseTimerId = setTimeout(schedulePulse, pulseInterval);
}
function updateAudioBtn(){
  const b = document.getElementById('btn-audio');
  b.textContent = audioOn ? 'Audio ON' : 'Audio OFF';
  b.style.background = audioOn ? '#27ae60' : '#555';
}
function toggleAudio(){
  audioOn = !audioOn;
  localStorage.setItem('focus.audio', audioOn ? '1' : '0');
  updateAudioBtn();
  if(audioOn){
    ensureAudioCtx();
    beepActivated();
    // Reanudar el pulso si ya hay señal activa.
    if(pulseTimerId === null && pulseInterval > 0) schedulePulse();
  }
}
function startCapture(){
  ensureAudioCtx();
  if(audioOn) beepStart();
  const overlay = document.getElementById('start-overlay');
  if(overlay) overlay.style.display = 'none';
  fetch('/start',{method:'POST'});
}
function finish(){
  if(!confirm('Finalizar y guardar reporte?'))return;
  const btn=document.getElementById('finbtn');
  btn.disabled=true;
  btn.textContent='Guardando reporte...';
  btn.style.background='#555';
  document.getElementById('status').innerHTML='<div style="color:#f1c40f;font-size:16px">Guardando reporte y cerrando cámaras...</div>';
  document.getElementById('stream').style.opacity='0.25';
  fetch('/finish',{method:'POST'});
}
updateAudioBtn();
// Seguir polleando /status después del finalize — el server lo
// actualiza con la card finalizada y detectamos eso vía el marker
// data-finalized.
setInterval(()=>{
  fetch('/status').then(r=>r.text()).then(t=>{
    document.getElementById('status').innerHTML=t;
    // Update del pulso: score normalizado al MIN_SCORE del backend.
    const scoreMatch = t.match(/data-pulse-score="([\d.]+)"/);
    updatePulse(scoreMatch ? parseFloat(scoreMatch[1]) : 0);
    if(!finalized && t.indexOf('data-finalized="1"')!==-1){
      finalized=true;
      // Cortar el pulso antes del beep de finalización para que no se solape.
      pulseInterval = 0;
      if(pulseTimerId !== null){ clearTimeout(pulseTimerId); pulseTimerId = null; }
      const btn=document.getElementById('finbtn');
      btn.textContent='Finalizado';
      btn.style.background='#27ae60';
      if(audioOn) beepFinish();
      const w=window.open('/report','_blank');
      if(!w){
        const note=document.createElement('div');
        note.style.cssText='color:#f1c40f;font-size:12px;margin-top:8px';
        note.textContent='El navegador bloqueó la apertura automática — usá el botón "Abrir reporte".';
        document.getElementById('status').appendChild(note);
      }
    }
  }).catch(()=>{})
},400);
</script></body></html>""".encode(
                    "utf-8"
                )
            )
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while not shutting_down:
                    with jpeg_lock:
                        frame = latest_jpeg
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with _status_lock:
                self.wfile.write(_status_html.encode("utf-8"))
        elif self.path == "/report":
            if _report_path_global is None or not _report_path_global.exists():
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write("Reporte no disponible todavía".encode("utf-8"))
                return
            try:
                body = _report_path_global.read_bytes()
            except OSError as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error leyendo reporte: {e}".encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def _resolve_resolution_from_device_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Si --resolution no fue pasado, leerlo de /etc/people-counter/config.yaml.

    Setup tools (focus, calib, preview, diagnose) tienen que matchear la
    resolución del runtime para que las calibraciones / decisiones de foco
    sean válidas. No hay fallback a defaults — si el config per-device no
    existe o no tiene ``vision.resolution`` definido, error.
    """
    if args.resolution is not None:
        return
    from src.config.loader import (
        DEFAULT_DEVICE_CONFIG_PATH,
        load_device_config,
    )

    try:
        cfg = load_device_config(DEFAULT_DEVICE_CONFIG_PATH)
    except FileNotFoundError:
        parser.error(
            f"--resolution no provisto y {DEFAULT_DEVICE_CONFIG_PATH} no "
            "existe. Pasá --resolution explícito o aprovisioná el config "
            "per-device."
        )
    res = cfg.get("vision", {}).get("resolution")
    if not res or not isinstance(res, list) or len(res) != 2:
        parser.error(
            f"vision.resolution no definido o malformado en "
            f"{DEFAULT_DEVICE_CONFIG_PATH}. Esperaba lista [W, H]."
        )
    args.resolution = [int(res[0]), int(res[1])]


def _resolve_mount_height_from_device_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Si --mount-height-m no fue pasado, leerlo de /etc/people-counter/
    config.yaml (vision.mounting_height_m). Si el config no existe o le
    falta el valor, caer al DEFAULT_MOUNT_HEIGHT_M con una advertencia
    (no es crítico: solo afecta la derivación del target distance, que
    el operador puede overridear con --target-distance-min/max-mm).
    """
    if args.mount_height_m is not None:
        return
    from src.config.loader import (
        DEFAULT_DEVICE_CONFIG_PATH,
        load_device_config,
    )

    try:
        cfg = load_device_config(DEFAULT_DEVICE_CONFIG_PATH)
    except FileNotFoundError:
        args.mount_height_m = DEFAULT_MOUNT_HEIGHT_M
        print(
            f"[focus] {DEFAULT_DEVICE_CONFIG_PATH} no existe — usando mount "
            f"height default {DEFAULT_MOUNT_HEIGHT_M:.2f}m. Pasá "
            f"--mount-height-m explícito o aprovisioná el config.",
            flush=True,
        )
        return
    mount = cfg.get("vision", {}).get("mounting_height_m")
    if mount is None:
        args.mount_height_m = DEFAULT_MOUNT_HEIGHT_M
        print(
            f"[focus] vision.mounting_height_m no definido en "
            f"{DEFAULT_DEVICE_CONFIG_PATH} — usando default "
            f"{DEFAULT_MOUNT_HEIGHT_M:.2f}m.",
            flush=True,
        )
        return
    args.mount_height_m = float(mount)


def main() -> None:
    global latest_jpeg, shutting_down, _status_html, finish_requested
    global _report_path_global, HW

    # Carga parámetros de hardware del config per-device (fallback a fleet
    # defaults si no hay config). Después de esto, todos los usos de
    # HW.full_res / HW.nominal_focal_full_px / HW.default_res / etc.
    # respetan el config del device.
    HW = load_hardware_params()

    parser = argparse.ArgumentParser(
        description="Asistente de foco guiado para cámaras estéreo"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--left",
        type=int,
        default=0,
        help="Índice de la cámara izquierda (lente izquierda "
        "mirando desde la cámara hacia la escena). "
        "Default 0 — matchea el wiring de la flota.",
    )
    parser.add_argument(
        "--right", type=int, default=1, help="Índice de la cámara derecha. Default 1."
    )
    parser.add_argument(
        "--low-light",
        action="store_true",
        help="Modo PoC para corridas en luz baja / cuartos "
        "chicos. Afloja sharpness de centro, sharpness "
        "de corners, balance L/R, rango de distancia, "
        "y fuerza scene=compact. Equivalente a pasar "
        "--min-score 80 --min-corner-score 30 "
        "--max-lr-diff-pct 50 --max-lr-zone-diff-pct "
        "100 --target-distance-min-mm 500 "
        "--target-distance-max-mm 5000 --scene compact. "
        "Los flags explícitos igual overridean este "
        "preset. El PASS resultante NO valida foco "
        "para producción — solo que la tool corre.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Variance Laplaciana mínima del centro "
        "(default 200, o 80 con --low-light)",
    )
    parser.add_argument(
        "--min-corner-score",
        type=float,
        default=None,
        help="Variance Laplaciana media mínima sobre las 4 "
        "zonas de corner (default 100, o 30 con "
        "--low-light). Métrica absoluta — ¿el corner "
        "tiene detalle? Reemplaza el ratio viejo "
        "bordes/centro que se disparaba mal en cuartos "
        "chicos donde el board domina el centro. "
        "Subir para más estricto.",
    )
    parser.add_argument(
        "--scene",
        choices=("auto", "compact", "full"),
        default="auto",
        help="Modo de escena. 'auto' (default): compact si "
        f"el bbox del ChArUco cubre más del {int(COMPACT_BBOX_THRESHOLD*100)}%% "
        "del frame. 'compact': siempre saltear el "
        "check de corners (para cuartos chicos de "
        "test donde los corners ven paredes a "
        "distancias no relacionadas). 'full': siempre "
        "enforzar el check de corners. --low-light "
        "fuerza compact.",
    )
    parser.add_argument(
        "--meter",
        choices=("matrix", "centre", "spot"),
        default="matrix",
        help="Modo de AE metering. 'matrix' (default) "
        "pondera todo el frame — funciona con luz "
        "pareja. 'centre' pondera fuerte el área "
        "central y 'spot' usa solo el centro — usar "
        "estos cuando hay zonas brillantes (ventanas, "
        "paredes detrás de un backdrop texturado) que "
        "arrastran la exposición abajo en el board. "
        "--low-light defaultea esto a 'centre'.",
    )
    parser.add_argument(
        "--lock-ae",
        action="store_true",
        help="Lockea AE/AWB en ambas cámaras tras 1s de "
        "settle. Sin lock, cada cámara corre AE/AWB "
        "independiente y pueden converger a estados "
        "distintos (exposición o tinte diferente entre "
        "L y R), lo que causa que el decoder ArUco "
        "lea los bits del marker bien en un lado y mal "
        "en el otro — manifestándose como detección "
        "asimétrica random. Recomendado siempre que se "
        "use focus_assist en serio. Espejo del flag del "
        "wizard de calibrate y del runtime principal.",
    )
    parser.add_argument(
        "--max-lr-diff-pct",
        type=float,
        default=None,
        help="Asimetría global máxima de sharpness L/R "
        "(default 15%%, o 50%% con --low-light)",
    )
    parser.add_argument(
        "--max-lr-zone-diff-pct",
        type=float,
        default=None,
        help="Asimetría per-zone L/R máxima (default 30%%, " "o 100%% con --low-light)",
    )
    parser.add_argument(
        "--mount-height-m",
        type=float,
        default=None,
        help="Altura de mount de la cámara desde el piso. "
        "Sin override, se lee vision.mounting_height_m "
        "del config per-device. La distancia target de "
        "foco se deriva de este valor (mount minus rango "
        "de altura de cabezas) salvo que se pasen "
        "--target-distance-min/max-mm explícitos.",
    )
    parser.add_argument(
        "--target-distance-min-mm",
        type=float,
        default=None,
        help=f"Distancia target mínima (mm) para validación "
        f"de foco. Default {TARGET_DISTANCE_MIN_MM:.0f}mm "
        f"(protocolo lab universal: foco a 1.5m ±20cm, "
        f"DoF cubre rango operativo 1.0-3.5m para "
        f"mount de flota 2.0-3.5m). 500mm con "
        f"--low-light.",
    )
    parser.add_argument(
        "--target-distance-max-mm",
        type=float,
        default=None,
        help=f"Distancia target máxima (mm) para validación "
        f"de foco. Default {TARGET_DISTANCE_MAX_MM:.0f}mm "
        f"(protocolo lab universal). 5000mm con "
        f"--low-light.",
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        default=HW.board_cols,
        help="Columnas (cuadrados) del ChArUco. "
        "Default desde vision.charuco.board_cols.",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=HW.board_rows,
        help="Filas (cuadrados) del ChArUco. "
        "Default desde vision.charuco.board_rows.",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=HW.square_mm,
        help="Tamaño del cuadrado ChArUco en mm. "
        "Default desde vision.charuco.square_mm.",
    )
    parser.add_argument(
        "--marker-mm",
        type=float,
        default=HW.marker_mm,
        help="Tamaño del marker ChArUco en mm. "
        "Default desde vision.charuco.marker_mm.",
    )
    parser.add_argument(
        "--dict",
        dest="aruco_dict",
        default=HW.aruco_dict,
        help="Nombre del dict ArUco. Default desde " "vision.charuco.dict.",
    )
    parser.add_argument(
        "--legacy-pattern",
        action=argparse.BooleanOptionalAction,
        default=HW.legacy_pattern,
        help="Usar enumeración de markers ChArUco pre-4.6 "
        "de OpenCV. Default desde vision.charuco."
        "legacy_pattern (True matchea calib.io).",
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help="Overridea el focal length en píxeles para "
        "la estimación de distancia. Usar si la f "
        "nominal del IMX708 da distancias erróneas "
        "(ej. modo de sensor con FOV parcial). Medir "
        "poniendo el board a distancia conocida y "
        "ajustando hasta que la distancia reportada "
        "matchee.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=None,
        help="Resolución de captura. Default: lee "
        "vision.resolution de /etc/people-counter/"
        "config.yaml (fuente única de verdad — "
        "garantiza match con el runtime). Pasá "
        "explícito solo para tests / dev workstation "
        "donde no hay config per-device.",
    )
    parser.add_argument(
        "--max-exposure-us",
        type=int,
        default=16000,
        help="Cap de exposure time en microsegundos via "
        "FrameDurationLimits (mismo cap que el runtime). "
        "Default 16000us (16ms) freezea micro-vibración "
        "del bracket que rompe el decoder ArUco "
        "asimétricamente entre L/R en luz baja con "
        "shutter largo. AE compensa con AnalogueGain "
        "más alto (más ruido pero cero blur). Setear "
        "0 para deshabilitar el cap.",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Usa el modo de foco estático clásico (board fijo en una "
        "posición al target). El DEFAULT es el mapa de foco: pasás el board "
        "por todo el cuadro y se acumula la nitidez por zona, así el check "
        "por zona / simetría L-R tiene board real en cada celda (no fondo).",
    )
    parser.add_argument(
        "--map-coverage-min",
        type=float,
        default=FOCUS_MAP_COVERAGE_MIN,
        help=f"(modo mapa) Varianza Laplaciana mínima para contar una zona "
        f"cubierta. Default {FOCUS_MAP_COVERAGE_MIN:.0f}.",
    )
    args = parser.parse_args()
    # mount_height_m tiene que estar resuelto antes de _apply_threshold_overrides
    # porque la derivación del target distance lo usa.
    _resolve_mount_height_from_device_config(args, parser)
    _apply_threshold_overrides(args)
    _resolve_resolution_from_device_config(args, parser)

    dict_attr = (
        args.aruco_dict
        if args.aruco_dict.startswith("DICT_")
        else f"DICT_{args.aruco_dict}"
    )
    if not hasattr(cv2.aruco, dict_attr):
        parser.error(f"Dict ArUco desconocido: {args.aruco_dict}")
    dict_id = getattr(cv2.aruco, dict_attr)

    from picamera2 import Picamera2
    from libcamera import controls as _libcam_controls

    cam_l = Picamera2(args.left)
    cam_r = Picamera2(args.right)
    res = (int(args.resolution[0]), int(args.resolution[1]))
    # Cap de exposure idéntico al runtime (vision.max_exposure_us del
    # config). 0 (o negativo) deshabilita el cap.
    max_exp = int(args.max_exposure_us) if args.max_exposure_us > 0 else None
    initial_controls = {"FrameDurationLimits": (max_exp, max_exp)} if max_exp else {}
    # Sensor raw mode desde el config per-device (sensor.default_res).
    # Anclar el mode evita que picamera2 caiga al Mode 0 cropeado del
    # IMX708 que reduce el HFOV a ~80°.
    raw_size = HW.default_res
    for cam in [cam_l, cam_r]:
        # Formato "RGB888": en los builds de RPi OS Trixie con los que
        # shippeamos, los nombres de formato de picamera2 están invertidos —
        # "RGB888" entrega BGR directo del ISP. Mismo patrón canónico que el
        # runtime (src/vision/capture.py), sin el cvtColor redundante.
        config = cam.create_still_configuration(
            main={"size": res, "format": "RGB888"},
            raw={"size": raw_size},
            controls=initial_controls,
        )
        cam.configure(config)
        cam.start()

    # Modo de AE metering: matrix (default) pondera todo el frame;
    # centre-weighted / spot ignoran la periferia y exponen para el
    # centro — útil cuando la escena tiene zonas brillantes fuera del
    # área del board (ventanas, paredes detrás de un backdrop
    # texturado) que arrastran la exposición abajo en el board mismo.
    meter_map = {
        "matrix": _libcam_controls.AeMeteringModeEnum.Matrix,
        "centre": _libcam_controls.AeMeteringModeEnum.CentreWeighted,
        "spot": _libcam_controls.AeMeteringModeEnum.Spot,
    }
    meter_mode = meter_map[args.meter]
    for cam in [cam_l, cam_r]:
        cam.set_controls({"AeMeteringMode": meter_mode})
    if args.meter != "matrix":
        print(
            f"[meter] AE metering = {args.meter} (centre/spot ignora "
            "los bordes del frame, útil cuando hay zonas brillantes "
            "rodeando el board)",
            flush=True,
        )
    # Lock provisional tras un settle inicial (vision.ae_lock.initial_settle_s
    # del config). La escena puede no tener el board todavía, pero un lock
    # estable evita la oscilación de AE auto durante el waiting. Cuando el
    # operador apreta Comenzar se re-settlea y re-lockea para que los valores
    # reflejen la escena real de medición.
    time.sleep(HW.ae_initial_settle_seconds)
    if args.lock_ae:
        for cam, name in [(cam_l, "left"), (cam_r, "right")]:
            metadata = cam.capture_metadata()
            cam.set_controls(
                {
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": metadata.get("ExposureTime", 30000),
                    "AnalogueGain": metadata.get("AnalogueGain", 1.0),
                    "ColourGains": metadata.get("ColourGains", (1.0, 1.0)),
                }
            )
        print(
            f"[lock-ae] Lock provisional tras {HW.ae_initial_settle_seconds:.1f}s settle "
            f"(L: exp={cam_l.capture_metadata().get('ExposureTime',0)}us "
            f"R: exp={cam_r.capture_metadata().get('ExposureTime',0)}us)",
            flush=True,
        )

    board = create_charuco_board(
        board_size=(args.board_cols, args.board_rows),
        square_length=args.square_mm,
        marker_length=args.marker_mm,
        dict_id=dict_id,
        legacy_pattern=args.legacy_pattern,
    )
    # ThreadingHTTPServer así el handler long-running de /stream no
    # bloquea que /finish (y /status) se sirvan en threads separados.
    # SO_REUSEADDR así una instancia previa Ctrl-C'eada no deja el
    # puerto en TIME_WAIT para la próxima corrida.
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"Focus assist — http://people-counter.local:{args.port}")
    print(
        f"Board: {args.board_cols}x{args.board_rows} / {args.square_mm:.0f}mm sq / "
        f"{args.marker_mm:.0f}mm mk / {dict_attr}"
    )
    print(
        f"Distancia target de foco: "
        f"{TARGET_DISTANCE_MIN_MM/1000:.2f}-{TARGET_DISTANCE_MAX_MM/1000:.2f}m "
        f"(mount {args.mount_height_m:.2f}m del config)"
    )
    print("Esperando que el operador haga click en Comenzar...")

    # Bloquear hasta que el operador apreta Comenzar — matchea el flow de calibrate.py.
    try:
        while not capture_started:
            if finish_requested:
                print("\nCancelado antes de comenzar.")
                try:
                    cam_l.stop()
                    cam_l.close()
                    cam_r.stop()
                    cam_r.close()
                except Exception:
                    pass
                try:
                    server.shutdown()
                except Exception:
                    pass
                return
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrumpido antes de comenzar (Ctrl-C).")
        try:
            cam_l.stop()
            cam_l.close()
            cam_r.stop()
            cam_r.close()
        except Exception:
            pass
        try:
            server.shutdown()
        except Exception:
            pass
        sys.exit(0)

    # Re-settle AE con el board ya posicionado y re-lock — los valores
    # provisionales del startup pudieron diferir de la escena real de
    # medición. Solo si --lock-ae está activo (matchea calibrate/diagnose_calibration).
    if args.lock_ae:
        print(
            f"[lock-ae] Re-settle con board en escena ({HW.ae_resettle_seconds:.1f}s)...",
            flush=True,
        )
        for cam in [cam_l, cam_r]:
            cam.set_controls({"AeEnable": True, "AwbEnable": True})
        time.sleep(HW.ae_resettle_seconds)
        for cam, name in [(cam_l, "left"), (cam_r, "right")]:
            metadata = cam.capture_metadata()
            cam.set_controls(
                {
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": metadata.get("ExposureTime", 30000),
                    "AnalogueGain": metadata.get("AnalogueGain", 1.0),
                    "ColourGains": metadata.get("ColourGains", (1.0, 1.0)),
                }
            )
        print(
            f"[lock-ae] Re-lock final (L: "
            f"exp={cam_l.capture_metadata().get('ExposureTime',0)}us "
            f"R: exp={cam_r.capture_metadata().get('ExposureTime',0)}us)",
            flush=True,
        )

    print("Poné el board ChArUco en ese rango frente al par. Ajustá los lentes.")
    print("Click Finalizar en la UI cuando las barras estén verdes.\n")

    frame_l = frame_r = None
    grid_l = grid_r = None
    ev: dict = {}
    pass_streak = 0
    peaks = PeakTracker(window_frames=40)
    lr_disparity_buffer: list[float] = []
    # Memoria de detección per-camera: la detección de ChArUco
    # flickea con contraste marginal (luz baja, backgrounds
    # ocupados), y los valores None per-frame hacían que el verdict
    # rebotara. Cargamos la última detección exitosa hacia adelante
    # por hasta DETECT_STALENESS_SEC, así los drops breves no
    # resetean el gate LISTO mientras que una pérdida real del
    # board igual aparece después del grace period.
    DETECT_STALENESS_SEC = 2.0
    last_dist_l: Optional[float] = None
    last_dist_l_t = 0.0
    last_dist_r: Optional[float] = None
    last_dist_r_t = 0.0

    # Estado del mapa de foco (default). Acumula la nitidez máxima por zona a
    # medida que el operador pasea el board por el cuadro; cuando las 9 zonas
    # están cubiertas en L y R, se evalúa el mapa completo (reusa
    # evaluate_focus). --static vuelve al chequeo de un solo frame estático.
    use_map = not getattr(args, "static", False)
    map_cov_min = float(getattr(args, "map_coverage_min", FOCUS_MAP_COVERAGE_MIN))
    map_l = np.zeros((3, 3))
    map_r = np.zeros((3, 3))
    covered_l = np.zeros((3, 3), dtype=bool)
    covered_r = np.zeros((3, 3), dtype=bool)
    map_dists: list[float] = []
    map_ev: Optional[dict] = None
    map_complete = False

    try:
        while not finish_requested:
            # "RGB888" ya entrega BGR (ver config arriba) — sin cvtColor.
            frame_l = cam_l.capture_array("main")
            frame_r = cam_r.capture_array("main")

            grid_l, valid_l = focus_grid(frame_l)
            grid_r, valid_r = focus_grid(frame_r)

            dist_l, ncorn_l, bbox_l, cx_l = estimate_charuco_distance_mm(
                frame_l, board, args.focal_px
            )
            dist_r, ncorn_r, bbox_r, cx_r = estimate_charuco_distance_mm(
                frame_r, board, args.focal_px
            )

            # Actualizar la memoria de detección per-camera; expira
            # con staleness así una cámara realmente ciega aparece
            # después de un par de segundos.
            now_ts = time.time()
            if dist_l is not None:
                last_dist_l = dist_l
                last_dist_l_t = now_ts
            if dist_r is not None:
                last_dist_r = dist_r
                last_dist_r_t = now_ts
            eff_dist_l = (
                last_dist_l if (now_ts - last_dist_l_t) < DETECT_STALENESS_SEC else None
            )
            eff_dist_r = (
                last_dist_r if (now_ts - last_dist_r_t) < DETECT_STALENESS_SEC else None
            )

            # Check de parity L/R — con las cámaras correctamente
            # mapeadas, la cámara L (físicamente a la izquierda) ve
            # un objeto shifteado a la DERECHA en su frame comparado
            # con R (parallax: baseline 14 cm). Así cx_l debería ser
            # MAYOR que cx_r cuando el wiring matchea la convención.
            # Disparidad negativa o cercana a cero → cámaras
            # swappeadas en software (el operador debería pasar
            # --left/--right invertidos).
            if cx_l is not None and cx_r is not None:
                lr_disparity_px = cx_l - cx_r
                lr_disparity_buffer.append(lr_disparity_px)
                if len(lr_disparity_buffer) > 30:
                    lr_disparity_buffer.pop(0)
            # Disparidad predicha para la profundidad actual, usada
            # para flagear magnitude_off (signo correcto pero valor
            # muy off — hint de baseline mismatch o un objeto no-board
            # detectado). Usa las distancias smootheadas así los
            # drops breves de detección no rebotan expected_px.
            expected_px = None
            if eff_dist_l is not None and frame_l is not None:
                expected_px = _expected_disparity_px(eff_dist_l, frame_l.shape[1])
            elif eff_dist_r is not None and frame_r is not None:
                expected_px = _expected_disparity_px(eff_dist_r, frame_r.shape[1])
            lr_status = _classify_lr(lr_disparity_buffer, expected_px=expected_px)

            # Resolución del scene-mode: "compact" fuerza
            # corners-skipped; "full" fuerza corners-checked; "auto"
            # tripea a compact cuando el board llena una fracción
            # significativa de cualquiera de los frames.
            if args.scene == "compact":
                is_compact = True
            elif args.scene == "full":
                is_compact = False
            else:
                is_compact = max(bbox_l, bbox_r) > COMPACT_BBOX_THRESHOLD

            ev = evaluate_focus(
                grid_l,
                grid_r,
                eff_dist_l,
                eff_dist_r,
                valid_l,
                valid_r,
                compact_scene=is_compact,
            )
            peaks.update(ev["center_l"], ev["center_r"])

            # Acumular el mapa de foco (default): nitidez máxima por zona +
            # cobertura. ev (arriba) sigue siendo del frame en vivo, para que
            # las barras del preview muestren la nitidez actual mientras
            # enfocás; el VEREDICTO en modo mapa sale del mapa acumulado.
            if use_map:
                focus_map_update(map_l, covered_l, grid_l, valid_l, map_cov_min)
                focus_map_update(map_r, covered_r, grid_r, valid_r, map_cov_min)
                if eff_dist_l is not None:
                    map_dists.append(eff_dist_l)
                elif eff_dist_r is not None:
                    map_dists.append(eff_dist_r)
                map_complete = bool(covered_l.all() and covered_r.all())
                if map_complete:
                    _md = float(np.median(map_dists)) if map_dists else None
                    map_ev = evaluate_focus(
                        map_l,
                        map_r,
                        _md,
                        _md,
                        covered_l,
                        covered_r,
                        compact_scene=False,
                    )

            preview = _compose_preview(
                frame_l, frame_r, ev, grid_l, grid_r, peaks=peaks
            )
            _, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()

            # HTML de status
            map_grid_html = ""
            if use_map:
                if (
                    map_complete
                    and map_ev
                    and map_ev["all_pass"]
                    and map_ev["distance_ok"]
                ):
                    color_status = "#2ecc71"
                    lead = "LISTO — mapa completo y nítido, fijá los lentes"
                elif map_complete:
                    color_status = "#e74c3c"
                    lead = (
                        map_ev["hints"][0]
                        if (map_ev and map_ev["hints"])
                        else "Mapa completo pero hay zonas blandas / distancia fuera de rango"
                    )
                else:
                    color_status = "#f1c40f"
                    _miss = focus_map_missing(covered_l, covered_r)
                    _shown = ", ".join(_miss[:4]) + ("…" if len(_miss) > 4 else "")
                    lead = f"Pasá el board por: {_shown}"
                map_grid_html = _focus_map_grid_html(covered_l, covered_r, map_l, map_r)
            else:
                if ev["all_pass_with_distance"]:
                    pass_streak += 1
                else:
                    pass_streak = 0

                if ev["all_pass_with_distance"] and pass_streak >= 3:
                    color_status = "#2ecc71"
                    lead = "LISTO — fijá los lentes y pasá a calibración"
                elif ev["all_pass_with_distance"]:
                    color_status = "#f1c40f"
                    lead = "Muy bien — mantené firme..."
                else:
                    color_status = "#e74c3c"
                    lead = ev["hints"][0] if ev["hints"] else "Ajustando..."

            # El análisis peak-vs-current ayuda al operador a
            # notar si el último ajuste overshooteó el mejor foco
            # visto recientemente. Solo se muestra cuando el valor
            # está materialmente off del pico (>15% abajo).
            state_l, state_r = peaks.state(ev["center_l"], ev["center_r"])
            peak_hints = []
            if state_l == "past_peak":
                peak_hints.append(
                    f"IZQ: superaste el pico reciente ({ev['center_l']:.0f} actual vs "
                    f"{peaks.peak_l:.0f} máximo) — revertí ligeramente el ajuste"
                )
            if state_r == "past_peak":
                peak_hints.append(
                    f"DER: superaste el pico reciente ({ev['center_r']:.0f} actual vs "
                    f"{peaks.peak_r:.0f} máximo) — revertí ligeramente el ajuste"
                )
            peak_html = ""
            if peak_hints:
                peak_html = (
                    '<div style="color:#3fb6f0;font-size:13px;margin-top:6px">'
                    + "<br>".join(peak_hints)
                    + "</div>"
                )

            lighting = live_lighting_warnings(frame_l, frame_r)
            lighting_html = ""
            if lighting:
                lighting_html = (
                    '<div style="color:#f1c40f">⚠ ' + " · ".join(lighting) + "</div>"
                )

            compact_html = ""
            if ev.get("compact_scene"):
                compact_html = (
                    '<div style="color:#3498db;font-size:13px;margin-top:6px">'
                    "Escena compacta — solo centro + simetría + distancia"
                    "</div>"
                )

            lr_html = ""
            if lr_status["state"] == "swapped":
                lr_html = (
                    '<div style="color:#e74c3c;font-size:14px;font-weight:600;margin-top:6px">'
                    f"⚠ L/R INVERTIDO — la cámara mapeada como L está físicamente "
                    f'a la derecha (disparidad mediana {lr_status["median_px"]:.0f}px). '
                    "Reiniciá con --left/--right swappeados."
                    "</div>"
                )
            elif lr_status["state"] == "magnitude_off":
                exp = lr_status["expected_px"] or 0
                lr_html = (
                    '<div style="color:#e67e22;font-size:13px;font-weight:600;margin-top:6px">'
                    f'⚠ L/R signo OK pero magnitud rara: {lr_status["median_px"]:.0f}px '
                    f"observados vs {exp:.0f}px esperados a esta distancia. "
                    "Posible baseline incorrecto, lente flojo, o detección espuria."
                    "</div>"
                )
            elif lr_status["state"] == "ok":
                exp = lr_status["expected_px"]
                exp_note = f" · esperado ~{exp:.0f}px" if exp else ""
                lr_html = (
                    '<div style="color:#2ecc71;font-size:13px;margin-top:6px">'
                    f'✓ L/R OK (disparidad {lr_status["median_px"]:.0f}px{exp_note})'
                    "</div>"
                )
            elif lr_status["state"] == "unknown" and lr_status["n"] > 0:
                lr_html = (
                    '<div style="color:#888;font-size:13px;margin-top:6px">'
                    f'L/R sin determinar (disparidad {lr_status["median_px"]:.0f}px '
                    f"< {LR_DISPARITY_OK_MIN_PX:.0f}px) — acercá el board para mejor señal"
                    "</div>"
                )

            # Score normalizado al threshold para el pulso del browser:
            # ratio del centro más débil contra MIN_SCORE. 0 = sin señal,
            # 1.0 = umbral de paso, ≥1.5 = lock holgado.
            if use_map:
                pulse_score = float((covered_l & covered_r).sum()) / 9.0
            else:
                pulse_score = min(ev["center_l"], ev["center_r"]) / max(1.0, MIN_SCORE)
            html = (
                f'<div data-pulse-score="{pulse_score:.2f}" '
                f'style="color:{color_status};font-size:18px;font-weight:700">{lead}</div>'
                f"{lighting_html}"
                f"{compact_html}"
                f"{lr_html}"
                f"{peak_html}"
                f"{map_grid_html}"
                f'<div style="color:#888;font-size:13px;margin-top:6px">'
                f"ChArUco IZQ:{ncorn_l} esq · DER:{ncorn_r} esq</div>"
            )
            with _status_lock:
                _status_html = html

            # One-liner de terminal
            verdict = "PASS" if ev["all_pass_with_distance"] else "FAIL"
            dist_str = (
                f"{ev['distance_avg_mm']/1000:.2f}m" if ev["distance_avg_mm"] else "-"
            )
            scene_tag = "C" if ev.get("compact_scene") else "F"
            print(
                f"\r  [{verdict}/{scene_tag}] L/R:{ev['lr_diff']:4.1f}% | "
                f"crn_L:{ev['corner_l']:5.0f} crn_R:{ev['corner_r']:5.0f} | "
                f"ctr_L:{ev['center_l']:6.0f} ctr_R:{ev['center_r']:6.0f} | d:{dist_str}    ",
                end="",
                flush=True,
            )
            # Sleep en chunks de 50ms así el loop puede reaccionar
            # a finish_requested rápido (capture+process a 2304x1296
            # ya toma ~500ms).
            for _ in range(2):
                if finish_requested:
                    break
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")

    shutting_down = True

    # En modo mapa, el reporte/veredicto salen del mapa acumulado (no del
    # último frame). Si el operador finalizó antes de completar, se evalúa lo
    # que haya quedado cubierto.
    report_ev = ev
    report_grid_l, report_grid_r = grid_l, grid_r
    if use_map and frame_l is not None:
        if map_ev is None:
            _md = float(np.median(map_dists)) if map_dists else None
            try:
                map_ev = evaluate_focus(
                    map_l, map_r, _md, _md, covered_l, covered_r, compact_scene=False
                )
            except Exception:
                map_ev = None
        report_ev = map_ev if map_ev is not None else ev
        report_grid_l, report_grid_r = map_l, map_r

    # Guardar el reporte si tenemos estado
    report_path = None
    if frame_l is not None and frame_r is not None and report_ev is not None:
        try:
            report_path = _save_report(
                frame_l, frame_r, report_grid_l, report_grid_r, report_ev
            )
            _report_path_global = report_path
        except Exception as e:
            print(f"\nNo se pudo escribir el reporte: {e}")

    try:
        cam_l.stop()
        cam_l.close()
        cam_r.stop()
        cam_r.close()
    except Exception:
        pass

    # Postear el summary de finalización a /status así el browser
    # — que sigue polleando — renderiza una pantalla "done" en
    # lugar de que el stream quede oscuro. Mantener el server vivo
    # por un grace period así el usuario tiene tiempo de leerlo y
    # apretar el link al reporte.
    verdict = (
        "PASS" if (report_ev and report_ev.get("all_pass_with_distance")) else "FAIL"
    )
    verdict_color = "#2ecc71" if verdict == "PASS" else "#e74c3c"
    report_html = ""
    if report_path is not None:
        report_html = (
            f'<div style="margin-top:14px;font-size:14px">'
            f"Reporte guardado en:<br>"
            f'<code style="background:#0b0b0d;padding:4px 8px;border-radius:4px;'
            f'display:inline-block;margin-top:4px;word-break:break-all">{report_path}</code>'
            f"</div>"
        )
    report_link_html = ""
    if report_path is not None:
        report_link_html = (
            '<a href="/report" target="_blank" '
            'style="display:inline-block;margin-top:14px;padding:10px 18px;'
            "background:#2980b9;color:#fff;text-decoration:none;border-radius:6px;"
            'font-weight:600;font-size:14px">Abrir reporte en nueva pestaña</a>'
        )
    final_html = (
        f'<div data-finalized="1">'
        f'<div style="color:{verdict_color};font-size:22px;font-weight:700;'
        f'margin-bottom:12px">Sesión finalizada — {verdict}</div>'
        f'<div style="color:#aaa;font-size:13px;line-height:1.6">'
        f"Las cámaras se cerraron y el reporte HTML fue generado."
        f"</div>"
        f"{report_link_html}"
        f"{report_html}"
        f"</div>"
    )
    with _status_lock:
        _status_html = final_html

    print("\n" + "=" * 60)
    print("SESIÓN DE FOCO TERMINADA")
    print("=" * 60)
    if ev:
        print(f"  Resultado: {verdict}")
        for hint in ev.get("hints", []):
            print(f"    -> {hint}")
    if report_path is not None:
        print(f"  Reporte HTML: {report_path}")
    print()

    # Grace period — suficiente para que el browser levante el
    # /status final y abra el reporte en una pestaña nueva.
    # Matchea el grace del wizard de calibrate así ambas tools se
    # comportan consistentes.
    time.sleep(10)
    os._exit(0)


if __name__ == "__main__":
    main()

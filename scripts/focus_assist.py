#!/usr/bin/env python3
"""Focus assist tool for stereo Arducam IMX708 cameras.

Guided focus tool that validates sharpness across 9 zones per camera,
checks L/R symmetry, detects the ChArUco board in-scene to validate
the target distance, and gives a clear visual PASS/FAIL verdict.

Usage:
    PYTHONPATH=. python3 scripts/focus_assist.py

    Then open: http://people-counter.local:8080

    Place the ChArUco calibration board (same one you'll use for
    calibration) at the target focus distance — the tool derives
    this from --mount-height-m as the range where heads will cross
    (mount - 1.85m to mount - 1.2m). Adjust each lens ring while
    watching the live bars. Click FINALIZAR when both cameras pass.
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

import cv2
import numpy as np

# Add project root so we can import src.vision.calibration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.calibration import (
    NOMINAL_FOCAL_PX,
    NOMINAL_FULL_RES,
    create_charuco_board,
    detect_charuco_corners,
    live_lighting_warnings,
)

latest_jpeg: bytes = b""
jpeg_lock = threading.Lock()
shutting_down = False
finish_requested = False
# Set to True when the operator clicks "Comenzar" in the browser. Blocks
# the main capture loop until that happens so (a) the operator has time to
# position themselves and (b) the click unlocks the browser audio context.
capture_started = False
capture_started_lock = threading.Lock()

MIN_SCORE = 200
MIN_CORNER_SCORE = 100.0          # Absolute Laplacian var for corner zones.
                                   # Replaces the old edges/center ratio: in
                                   # small test rooms the board fills the
                                   # center and drives the ratio near zero even
                                   # with good lenses. Measuring corners on
                                   # their own scale is honest across scenes.
MAX_LR_DIFF_PCT = 15.0
MAX_LR_ZONE_DIFF_PCT = 30.0
TARGET_DISTANCE_MIN_MM = 1800.0   # Lab protocol: focus at 2.0m ±20cm
TARGET_DISTANCE_MAX_MM = 2200.0   # DoF covers fleet range 1.15-3.30m
DEFAULT_MOUNT_HEIGHT_M = 3.0      # Mode of our door-height distribution
HEAD_HEIGHT_MAX_M = 1.85          # tall adult
HEAD_HEIGHT_MIN_M = 1.20          # short child
COMPACT_BBOX_THRESHOLD = 0.25     # Board bbox/frame area > this -> compact
                                   # scene (board fills the view, corners see
                                   # walls at wildly different depths, so
                                   # corner sharpness is not comparable to
                                   # center and we skip that check).


def _apply_threshold_overrides(args: argparse.Namespace) -> None:
    """Override focus-quality thresholds from CLI flags. Lets operators loosen
    checks for tricky environments (vidrio frontal, low-light) without editing
    code. Defaults preserve current behaviour.
    """
    global MIN_SCORE, MIN_CORNER_SCORE, MAX_LR_DIFF_PCT, MAX_LR_ZONE_DIFF_PCT
    global TARGET_DISTANCE_MIN_MM, TARGET_DISTANCE_MAX_MM
    MIN_SCORE = args.min_score
    MIN_CORNER_SCORE = args.min_corner_score
    MAX_LR_DIFF_PCT = args.max_lr_diff_pct
    MAX_LR_ZONE_DIFF_PCT = args.max_lr_zone_diff_pct
    TARGET_DISTANCE_MIN_MM = args.target_distance_min_mm
    TARGET_DISTANCE_MAX_MM = args.target_distance_max_mm


def focus_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


MIN_ZONE_STD = 8.0  # Below this std the zone is almost uniform (plain wall,
                    # flat surface) — Laplacian variance there is meaningless
                    # as a focus indicator, so we flag it as "no content".


def focus_grid(
    frame: np.ndarray, rows: int = 3, cols: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-zone focus scores + a validity mask.

    The mask is False where the zone has too little contrast (std < MIN_ZONE_STD
    on 0-255 scale) — there's no edge content to measure, so the Laplacian
    variance there doesn't reflect focus quality. Downstream calculations
    (uniformity, symmetry) skip invalid zones.
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
    frame: np.ndarray, board: cv2.aruco.CharucoBoard,
    focal_px_override: float | None = None,
) -> tuple[float | None, int, float]:
    """Detect ChArUco in frame, estimate distance via solvePnP with nominal K.

    Returns (distance_mm or None, n_corners_detected, bbox_ratio). bbox_ratio
    is the fraction of frame area covered by the detected-corners bounding
    rectangle — used upstream to auto-detect "compact scene" (board fills the
    view, corners see walls at unrelated depths). Uses nominal IMX708
    intrinsics — ±10% accuracy is fine for validating "is the board at 2.5-3m?".
    """
    corners, ids = detect_charuco_corners(
        frame, board, min_corners=4, lenient=True,
    )
    if corners is None or ids is None or len(corners) < 4:
        return None, 0, 0.0
    h, w = frame.shape[:2]
    pts = corners.reshape(-1, 2)
    bbox_w = float(pts[:, 0].max() - pts[:, 0].min())
    bbox_h = float(pts[:, 1].max() - pts[:, 1].min())
    frame_area = max(w * h, 1)
    bbox_ratio = (bbox_w * bbox_h) / frame_area
    if focal_px_override is not None:
        fx = fy = focal_px_override
    else:
        scale_x = w / NOMINAL_FULL_RES[0]
        scale_y = h / NOMINAL_FULL_RES[1]
        fx = NOMINAL_FOCAL_PX * scale_x
        fy = NOMINAL_FOCAL_PX * scale_y
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    obj_pts = board.getChessboardCorners()[ids.flatten()].astype(np.float32)
    img_pts = corners.reshape(-1, 2).astype(np.float32)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None, len(corners), bbox_ratio
    if not ok:
        return None, len(corners), bbox_ratio
    return float(tvec[2, 0]), len(corners), bbox_ratio


def evaluate_focus(
    grid_l: np.ndarray, grid_r: np.ndarray,
    distance_l_mm: float | None, distance_r_mm: float | None,
    valid_l: np.ndarray | None = None, valid_r: np.ndarray | None = None,
    compact_scene: bool = False,
) -> dict:
    if valid_l is None:
        valid_l = np.ones_like(grid_l, dtype=bool)
    if valid_r is None:
        valid_r = np.ones_like(grid_r, dtype=bool)

    center_l, center_r = grid_l[1, 1], grid_r[1, 1]
    corners_idx = [(0, 0), (0, 2), (2, 0), (2, 2)]

    # Only include corners with enough contrast to carry a meaningful
    # sharpness signal. A plain wall zone with std~0 has Laplacian variance
    # near 0 regardless of focus — including it distorts the edge average.
    corners_l_vals = [grid_l[r, c] for r, c in corners_idx if valid_l[r, c]]
    corners_r_vals = [grid_r[r, c] for r, c in corners_idx if valid_r[r, c]]
    n_valid_corners_l = len(corners_l_vals)
    n_valid_corners_r = len(corners_r_vals)
    # Absolute corner sharpness (mean Laplacian var across valid corners).
    # Directly comparable to MIN_CORNER_SCORE — a lens that actually resolves
    # detail in the corners will read >= threshold regardless of what the
    # center shows. The old ratio (edges/center) failed in small rooms because
    # a bright ChArUco centered in the frame inflated the denominator.
    corner_l = float(np.mean(corners_l_vals)) if corners_l_vals else 0.0
    corner_r = float(np.mean(corners_r_vals)) if corners_r_vals else 0.0

    # Global average and zone diffs only over zones valid on BOTH sides, so
    # comparing cameras doesn't penalise a zone where only one happens to
    # have content.
    both_valid = valid_l & valid_r
    if both_valid.any():
        global_l = float(grid_l[both_valid].mean())
        global_r = float(grid_r[both_valid].mean())
        zone_diffs_full = (
            np.abs(grid_l - grid_r) / np.maximum(grid_l, grid_r).clip(1) * 100
        )
        # Only consider zones where at least one side carries meaningful
        # sharpness. Two near-zero zones (shadow, low-contrast patch) can
        # produce huge relative diffs for a tiny absolute difference — not
        # a real lens asymmetry.
        SIGNIFICANCE_THRESHOLD = 100.0
        significant = (
            (np.maximum(grid_l, grid_r) >= SIGNIFICANCE_THRESHOLD) & both_valid
        )
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

    distance_ok = (
        distance_avg is not None
        and TARGET_DISTANCE_MIN_MM <= distance_avg <= TARGET_DISTANCE_MAX_MM
    )

    # Uniformity check becomes "pass by default" when we didn't have enough
    # valid corners OR we're in a compact scene where corners see walls at
    # depths unrelated to the board plane (the check would fail structurally,
    # not because the lens is bad). Flag via uniformity_measurable=False.
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

    # Actionable hints
    hints: list[str] = []
    target_lo = TARGET_DISTANCE_MIN_MM / 1000
    target_hi = TARGET_DISTANCE_MAX_MM / 1000
    if distance_avg is None:
        hints.append("Poné el board ChArUco en la escena para validar la distancia")
    elif not distance_ok:
        if distance_avg < TARGET_DISTANCE_MIN_MM:
            hints.append(f"Board muy cerca ({distance_avg/1000:.2f}m). Objetivo: {target_lo:.2f}-{target_hi:.2f}m")
        else:
            hints.append(f"Board muy lejos ({distance_avg/1000:.2f}m). Objetivo: {target_lo:.2f}-{target_hi:.2f}m")
    if not checks["center_l"]:
        hints.append(f"IZQ: centro débil ({center_l:.0f}<{MIN_SCORE}) — girá el lente izquierdo")
    if not checks["center_r"]:
        hints.append(f"DER: centro débil ({center_r:.0f}<{MIN_SCORE}) — girá el lente derecho")
    if checks["center_l"] and uniformity_l_measurable and not checks["uniformity_l"]:
        hints.append(f"IZQ: corners débiles ({corner_l:.0f}<{MIN_CORNER_SCORE:.0f}) — revisá foco / agregá textura en los bordes")
    if checks["center_r"] and uniformity_r_measurable and not checks["uniformity_r"]:
        hints.append(f"DER: corners débiles ({corner_r:.0f}<{MIN_CORNER_SCORE:.0f}) — revisá foco / agregá textura en los bordes")
    # Only nag about low-texture corners in "full" mode — in compact mode the
    # UI shows a dedicated banner explaining corners are intentionally skipped.
    if not compact_scene and (not uniformity_l_measurable or not uniformity_r_measurable):
        hints.append(
            "Escena con poca textura en los bordes — agregá detalle (poster/empapelado) "
            "detrás de la zona de medición para validar uniformidad"
        )
    if not checks["lr_global"]:
        side = "IZQ" if global_l < global_r else "DER"
        hints.append(f"Cámara {side} más blanda que la otra ({lr_diff:.0f}% de diferencia)")
    if not checks["lr_zones"]:
        hints.append(
            f"Asimetría por zona L/R: hasta {max_zone_diff:.0f}% entre zonas "
            f"correspondientes (máx permitido {MAX_LR_ZONE_DIFF_PCT:.0f}%). "
            f"Revisá alineación mecánica del par."
        )
    if all_pass_with_distance:
        hints = ["LISTO — fijá los lentes y pasá a calibración"]

    return {
        "center_l": center_l, "center_r": center_r,
        "corner_l": corner_l, "corner_r": corner_r,
        "lr_diff": lr_diff, "max_zone_diff": max_zone_diff,
        "distance_l_mm": distance_l_mm, "distance_r_mm": distance_r_mm,
        "distance_avg_mm": distance_avg, "distance_ok": distance_ok,
        "checks": checks, "all_pass": all_pass,
        "all_pass_with_distance": all_pass_with_distance,
        "hints": hints,
        "global_l": global_l, "global_r": global_r,
        "uniformity_l_measurable": uniformity_l_measurable,
        "uniformity_r_measurable": uniformity_r_measurable,
        "n_valid_corners_l": n_valid_corners_l,
        "n_valid_corners_r": n_valid_corners_r,
        "compact_scene": compact_scene,
    }


def _ascii(text: str) -> str:
    """Fold to ASCII for cv2.putText (Hershey fonts have no unicode support)."""
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return stripped.replace("—", "-").replace("–", "-").replace("¿", "?").replace("¡", "!")


class PeakTracker:
    """Rolling-window peak tracker for focus scores per camera.

    Holds the max value seen in the last `window_frames` samples. Used to
    tell the operator whether the current adjustment is moving toward or
    away from the best focus recently achieved — very useful when the M12
    lens ring is sensitive and easy to overshoot.

    State thresholds:
        state == "at_peak"   → current within 5% of the recent peak
        state == "past_peak" → current < 85% of the recent peak
        state == "climbing"  → in between (actively improving or still far)
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


def _draw_bar(img: np.ndarray, x: int, y: int, w: int, h: int,
              value: float, target: float, label: str,
              max_value: float | None = None,
              peak: float | None = None,
              passing: bool | None = None) -> None:
    """Draw a horizontal bar: current value fills; green zone marks the target.

    Optional `peak` renders a thin vertical marker at the recent peak so the
    operator can see whether the current adjustment is climbing toward or
    moving away from the best value seen recently.

    Optional `passing` overrides the pass/fail colour when the bar depends
    on multiple checks (e.g. L/R symmetry failing on max-zone-diff even
    though the global diff is fine).
    """
    if max_value is None:
        candidate = max(target * 2.5, value * 1.2, 1.0)
        if peak is not None:
            candidate = max(candidate, peak * 1.1)
        max_value = candidate
    # Track
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    # Target zone: from target to max
    target_x = int(x + (target / max_value) * w)
    cv2.rectangle(img, (target_x, y), (x + w, y + h), (40, 90, 40), -1)
    # Current fill
    fill = max(0.0, min(1.0, value / max_value))
    fill_x = int(x + fill * w)
    is_passing = passing if passing is not None else value >= target
    color = (0, 220, 60) if is_passing else (0, 90, 220)
    cv2.rectangle(img, (x, y), (fill_x, y + h), color, -1)
    # Peak marker (amber vertical line)
    if peak is not None and peak > 0:
        peak_ratio = max(0.0, min(1.0, peak / max_value))
        peak_x = int(x + peak_ratio * w)
        cv2.line(img, (peak_x, y), (peak_x, y + h), (60, 200, 240), 2)
    # Border
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    # Label + value (+ peak value if provided)
    if peak is not None and peak > 0:
        txt = f"{label}: {value:.2f} (pico {peak:.2f})"
    else:
        txt = f"{label}: {value:.2f}"
    cv2.putText(img, _ascii(txt), (x + 4, y + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _compose_preview(frame_l: np.ndarray, frame_r: np.ndarray,
                     ev: dict, grid_l: np.ndarray, grid_r: np.ndarray,
                     peaks: PeakTracker | None = None) -> np.ndarray:
    """Build the HTTP preview image with visual bars + distance + big banner."""
    h = 360  # preview half height
    w = 640  # preview half width (16:9 to match sensor native aspect)
    vis_l = cv2.resize(frame_l, (w, h))
    vis_r = cv2.resize(frame_r, (w, h))

    # Overlay focus grid as semi-transparent rectangles on the image itself
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

    # Bars panel
    panel_h = 180
    total_w = w * 2
    panel = np.full((panel_h, total_w, 3), 18, dtype=np.uint8)

    # Row 1: center sharpness bars (with peak markers)
    bar_w = w - 40
    peak_l_scaled = peaks.peak_l / 10.0 if peaks is not None else None
    peak_r_scaled = peaks.peak_r / 10.0 if peaks is not None else None
    _draw_bar(panel, 20, 14, bar_w, 22,
              ev["center_l"] / 10.0, MIN_SCORE / 10.0,
              f"IZQ nitidez centro: {ev['center_l']:.0f}",
              max_value=max(MIN_SCORE / 10 * 2.5, ev["center_l"] / 10 * 1.2, 1.0),
              peak=peak_l_scaled)
    _draw_bar(panel, w + 20, 14, bar_w, 22,
              ev["center_r"] / 10.0, MIN_SCORE / 10.0,
              f"DER nitidez centro: {ev['center_r']:.0f}",
              max_value=max(MIN_SCORE / 10 * 2.5, ev["center_r"] / 10 * 1.2, 1.0),
              peak=peak_r_scaled)

    # Row 2: corner sharpness (absolute). In compact scenes this check is
    # skipped, so we grey the bar label and force-pass its colour.
    compact = bool(ev.get("compact_scene"))
    corner_l_label = "IZQ corners (omitido)" if compact else f"IZQ corners: {ev['corner_l']:.0f}"
    corner_r_label = "DER corners (omitido)" if compact else f"DER corners: {ev['corner_r']:.0f}"
    corner_max = max(MIN_CORNER_SCORE * 2.5, ev["corner_l"] * 1.2, ev["corner_r"] * 1.2, 100.0)
    _draw_bar(panel, 20, 50, bar_w, 22,
              ev["corner_l"], MIN_CORNER_SCORE,
              corner_l_label, max_value=corner_max,
              passing=ev["checks"]["uniformity_l"])
    _draw_bar(panel, w + 20, 50, bar_w, 22,
              ev["corner_r"], MIN_CORNER_SCORE,
              corner_r_label, max_value=corner_max,
              passing=ev["checks"]["uniformity_r"])

    # Row 3: L/R symmetry — bar tracks global diff, but fails red also when
    # per-zone diff exceeds threshold (single visual indicator for both
    # lr_global and lr_zones checks).
    sym_val = max(0.0, 100.0 - ev["lr_diff"])  # 100 = perfect, 0 = very asymmetric
    sym_passing = ev["checks"]["lr_global"] and ev["checks"]["lr_zones"]
    _draw_bar(panel, 20, 86, total_w - 40, 22,
              sym_val, 100 - MAX_LR_DIFF_PCT,
              f"Simetría L/R: {100-ev['lr_diff']:.0f}% (max zona diff {ev['max_zone_diff']:.0f}%)",
              max_value=100.0,
              passing=sym_passing)

    # Row 4: distance — show L, R, and avg so operator can spot divergence
    if ev["distance_avg_mm"] is not None:
        dist_m = ev["distance_avg_mm"] / 1000
        color = (0, 220, 60) if ev["distance_ok"] else (50, 180, 240)
        dl = ev["distance_l_mm"]
        dr = ev["distance_r_mm"]
        dl_str = f"{dl/1000:.2f}m" if dl is not None else "—"
        dr_str = f"{dr/1000:.2f}m" if dr is not None else "—"
        # Divergence warning: >10% diff between L and R hints at a distortion
        # problem on one lens (bad focus or physical misalignment).
        divergence_pct = 0.0
        if dl is not None and dr is not None:
            divergence_pct = abs(dl - dr) / max(dl, dr) * 100
        cv2.putText(panel, _ascii(f"Distancia board: {dist_m:.2f} m  (IZQ {dl_str} / DER {dr_str})"),
                    (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if divergence_pct > 10:
            cv2.putText(panel, _ascii(f"! IZQ/DER difieren {divergence_pct:.0f}% — revisar lente asimétrico"),
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 180, 240), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, _ascii(f"(objetivo {TARGET_DISTANCE_MIN_MM/1000:.2f}-{TARGET_DISTANCE_MAX_MM/1000:.2f} m)"),
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    else:
        cv2.putText(panel, _ascii("Board ChArUco no detectado — ponelo frente al par"),
                    (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 180, 240), 2, cv2.LINE_AA)

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
    cv2.putText(frame, _ascii(banner_text), (12, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


_status_html = ""
_status_lock = threading.Lock()
_last_eval: dict | None = None
_eval_lock = threading.Lock()
_report_path_global: Path | None = None


def _save_report(frame_l: np.ndarray, frame_r: np.ndarray,
                 grid_l: np.ndarray, grid_r: np.ndarray, ev: dict) -> Path:
    """Save HTML report with final frames + metrics."""
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

    dist_text = (f"{ev['distance_avg_mm']/1000:.2f} m"
                 if ev['distance_avg_mm'] is not None else "no detectado")
    compact_note = ""
    if ev.get("compact_scene"):
        compact_note = (
            '<p style="background:#fff3cd;border:1px solid #ffe39a;'
            'padding:8px 12px;border-radius:6px;color:#7a5d00">'
            'Escena compacta detectada — el check de corners fue omitido '
            'porque los bordes del frame ven superficies a distancia '
            'distinta del board. Validación basada en centro + simetría + '
            'distancia.</p>'
        )
    corner_l_label = (
        "IZQ corners: omitido (escena compacta)" if ev.get("compact_scene")
        else f"IZQ corners: {ev['corner_l']:.0f} (≥{MIN_CORNER_SCORE:.0f})"
    )
    corner_r_label = (
        "DER corners: omitido (escena compacta)" if ev.get("compact_scene")
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
            self.send_response(204); self.end_headers()
        elif self.path == "/start":
            with capture_started_lock:
                capture_started = True
            self.send_response(204); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(r"""<!DOCTYPE html>
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
    del navegador para las indicaciones de voz.
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
// Default ON — operator can turn it off and the preference persists.
let audioOn = localStorage.getItem('focus.audio') !== '0';
let audioCtx = null;
let lastSpokenHint = '';
let lastSpokenAt = 0;
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
function speak(text){
  // Queue utterances; don't cancel in-flight speech so hints flow.
  if(!audioOn || !window.speechSynthesis) return;
  try{
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'es-ES'; u.rate = 1.05;
    window.speechSynthesis.speak(u);
  }catch(e){}
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
  if(audioOn){ ensureAudioCtx(); speak('Audio activado'); beep(800, 80); }
  else if(window.speechSynthesis){
    // Cancel any in-flight utterance + the queue so turning off is immediate.
    try{ window.speechSynthesis.cancel(); }catch(e){}
  }
}
function startCapture(){
  ensureAudioCtx();
  if(audioOn) speak('Comenzando ajuste de foco');
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
// Keep polling /status after finalize — the server updates it with the
// finalised card and we detect that via the data-finalized marker.
setInterval(()=>{
  fetch('/status').then(r=>r.text()).then(t=>{
    document.getElementById('status').innerHTML=t;
    // TTS the primary hint when it changes (throttled 3s). We compare the
    // hint with digits stripped so numeric fluctuations ("315 < 200" vs
    // "318 < 200") don't re-trigger — only semantic changes do.
    const hintMatch = t.match(/data-audio-hint="([^"]*)"/);
    if(audioOn && hintMatch){
      const hint = hintMatch[1];
      const hintKey = hint.replace(/[\d.,]+/g, '').trim();
      const now = Date.now();
      if(hint && hintKey !== lastSpokenHint && now - lastSpokenAt > 3000){
        lastSpokenHint = hintKey;
        lastSpokenAt = now;
        speak(hint);
      }
    }
    if(!finalized && t.indexOf('data-finalized="1"')!==-1){
      finalized=true;
      const btn=document.getElementById('finbtn');
      btn.textContent='Finalizado';
      btn.style.background='#27ae60';
      if(audioOn) speak('Sesión finalizada');
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
</script></body></html>""".encode("utf-8"))
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
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
            self.send_response(404); self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def main() -> None:
    global latest_jpeg, shutting_down, _status_html, finish_requested
    global _report_path_global

    parser = argparse.ArgumentParser(description="Guided focus assist for stereo cameras")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE,
                        help="Minimum center Laplacian variance (default 200)")
    parser.add_argument("--min-corner-score", type=float, default=MIN_CORNER_SCORE,
                        help="Minimum mean Laplacian variance across the 4 corner zones "
                             "(default 100). Absolute metric — does the corner have detail? "
                             "Replaces the old edges/center ratio which mis-fired in small "
                             "rooms where the board dominates the centre. Raise for stricter.")
    parser.add_argument("--scene", choices=("auto", "compact", "full"), default="auto",
                        help="Scene mode. 'auto' (default): compact if the ChArUco bbox "
                             f"covers more than {int(COMPACT_BBOX_THRESHOLD*100)}%% of the "
                             "frame. 'compact': always skip the corner check (for small "
                             "test rooms where corners see walls at unrelated depths). "
                             "'full': always enforce the corner check.")
    parser.add_argument("--max-lr-diff-pct", type=float, default=MAX_LR_DIFF_PCT,
                        help="Max global L/R sharpness asymmetry (default 15%%)")
    parser.add_argument("--max-lr-zone-diff-pct", type=float, default=MAX_LR_ZONE_DIFF_PCT,
                        help="Max per-zone L/R asymmetry (default 30%%)")
    parser.add_argument("--mount-height-m", type=float, default=DEFAULT_MOUNT_HEIGHT_M,
                        help=f"Informational only: camera mount height from floor. "
                             f"Focus is done once in the lab at a fixed distance "
                             f"(see --target-distance-min/max-mm) — this flag does "
                             f"not affect the target range. "
                             f"Default {DEFAULT_MOUNT_HEIGHT_M}m.")
    parser.add_argument("--target-distance-min-mm", type=float,
                        default=TARGET_DISTANCE_MIN_MM,
                        help=f"Min target distance (mm) for focus validation. "
                             f"Default {TARGET_DISTANCE_MIN_MM:.0f}mm (lab protocol: "
                             f"focus at 2.0m ±20cm, DoF covers fleet 1.15-3.30m).")
    parser.add_argument("--target-distance-max-mm", type=float,
                        default=TARGET_DISTANCE_MAX_MM,
                        help=f"Max target distance (mm) for focus validation. "
                             f"Default {TARGET_DISTANCE_MAX_MM:.0f}mm (lab protocol).")
    parser.add_argument("--board-cols", type=int, default=9,
                        help="ChArUco columns (squares). Default 9 (canonical board)")
    parser.add_argument("--board-rows", type=int, default=6,
                        help="ChArUco rows (squares). Default 6 (canonical board)")
    parser.add_argument("--square-mm", type=float, default=45.0,
                        help="ChArUco square size in mm. Default 45")
    parser.add_argument("--marker-mm", type=float, default=33.0,
                        help="ChArUco marker size in mm. Default 33")
    parser.add_argument("--dict", dest="aruco_dict", default="DICT_4X4_100",
                        help="ArUco dictionary name (e.g. DICT_4X4_100, DICT_5X5_100). "
                             "Default DICT_4X4_100 (canonical board)")
    parser.add_argument("--legacy-pattern", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use OpenCV pre-4.6 ChArUco marker enumeration. Default "
                             "True matches calib.io's canonical board layout. Pass "
                             "--no-legacy-pattern only if you generated a board with "
                             "a post-4.6 OpenCV CharucoBoard without calling "
                             "setLegacyPattern(True).")
    parser.add_argument("--focal-px", type=float, default=None,
                        help="Override focal length in pixels for distance estimation. "
                             "Use if nominal IMX708 f gives wrong distances (e.g. sensor "
                             "mode with partial FOV). Measure by placing board at known "
                             "distance and adjusting until reported distance matches.")
    args = parser.parse_args()
    _apply_threshold_overrides(args)

    dict_attr = args.aruco_dict if args.aruco_dict.startswith("DICT_") else f"DICT_{args.aruco_dict}"
    if not hasattr(cv2.aruco, dict_attr):
        parser.error(f"Unknown ArUco dict: {args.aruco_dict}")
    dict_id = getattr(cv2.aruco, dict_attr)

    from picamera2 import Picamera2

    cam_l = Picamera2(1)
    cam_r = Picamera2(0)
    for cam in [cam_l, cam_r]:
        # Capture at the IMX708 native full-FOV binned mode (2304x1296 @ 56fps,
        # 16:9) so there's no sensor-mode ambiguity, no aspect-ratio cropping,
        # and markers at 2.5-3m have enough pixels to be detected.
        config = cam.create_still_configuration(
            main={"size": (2304, 1296), "format": "BGR888"},
            raw={"size": (2304, 1296)},
        )
        cam.configure(config)
        cam.start()
    time.sleep(1)

    board = create_charuco_board(
        board_size=(args.board_cols, args.board_rows),
        square_length=args.square_mm,
        marker_length=args.marker_mm,
        dict_id=dict_id,
        legacy_pattern=args.legacy_pattern,
    )
    # ThreadingHTTPServer so the long-running /stream handler doesn't block
    # /finish (and /status) from being served on separate threads.
    # SO_REUSEADDR so a Ctrl-C'd previous instance doesn't leave the port
    # in TIME_WAIT for the next run.
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"Focus assist — http://people-counter.local:{args.port}")
    print(f"Board: {args.board_cols}x{args.board_rows} / {args.square_mm:.0f}mm sq / "
          f"{args.marker_mm:.0f}mm mk / {dict_attr}")
    print(f"Target focus distance: "
          f"{TARGET_DISTANCE_MIN_MM/1000:.2f}-{TARGET_DISTANCE_MAX_MM/1000:.2f}m "
          f"(mount height {args.mount_height_m:.2f}m — informational)")
    print("Esperando que el operador haga click en Comenzar...")

    # Block until the operator clicks Comenzar — matches calibrate.py flow.
    try:
        while not capture_started:
            if finish_requested:
                print("\nCancelado antes de comenzar.")
                try:
                    cam_l.stop(); cam_l.close()
                    cam_r.stop(); cam_r.close()
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
            cam_l.stop(); cam_l.close()
            cam_r.stop(); cam_r.close()
        except Exception:
            pass
        try:
            server.shutdown()
        except Exception:
            pass
        sys.exit(0)
    print("Poné el board ChArUco en ese rango frente al par. Ajustá los lentes.")
    print("Click Finalizar en la UI cuando los bars estén verdes.\n")

    frame_l = frame_r = None
    grid_l = grid_r = None
    ev: dict = {}
    pass_streak = 0
    peaks = PeakTracker(window_frames=40)

    try:
        while not finish_requested:
            frame_l = cv2.cvtColor(cam_l.capture_array("main"), cv2.COLOR_RGB2BGR)
            frame_r = cv2.cvtColor(cam_r.capture_array("main"), cv2.COLOR_RGB2BGR)

            grid_l, valid_l = focus_grid(frame_l)
            grid_r, valid_r = focus_grid(frame_r)

            dist_l, ncorn_l, bbox_l = estimate_charuco_distance_mm(frame_l, board, args.focal_px)
            dist_r, ncorn_r, bbox_r = estimate_charuco_distance_mm(frame_r, board, args.focal_px)

            # Scene-mode resolution: "compact" forces corners-skipped; "full"
            # forces corners-checked; "auto" trips to compact when the board
            # fills a meaningful fraction of either frame.
            if args.scene == "compact":
                is_compact = True
            elif args.scene == "full":
                is_compact = False
            else:
                is_compact = max(bbox_l, bbox_r) > COMPACT_BBOX_THRESHOLD

            ev = evaluate_focus(
                grid_l, grid_r, dist_l, dist_r, valid_l, valid_r,
                compact_scene=is_compact,
            )
            peaks.update(ev["center_l"], ev["center_r"])

            preview = _compose_preview(frame_l, frame_r, ev, grid_l, grid_r, peaks=peaks)
            _, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()

            # Status HTML
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

            # Peak-vs-current analysis helps the operator notice if the last
            # adjustment overshot the best focus seen recently. Only show
            # when the value is materially off the peak (>15% below).
            state_l, state_r = peaks.state(ev["center_l"], ev["center_r"])
            peak_hints = []
            if state_l == "past_peak":
                peak_hints.append(
                    f"IZQ: superaste el pico reciente ({ev['center_l']:.0f} actual vs "
                    f"{peaks.peak_l:.0f} máximo) — revertí ligeramente el ajuste")
            if state_r == "past_peak":
                peak_hints.append(
                    f"DER: superaste el pico reciente ({ev['center_r']:.0f} actual vs "
                    f"{peaks.peak_r:.0f} máximo) — revertí ligeramente el ajuste")
            peak_html = ""
            if peak_hints:
                peak_html = (
                    '<div style="color:#3fb6f0;font-size:13px;margin-top:6px">'
                    + "<br>".join(peak_hints) + "</div>"
                )

            lighting = live_lighting_warnings(frame_l, frame_r)
            lighting_html = ""
            if lighting:
                lighting_html = '<div style="color:#f1c40f">⚠ ' + " · ".join(lighting) + "</div>"

            compact_html = ""
            if ev.get("compact_scene"):
                compact_html = (
                    '<div style="color:#3498db;font-size:13px;margin-top:6px">'
                    'Escena compacta — solo centro + simetría + distancia'
                    '</div>'
                )

            # data-audio-hint: what the browser TTS will read aloud.
            # Sanitise quotes so the attribute parses cleanly.
            audio_hint = lead.replace('"', '')
            html = (
                f'<div data-audio-hint="{audio_hint}" '
                f'style="color:{color_status};font-size:18px;font-weight:700">{lead}</div>'
                f'{lighting_html}'
                f'{compact_html}'
                f'{peak_html}'
                f'<div style="color:#888;font-size:13px;margin-top:6px">'
                f'ChArUco IZQ:{ncorn_l} esq · DER:{ncorn_r} esq</div>'
            )
            with _status_lock:
                _status_html = html

            # Terminal one-liner
            verdict = "PASS" if ev["all_pass_with_distance"] else "FAIL"
            dist_str = f"{ev['distance_avg_mm']/1000:.2f}m" if ev['distance_avg_mm'] else "-"
            scene_tag = "C" if ev.get("compact_scene") else "F"
            print(
                f"\r  [{verdict}/{scene_tag}] L/R:{ev['lr_diff']:4.1f}% | "
                f"crn_L:{ev['corner_l']:5.0f} crn_R:{ev['corner_r']:5.0f} | "
                f"ctr_L:{ev['center_l']:6.0f} ctr_R:{ev['center_r']:6.0f} | d:{dist_str}    ",
                end="", flush=True,
            )
            # Sleep in 50ms chunks so the loop can react to finish_requested
            # promptly (capture+process at 2304x1296 already takes ~500ms).
            for _ in range(2):
                if finish_requested:
                    break
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")

    shutting_down = True

    # Save report if we have state
    report_path = None
    if frame_l is not None and frame_r is not None and grid_l is not None:
        try:
            report_path = _save_report(frame_l, frame_r, grid_l, grid_r, ev)
            _report_path_global = report_path
        except Exception as e:
            print(f"\nNo se pudo escribir el reporte: {e}")

    try:
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
    except Exception:
        pass

    # Post the finalisation summary to /status so the browser — still
    # polling — renders a "done" screen instead of the stream going dark.
    # Keep the server alive for a grace period so the user has time to
    # read it and click the report link.
    verdict = "PASS" if (ev and ev.get("all_pass_with_distance")) else "FAIL"
    verdict_color = "#2ecc71" if verdict == "PASS" else "#e74c3c"
    report_html = ""
    if report_path is not None:
        report_html = (
            f'<div style="margin-top:14px;font-size:14px">'
            f'Reporte guardado en:<br>'
            f'<code style="background:#0b0b0d;padding:4px 8px;border-radius:4px;'
            f'display:inline-block;margin-top:4px;word-break:break-all">{report_path}</code>'
            f'</div>'
        )
    report_link_html = ""
    if report_path is not None:
        report_link_html = (
            '<a href="/report" target="_blank" '
            'style="display:inline-block;margin-top:14px;padding:10px 18px;'
            'background:#2980b9;color:#fff;text-decoration:none;border-radius:6px;'
            'font-weight:600;font-size:14px">Abrir reporte en nueva pestaña</a>'
        )
    final_html = (
        f'<div data-finalized="1">'
        f'<div style="color:{verdict_color};font-size:22px;font-weight:700;'
        f'margin-bottom:12px">Sesión finalizada — {verdict}</div>'
        f'<div style="color:#aaa;font-size:13px;line-height:1.6">'
        f'Las cámaras se cerraron y el reporte HTML fue generado.'
        f'</div>'
        f'{report_link_html}'
        f'{report_html}'
        f'</div>'
    )
    with _status_lock:
        _status_html = final_html

    print("\n" + "=" * 60)
    print("FOCUS SESSION TERMINADA")
    print("=" * 60)
    if ev:
        print(f"  Resultado: {verdict}")
        for hint in ev.get("hints", []):
            print(f"    -> {hint}")
    if report_path is not None:
        print(f"  Reporte HTML: {report_path}")
    print()

    # Grace period — enough for the browser to pick up the final /status
    # and open the report in a new tab. Matches the calibrate wizard grace
    # so both tools behave consistently.
    time.sleep(10)
    os._exit(0)


if __name__ == "__main__":
    main()

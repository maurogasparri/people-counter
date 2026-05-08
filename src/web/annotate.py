"""Helpers de anotación para el viewer web del runtime.

Funciones puras que dibujan sobre frames BGR. Sin estado, sin I/O. Se
mantienen fuera de ``viewer.py`` así el módulo de streaming queda chico y
fácil de testear aislado.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Paleta de colores BGR
_COLOR_CONFIRMED = (0, 255, 0)
_COLOR_PENDING = (0, 165, 255)
_COLOR_CANDIDATE = (180, 180, 180)
_COLOR_DET = (90, 90, 90)
_COLOR_ROI = (0, 0, 255)         # rojo
_COLOR_TEXT = (255, 255, 255)
_COLOR_OVERLAY_BG = (0, 0, 0)

# Paleta de la counting-line por label de dirección. Cualquier otra cosa cae
# a blanco así que un label exótico igual renderiza visiblemente. Verdes para
# eventos IN-side, azules para OUT-side — matchea el modelo mental del operador.
_LINE_COLOR_BY_LABEL = {
    "ingress": (0, 255, 0),       # verde: IN
    "egress": (255, 0, 0),        # azul: OUT
    "in": (0, 255, 0),
    "out": (255, 0, 0),
    "enter": (0, 255, 0),
    "leave": (255, 0, 0),
}
_LINE_COLOR_FALLBACK = (255, 255, 255)


def annotate_left(
    frame: np.ndarray,
    detections: list,
    tracks: dict,
    counter: Optional[Any],
    fps: float = 0.0,
) -> np.ndarray:
    """Dibuja ROI, línea, detecciones raw y tracks sobre una copia de ``frame``.

    Args:
        frame: Frame BGR de la cámara izquierda (rectificado).
        detections: objetos Detection (tienen ``.bbox`` y ``.centroid``).
        tracks: dict[int, Track] de EuclideanTracker.
        counter: instancia LineCounter o ROICounter (se lee para el
            overlay de geometría + totales). Puede ser None.
        fps: estimación de FPS del pipeline para el overlay inferior.
    """
    out = frame.copy()

    # Geometría primero así las detecciones quedan arriba.
    _draw_counter_geometry(out, counter)

    # Detecciones raw en gris sutil. Las posiciones trackeadas reciben
    # un marker coloreado encima así el operador puede ver qué
    # detecciones produjeron tracks y cuáles no.
    for det in detections:
        try:
            x1, y1, x2, y2 = det.bbox
        except Exception:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      _COLOR_DET, 1)

    # Diferir el import así un import circular en init parcial no
    # vuela el módulo viewer.
    from src.tracking.tracker import CONFIRMED, PENDING, CANDIDATE
    state_colour = {
        CONFIRMED: _COLOR_CONFIRMED,
        PENDING: _COLOR_PENDING,
        CANDIDATE: _COLOR_CANDIDATE,
    }
    # Fuentes más grandes así el operador puede leer el overlay
    # mientras camina bajo las cámaras durante un check de piloto (el
    # live viewer está pensado para debug on-site, no para archivar).
    for tid, track in tracks.items():
        colour = state_colour.get(getattr(track, "state", None))
        if colour is None:
            continue
        positions = getattr(track, "positions", None)
        if not positions:
            continue
        cx, cy = int(positions[-1][0]), int(positions[-1][1])
        cv2.circle(out, (cx, cy), 10, colour, -1)
        cv2.putText(out, f"#{tid}", (cx + 14, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2,
                    cv2.LINE_AA)
        # Label de altura de la metadata de detección más reciente del track.
        meta = getattr(track, "meta", None)
        history = meta.get("detection_history") if isinstance(meta, dict) else None
        if history:
            last = history[-1]
            head_mm = last.get("head_height_mm")
            cls = last.get("height_class") or "unknown"
            if isinstance(head_mm, (int, float)) and head_mm > 0:
                label = f"{head_mm/1000:.2f}m {cls}"
            else:
                label = cls
            cv2.putText(out, label, (cx + 14, cy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2,
                        cv2.LINE_AA)

    _draw_counter_overlay(out, counter, fps)
    return out


def depth_to_colormap(
    depth_map: Optional[np.ndarray],
    vmin_mm: float = 500.0,
    vmax_mm: float = 5000.0,
) -> np.ndarray:
    """Renderiza un mapa depth-en-mm como un colormap BGR JET.

    Cero (= disparity inválido) renderiza en negro. Input None /
    vacío devuelve un frame chico gris oscuro así el panel es
    visualmente distinto de un depth map real y el caller no necesita
    chequear None.
    """
    if depth_map is None or depth_map.size == 0:
        return np.full((100, 100, 3), 30, dtype=np.uint8)
    d = depth_map.astype(np.float32)
    invalid = d <= 0
    d = np.clip(d, vmin_mm, vmax_mm)
    norm = ((d - vmin_mm) / (vmax_mm - vmin_mm) * 255.0).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    if np.any(invalid):
        coloured[invalid] = (0, 0, 0)
    return coloured


def compose_3panel(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    target_height: int = 480,
) -> np.ndarray:
    """Composite de dos filas: fila superior L | R side-by-side, fila
    inferior el panel de depth que span el mismo ancho.

    La fila superior L/R le da al operador la misma vista que ven
    las cámaras; la fila de depth abajo queda aproximadamente
    cuadrada así el colormap es legible. Los paneles faltantes /
    vacíos se llenan con un placeholder gris oscuro así el composite
    nunca crashea ante input parcial.
    """
    def _to_bgr_height(img: Optional[np.ndarray], h_target: int,
                       w_fallback: int) -> np.ndarray:
        if img is None or img.size == 0:
            return np.full((h_target, w_fallback, 3), 30, dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        if h != h_target:
            scale = h_target / h
            img = cv2.resize(
                img, (max(1, int(w * scale)), h_target),
                interpolation=cv2.INTER_AREA,
            )
        return img

    # Fila superior: L y R a target_height cada una.
    l_img = _to_bgr_height(left, target_height, target_height)
    r_img = _to_bgr_height(right, target_height, target_height)
    top = cv2.hconcat([l_img, r_img])
    top_w = top.shape[1]

    # Fila inferior: depth redimensionado para spanear exactamente el
    # ancho del top. Mantenerlo aproximadamente la mitad de la altura
    # así el layout se ve balanceado.
    depth_h = max(1, target_height // 2)
    if depth is None or depth.size == 0:
        bottom = np.full((depth_h, top_w, 3), 30, dtype=np.uint8)
    else:
        d = depth
        if d.ndim == 2:
            d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        bottom = cv2.resize(
            d, (top_w, depth_h), interpolation=cv2.INTER_AREA,
        )
    return cv2.vconcat([top, bottom])


# ----------------------------------------------------------------- internals
def _draw_counter_geometry(frame: np.ndarray, counter: Optional[Any]) -> None:
    """Overlay del rectángulo del ROI (si hay) y cada segmento de
    línea con una flecha perpendicular mostrando la dirección contada.

    Cada línea se dibuja en el color que matchea su label dominante
    (``ingress`` -> verde, ``egress`` -> azul, cualquier otra cosa en
    blanco). Una flecha por dirección con label, perpendicular al
    segmento, apuntando hacia el lado en el que un cruce tiene que
    terminar para que dispare ese label.
    """
    if counter is None:
        return
    roi = getattr(counter, "roi", None)
    if roi is not None:
        try:
            x_min, x_max, y_min, y_max = roi
            cv2.rectangle(frame, (int(x_min), int(y_min)),
                          (int(x_max), int(y_max)), _COLOR_ROI, 4)
        except Exception:
            logger.debug("ROI overlay failed", exc_info=True)

    lines = getattr(counter, "lines", None) or []
    for line in lines:
        try:
            _draw_line_with_arrows(frame, line)
        except Exception:
            logger.debug("Line overlay failed", exc_info=True)


def _draw_line_with_arrows(frame: np.ndarray, line: Any) -> None:
    """Dibuja una línea de conteo + flechas per-dirección.

    El largo de la flecha escala con el largo del segmento así sigue
    siendo visible en ROIs chicos sin desbordar en grandes. Si ambas
    direcciones tienen label el segmento renderiza en blanco (neutral)
    y el color per-flecha codifica qué lado de la línea es qué evento.
    """
    x1, y1 = int(line.from_xy[0]), int(line.from_xy[1])
    x2, y2 = int(line.to_xy[0]), int(line.to_xy[1])
    orientation = line.orientation
    labels: dict[str, str] = line.labels

    seg_len = max(1, abs(x2 - x1) + abs(y2 - y1))
    arrow_len = max(20, min(60, seg_len // 6))
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2

    if len(labels) >= 2:
        seg_color = _LINE_COLOR_FALLBACK
    else:
        only_label = next(iter(labels.values()), None)
        seg_color = _LINE_COLOR_BY_LABEL.get(
            only_label or "", _LINE_COLOR_FALLBACK,
        )
    cv2.line(frame, (x1, y1), (x2, y2), seg_color, 4)

    # La cola de la flecha se ancla sobre la línea y la punta se
    # extiende un ``arrow_len`` hacia el lado donde el cruce tiene
    # que terminar. Esto mantiene la flecha estrictamente de un lado
    # del segmento en vez de cruzarlo, lo cual hace que la lectura
    # "andá para acá" sea inmediata.
    if orientation == "horizontal":
        for direction, label in labels.items():
            color = _LINE_COLOR_BY_LABEL.get(label, _LINE_COLOR_FALLBACK)
            if direction == "top_to_bottom":
                tail = (mx, my)
                tip = (mx, my + arrow_len)
            else:  # bottom_to_top
                tail = (mx, my)
                tip = (mx, my - arrow_len)
            cv2.arrowedLine(frame, tail, tip, color, 4, tipLength=0.35)
    else:
        for direction, label in labels.items():
            color = _LINE_COLOR_BY_LABEL.get(label, _LINE_COLOR_FALLBACK)
            if direction == "left_to_right":
                tail = (mx, my)
                tip = (mx + arrow_len, my)
            else:  # right_to_left
                tail = (mx, my)
                tip = (mx - arrow_len, my)
            cv2.arrowedLine(frame, tail, tip, color, 4, tipLength=0.35)


def _draw_counter_overlay(
    frame: np.ndarray, counter: Optional[Any], fps: float,
) -> None:
    h, w = frame.shape[:2]
    in_n = getattr(counter, "total_in", 0) if counter else 0
    out_n = getattr(counter, "total_out", 0) if counter else 0
    text = f"IN: {in_n}  OUT: {out_n}  FPS: {fps:.1f}"
    bar_h = 56
    cv2.rectangle(frame, (0, h - bar_h), (w, h), _COLOR_OVERLAY_BG, -1)
    cv2.putText(frame, text, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, _COLOR_TEXT, 3,
                cv2.LINE_AA)

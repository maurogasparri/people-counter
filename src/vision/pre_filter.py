"""Filtro pre-tracker por polígono.

Descarta detecciones cuyo centroide cae fuera del polígono de
"tracking zone". Reduce FPs estructurales en sites con clutter intenso
(percheros, mostradores, vidrieras con tráfico exterior) sin tocar la
``counting_zone`` semántica del counter.

Análogo conceptualmente al ``pointPolygonTest`` del incumbent FFC, pero:

- **Opt-in per-site**: default OFF (back-compat). El pipeline post-gate
  funciona bien en sites limpios y conserva el lead-in del approach,
  fundamental para el rescue cascade del counter.
- **Polígono más AMPLIO que el counting_zone**: típicamente
  ``counting_zone`` + margen de ~200-300 px en cada lado, suficiente
  para capturar 1-2 m de approach antes de la línea de cruce. Eso
  preserva ``last_outside_pos`` snap-eado desde un frame real outside
  counting_zone (pero inside tracking_zone).

El polígono se especifica como lista de N >= 3 vértices ``[(x, y), ...]``.
Para counting zones rectangulares, ``derive_polygon_from_counting_zone``
deriva un rectángulo automáticamente expandiendo cada lado por un margen.

Ver ``docs/tracker_tuning.md`` patrón 6 (clutter estructural en retail).
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

import cv2
import numpy as np


class _HasCentroid(Protocol):
    """Cualquier objeto que expone ``centroid`` como (x, y)."""

    @property
    def centroid(self) -> tuple[float, float]: ...


def derive_polygon_from_counting_zone(
    counting_zone: tuple[float, float, float, float],
    margin_px: float,
    frame_size: Optional[tuple[int, int]] = None,
) -> list[tuple[float, float]]:
    """Construye un polígono rectangular expandiendo ``counting_zone`` por
    ``margin_px`` en cada lado. Recorta al tamaño del frame si se provee.

    **Caveat**: cuando el ``counting_zone`` es grande relativo al frame,
    el ``margin_px`` puede chocar con los bordes y producir un polígono
    igual al frame entero (sin efecto de filtro). En ese caso conviene
    usar ``derive_polygon_from_frame_margin`` que es más predecible.

    Args:
        counting_zone: ``(x_min, x_max, y_min, y_max)``.
        margin_px: margen a expandir (px) en cada lado.
        frame_size: ``(width, height)`` opcional para clampear el polígono
            a los bordes del frame.

    Returns:
        Lista de 4 vértices ``[(x, y), ...]`` en orden horario, lista para
        pasarse a ``filter_detections_by_polygon``.
    """
    x_min, x_max, y_min, y_max = counting_zone
    ex_min = x_min - margin_px
    ex_max = x_max + margin_px
    ey_min = y_min - margin_px
    ey_max = y_max + margin_px
    if frame_size is not None:
        w, h = frame_size
        ex_min = max(0.0, ex_min)
        ex_max = min(float(w), ex_max)
        ey_min = max(0.0, ey_min)
        ey_max = min(float(h), ey_max)
    return [
        (float(ex_min), float(ey_min)),
        (float(ex_max), float(ey_min)),
        (float(ex_max), float(ey_max)),
        (float(ex_min), float(ey_max)),
    ]


def derive_polygon_from_frame_margin(
    frame_size: tuple[int, int],
    frame_margin_px: float,
    counting_zone: Optional[tuple[float, float, float, float]] = None,
    lead_in_buffer_px: float = 30.0,
) -> list[tuple[float, float]]:
    """Construye un polígono rectangular insetado ``frame_margin_px`` desde
    cada borde del frame. Modo predecible y simétrico (independiente del
    tamaño del counting_zone).

    Si se provee ``counting_zone``, el margen se REDUCE automáticamente
    cuando chocaría con el counting_zone (con un buffer adicional de
    ``lead_in_buffer_px`` para preservar el approach del rescue cascade).
    Esto evita que el polígono corte detecciones legítimas justo afuera
    del counting_zone que sirven para snap-ear ``last_outside_pos`` en
    la entry-fresca.

    Args:
        frame_size: ``(width, height)`` del frame en pixels.
        frame_margin_px: margen desde cada borde del frame en pixels.
        counting_zone: opcional ``(x_min, x_max, y_min, y_max)``. Cuando
            se provee, el margen se ajusta para preservar lead-in.
        lead_in_buffer_px: margen mínimo de lead-in que se preserva entre
            el polígono y el counting_zone. Default 30 px (~150 ms de
            approach a velocidad de caminata).

    Returns:
        Lista de 4 vértices ``[(x, y), ...]`` en orden horario.
    """
    w, h = frame_size
    if counting_zone is not None:
        cz_x_min, cz_x_max, cz_y_min, cz_y_max = counting_zone
        safe_left = min(frame_margin_px, max(0.0, cz_x_min - lead_in_buffer_px))
        safe_right = min(frame_margin_px, max(0.0, w - cz_x_max - lead_in_buffer_px))
        safe_top = min(frame_margin_px, max(0.0, cz_y_min - lead_in_buffer_px))
        safe_bottom = min(frame_margin_px, max(0.0, h - cz_y_max - lead_in_buffer_px))
    else:
        safe_left = safe_right = safe_top = safe_bottom = float(frame_margin_px)
    return [
        (float(safe_left), float(safe_top)),
        (float(w - safe_right), float(safe_top)),
        (float(w - safe_right), float(h - safe_bottom)),
        (float(safe_left), float(h - safe_bottom)),
    ]


def filter_detections_by_polygon(
    detections: Sequence[_HasCentroid],
    polygon: Sequence[tuple[float, float]],
) -> list[_HasCentroid]:
    """Devuelve las detecciones cuyo centroide cae dentro del polígono.

    Usa ``cv2.pointPolygonTest`` (implementación C++ optimizada, ~10 µs
    por test en hardware típico de la RPi5). Para polígonos con menos
    de 3 vértices, devuelve la lista sin filtrar (no-op safety: un
    polígono malformado no debe romper el pipeline).

    Args:
        detections: secuencia de objetos con atributo ``centroid``
            como tupla ``(x, y)``.
        polygon: lista de N >= 3 vértices ``[(x, y), ...]``.

    Returns:
        Lista filtrada — solo las detecciones cuyo centroide cae adentro
        o exactamente en el borde del polígono.
    """
    det_list = list(detections)
    if len(polygon) < 3:
        return det_list
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    kept: list[_HasCentroid] = []
    for d in det_list:
        cx = float(d.centroid[0])
        cy = float(d.centroid[1])
        # measureDist=False → -1 (outside), 0 (en el borde), +1 (inside).
        # Aceptamos >= 0 (inside o en el borde).
        if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
            kept.append(d)
    return kept

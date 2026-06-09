"""Tests del filtro pre-tracker por polígono."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.vision.pre_filter import (
    derive_polygon_from_counting_zone,
    derive_polygon_from_frame_margin,
    filter_detections_by_polygon,
)


@dataclass
class _Detection:
    """Stub mínimo con la interfaz que ``filter_detections_by_polygon`` espera."""

    centroid: tuple[float, float]
    confidence: float = 0.5


# ---------------------------------------------------------------------------
# derive_polygon_from_counting_zone
# ---------------------------------------------------------------------------


def test_derive_polygon_basic_rectangle():
    """Polígono = counting_zone + margin en cada lado, 4 vértices en orden."""
    poly = derive_polygon_from_counting_zone(
        counting_zone=(200.0, 800.0, 100.0, 500.0),
        margin_px=50.0,
    )
    assert len(poly) == 4
    # Esperado: (x_min - margin, y_min - margin) → ... → (x_min - margin, y_max + margin)
    assert poly[0] == (150.0, 50.0)
    assert poly[1] == (850.0, 50.0)
    assert poly[2] == (850.0, 550.0)
    assert poly[3] == (150.0, 550.0)


def test_derive_polygon_clamps_to_frame_size():
    """Si counting_zone + margin sale del frame, se recorta a (0,0)-(w,h)."""
    poly = derive_polygon_from_counting_zone(
        counting_zone=(50.0, 1100.0, 50.0, 600.0),
        margin_px=200.0,
        frame_size=(1152, 648),
    )
    # x_min - 200 = -150 → clampea a 0.
    # x_max + 200 = 1300 → clampea a 1152.
    # y_min - 200 = -150 → clampea a 0.
    # y_max + 200 = 800 → clampea a 648.
    assert poly[0] == (0.0, 0.0)
    assert poly[1] == (1152.0, 0.0)
    assert poly[2] == (1152.0, 648.0)
    assert poly[3] == (0.0, 648.0)


def test_derive_polygon_zero_margin_returns_counting_zone():
    """Margen 0 = polígono coincide exactamente con el counting_zone."""
    poly = derive_polygon_from_counting_zone(
        counting_zone=(100.0, 500.0, 200.0, 400.0),
        margin_px=0.0,
    )
    assert poly == [(100.0, 200.0), (500.0, 200.0), (500.0, 400.0), (100.0, 400.0)]


# ---------------------------------------------------------------------------
# derive_polygon_from_frame_margin
# ---------------------------------------------------------------------------


def test_derive_polygon_from_frame_margin_simple():
    """Sin counting_zone, el margen es simétrico desde cada borde."""
    poly = derive_polygon_from_frame_margin(
        frame_size=(1152, 648),
        frame_margin_px=100.0,
    )
    assert poly == [(100.0, 100.0), (1052.0, 100.0), (1052.0, 548.0), (100.0, 548.0)]


def test_derive_polygon_from_frame_margin_respects_counting_zone():
    """Si el margen chocaría con counting_zone (+ buffer), se reduce
    automáticamente para preservar lead-in del approach."""
    # counting_zone (246, 906, 204, 444) en frame 1152×648.
    # Distancias del counting_zone a los bordes del frame:
    #   left = 246, right = 246, top = 204, bottom = 204.
    # Con margin=300 y lead_in_buffer=30, el límite es (distance - 30):
    #   safe_left  = min(300, 246-30) = 216
    #   safe_right = min(300, 246-30) = 216
    #   safe_top   = min(300, 204-30) = 174
    #   safe_bottom= min(300, 204-30) = 174
    poly = derive_polygon_from_frame_margin(
        frame_size=(1152, 648),
        frame_margin_px=300.0,
        counting_zone=(246.0, 906.0, 204.0, 444.0),
    )
    assert poly == [(216.0, 174.0), (936.0, 174.0), (936.0, 474.0), (216.0, 474.0)]


def test_derive_polygon_from_frame_margin_unrestricted_when_small():
    """Si el margen NO choca con counting_zone, se usa tal cual."""
    poly = derive_polygon_from_frame_margin(
        frame_size=(1152, 648),
        frame_margin_px=80.0,
        counting_zone=(246.0, 906.0, 204.0, 444.0),
    )
    # 80 < (204 - 30) → no choca → se usa 80 en todos los bordes.
    assert poly == [(80.0, 80.0), (1072.0, 80.0), (1072.0, 568.0), (80.0, 568.0)]


def test_derive_polygon_from_frame_margin_custom_lead_in_buffer():
    """El buffer de lead-in es configurable (más grande = polígono más
    restringido respecto al counting_zone)."""
    poly = derive_polygon_from_frame_margin(
        frame_size=(1152, 648),
        frame_margin_px=300.0,
        counting_zone=(246.0, 906.0, 204.0, 444.0),
        lead_in_buffer_px=100.0,
    )
    # safe_top = min(300, 204-100) = 104.
    assert poly[0] == (146.0, 104.0)


# ---------------------------------------------------------------------------
# filter_detections_by_polygon
# ---------------------------------------------------------------------------


@pytest.fixture
def square_polygon():
    """Cuadrado simple [(100,100), (400,100), (400,400), (100,400)]."""
    return [(100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)]


def test_filter_keeps_detections_inside(square_polygon):
    """Detecciones cuyo centroide cae dentro se conservan."""
    dets = [
        _Detection(centroid=(200, 200)),  # adentro
        _Detection(centroid=(350, 350)),  # adentro
        _Detection(centroid=(101, 101)),  # adentro (apenas)
    ]
    kept = filter_detections_by_polygon(dets, square_polygon)
    assert len(kept) == 3


def test_filter_drops_detections_outside(square_polygon):
    """Detecciones cuyo centroide cae fuera se descartan."""
    dets = [
        _Detection(centroid=(50, 200)),  # afuera izquierda
        _Detection(centroid=(500, 200)),  # afuera derecha
        _Detection(centroid=(200, 50)),  # afuera arriba
        _Detection(centroid=(200, 500)),  # afuera abajo
    ]
    kept = filter_detections_by_polygon(dets, square_polygon)
    assert kept == []


def test_filter_mixed_inside_outside(square_polygon):
    """Mix inside/outside: solo los inside sobreviven, preservando el orden."""
    dets = [
        _Detection(centroid=(50, 50), confidence=0.9),  # afuera
        _Detection(centroid=(250, 250), confidence=0.5),  # adentro
        _Detection(centroid=(500, 250), confidence=0.7),  # afuera
        _Detection(centroid=(150, 300), confidence=0.3),  # adentro
    ]
    kept = filter_detections_by_polygon(dets, square_polygon)
    assert len(kept) == 2
    assert kept[0].confidence == 0.5
    assert kept[1].confidence == 0.3


def test_filter_accepts_border_points(square_polygon):
    """Puntos exactamente en el borde se consideran inside (>= 0)."""
    dets = [
        _Detection(centroid=(100, 250)),  # borde izquierdo
        _Detection(centroid=(400, 250)),  # borde derecho
        _Detection(centroid=(250, 100)),  # borde superior
        _Detection(centroid=(100, 100)),  # vértice
    ]
    kept = filter_detections_by_polygon(dets, square_polygon)
    assert len(kept) == 4


def test_filter_empty_polygon_is_noop_safety():
    """Polígono con < 3 vértices = sin filter (no rompe el pipeline)."""
    dets = [
        _Detection(centroid=(50, 50)),
        _Detection(centroid=(5000, 5000)),
    ]
    assert filter_detections_by_polygon(dets, []) == dets
    assert filter_detections_by_polygon(dets, [(0, 0)]) == dets
    assert filter_detections_by_polygon(dets, [(0, 0), (10, 10)]) == dets


def test_filter_with_non_rectangular_polygon():
    """Polígono arbitrario (triángulo) — el filtro respeta la forma exacta.

    Triángulo con base de y=0 (x=0..400) y vértice en (200, 400). En y=350
    el ancho del triángulo es x ∈ [175, 225] (lados convergen al vértice).
    """
    triangle = [(0.0, 0.0), (400.0, 0.0), (200.0, 400.0)]
    dets = [
        _Detection(centroid=(200, 100)),  # adentro del triángulo
        _Detection(centroid=(50, 50)),  # adentro (cerca de un vértice)
        _Detection(centroid=(350, 50)),  # adentro
        _Detection(centroid=(50, 350)),  # afuera (lateral, fuera del cono)
        _Detection(centroid=(350, 350)),  # afuera (lateral derecho)
        _Detection(centroid=(200, 450)),  # afuera (debajo del vértice)
    ]
    kept = filter_detections_by_polygon(dets, triangle)
    kept_centroids = {d.centroid for d in kept}
    assert (200, 100) in kept_centroids
    assert (50, 50) in kept_centroids
    assert (350, 50) in kept_centroids
    assert (50, 350) not in kept_centroids
    assert (350, 350) not in kept_centroids
    assert (200, 450) not in kept_centroids


def test_filter_empty_detections_list(square_polygon):
    """Lista vacía de detecciones devuelve lista vacía."""
    assert filter_detections_by_polygon([], square_polygon) == []


# ---------------------------------------------------------------------------
# Integración: derive + filter end-to-end
# ---------------------------------------------------------------------------


def test_auto_derived_polygon_filters_periphery():
    """Caso típico: counting_zone retail centrado, margin 250, detecciones
    en el perchero (lateral del frame) se filtran; adentro del approach se
    preservan."""
    counting_zone = (400.0, 800.0, 250.0, 450.0)  # ej. site retail típico
    poly = derive_polygon_from_counting_zone(
        counting_zone,
        margin_px=250.0,
        frame_size=(1152, 648),
    )
    # poly = (150, 0), (1050, 0), (1050, 648), (150, 648)   tras clamp
    dets = [
        _Detection(centroid=(600, 350)),  # dentro counting_zone (cuenta)
        _Detection(centroid=(200, 350)),  # dentro del lead-in (approach OK)
        _Detection(centroid=(50, 350)),  # afuera (perchero izquierdo) — filtrado
        _Detection(centroid=(1100, 350)),  # afuera (mostrador derecho) — filtrado
    ]
    kept = filter_detections_by_polygon(dets, poly)
    kept_centroids = {d.centroid for d in kept}
    assert (600, 350) in kept_centroids
    assert (200, 350) in kept_centroids
    assert (50, 350) not in kept_centroids
    assert (1100, 350) not in kept_centroids

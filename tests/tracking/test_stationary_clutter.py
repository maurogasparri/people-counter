"""Tests del filtro de clutter estático (stationary_track_ids)."""

import numpy as np

from src.tracking.tracker import stationary_track_ids

# counting zone (x_min, x_max, y_min, y_max)
COUNTING_ZONE = (100.0, 1050.0, 150.0, 500.0)


class _FakeTrack:
    def __init__(self, positions):
        self.positions = [np.array([x, y, 0.0]) for x, y in positions]


def test_static_off_counting_zone_track_is_clutter():
    # Fuera de la counting zone (y=620 > 500), sin moverse, suficientes frames.
    tracks = {7: _FakeTrack([(600.0, 620.0)] * 25)}
    assert stationary_track_ids(tracks, COUNTING_ZONE, min_frames=20, max_movement_px=50) == {7}


def test_moving_track_not_clutter():
    # Fuera de la counting zone pero caminando -> se mueve -> no es clutter.
    tracks = {7: _FakeTrack([(600.0, 600.0 + i * 10.0) for i in range(25)])}
    assert stationary_track_ids(tracks, COUNTING_ZONE) == set()


def test_static_inside_counting_zone_protected():
    # Estático PERO dentro de la counting zone -> nunca se trata como clutter (protege
    # a una persona parada en el umbral).
    tracks = {7: _FakeTrack([(500.0, 300.0)] * 25)}
    assert stationary_track_ids(tracks, COUNTING_ZONE) == set()


def test_too_young_not_clutter():
    # Estático fuera de la counting zone pero todavía con poca historia.
    tracks = {7: _FakeTrack([(600.0, 620.0)] * 5)}
    assert stationary_track_ids(tracks, COUNTING_ZONE, min_frames=20) == set()


def test_no_counting_zone_returns_empty():
    # Sin counting zone no filtramos nada.
    tracks = {7: _FakeTrack([(600.0, 620.0)] * 25)}
    assert stationary_track_ids(tracks, None) == set()


def test_mixed_only_clutter_flagged():
    tracks = {
        1: _FakeTrack([(600.0, 620.0)] * 25),                       # clutter
        2: _FakeTrack([(600.0, 600.0 + i * 10.0) for i in range(25)]),  # caminante
        3: _FakeTrack([(500.0, 300.0)] * 25),                       # parado en counting zone
    }
    assert stationary_track_ids(tracks, COUNTING_ZONE) == {1}

"""Tests del filtro de clutter estático (stationary_track_ids)."""

import numpy as np

from src.tracking.tracker import stationary_track_ids

# ROI (x_min, x_max, y_min, y_max)
ROI = (100.0, 1050.0, 150.0, 500.0)


class _FakeTrack:
    def __init__(self, positions):
        self.positions = [np.array([x, y, 0.0]) for x, y in positions]


def test_static_off_roi_track_is_clutter():
    # Fuera del ROI (y=620 > 500), sin moverse, suficientes frames.
    tracks = {7: _FakeTrack([(600.0, 620.0)] * 25)}
    assert stationary_track_ids(tracks, ROI, min_frames=20, max_movement_px=50) == {7}


def test_moving_track_not_clutter():
    # Fuera del ROI pero caminando -> se mueve -> no es clutter.
    tracks = {7: _FakeTrack([(600.0, 600.0 + i * 10.0) for i in range(25)])}
    assert stationary_track_ids(tracks, ROI) == set()


def test_static_inside_roi_protected():
    # Estático PERO dentro del ROI -> nunca se trata como clutter (protege
    # a una persona parada en el umbral).
    tracks = {7: _FakeTrack([(500.0, 300.0)] * 25)}
    assert stationary_track_ids(tracks, ROI) == set()


def test_too_young_not_clutter():
    # Estático fuera del ROI pero todavía con poca historia.
    tracks = {7: _FakeTrack([(600.0, 620.0)] * 5)}
    assert stationary_track_ids(tracks, ROI, min_frames=20) == set()


def test_no_roi_returns_empty():
    # Sin ROI no filtramos nada.
    tracks = {7: _FakeTrack([(600.0, 620.0)] * 25)}
    assert stationary_track_ids(tracks, None) == set()


def test_mixed_only_clutter_flagged():
    tracks = {
        1: _FakeTrack([(600.0, 620.0)] * 25),                       # clutter
        2: _FakeTrack([(600.0, 600.0 + i * 10.0) for i in range(25)]),  # caminante
        3: _FakeTrack([(500.0, 300.0)] * 25),                       # parado en ROI
    }
    assert stationary_track_ids(tracks, ROI) == {1}

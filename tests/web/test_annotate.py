"""Smoke tests for src/web/annotate.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.tracking.tracker import CONFIRMED, PENDING, CANDIDATE
from src.web.annotate import (
    annotate_left,
    compose_3panel,
    depth_to_colormap,
)


@dataclass
class _FakeDet:
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    confidence: float = 0.9


@dataclass
class _FakeTrack:
    track_id: int
    state: str
    positions: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class _FakeROICounter:
    def __init__(self):
        self._roi = (50.0, 250.0, 100.0, 200.0)
        self._orientation = "horizontal"
        self._line_pos = 150.0
        self.total_in = 7
        self.total_out = 3


class _FakeLineCounter:
    def __init__(self):
        self.line_y = 240.0
        self.total_in = 2
        self.total_out = 1


def test_compose_3panel_handles_none_panels():
    out = compose_3panel(None, None, None, target_height=120)
    assert out.shape[0] == 120
    assert out.shape[2] == 3


def test_compose_3panel_resizes_to_target_height():
    a = np.zeros((300, 400, 3), dtype=np.uint8)
    b = np.zeros((100, 200, 3), dtype=np.uint8)
    c = np.zeros((600, 800, 3), dtype=np.uint8)
    out = compose_3panel(a, b, c, target_height=120)
    assert out.shape[0] == 120
    # 3 panels concatenated horizontally
    assert out.shape[1] >= 120 * 3


def test_depth_colormap_handles_none():
    out = depth_to_colormap(None)
    assert out.shape[2] == 3
    assert out.dtype == np.uint8


def test_depth_colormap_marks_zero_invalid_black():
    depth = np.full((10, 10), 2000.0, dtype=np.float32)
    depth[0, 0] = 0  # invalid
    out = depth_to_colormap(depth)
    assert tuple(out[0, 0].tolist()) == (0, 0, 0)
    # non-invalid pixels are not all zero
    assert tuple(out[5, 5].tolist()) != (0, 0, 0)


def test_annotate_left_with_empty_inputs_does_not_crash():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = annotate_left(frame, [], {}, None, fps=0.0)
    assert out.shape == frame.shape
    # Frame must remain a copy (not the original buffer).
    assert out is not frame


def test_annotate_left_draws_roi_and_tracks():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    counter = _FakeROICounter()
    dets = [_FakeDet(bbox=(150, 130, 200, 180), centroid=(175, 155))]
    tracks = {
        1: _FakeTrack(
            track_id=1, state=CONFIRMED,
            positions=[np.array([175.0, 155.0, 0.0])],
            meta={"detection_history": [
                {"head_height_mm": 1620.0, "height_class": "adult"},
            ]},
        ),
        2: _FakeTrack(
            track_id=2, state=PENDING,
            positions=[np.array([100.0, 110.0, 0.0])],
            meta={},
        ),
        3: _FakeTrack(
            track_id=3, state=CANDIDATE,
            positions=[np.array([60.0, 60.0, 0.0])],
            meta={},
        ),
    }
    out = annotate_left(frame, dets, tracks, counter, fps=15.0)
    assert out.shape == frame.shape
    # Bottom overlay region must have been drawn (non-zero pixels at the
    # text band).
    assert out[290:298, 10:30].any()


def test_annotate_left_with_line_counter():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = annotate_left(frame, [], {}, _FakeLineCounter(), fps=0.0)
    # Line band must have non-zero pixels somewhere along y=240.
    assert out[240, :, :].any()

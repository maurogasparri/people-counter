"""Smoke tests for src/web/annotate.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.tracking.counter import Counter, Line
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


def _two_way_counter() -> Counter:
    return Counter(
        lines=[Line(
            from_xy=(50, 150), to_xy=(250, 150),
            labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
        )],
        roi={"x_min": 50, "x_max": 250, "y_min": 100, "y_max": 200},
    )


def _no_roi_counter() -> Counter:
    return Counter(
        lines=[Line(
            from_xy=(0, 240), to_xy=(400, 240),
            labels={"top_to_bottom": "ingress"},
        )],
    )


def test_compose_3panel_handles_none_panels():
    out = compose_3panel(None, None, None, target_height=120)
    # 2-row layout: top is target_height, bottom is target_height // 2.
    assert out.shape[0] == 120 + 120 // 2
    assert out.shape[2] == 3


def test_compose_3panel_two_row_layout():
    a = np.zeros((300, 400, 3), dtype=np.uint8)   # left
    b = np.zeros((100, 200, 3), dtype=np.uint8)   # right
    c = np.zeros((600, 800, 3), dtype=np.uint8)   # depth
    out = compose_3panel(a, b, c, target_height=120)
    # Top row height = target_height; bottom row spans the full top width.
    expected_h = 120 + 120 // 2
    assert out.shape[0] == expected_h
    # Top row holds two panels side by side, both at target_height after
    # resize, so total width is at least 2 * smallest aspect ratio.
    assert out.shape[1] > 120


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
    counter = _two_way_counter()
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
    # Anything was drawn (the all-zero input makes that easy to detect).
    assert out.any()
    # The bottom overlay band (last ~60 px) must contain text pixels.
    assert out[-60:, :, :].any()


def test_annotate_left_with_no_roi_counter():
    """Counter without ROI should still draw the line + arrow."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = annotate_left(frame, [], {}, _no_roi_counter(), fps=0.0)
    # The line at y=240 should produce non-zero pixels along that row.
    assert out[240, :, :].any()

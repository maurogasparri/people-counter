"""Tests for virtual line crossing counter."""
import numpy as np

from src.tracking.counter import LineCounter
from src.tracking.tracker import Track


def _make_track(track_id: int, positions: list[list[float]]) -> Track:
    return Track(
        track_id=track_id,
        positions=[np.array(p) for p in positions],
    )


def test_ingress_top_to_bottom():
    counter = LineCounter(line_y=300)
    track = _make_track(1, [[150, 290, 3000], [150, 310, 3000]])
    event = counter.check(track)
    assert event is not None
    assert event.direction == "in"
    assert counter.total_in == 1


def test_egress_bottom_to_top():
    counter = LineCounter(line_y=300)
    track = _make_track(1, [[150, 310, 3000], [150, 290, 3000]])
    event = counter.check(track)
    assert event is not None
    assert event.direction == "out"
    assert counter.total_out == 1


def test_no_crossing():
    counter = LineCounter(line_y=300)
    track = _make_track(1, [[150, 280, 3000], [150, 285, 3000]])
    event = counter.check(track)
    assert event is None


def test_same_track_counted_once():
    counter = LineCounter(line_y=300)
    track = _make_track(1, [[150, 290, 3000], [150, 310, 3000]])
    counter.check(track)
    track.positions.append(np.array([150, 320, 3000]))
    event = counter.check(track)
    assert event is None  # already counted


def test_single_position_no_event():
    counter = LineCounter(line_y=300)
    track = _make_track(1, [[150, 310, 3000]])
    event = counter.check(track)
    assert event is None


def test_check_all():
    counter = LineCounter(line_y=300)
    tracks = {
        1: _make_track(1, [[150, 290, 3000], [150, 310, 3000]]),  # in
        2: _make_track(2, [[250, 310, 3000], [250, 290, 3000]]),  # out
        3: _make_track(3, [[350, 280, 3000], [350, 285, 3000]]),  # no cross
    }
    events = counter.check_all(tracks)
    assert len(events) == 2
    assert counter.total_in == 1
    assert counter.total_out == 1


def test_reset_daily():
    counter = LineCounter(line_y=300)
    track = _make_track(1, [[150, 290, 3000], [150, 310, 3000]])
    counter.check(track)
    assert counter.total_in == 1
    counter.reset_daily()
    assert counter.total_in == 0
    # Can count same track again after reset
    event = counter.check(track)
    assert event is not None

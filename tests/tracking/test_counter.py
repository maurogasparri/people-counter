"""Tests for counting logic (legacy LineCounter + new ROICounter)."""
import numpy as np
import pytest

from src.tracking.counter import LineCounter, ROICounter, build_counter
from src.tracking.tracker import CONFIRMED, LOST, PENDING, Track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(
    track_id: int,
    positions: list[list[float]],
    state: str = CONFIRMED,
) -> Track:
    return Track(
        track_id=track_id,
        positions=[np.array(p, dtype=float) for p in positions],
        state=state,
    )


def _advance(counter: ROICounter, track: Track, position: list[float]):
    """Append a position to the track and run the counter on it. Returns the
    event emitted on this step (if any)."""
    track.positions.append(np.array(position, dtype=float))
    return counter._process_track(track)


# ---------------------------------------------------------------------------
# Legacy LineCounter tests (preserved verbatim for back-compat)
# ---------------------------------------------------------------------------


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
    event = counter.check(track)
    assert event is not None


# ---------------------------------------------------------------------------
# ROICounter — config validation
# ---------------------------------------------------------------------------

ROI = {"x_min": 100, "x_max": 500, "y_min": 200, "y_max": 400}
LINE_H = {"orientation": "horizontal", "position": 300}
LINE_V = {"orientation": "vertical", "position": 300}


def test_line_outside_roi_raises():
    with pytest.raises(ValueError, match="strictly inside"):
        ROICounter(roi=ROI, line={"orientation": "horizontal", "position": 500})
    with pytest.raises(ValueError, match="strictly inside"):
        ROICounter(roi=ROI, line={"orientation": "vertical", "position": 100})


def test_invalid_orientation_raises():
    with pytest.raises(ValueError, match="orientation"):
        ROICounter(roi=ROI, line={"orientation": "diagonal", "position": 300})


def test_invalid_roi_raises():
    with pytest.raises(ValueError):
        ROICounter(
            roi={"x_min": 500, "x_max": 100, "y_min": 200, "y_max": 400},
            line=LINE_H,
        )


# ---------------------------------------------------------------------------
# ROICounter — horizontal line
# ---------------------------------------------------------------------------


def test_roi_ingress_counts_on_exit():
    """Track enters side_a (y<300), crosses the line, exits side_b (y>400)."""
    counter = ROICounter(roi=ROI, line=LINE_H)
    track = _make_track(1, [[300, 210, 3000]])  # entered ROI, side A
    assert counter._process_track(track) is None

    ev = _advance(counter, track, [300, 280, 3000])  # still inside, side A
    assert ev is None
    ev = _advance(counter, track, [300, 320, 3000])  # crossed, side B
    assert ev is None
    ev = _advance(counter, track, [300, 420, 3000])  # exited on side B
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_roi_egress_counts_on_exit():
    counter = ROICounter(roi=ROI, line=LINE_H)
    track = _make_track(1, [[300, 380, 3000]])  # inside ROI, side B
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])  # still inside, B
    _advance(counter, track, [300, 290, 3000])  # crossed to A
    ev = _advance(counter, track, [300, 180, 3000])  # exited on side A
    assert ev is not None
    assert ev.direction == "egress"
    assert counter.total_out == 1
    assert counter.total_in == 0


def test_roi_indeciso_same_side_after_crossing():
    """Enters A, crosses to B, comes back, exits on A -> no count."""
    counter = ROICounter(roi=ROI, line=LINE_H)
    track = _make_track(1, [[300, 250, 3000]])  # inside, A
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])  # crossed to B
    _advance(counter, track, [300, 280, 3000])  # back to A
    ev = _advance(counter, track, [300, 180, 3000])  # exit on A
    assert ev is None
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_roi_indeciso_no_crossing():
    """Enters A, lingers, exits A without ever crossing -> no count."""
    counter = ROICounter(roi=ROI, line=LINE_H)
    track = _make_track(1, [[300, 220, 3000]])  # inside, A
    counter._process_track(track)
    _advance(counter, track, [300, 240, 3000])
    _advance(counter, track, [300, 230, 3000])
    ev = _advance(counter, track, [300, 180, 3000])  # exit on A
    assert ev is None
    assert counter.total_in == 0


def test_roi_lost_inside_does_not_count():
    """Track inside ROI goes LOST mid-crossing -> no count."""
    counter = ROICounter(roi=ROI, line=LINE_H)
    track = _make_track(1, [[300, 250, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])  # crossed, still inside
    # Tracker drops the track (LOST -> removed from dict). Counter sees it
    # disappear; totals remain zero.
    track.state = LOST
    tracks: dict[int, Track] = {}
    counter.check_all(tracks)
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_roi_preserves_state_across_pending():
    """Re-identified track (PENDING -> CONFIRMED) keeps entry side + crossed flag."""
    counter = ROICounter(roi=ROI, line=LINE_H)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    assert track.meta["roi_counter"]["entry_side"] == "a"
    _advance(counter, track, [300, 310, 3000])  # crossed to B
    assert track.meta["roi_counter"]["crossed"] is True

    # Simulate a PENDING frame (tracker-level miss); meta persists on track.
    track.state = PENDING
    # No position update — counter must not lose entry_side / crossed flag.
    counter._process_track(track)
    assert track.meta["roi_counter"]["entry_side"] == "a"
    assert track.meta["roi_counter"]["crossed"] is True

    # Recovery + exit on side B -> count.
    track.state = CONFIRMED
    ev = _advance(counter, track, [300, 420, 3000])
    assert ev is not None
    assert ev.direction == "ingress"


def test_roi_candidate_not_counted():
    """CANDIDATE tracks are ignored by the counter."""
    counter = ROICounter(roi=ROI, line=LINE_H)
    track = _make_track(1, [[300, 220, 3000]], state="candidate")
    assert counter._process_track(track) is None
    _advance(counter, track, [300, 310, 3000])
    ev = _advance(counter, track, [300, 420, 3000])
    assert ev is None
    assert counter.total_in == 0


# ---------------------------------------------------------------------------
# ROICounter — vertical line (mirror of horizontal)
# ---------------------------------------------------------------------------


def test_roi_vertical_line_ingress():
    counter = ROICounter(roi=ROI, line=LINE_V)
    # Enter on side A (x<300), cross to B (x>300), exit on B (x>500).
    track = _make_track(1, [[150, 300, 3000]])
    counter._process_track(track)
    _advance(counter, track, [280, 300, 3000])  # still A
    _advance(counter, track, [320, 300, 3000])  # crossed to B
    ev = _advance(counter, track, [520, 300, 3000])  # exit B
    assert ev is not None
    assert ev.direction == "ingress"


def test_roi_vertical_line_egress():
    counter = ROICounter(roi=ROI, line=LINE_V)
    track = _make_track(1, [[450, 300, 3000]])  # enter on B
    counter._process_track(track)
    _advance(counter, track, [280, 300, 3000])  # cross to A
    ev = _advance(counter, track, [90, 300, 3000])  # exit A
    assert ev is not None
    assert ev.direction == "egress"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_counter_prefers_roi():
    cfg = {
        "counter": {"roi": ROI, "line": LINE_H},
        "vision": {"counting_line_y": 0.5},
    }
    c = build_counter(cfg, frame_height=480)
    assert isinstance(c, ROICounter)


def test_build_counter_falls_back_to_legacy():
    cfg = {"vision": {"counting_line_y": 0.5}}
    c = build_counter(cfg, frame_height=480)
    assert isinstance(c, LineCounter)
    assert c.line_y == 240.0


def test_build_counter_no_config_raises():
    with pytest.raises(ValueError):
        build_counter({}, frame_height=480)


def test_roi_direction_labels_customisable():
    counter = ROICounter(
        roi=ROI,
        line=LINE_H,
        direction_labels={"side_a_to_b": "enter", "side_b_to_a": "leave"},
    )
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    ev = _advance(counter, track, [300, 420, 3000])
    assert ev.direction == "enter"
    assert counter.totals["enter"] == 1
    assert counter.totals["leave"] == 0


def test_roi_reset_daily():
    counter = ROICounter(roi=ROI, line=LINE_H)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    _advance(counter, track, [300, 420, 3000])
    assert counter.total_in == 1
    counter.reset_daily()
    assert counter.total_in == 0
    assert counter.totals == {"ingress": 0, "egress": 0}

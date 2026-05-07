"""Tests for the unified Counter (ROI + N directional lines)."""
import numpy as np
import pytest

from src.tracking.counter import Counter, Line, build_counter
from src.tracking.tracker import CONFIRMED, LOST, PENDING, Track


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


ROI = {"x_min": 100, "x_max": 500, "y_min": 200, "y_max": 400}


def _line_h(
    line_y: float = 300,
    *,
    x1: float = 100,
    x2: float = 500,
    ingress: str | None = "ingress",
    egress: str | None = "egress",
) -> Line:
    """Horizontal line at ``y=line_y`` with default ingress/egress labels.

    Pass ``ingress=None`` to make the line a one-way egress gate (or vice
    versa) — matches the "two physically separate doors" use case.
    """
    labels: dict[str, str] = {}
    if ingress is not None:
        labels["top_to_bottom"] = ingress
    if egress is not None:
        labels["bottom_to_top"] = egress
    return Line(from_xy=(x1, line_y), to_xy=(x2, line_y), labels=labels)


def _line_v(
    line_x: float = 300,
    *,
    y1: float = 200,
    y2: float = 400,
    right: str | None = "ingress",
    left: str | None = "egress",
) -> Line:
    labels: dict[str, str] = {}
    if right is not None:
        labels["left_to_right"] = right
    if left is not None:
        labels["right_to_left"] = left
    return Line(from_xy=(line_x, y1), to_xy=(line_x, y2), labels=labels)


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


def _advance(counter: Counter, track: Track, position: list[float]):
    """Append a position to the track and run the counter on it. Returns
    the event emitted on this step (if any)."""
    track.positions.append(np.array(position, dtype=float))
    return counter._process_track(track)


# ---------------------------------------------------------------------------
# Line — validation
# ---------------------------------------------------------------------------


def test_line_oblique_raises():
    with pytest.raises(ValueError, match="axis-aligned"):
        Line(from_xy=(0, 0), to_xy=(10, 10),
             labels={"top_to_bottom": "ingress"})


def test_line_zero_length_raises():
    with pytest.raises(ValueError, match="axis-aligned"):
        Line(from_xy=(5, 5), to_xy=(5, 5),
             labels={"top_to_bottom": "ingress"})


def test_line_invalid_direction_for_horizontal_raises():
    with pytest.raises(ValueError, match="invalid for horizontal"):
        Line(from_xy=(0, 50), to_xy=(100, 50),
             labels={"left_to_right": "ingress"})


def test_line_invalid_direction_for_vertical_raises():
    with pytest.raises(ValueError, match="invalid for vertical"):
        Line(from_xy=(50, 0), to_xy=(50, 100),
             labels={"top_to_bottom": "ingress"})


def test_line_no_labels_raises():
    with pytest.raises(ValueError, match="no direction labels"):
        Line(from_xy=(0, 50), to_xy=(100, 50), labels={})


def test_line_side_of_horizontal():
    line = _line_h(line_y=100)
    assert line.side_of(50, 99) == -1   # above
    assert line.side_of(50, 101) == 1   # below
    assert line.side_of(50, 100) == 0   # on line


def test_line_within_segment():
    line = _line_h(line_y=100, x1=10, x2=90)
    assert line.within_segment(50, 100) is True
    assert line.within_segment(5, 100) is False
    assert line.within_segment(95, 100) is False


# ---------------------------------------------------------------------------
# Counter — config validation
# ---------------------------------------------------------------------------


def test_counter_no_lines_raises():
    with pytest.raises(ValueError, match="at least one line"):
        Counter(lines=[])


def test_counter_invalid_roi_raises():
    with pytest.raises(ValueError):
        Counter(
            lines=[_line_h()],
            roi={"x_min": 500, "x_max": 100, "y_min": 200, "y_max": 400},
        )


# ---------------------------------------------------------------------------
# Single horizontal line — ingress / egress / indeciso / oscillation
# ---------------------------------------------------------------------------


def test_ingress_counts_on_exit():
    """Track enters above the line, crosses, exits below — counts ingress."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 210, 3000]])  # entered ROI above line
    assert counter._process_track(track) is None

    ev = _advance(counter, track, [300, 280, 3000])  # still inside, above
    assert ev is None
    ev = _advance(counter, track, [300, 320, 3000])  # crossed, below
    assert ev is None
    ev = _advance(counter, track, [300, 420, 3000])  # exited below
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_egress_counts_on_exit():
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 380, 3000]])  # inside, below
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])  # still inside, below
    _advance(counter, track, [300, 290, 3000])  # crossed to above
    ev = _advance(counter, track, [300, 180, 3000])  # exit above
    assert ev is not None
    assert ev.direction == "egress"
    assert counter.total_out == 1
    assert counter.total_in == 0


def test_same_track_can_count_two_full_cycles():
    """A track that completes two full cycles must produce two events.

    Antiglitch: the counter resets meta after each emitted event so a
    legitimate "person walks in and immediately walks out without leaving
    the camera frame long enough for the track to die" case still
    registers both events.
    """
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 210, 3000]])

    # Cycle 1: above -> below (ingress)
    counter._process_track(track)
    _advance(counter, track, [300, 280, 3000])
    _advance(counter, track, [300, 320, 3000])
    ev1 = _advance(counter, track, [300, 420, 3000])
    assert ev1 is not None and ev1.direction == "ingress"
    assert counter.total_in == 1
    assert counter.total_out == 0

    # Cycle 2: same track re-enters from below and crosses to above
    _advance(counter, track, [300, 380, 3000])
    _advance(counter, track, [300, 310, 3000])
    _advance(counter, track, [300, 290, 3000])
    ev2 = _advance(counter, track, [300, 180, 3000])
    assert ev2 is not None and ev2.direction == "egress"
    assert counter.total_in == 1
    assert counter.total_out == 1


def test_oscillation_without_full_cycle_does_not_count_twice():
    """A track that exits the ROI and re-enters from the same side without
    reaching the line again must not produce a second event."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 210, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 320, 3000])
    ev1 = _advance(counter, track, [300, 420, 3000])
    assert ev1 is not None
    assert counter.total_in == 1

    # Re-enter below, never reach the line, exit below again.
    _advance(counter, track, [300, 380, 3000])
    _advance(counter, track, [300, 360, 3000])
    ev2 = _advance(counter, track, [300, 420, 3000])
    assert ev2 is None
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_indeciso_with_two_way_line_counts_last_crossing():
    """A line labelled in BOTH directions counts the most recent crossing.

    Track enters above, crosses below (ingress), comes back above (egress),
    exits above. With the old ROICounter this was an "indeciso" non-event;
    with the unified counter the last crossing wins, so the track is counted
    as egress (its net movement is "back to where it came from" but the
    final transit was bottom->top, which is the egress label).
    """
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 250, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])  # crossed below: ingress
    _advance(counter, track, [300, 280, 3000])  # back above: egress
    ev = _advance(counter, track, [300, 180, 3000])  # exit above
    assert ev is not None
    assert ev.direction == "egress"


def test_indeciso_with_one_way_line_does_not_count():
    """If the line only labels ONE direction (one-way gate), an indeciso
    track that wanders back through the unlabelled side leaves the counter
    in its last labelled state. To cleanly model "person walks in, walks
    back out without committing" use one-way lines: the return crossing
    has no label, so it does not overwrite the previous one — but in this
    specific case the track never crossed into the labelled direction at
    all, so nothing is counted."""
    one_way_in = _line_h(egress=None)  # only top_to_bottom is labelled
    counter = Counter(lines=[one_way_in], roi=ROI)
    track = _make_track(1, [[300, 250, 3000]])  # above (no label yet)
    counter._process_track(track)
    _advance(counter, track, [300, 280, 3000])  # still above
    _advance(counter, track, [300, 240, 3000])  # still above
    ev = _advance(counter, track, [300, 180, 3000])  # exit above
    assert ev is None
    assert counter.total_in == 0


def test_indeciso_no_crossing():
    """Enters above, lingers, exits above without ever crossing — no count."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 240, 3000])
    _advance(counter, track, [300, 230, 3000])
    ev = _advance(counter, track, [300, 180, 3000])
    assert ev is None
    assert counter.total_in == 0


def test_lost_inside_does_not_count():
    """Track inside the ROI goes LOST mid-crossing — no count is emitted."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 250, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    track.state = LOST
    counter.check_all({})  # tracker dropped it
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_pending_state_counted():
    """PENDING tracks are eligible for counting (they keep their meta)."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    track.state = PENDING
    ev = _advance(counter, track, [300, 420, 3000])
    assert ev is not None
    assert ev.direction == "ingress"


def test_candidate_not_counted():
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 220, 3000]], state="candidate")
    assert counter._process_track(track) is None
    _advance(counter, track, [300, 310, 3000])
    ev = _advance(counter, track, [300, 420, 3000])
    assert ev is None
    assert counter.total_in == 0


# ---------------------------------------------------------------------------
# Vertical line — mirror of horizontal
# ---------------------------------------------------------------------------


def test_vertical_line_ingress():
    counter = Counter(lines=[_line_v()], roi=ROI)
    track = _make_track(1, [[150, 300, 3000]])  # enter left
    counter._process_track(track)
    _advance(counter, track, [280, 300, 3000])  # still left
    _advance(counter, track, [320, 300, 3000])  # crossed right
    ev = _advance(counter, track, [520, 300, 3000])  # exit right
    assert ev is not None
    assert ev.direction == "ingress"


def test_vertical_line_egress():
    counter = Counter(lines=[_line_v()], roi=ROI)
    track = _make_track(1, [[450, 300, 3000]])  # enter right
    counter._process_track(track)
    _advance(counter, track, [280, 300, 3000])  # cross to left
    ev = _advance(counter, track, [90, 300, 3000])  # exit left
    assert ev is not None
    assert ev.direction == "egress"


# ---------------------------------------------------------------------------
# Multi-line — physically separate doors (the use case that drove the
# unified design: one line is a one-way IN gate, another is a one-way OUT
# gate, in different parts of the frame)
# ---------------------------------------------------------------------------


def test_two_one_way_lines_separate_doors():
    """Two horizontal lines in different x ranges of the ROI: the left one
    only counts ingress (top->bottom), the right one only counts egress
    (bottom->top). A track that crosses the wrong direction at the wrong
    line is silently ignored."""
    line_in = Line(
        from_xy=(120, 300), to_xy=(280, 300),
        labels={"top_to_bottom": "ingress"},
    )
    line_out = Line(
        from_xy=(320, 300), to_xy=(480, 300),
        labels={"bottom_to_top": "egress"},
    )
    counter = Counter(lines=[line_in, line_out], roi=ROI)

    # Track A walks down through the ingress door
    t_in = _make_track(1, [[200, 220, 3000]])
    counter._process_track(t_in)
    _advance(counter, t_in, [200, 320, 3000])
    ev_in = _advance(counter, t_in, [200, 420, 3000])
    assert ev_in is not None
    assert ev_in.direction == "ingress"

    # Track B walks up through the egress door
    t_out = _make_track(2, [[400, 380, 3000]])
    counter._process_track(t_out)
    _advance(counter, t_out, [400, 290, 3000])
    ev_out = _advance(counter, t_out, [400, 180, 3000])
    assert ev_out is not None
    assert ev_out.direction == "egress"

    # Track C tries to walk DOWN through the egress (one-way) door — ignored
    t_wrong = _make_track(3, [[400, 220, 3000]])
    counter._process_track(t_wrong)
    _advance(counter, t_wrong, [400, 320, 3000])
    ev_wrong = _advance(counter, t_wrong, [400, 420, 3000])
    assert ev_wrong is None

    assert counter.total_in == 1
    assert counter.total_out == 1


def test_crossing_outside_segment_is_ignored():
    """A line restricted to x in [120, 280] must NOT count a crossing at
    x=400, even if the centroid trajectory crosses the y=300 plane."""
    line = Line(
        from_xy=(120, 300), to_xy=(280, 300),
        labels={"top_to_bottom": "ingress"},
    )
    counter = Counter(lines=[line], roi=ROI)
    track = _make_track(1, [[400, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [400, 320, 3000])  # crosses y=300 at x=400
    ev = _advance(counter, track, [400, 420, 3000])
    assert ev is None
    assert counter.total_in == 0


# ---------------------------------------------------------------------------
# No-ROI mode (gate purely by line crossings — useful when the camera FOV
# itself is the area of interest and no rectangular gate is needed)
# ---------------------------------------------------------------------------


def test_no_roi_uses_full_frame():
    """Without ROI, the counter treats the whole frame as 'inside' and
    triggers counting transitions only when the track disappears (LOST).
    For tests we approximate the LOST signal by removing the track from
    check_all — but for the standard exit-from-ROI path, the no-ROI
    config has nowhere to exit to, so single-line crossings can't count
    until the track dies. This is by design: in real deploys you usually
    DO want a ROI."""
    counter = Counter(lines=[_line_h()])  # no ROI
    track = _make_track(1, [[200, 200, 3000]])
    # The track is "always inside" (no ROI). A single crossing won't fire
    # an event — only the exit transition does, and there's no exit.
    counter._process_track(track)
    _advance(counter, track, [200, 350, 3000])
    assert counter.total_in == 0  # no exit yet


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_counter_full_schema():
    cfg = {
        "counter": {
            "roi": ROI,
            "lines": [
                {
                    "from": [100, 300],
                    "to":   [500, 300],
                    "labels": {
                        "top_to_bottom": "ingress",
                        "bottom_to_top": "egress",
                    },
                },
            ],
        },
    }
    c = build_counter(cfg, frame_height=480)
    assert isinstance(c, Counter)
    assert c.roi == (100.0, 500.0, 200.0, 400.0)
    assert len(c.lines) == 1
    assert c.lines[0].labels == {
        "top_to_bottom": "ingress",
        "bottom_to_top": "egress",
    }


def test_build_counter_two_lines():
    cfg = {
        "counter": {
            "roi": ROI,
            "lines": [
                {
                    "from": [120, 300],
                    "to":   [280, 300],
                    "labels": {"top_to_bottom": "ingress"},
                },
                {
                    "from": [320, 300],
                    "to":   [480, 300],
                    "labels": {"bottom_to_top": "egress"},
                },
            ],
        },
    }
    c = build_counter(cfg)
    assert len(c.lines) == 2


def test_build_counter_no_lines_raises():
    with pytest.raises(ValueError, match="counter.lines"):
        build_counter({"counter": {"roi": ROI}})


def test_build_counter_empty_lines_raises():
    with pytest.raises(ValueError, match="counter.lines"):
        build_counter({"counter": {"roi": ROI, "lines": []}})


def test_build_counter_no_config_raises():
    with pytest.raises(ValueError, match="counter.lines"):
        build_counter({})


def test_build_counter_optional_roi():
    cfg = {
        "counter": {
            "lines": [
                {
                    "from": [0, 100],
                    "to":   [200, 100],
                    "labels": {"top_to_bottom": "ingress"},
                },
            ],
        },
    }
    c = build_counter(cfg)
    assert c.roi is None


# ---------------------------------------------------------------------------
# reset_daily / custom labels
# ---------------------------------------------------------------------------


def test_custom_labels():
    line = Line(
        from_xy=(100, 300), to_xy=(500, 300),
        labels={"top_to_bottom": "enter", "bottom_to_top": "leave"},
    )
    counter = Counter(lines=[line], roi=ROI)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    ev = _advance(counter, track, [300, 420, 3000])
    assert ev.direction == "enter"
    assert counter.totals["enter"] == 1
    assert counter.totals["leave"] == 0


def test_reset_daily():
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    _advance(counter, track, [300, 420, 3000])
    assert counter.total_in == 1
    counter.reset_daily()
    assert counter.total_in == 0
    assert counter.totals == {"ingress": 0, "egress": 0}


def test_check_all_processes_multiple_tracks():
    counter = Counter(lines=[_line_h()], roi=ROI)
    # Track 1: full ingress cycle
    t1 = _make_track(1, [[200, 220, 3000]])
    counter._process_track(t1)
    t1.positions.append(np.array([200, 320, 3000]))
    # Track 2: full egress cycle
    t2 = _make_track(2, [[300, 380, 3000]])
    counter._process_track(t2)
    t2.positions.append(np.array([300, 290, 3000]))
    # Both exit on the next check_all
    t1.positions.append(np.array([200, 420, 3000]))
    t2.positions.append(np.array([300, 180, 3000]))
    events = counter.check_all({1: t1, 2: t2})
    directions = {e.direction for e in events}
    assert directions == {"ingress", "egress"}
    assert counter.total_in == 1
    assert counter.total_out == 1

"""Tests para el Counter unificado (ROI + N líneas direccionales)."""

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
        Line(from_xy=(0, 0), to_xy=(10, 10), labels={"top_to_bottom": "ingress"})


def test_line_zero_length_raises():
    with pytest.raises(ValueError, match="axis-aligned"):
        Line(from_xy=(5, 5), to_xy=(5, 5), labels={"top_to_bottom": "ingress"})


def test_line_invalid_direction_for_horizontal_raises():
    with pytest.raises(ValueError, match="invalid for horizontal"):
        Line(from_xy=(0, 50), to_xy=(100, 50), labels={"left_to_right": "ingress"})


def test_line_invalid_direction_for_vertical_raises():
    with pytest.raises(ValueError, match="invalid for vertical"):
        Line(from_xy=(50, 0), to_xy=(50, 100), labels={"top_to_bottom": "ingress"})


def test_line_no_labels_raises():
    with pytest.raises(ValueError, match="no direction labels"):
        Line(from_xy=(0, 50), to_xy=(100, 50), labels={})


def test_line_side_of_horizontal():
    line = _line_h(line_y=100)
    assert line.side_of(50, 99) == -1  # above
    assert line.side_of(50, 101) == 1  # below
    assert line.side_of(50, 100) == 0  # on line


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
        from_xy=(120, 300),
        to_xy=(280, 300),
        labels={"top_to_bottom": "ingress"},
    )
    line_out = Line(
        from_xy=(320, 300),
        to_xy=(480, 300),
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
        from_xy=(120, 300),
        to_xy=(280, 300),
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
                    "to": [500, 300],
                    "labels": {
                        "top_to_bottom": "ingress",
                        "bottom_to_top": "egress",
                    },
                },
            ],
        },
    }
    c = build_counter(cfg)
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
                    "to": [280, 300],
                    "labels": {"top_to_bottom": "ingress"},
                },
                {
                    "from": [320, 300],
                    "to": [480, 300],
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
                    "to": [200, 100],
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
        from_xy=(100, 300),
        to_xy=(500, 300),
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


# ---------------------------------------------------------------------------
# Parallax-corrected footpoint tracking
# ---------------------------------------------------------------------------
#
# Estos tests pinean el comportamiento de diseño: en fisheye cenital el bbox
# envuelve cabeza + hombros + torso (sin pies), así que usar el centroide
# como punto de tracking dispara cruces de línea decenas de centímetros antes
# de que los pies efectivamente crucen. Con mount + principal point el counter
# escala el head pixel hacia el principal point por Z_head/H y recupera el
# pixel del *pie*.
#
# Los tests están armados para que el shift de parallax sea lo suficientemente
# grande como para que el path de centroide cuente un cruce y el de footpoint
# NO, o viceversa — es la única forma de verificar que la convención se está
# aplicando efectivamente (counts iguales bajo ambas convenciones no probaría
# nada).


def _meta_for_track(
    bbox: tuple[float, float, float, float],
    head_height_mm: float | None,
) -> dict:
    """Detection-history record with the keys the counter reads."""
    return {
        "bbox": bbox,
        "head_height_mm": head_height_mm,
        "near_depth_mm": 1500.0,  # arbitrary — counter doesn't read this
        "confidence": 0.8,
        "height_class": "adult",
    }


def _attach_history(track: Track, records: list[dict]) -> None:
    track.meta.setdefault("detection_history", []).extend(records)


# Mount + principal point used across the tests below — 3 m mount, 1152
# px wide rectified frame so the principal point sits at (576, 324). At
# this geometry the parallax scale factor for a 1.7 m head is
# (3000-1700)/3000 = 0.4333.
MOUNT_MM = 3000.0
HEAD_MM = 1700.0
CX, CY = 576.0, 324.0


def test_periphery_crossing_centroid_path_overcounts_early():
    """No-calibration baseline: with the legacy centroid path, the
    crossing fires when the bbox center clears the line — even though
    in zenith fisheye that's the shoulder/torso projection, not the
    feet. Pinning this on purpose so the contrast vs the projected
    test below is explicit: same trajectory, same exit, but the
    *moment of crossing* differs.

    Track on the right of the principal point. The line lives at the
    centroid plane (x=600, vertical line). The track centroid moves
    580 → 620 (centroid crosses) → exits the ROI on the right.
    """
    line = Line(
        from_xy=(600, 200),
        to_xy=(600, 500),
        labels={"left_to_right": "ingress", "right_to_left": "egress"},
    )
    counter = Counter(
        lines=[line],
        roi={"x_min": 100, "x_max": 1000, "y_min": 200, "y_max": 500},
        # No mount / principal_point → legacy centroid path.
    )
    # Enter the ROI to the LEFT of the line (centroid x=580).
    track = _make_track(1, [[580, 350, 3000]])
    counter._process_track(track)
    # Step centroid past the line (x=620) — counted as ingress.
    track.positions.append(np.array([620, 350, 3000], dtype=float))
    counter._process_track(track)
    # Exit the ROI on the right (x>1000). On the exit transition the
    # event is emitted.
    track.positions.append(np.array([1100, 350, 3000], dtype=float))
    counter._process_track(track)
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_periphery_crossing_footpoint_path_does_not_overcount():
    """Same geometry, but with calibration → projected footpoint. With
    1.7 m head height under 3 m mount, the parallax scale is 0.4333:
    a centroid at x=620 (44 px right of nadir x=576) projects the foot
    to x ≈ 576 + 44*0.4333 = 595 — still LEFT of the line at x=600.
    The centroid-path crossing is therefore an artefact of the
    geometry; the foot actually hasn't crossed.

    With foot tracking, no ingress fires while the centroid is at 620.
    Pinning the asymmetry: same trajectory, different verdict.
    Eventually the centroid (and foot) clear the line on the way to
    the ROI exit, so exactly one ingress is counted.
    """
    line = Line(
        from_xy=(600, 200),
        to_xy=(600, 500),
        labels={"left_to_right": "ingress", "right_to_left": "egress"},
    )
    counter = Counter(
        lines=[line],
        roi={"x_min": 100, "x_max": 1140, "y_min": 200, "y_max": 500},
        mounting_height_mm=MOUNT_MM,
        principal_point=(CX, CY),
    )

    # Enter ROI. Foot pixel for centroid x=580: 576 + 4*0.4333 = 577.7
    # — left of 600.
    track = _make_track(1, [[580, 350, 3000]])
    bbox = (560.0, 280.0, 600.0, 420.0)
    _attach_history(track, [_meta_for_track(bbox, HEAD_MM)])
    counter._process_track(track)

    # Centroid at 620: foot at ~595 (still left of the line).
    track.positions.append(np.array([620, 350, 3000], dtype=float))
    bbox = (600.0, 280.0, 640.0, 420.0)
    _attach_history(track, [_meta_for_track(bbox, HEAD_MM)])
    counter._process_track(track)

    # Step further: centroid at 700, foot at 576 + 124*0.4333 = 630
    # — NOW the foot has crossed.
    track.positions.append(np.array([700, 350, 3000], dtype=float))
    bbox = (680.0, 280.0, 720.0, 420.0)
    _attach_history(track, [_meta_for_track(bbox, HEAD_MM)])
    counter._process_track(track)

    # Exit ROI to the right. Need foot_x outside x_max=1140.
    # foot_x = 576 + (head_x - 576)*0.4333 > 1140 → head_x > 1879.
    # Use head_x = 2200 → foot_x ≈ 1279.
    track.positions.append(np.array([2200, 350, 3000], dtype=float))
    bbox = (2180.0, 280.0, 2220.0, 420.0)
    _attach_history(track, [_meta_for_track(bbox, HEAD_MM)])
    counter._process_track(track)
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_nadir_centroid_and_footpoint_agree():
    """Track moving exactly through the principal point: the parallax
    scale is irrelevant (offset from principal point is zero), so
    centroid and footpoint produce identical line-crossing decisions.
    Pins the nadir-invariant: enabling projection MUST NOT change
    counts for tracks that pass through the optical axis.
    """
    # Horizontal line at y=CY (so the cross happens exactly under
    # nadir). Track moves top → bottom through (CX, CY).
    line = Line(
        from_xy=(50, CY),
        to_xy=(1100, CY),
        labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
    )
    counter_proj = Counter(
        lines=[line],
        roi={"x_min": 50, "x_max": 1100, "y_min": 100, "y_max": 600},
        mounting_height_mm=MOUNT_MM,
        principal_point=(CX, CY),
    )
    counter_centroid = Counter(
        lines=[line],
        roi={"x_min": 50, "x_max": 1100, "y_min": 100, "y_max": 600},
    )

    def _run(counter: Counter) -> None:
        # Inside ROI, above line. Bbox top above the line.
        track = _make_track(1, [[CX, 250, 3000]])
        bbox_above = (CX - 30, 200, CX + 30, 280)
        _attach_history(track, [_meta_for_track(bbox_above, HEAD_MM)])
        counter._process_track(track)

        # Step across the line. Centroid at CY+30 → bbox_top at
        # CY-30=294. For projection: foot_v = 324 + (294-324)*0.4333
        # = 311 (still above line at 324) — but the centroid is past.
        # To make BOTH conventions cross the line, drop the bbox top
        # well below CY so the foot projection clears it too.
        track.positions.append(np.array([CX, CY + 80, 3000], dtype=float))
        bbox_cross = (CX - 30, CY + 30, CX + 30, CY + 130)
        _attach_history(track, [_meta_for_track(bbox_cross, HEAD_MM)])
        counter._process_track(track)

        # Exit ROI at the bottom (y_max=600). Push the bbox top deep
        # so projected y also escapes. bbox_top=900 → foot_v ≈ 574 +
        # … well past 600 once we add the centroid offset. Use a far
        # bottom value to guarantee exit under both conventions.
        track.positions.append(np.array([CX, 1200, 3000], dtype=float))
        bbox_below = (CX - 30, 1100, CX + 30, 1300)
        _attach_history(track, [_meta_for_track(bbox_below, HEAD_MM)])
        counter._process_track(track)

    _run(counter_proj)
    _run(counter_centroid)
    assert counter_proj.total_in == counter_centroid.total_in == 1
    assert counter_proj.total_out == counter_centroid.total_out == 0


def test_no_head_height_falls_back_to_centroid():
    """Track has no head_height_mm (depth disabled, classifier off).
    Counter must fall back to centroid behaviour — the projection
    feature is opt-in via metadata presence, not failure-loud.
    """
    line = Line(
        from_xy=(100, 300),
        to_xy=(900, 300),
        labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
    )
    counter = Counter(
        lines=[line],
        roi={"x_min": 50, "x_max": 900, "y_min": 200, "y_max": 500},
        mounting_height_mm=MOUNT_MM,
        principal_point=(CX, CY),
    )
    track = _make_track(1, [[300, 220, 3000]])  # well off-axis
    # No detection_history attached → no head_height_mm available.
    counter._process_track(track)
    track.positions.append(np.array([300, 320, 3000], dtype=float))
    counter._process_track(track)
    # Exit the ROI at the bottom (y > 500).
    track.positions.append(np.array([300, 550, 3000], dtype=float))
    ev = counter._process_track(track)
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1


def test_only_one_event_per_full_cycle_with_projection():
    """Sanity: enabling projection must not double-count. Same person,
    same trajectory, single line, projection ON → exactly one event,
    fired only on ROI exit.

    The line sits at the foot-Y for a person whose centroid is below
    it: that's the regime where only the projection-corrected foot can
    eventually clear the line. The track exits the ROI at the bottom,
    triggering the cycle.
    """
    line = Line(
        from_xy=(50, 300),
        to_xy=(1100, 300),
        labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
    )
    counter = Counter(
        lines=[line],
        roi={"x_min": 50, "x_max": 1100, "y_min": 200, "y_max": 500},
        mounting_height_mm=MOUNT_MM,
        principal_point=(CX, CY),
    )
    # Position #0 sits inside the ROI above the line. Bbox top is far
    # above the line (foot projection lands above too).
    track = _make_track(1, [[800, 220, 3000]])
    bbox = (770.0, 180.0, 830.0, 270.0)
    _attach_history(track, [_meta_for_track(bbox, HEAD_MM)])
    counter._process_track(track)

    events = []
    # Walk the centroid down through the line and out of the ROI. The
    # last step puts the bbox top so far down (y=900) that even after
    # the parallax shrink (×0.4333) the foot pixel is well below
    # y_max=500, triggering the ROI-exit transition + crossing emit.
    steps = [
        # (cx, cy_centroid, bbox_top_y)
        (800, 260, 200),  # foot_v ~270 — above line
        (800, 360, 300),  # foot_v ~314 — past line
        (800, 460, 400),  # foot_v ~357 — well past
        (800, 700, 900),  # foot_v ~574 — outside ROI bottom
    ]
    for cx_, cy_, bbox_top in steps:
        track.positions.append(np.array([cx_, cy_, 3000], dtype=float))
        b = (770.0, float(bbox_top), 830.0, float(bbox_top + 100))
        _attach_history(track, [_meta_for_track(b, HEAD_MM)])
        ev = counter._process_track(track)
        if ev is not None:
            events.append(ev)
    assert len(events) == 1
    assert events[0].direction == "ingress"
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_convention_flip_does_not_emit_phantom_crossing():
    """Track starts inside the ROI without head_height_mm (centroid
    convention) and gains a valid head height mid-visit (projected
    convention). The counter must re-snapshot sides on the flipping
    frame and NOT fire a fake crossing just because the projection
    moves the tracking point.
    """
    line = Line(
        from_xy=(50, 300),
        to_xy=(1100, 300),
        labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
    )
    counter = Counter(
        lines=[line],
        roi={"x_min": 50, "x_max": 1100, "y_min": 200, "y_max": 500},
        mounting_height_mm=MOUNT_MM,
        principal_point=(CX, CY),
    )
    # Start near the right edge, centroid path puts the tracking point
    # below the line. Switch to projection mid-visit; the foot pixel
    # ends up ABOVE the line (large parallax shift toward principal
    # point) — under naive comparison that'd look like a crossing.
    track = _make_track(1, [[1000, 320, 3000]])
    counter._process_track(track)  # centroid: side=+1, no projection

    # Linger inside, still no head height.
    track.positions.append(np.array([1000, 330, 3000], dtype=float))
    counter._process_track(track)

    # Now head height is suddenly available. Foot projects to
    # 576 + (1000-576)*0.4333 ≈ 760, but the *vertical* coordinate is
    # what matters for the y=300 line: bbox top at y=270 projects to
    # 324 + (270-324)*0.4333 ≈ 300.6 — almost exactly on the line.
    # The convention flip happens; no event must fire on this frame.
    bbox = (970.0, 270.0, 1030.0, 380.0)
    _attach_history(track, [_meta_for_track(bbox, HEAD_MM)])
    track.positions.append(np.array([1000, 320, 3000], dtype=float))
    ev = counter._process_track(track)
    assert ev is None
    assert counter.total_in == 0
    assert counter.total_out == 0


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


# ---------------------------------------------------------------------------
# Debounce — min_crossing_movement_px
# ---------------------------------------------------------------------------


def test_debounce_filters_jitter_around_line():
    """Track parado al lado de la línea con micro-jitter no debe contar.

    Sin debounce: cada flip de lado por jitter actualiza last_label, y al
    salir del ROI el contador dispara un evento espurio. Con
    min_crossing_movement_px=3.0, los movimientos sub-3px (medidos contra
    la última posición no-debounced) se ignoran y no fabrican cruces.
    """
    counter = Counter(
        lines=[_line_h()],
        roi=ROI,
        min_crossing_movement_px=3.0,
    )
    # Track entra arriba de la línea (y=210, line en y=300).
    track = _make_track(1, [[300, 210, 3000]])
    counter._process_track(track)
    # Camina hacia la línea hasta y=298 (todavía arriba, sin cruce).
    # last_track_pos queda en (300, 298).
    _advance(counter, track, [300, 250, 3000])
    _advance(counter, track, [300, 298, 3000])
    # Jitter alrededor de la línea: todos los valores caen dentro de 3px
    # de y=298 (la referencia cacheada), así cada frame es debounced y
    # last_track_pos no se actualiza. Sin debounce, esos flips de lado
    # quedarían registrados como cruces.
    for y in (299.5, 300.5, 299.0, 300.7, 297.5, 300.2, 298.8):
        _advance(counter, track, [300, y, 3000])
    # Sale del ROI por arriba (regreso). Si el debounce funcionó, no
    # tiene last_label seteado y no dispara evento.
    ev = _advance(counter, track, [300, 180, 3000])
    assert ev is None
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_debounce_does_not_block_real_crossing():
    """Movimiento normal (>threshold) sigue contando con debounce activo."""
    counter = Counter(
        lines=[_line_h()],
        roi=ROI,
        min_crossing_movement_px=3.0,
    )
    track = _make_track(1, [[300, 210, 3000]])
    counter._process_track(track)
    # Movimiento normal: 10 px/frame (típico a 1m/s, 30fps).
    _advance(counter, track, [300, 220, 3000])
    _advance(counter, track, [300, 280, 3000])  # cruce dentro de un solo frame
    _advance(counter, track, [300, 320, 3000])  # del lado nuevo
    ev = _advance(counter, track, [300, 420, 3000])  # exit abajo
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1


def test_debounce_default_disabled():
    """min_crossing_movement_px=0 (default) preserva comportamiento legacy."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    # Mismo escenario que test_debounce_filters_jitter_around_line, pero
    # sin threshold — el último flip dentro del ROI gana, así que el exit
    # dispara según el último cruce registrado.
    track = _make_track(1, [[300, 210, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 298, 3000])
    # Cruce real abajo, después oscila por arriba — el último cruce gana.
    _advance(counter, track, [300, 305, 3000])  # cruzó, ahora abajo (last_label=ingress)
    _advance(counter, track, [300, 295, 3000])  # cruzó al revés (last_label=egress)
    ev = _advance(counter, track, [300, 180, 3000])  # exit arriba
    # Sin debounce el último cruce dentro del ROI es egress, y al salir
    # arriba ese label se emite — el legacy permite que jitter cuente.
    assert ev is not None and ev.direction == "egress"


def test_entry_side_uses_last_outside_pos_under_detection_gap():
    """Con ROI chico y detector miss en la zona pre-línea, la primera
    detección inside puede caer ya del otro lado. El counter debe
    snapshotear sides[] desde la última posición outside-ROI conocida
    (lado de approach, inequívoco), no desde la primera detección inside.

    Reproduce el bug operativo observado en piloto: ROI 264-384 vertical
    + línea en y=324 + detector que pierde el frame inside-pre-línea.
    Sin fix, sides[0] se cachea ya como ``below`` y el cruce nunca se
    detecta — el track sale del ROI sin emitir.
    """
    # ROI vertical chico, simulando la setup de piloto antes del enlarge.
    small_roi = {"x_min": 100, "x_max": 500, "y_min": 264, "y_max": 384}
    counter = Counter(lines=[_line_h(line_y=324)], roi=small_roi)

    # Frame 1: track outside-ROI arriba de la línea (approach).
    track = _make_track(1, [[300, 240, 3000]])
    assert counter._process_track(track) is None

    # Frame 2: detección MISS en y=290 (would be inside, above line) —
    # simula `det=N`. No se llama _process_track; el siguiente frame
    # vendrá con la posición real cuando el detector firee otra vez.

    # Frame 3: primera detección inside ROI, ya pasada la línea (y=350).
    # Sin el fix, sides[0] se cachearía aquí como "below" y todo lo
    # que sigue mantiene "below" → no se observa cruce.
    _advance(counter, track, [300, 350, 3000])

    # Frame 4: track sale del ROI por abajo.
    ev = _advance(counter, track, [300, 400, 3000])

    # Con el fix: sides[0] se snapshoteó desde la última outside-pos
    # (y=240, above line). El frame inside @ y=350 detectó el flip
    # above→below durante el inside-loop, capturó label=ingress. El
    # exit @ y=400 emite el evento.
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_entry_side_fallback_when_track_born_inside_roi():
    """Si el track aparece directamente dentro del ROI (sin posición
    outside previa registrada), el counter cae al comportamiento legacy:
    snapshotear sides[] desde la posición actual. Sin regresión para
    tracks que nacen inside.
    """
    counter = Counter(lines=[_line_h()], roi=ROI)
    # Track nace inside ROI, arriba de la línea (y=210 < line y=300).
    track = _make_track(1, [[300, 210, 3000]])
    assert counter._process_track(track) is None
    # Camina cruzando la línea.
    _advance(counter, track, [300, 320, 3000])
    ev = _advance(counter, track, [300, 420, 3000])  # exit abajo
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1

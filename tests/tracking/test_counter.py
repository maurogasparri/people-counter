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


def test_same_track_two_full_cycles_cancel_via_uturn(monkeypatch):
    """A track that completes two full cycles back-to-back se cancela
    mutuamente vía U-turn (el ROI ES la zona de cancelación).

    Antiglitch: el counter resetea meta después de cada evento emitido
    así el mismo track puede producir cycles separados; pero si los
    cycles ocurren dentro de la ventana U-turn (default 5s), se cancelan
    — modela "persona entró, dudó, volvió a salir" como neutral, no
    como 1 IN + 1 OUT espurios. Para validar el mecanismo de reset de
    meta independiente de U-turn, se patchea la ventana a 0."""
    monkeypatch.setattr(Counter, "_try_cancel_uturn", lambda *a, **kw: False)
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

    # Cycle 2: same track re-enters from below and crosses to above.
    # Con U-turn deshabilitado, el reset de meta permite contar el
    # segundo cycle como egress independiente.
    _advance(counter, track, [300, 380, 3000])
    _advance(counter, track, [300, 310, 3000])
    _advance(counter, track, [300, 290, 3000])
    ev2 = _advance(counter, track, [300, 180, 3000])
    assert ev2 is not None and ev2.direction == "egress"
    assert counter.total_in == 1
    assert counter.total_out == 1


def test_same_track_two_full_cycles_within_window_cancel():
    """Con U-turn enabled (default — ROI configurado), dos cycles
    back-to-back del mismo track dentro de la ventana se cancelan
    mutuamente: 1 IN emitido, 1 OUT cancela el IN, net 0/0."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 210, 3000]])

    counter._process_track(track)
    _advance(counter, track, [300, 280, 3000])
    _advance(counter, track, [300, 320, 3000])
    ev1 = _advance(counter, track, [300, 420, 3000])
    assert ev1 is not None and ev1.direction == "ingress"
    assert counter.total_in == 1

    _advance(counter, track, [300, 380, 3000])
    _advance(counter, track, [300, 310, 3000])
    _advance(counter, track, [300, 290, 3000])
    ev2 = _advance(counter, track, [300, 180, 3000])
    # El egress cancela el ingress previo: ev2=None, totals vuelven a 0.
    assert ev2 is None
    assert counter.total_in == 0
    assert counter.total_out == 0
    # El breakdown horario queda consistente con los totales: el ingress que
    # se contó fue revertido al cancelarse (antes quedaba colgado +1).
    assert counter.hourly_in_out() == []
    assert sum(h["in"] for h in counter.hourly_in_out()) == counter.total_in


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


def test_lost_inside_without_crossing_does_not_count():
    """Track entra al ROI pero muere antes de cruzar la línea — no
    cuenta. Sin cruce no hay dirección que contar; y el conteo exige
    salir del ROI de todos modos."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 220, 3000]])  # entry above line at y=300
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000]))  # still above line
    counter.check_all({1: track})
    track.state = LOST
    events = counter.check_all({})  # tracker dropped it
    assert events == []
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_no_count_on_track_death_inside_roi_after_crossing():
    """Track entra al ROI, cruza la línea, y muere SIN salir del ROI —
    NO se cuenta. Es el caso de una persona que duda/se para/se sienta en
    el ROI (o que el detector pierde por pose fuera de distribución, o que
    mueve la cabeza cruzando la línea sin trasladarse). La semántica
    canónica exige entrar -> cruzar -> SALIR del ROI; no hay "salida
    sintética" por muerte del track adentro. Evita el falso positivo de
    gente lingering en la puerta."""
    counter = Counter(lines=[_line_h()], roi=ROI)

    # Frame 1: outside ROI, arriba de la línea.
    track = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: track})

    # Frame 2: entry inside ROI, todavía arriba de la línea.
    track.positions.append(np.array([300, 250, 3000]))
    counter.check_all({1: track})

    # Frame 3: cross — abajo de la línea, still inside ROI (nunca sale).
    track.positions.append(np.array([300, 350, 3000]))
    counter.check_all({1: track})

    # Frame 4: track desaparece del tracker SIN haber salido del ROI.
    events = counter.check_all({})

    # Sin salida sintética: el cruce sin salida del ROI NO cuenta.
    assert events == []
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_uturn_zone_cancels_in_followed_by_out():
    """Persona entra al ROI cruzando IN, sale del ROI (emite ingress),
    re-entra al ROI cruzando OUT dentro de la ventana → U-turn detectado
    (el ROI es la zona de cancelación), el ingress previo se cancela y
    el OUT NO se emite. Total IN/OUT queda en 0 (net neutral)."""
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 150, 3000]])  # outside, above line
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000]))  # entry above line
    counter.check_all({1: track})
    track.positions.append(np.array([300, 350, 3000]))  # cross down (IN)
    counter.check_all({1: track})
    track.positions.append(np.array([300, 450, 3000]))  # exit south
    events = counter.check_all({1: track})
    assert len(events) == 1
    assert events[0].direction == "ingress"
    assert counter.total_in == 1

    # Persona re-entra, cruza OUT, sale north.
    track2 = _make_track(2, [[300, 450, 3000]])
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 350, 3000]))  # entry south
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 250, 3000]))  # cross up (OUT)
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 150, 3000]))  # exit north
    events2 = counter.check_all({2: track2})

    # El OUT debería cancelar el IN previo: 0 eventos emitidos, totals
    # vuelven a 0.
    assert events2 == []
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_uturn_zone_does_not_cancel_outside_window(monkeypatch):
    """Eventos opuestos fuera de la ventana NO se cancelan — visitantes
    legítimos que entran y salen mucho después siguen contando como
    1 IN + 1 OUT independientes."""
    import time as _time

    # Patchear la constante de clase para no esperar 5s reales en test.
    monkeypatch.setattr(Counter, "UTURN_WINDOW_SECONDS", 0.1)
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000]))
    counter.check_all({1: track})
    track.positions.append(np.array([300, 350, 3000]))
    counter.check_all({1: track})
    track.positions.append(np.array([300, 450, 3000]))
    counter.check_all({1: track})
    assert counter.total_in == 1

    # Esperar más que la ventana.
    _time.sleep(0.15)

    track2 = _make_track(2, [[300, 450, 3000]])
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 350, 3000]))
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 250, 3000]))
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 150, 3000]))
    events2 = counter.check_all({2: track2})

    # Ambos eventos legítimos — la window expiró antes del 2do.
    assert len(events2) == 1
    assert events2[0].direction == "egress"
    assert counter.total_in == 1
    assert counter.total_out == 1


def test_uturn_does_not_cancel_when_window_disabled(monkeypatch):
    """Con UTURN_WINDOW_SECONDS=0, la cancelación está efectivamente
    deshabilitada — dos eventos opuestos consecutivos cuentan ambos.
    Documenta cómo escapar del comportamiento default si algún site
    necesita desactivar la cancelación."""
    monkeypatch.setattr(Counter, "_try_cancel_uturn", lambda *a, **kw: False)
    counter = Counter(lines=[_line_h()], roi=ROI)
    track = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000]))
    counter.check_all({1: track})
    track.positions.append(np.array([300, 350, 3000]))
    counter.check_all({1: track})
    track.positions.append(np.array([300, 450, 3000]))
    counter.check_all({1: track})

    track2 = _make_track(2, [[300, 450, 3000]])
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 350, 3000]))
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 250, 3000]))
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 150, 3000]))
    counter.check_all({2: track2})

    assert counter.total_in == 1
    assert counter.total_out == 1


def test_legitimate_exit_counts_once_and_death_does_not_recount():
    """Cuando un track sale del ROI legítimamente (entrar -> cruzar ->
    SALIR), se cuenta UNA vez en el frame de salida. Si después el track
    desaparece del tracker, NO se vuelve a contar (sin salida sintética
    no hay doble-conteo del mismo cruce)."""
    counter = Counter(lines=[_line_h()], roi=ROI)

    # Cycle completo: outside → entry → cross → exit.
    track = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000]))  # entry
    counter.check_all({1: track})
    track.positions.append(np.array([300, 350, 3000]))  # cross (still inside)
    counter.check_all({1: track})

    track.positions.append(np.array([300, 450, 3000]))  # exit (sale del ROI)
    events = counter.check_all({1: track})
    assert len(events) == 1  # count event geométrico en la salida
    assert events[0].direction == "ingress"

    # Track muere — NO debe re-contar (ya se contó en la salida).
    events = counter.check_all({})
    assert events == []
    assert counter.total_in == 1  # sigue siendo 1, no se duplicó


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


def test_no_roi_single_crossing_does_not_count():
    """Without a ROI the whole frame is 'inside', so there is no exit
    gate: a single-line crossing alone never counts. (The previous
    track-death / synthetic-exit fallback that made no-ROI count was
    removed to stop false positives from people lingering inside the ROI
    — see counter docstring.) In real deploys you ALWAYS configure a ROI;
    the count requires entrar -> cruzar -> SALIR del ROI."""
    counter = Counter(lines=[_line_h()])  # no ROI
    track = _make_track(1, [[200, 200, 3000]])
    counter._process_track(track)
    _advance(counter, track, [200, 350, 3000])
    assert counter.total_in == 0  # sin ROI no hay salida que dispare el conteo


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


def test_check_all_processes_multiple_tracks(monkeypatch):
    # Patcheamos la ventana U-turn a 0 — el test cubre el procesamiento
    # batch de múltiples tracks. Con U-turn enabled (default), un IN +
    # OUT en el mismo ROI dentro de la ventana cancelarían (cubierto
    # por test_uturn_zone_cancels_in_followed_by_out). Acá queremos
    # verificar que check_all efectivamente emite ambos eventos cuando
    # la cancelación no aplica.
    monkeypatch.setattr(Counter, "_try_cancel_uturn", lambda *a, **kw: False)
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

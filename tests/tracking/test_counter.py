"""Tests para el Counter unificado (counting zone + N líneas direccionales)."""

import numpy as np
import pytest

from src.tracking.counter import Counter, Line, build_counter
from src.tracking.tracker import CONFIRMED, LOST, PENDING, Track


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


COUNTING_ZONE = {"x_min": 100, "x_max": 500, "y_min": 200, "y_max": 400}


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


def test_counter_invalid_counting_zone_raises():
    with pytest.raises(ValueError):
        Counter(
            lines=[_line_h()],
            counting_zone={"x_min": 500, "x_max": 100, "y_min": 200, "y_max": 400},
        )


# ---------------------------------------------------------------------------
# Single horizontal line — ingress / egress / indeciso / oscillation
# ---------------------------------------------------------------------------


def test_ingress_counts_on_exit():
    """Track enters above the line, crosses, exits below — counts ingress."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 210, 3000]])  # entered counting zone above line
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
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 380, 3000]])  # inside, below
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])  # still inside, below
    _advance(counter, track, [300, 290, 3000])  # crossed to above
    ev = _advance(counter, track, [300, 180, 3000])  # exit above
    assert ev is not None
    assert ev.direction == "egress"
    assert counter.total_out == 1
    assert counter.total_in == 0


def test_same_track_two_full_cycles_both_count():
    """A track that completes two full cycles back-to-back cuenta AMBOS.

    Antiglitch: el counter resetea meta después de cada evento emitido
    así el mismo track puede producir cycles separados. Una persona que
    entra, cruza y sale (ingress) y luego vuelve a entrar, cruzar y salir
    por el otro lado (egress) es un round-trip REAL — cuenta 1 IN + 1 OUT.
    (Antes el U-turn cancellation lo neutralizaba; se removió porque
    cancelaba round-trips legítimos.)"""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
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
    # El reset de meta permite contar el segundo cycle como egress
    # independiente.
    _advance(counter, track, [300, 380, 3000])
    _advance(counter, track, [300, 310, 3000])
    _advance(counter, track, [300, 290, 3000])
    ev2 = _advance(counter, track, [300, 180, 3000])
    assert ev2 is not None and ev2.direction == "egress"
    assert counter.total_in == 1
    assert counter.total_out == 1

    # El breakdown horario queda consistente con los totales.
    assert sum(h["in"] for h in counter.hourly_in_out()) == counter.total_in
    assert sum(h["out"] for h in counter.hourly_in_out()) == counter.total_out


def test_oscillation_without_full_cycle_does_not_count_twice():
    """A track that exits the counting zone and re-enters from the same side without
    reaching the line again must not produce a second event."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
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


def test_indeciso_with_two_way_line_cancels_net_zero():
    """A line labelled in BOTH directions: cruzar y re-cruzar dentro del
    counting zone se cancela (balance neto 0) → NO cuenta.

    Track entra arriba, cruza abajo (ingress, neto +1), vuelve arriba
    (egress, neto 0), sale arriba. Es la "duda en la puerta": entró,
    cruzó, se arrepintió y volvió sin entrar a la tienda. El balance neto
    queda en 0 → ningún evento. (El viejo comportamiento "gana el último
    cruce" contaba egress acá, que era incorrecto.)
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 250, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])  # crossed below: ingress (+1)
    _advance(counter, track, [300, 280, 3000])  # back above: egress (net 0)
    ev = _advance(counter, track, [300, 180, 3000])  # exit above
    assert ev is None
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_cross_then_linger_then_exit_counts_once():
    """Persona entra a la counting zone, cruza la línea (ingress), deambula DENTRO
    de la counting zone sin volver a cruzar (mira un cartel, etc.) y luego sale por
    el lado de adentro. Debe contar 1 ingress — el lingering sin re-cruce
    no afecta el balance neto. (Asume que el track SOBREVIVE el lingering;
    si muere adentro de la counting zone no hay exit y no cuenta — ver tracking.)"""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 250, 3000]])  # entry above line (y=300)
    counter._process_track(track)
    _advance(counter, track, [300, 350, 3000])  # cruza abajo: ingress (+1)
    # Deambula dentro de la counting zone, abajo de la línea, sin re-cruzar.
    _advance(counter, track, [320, 360, 3000])
    _advance(counter, track, [340, 355, 3000])
    _advance(counter, track, [300, 380, 3000])
    ev = _advance(counter, track, [300, 520, 3000])  # exit abajo de la counting zone
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_indeciso_with_one_way_line_does_not_count():
    """If the line only labels ONE direction (one-way gate), an indeciso
    track that wanders back through the unlabelled side leaves the counter
    in its last labelled state. To cleanly model "person walks in, walks
    back out without committing" use one-way lines: the return crossing
    has no label, so it does not overwrite the previous one — but in this
    specific case the track never crossed into the labelled direction at
    all, so nothing is counted."""
    one_way_in = _line_h(egress=None)  # only top_to_bottom is labelled
    counter = Counter(lines=[one_way_in], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 250, 3000]])  # above (no label yet)
    counter._process_track(track)
    _advance(counter, track, [300, 280, 3000])  # still above
    _advance(counter, track, [300, 240, 3000])  # still above
    ev = _advance(counter, track, [300, 180, 3000])  # exit above
    assert ev is None
    assert counter.total_in == 0


def test_indeciso_no_crossing():
    """Enters above, lingers, exits above without ever crossing — no count."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 240, 3000])
    _advance(counter, track, [300, 230, 3000])
    ev = _advance(counter, track, [300, 180, 3000])
    assert ev is None
    assert counter.total_in == 0


def test_lost_inside_without_crossing_does_not_count():
    """Track entra a la counting zone pero muere antes de cruzar la línea — no
    cuenta. Sin cruce no hay dirección que contar; y el conteo exige
    salir de la counting zone de todos modos."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 220, 3000]])  # entry above line at y=300
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000]))  # still above line
    counter.check_all({1: track})
    track.state = LOST
    events = counter.check_all({})  # tracker dropped it
    assert events == []
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_count_on_track_death_inside_counting_zone_after_crossing():
    """Track entra a la counting zone, cruza la línea con detección real, y muere SIN
    salir de la counting zone — IGUAL se cuenta (death-emit-if-crossed).

    El caso real: el detector pierde a la persona después del cruce y el
    Kalman parka adentro de la counting zone antes de alcanzar el borde → el exit nunca
    dispara y, sin el death-emit, el conteo se perdía. El gate sigue siendo
    selectivo: si entró pero NO cruzó (el caso del que duda/lingera en la
    entrada), `test_death_inside_counting_zone_without_crossing_no_count` cubre que
    NO se cuente."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)

    # Frame 1: outside counting zone, arriba de la línea.
    track = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: track})

    # Frame 2: entry inside counting zone, todavía arriba de la línea.
    track.positions.append(np.array([300, 250, 3000]))
    counter.check_all({1: track})

    # Frame 3: cross — abajo de la línea, still inside counting zone (nunca sale).
    track.positions.append(np.array([300, 350, 3000]))
    counter.check_all({1: track})

    # Frame 4: track desaparece del tracker SIN haber salido de la counting zone.
    # El death-emit ahora está DIFERIDO por DEATH_EMIT_GRACE_FRAMES (espera
    # ghost adoption); el primer check_all post-muerte NO emite.
    events = counter.check_all({})
    assert events == []

    # Tras avanzar la grace window sin re-aparición → fire del death-emit.
    for _ in range(Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES + 1):
        events = counter.check_all({})
    assert counter.total_in == 1


def test_in_followed_by_out_both_count():
    """Persona entra a la counting zone cruzando IN y sale (emite ingress), luego
    re-entra cruzando OUT y sale (emite egress) → 1 IN + 1 OUT. Es un
    round-trip real y debe contar ambos. (Antes el U-turn cancellation
    cancelaba el IN cuando el OUT caía dentro de los 5s — se removió
    porque ese escenario es tráfico legítimo, no una duda en la puerta.)"""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
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

    # Persona re-entra (track nuevo), cruza OUT, sale north.
    track2 = _make_track(2, [[300, 450, 3000]])
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 350, 3000]))  # entry south
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 250, 3000]))  # cross up (OUT)
    counter.check_all({2: track2})
    track2.positions.append(np.array([300, 150, 3000]))  # exit north
    events2 = counter.check_all({2: track2})

    # Sin cancelación: el OUT cuenta como egress independiente.
    assert len(events2) == 1
    assert events2[0].direction == "egress"
    assert counter.total_in == 1
    assert counter.total_out == 1


def test_legitimate_exit_counts_once_and_death_does_not_recount():
    """Cuando un track sale de la counting zone legítimamente (entrar -> cruzar ->
    SALIR), se cuenta UNA vez en el frame de salida. Si después el track
    desaparece del tracker, NO se vuelve a contar (sin salida sintética
    no hay doble-conteo del mismo cruce)."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)

    # Cycle completo: outside → entry → cross → exit.
    track = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000]))  # entry
    counter.check_all({1: track})
    track.positions.append(np.array([300, 350, 3000]))  # cross (still inside)
    counter.check_all({1: track})

    track.positions.append(np.array([300, 450, 3000]))  # exit (sale de la counting zone)
    events = counter.check_all({1: track})
    assert len(events) == 1  # count event geométrico en la salida
    assert events[0].direction == "ingress"

    # Track muere — NO debe re-contar (ya se contó en la salida).
    events = counter.check_all({})
    assert events == []
    assert counter.total_in == 1  # sigue siendo 1, no se duplicó


def test_pending_state_counted():
    """PENDING tracks are eligible for counting (they keep their meta)."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    track.state = PENDING
    ev = _advance(counter, track, [300, 420, 3000])
    assert ev is not None
    assert ev.direction == "ingress"


def test_candidate_not_counted():
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
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
    counter = Counter(lines=[_line_v()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[150, 300, 3000]])  # enter left
    counter._process_track(track)
    _advance(counter, track, [280, 300, 3000])  # still left
    _advance(counter, track, [320, 300, 3000])  # crossed right
    ev = _advance(counter, track, [520, 300, 3000])  # exit right
    assert ev is not None
    assert ev.direction == "ingress"


def test_vertical_line_egress():
    counter = Counter(lines=[_line_v()], counting_zone=COUNTING_ZONE)
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
    """Two horizontal lines in different x ranges of the counting zone: the left one
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
    counter = Counter(lines=[line_in, line_out], counting_zone=COUNTING_ZONE)

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
    counter = Counter(lines=[line], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[400, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [400, 320, 3000])  # crosses y=300 at x=400
    ev = _advance(counter, track, [400, 420, 3000])
    assert ev is None
    assert counter.total_in == 0


# ---------------------------------------------------------------------------
# No-counting-zone mode (gate purely by line crossings — useful when the camera FOV
# itself is the area of interest and no rectangular gate is needed)
# ---------------------------------------------------------------------------


def test_no_counting_zone_single_crossing_does_not_count():
    """Without a counting zone the whole frame is 'inside', so there is no exit
    gate: a single-line crossing alone never counts. (The previous
    track-death / synthetic-exit fallback that made no-counting-zone count was
    removed to stop false positives from people lingering inside the counting zone
    — see counter docstring.) In real deploys you ALWAYS configure a counting zone;
    the count requires entrar -> cruzar -> SALIR de la counting zone."""
    counter = Counter(lines=[_line_h()])  # no counting zone
    track = _make_track(1, [[200, 200, 3000]])
    counter._process_track(track)
    _advance(counter, track, [200, 350, 3000])
    assert counter.total_in == 0  # sin counting zone no hay salida que dispare el conteo


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_counter_full_schema():
    cfg = {
        "counter": {
            "counting_zone": COUNTING_ZONE,
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
    assert c.counting_zone == (100.0, 500.0, 200.0, 400.0)
    assert len(c.lines) == 1
    assert c.lines[0].labels == {
        "top_to_bottom": "ingress",
        "bottom_to_top": "egress",
    }


def test_build_counter_two_lines():
    cfg = {
        "counter": {
            "counting_zone": COUNTING_ZONE,
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
        build_counter({"counter": {"counting_zone": COUNTING_ZONE}})


def test_build_counter_empty_lines_raises():
    with pytest.raises(ValueError, match="counter.lines"):
        build_counter({"counter": {"counting_zone": COUNTING_ZONE, "lines": []}})


def test_build_counter_no_config_raises():
    with pytest.raises(ValueError, match="counter.lines"):
        build_counter({})


def test_build_counter_optional_counting_zone():
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
    assert c.counting_zone is None


# ---------------------------------------------------------------------------
# reset_daily / custom labels
# ---------------------------------------------------------------------------


def test_custom_labels():
    line = Line(
        from_xy=(100, 300),
        to_xy=(500, 300),
        labels={"top_to_bottom": "enter", "bottom_to_top": "leave"},
    )
    counter = Counter(lines=[line], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    ev = _advance(counter, track, [300, 420, 3000])
    assert ev.direction == "enter"
    assert counter.totals["enter"] == 1
    assert counter.totals["leave"] == 0


def test_reset_daily():
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 220, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 310, 3000])
    _advance(counter, track, [300, 420, 3000])
    assert counter.total_in == 1
    counter.reset_daily()
    assert counter.total_in == 0
    assert counter.totals == {"ingress": 0, "egress": 0}


def test_check_all_processes_multiple_tracks():
    # check_all debe emitir un evento por cada track que completa su ciclo
    # en el mismo batch.
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
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

    Con un número IMPAR de flips por jitter, el balance neto no se cancela
    solo; el debounce es la red que evita que esos micro-flips se registren
    como cruces. Con min_crossing_movement_px=3.0, los movimientos sub-3px
    (medidos contra la última posición no-debounced) se ignoran.
    """
    counter = Counter(
        lines=[_line_h()],
        counting_zone=COUNTING_ZONE,
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
    # Sale de la counting zone por arriba (regreso). Si el debounce funcionó, ningún
    # cruce se registró (balance neto 0) y no dispara evento.
    ev = _advance(counter, track, [300, 180, 3000])
    assert ev is None
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_debounce_does_not_block_real_crossing():
    """Movimiento normal (>threshold) sigue contando con debounce activo."""
    counter = Counter(
        lines=[_line_h()],
        counting_zone=COUNTING_ZONE,
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


def test_oscillation_nets_to_zero_without_debounce():
    """Aún con debounce desactivado (min_crossing_movement_px=0), una
    oscilación simétrica (cruzar y re-cruzar) NO cuenta: el balance neto
    de la visita queda en 0. El debounce sigue siendo útil para jitter
    asimétrico, pero la cancelación neta cubre el caso simétrico sola."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 210, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 298, 3000])  # arriba
    _advance(counter, track, [300, 305, 3000])  # cruzó abajo (neto +1)
    _advance(counter, track, [300, 295, 3000])  # cruzó arriba (neto 0)
    ev = _advance(counter, track, [300, 180, 3000])  # exit arriba
    # Neto 0 → no cuenta, sin importar el debounce.
    assert ev is None
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_entry_side_uses_last_outside_pos_under_detection_gap():
    """Con counting zone chico y detector miss en la zona pre-línea, la primera
    detección inside puede caer ya del otro lado. El counter debe
    snapshotear sides[] desde la última posición outside-counting-zone conocida
    (lado de approach, inequívoco), no desde la primera detección inside.

    Reproduce el bug operativo observado en piloto: counting zone 264-384 vertical
    + línea en y=324 + detector que pierde el frame inside-pre-línea.
    Sin fix, sides[0] se cachea ya como ``below`` y el cruce nunca se
    detecta — el track sale de la counting zone sin emitir.
    """
    # counting zone vertical chico, simulando la setup de piloto antes del enlarge.
    small_zone = {"x_min": 100, "x_max": 500, "y_min": 264, "y_max": 384}
    counter = Counter(lines=[_line_h(line_y=324)], counting_zone=small_zone)

    # Frame 1: track outside-counting-zone arriba de la línea (approach).
    track = _make_track(1, [[300, 240, 3000]])
    assert counter._process_track(track) is None

    # Frame 2: detección MISS en y=290 (would be inside, above line) —
    # simula `det=N`. No se llama _process_track; el siguiente frame
    # vendrá con la posición real cuando el detector firee otra vez.

    # Frame 3: primera detección inside counting zone, ya pasada la línea (y=350).
    # Sin el fix, sides[0] se cachearía aquí como "below" y todo lo
    # que sigue mantiene "below" → no se observa cruce.
    _advance(counter, track, [300, 350, 3000])

    # Frame 4: track sale de la counting zone por abajo.
    ev = _advance(counter, track, [300, 400, 3000])

    # Con el fix: sides[0] se snapshoteó desde la última outside-pos
    # (y=240, above line). El frame inside @ y=350 detectó el flip
    # above→below durante el inside-loop, capturó label=ingress. El
    # exit @ y=400 emite el evento.
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_entry_side_fallback_when_track_born_inside_counting_zone():
    """Si el track aparece directamente dentro de la counting zone (sin posición
    outside previa registrada), el counter cae al comportamiento legacy:
    snapshotear sides[] desde la posición actual. Sin regresión para
    tracks que nacen inside.
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Track nace inside counting zone, arriba de la línea (y=210 < line y=300).
    track = _make_track(1, [[300, 210, 3000]])
    assert counter._process_track(track) is None
    # Camina cruzando la línea.
    _advance(counter, track, [300, 320, 3000])
    ev = _advance(counter, track, [300, 420, 3000])  # exit abajo
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1


def test_lateral_entry_crosses_line_lateral_exit_counts():
    """Entry y exit por aristas LATERALES de la counting zone (no por arriba/abajo),
    con un cruce horizontal en el medio.

    Geometría: persona caminando paralelo al frente del local. Pre-counting-zone
    viene del costado-arriba (x<x_min, y<y_line) → entra a la counting zone por la
    arista izquierda → cruza la línea horizontal de arriba hacia abajo
    → sale de la counting zone por la arista derecha. Debe contar como ingress (la
    dirección del cruce respecto a la línea, NO respecto a la counting zone, es lo
    que define IN/OUT).

    Cubre el caso que ``test_entry_side_uses_last_outside_pos_under_
    detection_gap`` no toca: ese test prueba approach VERTICAL (arriba
    de la línea, baja a través de la arista superior de la counting zone). Este
    prueba approach HORIZONTAL (al costado de la línea, entra por la
    arista lateral de la counting zone).
    """
    counter = Counter(lines=[_line_h(line_y=300)], counting_zone=COUNTING_ZONE)

    # Frame 1: outside counting zone por la izquierda, ARRIBA de la línea
    # (y=250 < line_y=300). Esto setea last_outside_pos para el
    # snapshot de sides[].
    track = _make_track(1, [[50, 250, 3000]])
    assert counter._process_track(track) is None

    # Frame 2: entry-fresca por la arista izquierda, todavía arriba de
    # la línea. sides[0] se snapshotea desde last_outside_pos=(50,250):
    # side_of(50, 250) = -1 (above). No emite todavía.
    assert _advance(counter, track, [150, 270, 3000]) is None

    # Frame 3: cruza la línea horizontal hacia abajo (y > 300). La
    # transición -1 → +1 con label "top_to_bottom" registra +1 en
    # crossing_net[0]. No emite aún (sigue inside).
    assert _advance(counter, track, [300, 340, 3000]) is None

    # Frame 4: sale de la counting zone por la arista DERECHA (x > x_max=500). El
    # exit-transition consume crossing_net=[+1] y emite ingress.
    ev = _advance(counter, track, [550, 340, 3000])
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1
    assert counter.total_out == 0


def test_lateral_entry_crosses_line_downside_egress():
    """Variante del test lateral: approach desde el costado pero ABAJO
    de la línea (y > y_line). El cruce hacia arriba debe contar como
    egress. Confirma que la dirección del cruce se determina por el
    side de last_outside_pos vs y_line, independientemente de por dónde
    entró a la counting zone (arista lateral).
    """
    counter = Counter(lines=[_line_h(line_y=300)], counting_zone=COUNTING_ZONE)

    # Frame 1: outside counting zone por la izquierda, ABAJO de la línea (y=350).
    track = _make_track(1, [[50, 350, 3000]])
    assert counter._process_track(track) is None

    # Frame 2: entry lateral por la izquierda. sides[0] = side_of(50,
    # 350) = +1 (below). Todavía abajo de la línea inside counting zone.
    assert _advance(counter, track, [150, 340, 3000]) is None

    # Frame 3: cruza hacia arriba (y < 300). Transición +1 → -1 con
    # label "bottom_to_top" → egress, registra -1 en crossing_net[0].
    assert _advance(counter, track, [300, 260, 3000]) is None

    # Frame 4: sale por la arista DERECHA. crossing_net=[-1] → emite
    # egress.
    ev = _advance(counter, track, [550, 260, 3000])
    assert ev is not None
    assert ev.direction == "egress"
    assert counter.total_in == 0
    assert counter.total_out == 1


# ---------------------------------------------------------------------------
# Gate de detección real: los cruces solo cuentan con detección real, no con
# predicción Kalman (anti doble-conteo del drift + no perder el conteo del
# que cruzó y se perdió).
# ---------------------------------------------------------------------------


def test_inside_was_inside_prediction_does_not_register_cross():
    """En la rama inside-was-inside, los frames de PURA PREDICCIÓN (track
    PENDING) NO actualizan sides/net (gate estricto). Se preserva el lado
    desde la última detección real así un drift Kalman cruzando-recruzando la
    línea adentro de la counting zone no acumula cruces espurios."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 295, 3000]])  # entry justo arriba (real)
    counter._process_track(track)  # entry, last_track_pos=(300,295), sides=-1
    # Frame de predicción: el track "cruza" hacia abajo por extrapolación.
    # Inside-was-inside con is_real=False → early return → sides PRESERVADO
    # en -1 (no se registra cruce per-frame). Sin esto, un drift que
    # oscila atravesando la línea contaría múltiples crosses espurios.
    track.disappeared = 5
    ev = _advance(counter, track, [300, 305, 3000])  # cruzaría a +1, pero predicción
    assert ev is None  # sin emit en inside-was-inside


def test_crossing_real_then_exit_on_prediction_still_counts():
    """Una persona que cruza con detección REAL y después se pierde (el track
    sale de la counting zone por extrapolación) IGUAL cuenta al salir — no se pierde el
    conteo (era el bug del freeze).

    Pre-cond del test: arrancar OUTSIDE counting zone para que had_outside_pos=True
    al entrar — sin esto el guard de exit-por-Kalman (defense-in-depth
    contra el sitter pegado a la línea) descartaría el count.
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 0: outside counting zone (y=150 < y_min=200) — establece last_outside_pos
    # y por tanto had_outside_pos=True en la entry posterior.
    track = _make_track(1, [[300, 150, 3000]])
    counter._process_track(track)
    _advance(counter, track, [300, 250, 3000])  # entry inside (real)
    _advance(counter, track, [300, 350, 3000])  # cruza abajo con detección REAL (net+1)
    # El detector la pierde; el track extrapola y sale de la counting zone (predicción).
    track.disappeared = 5
    ev = _advance(counter, track, [300, 520, 3000])  # exit por extrapolación
    assert ev is not None
    assert ev.direction == "ingress"
    assert counter.total_in == 1


# ---------------------------------------------------------------------------
# Death-emit-if-crossed: si un track desaparece habiendo cruzado pero sin
# haber salido de la counting zone (el detector lo pierde y el Kalman parka adentro),
# igual se emite el conteo. Cubre el lost-count residual.
# ---------------------------------------------------------------------------


def test_death_inside_counting_zone_after_crossing_emits_count():
    """Track cruza la línea (detección real) y después desaparece de la dict
    sin haber salido de la counting zone → emite el conteo en la muerte (post grace).
    Pre-cond: arranca FUERA de la counting zone (had_outside_pos=True) y recorre >80px
    para pasar los guards anti-falso-positivo del death-emit."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 0: outside counting zone (y=150 < y_min=200) — fija last_outside_pos.
    track = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000], dtype=float))
    counter.check_all({1: track})  # entry inside counting zone, had_outside_pos=True
    track.positions.append(np.array([300, 350, 3000], dtype=float))
    events = counter.check_all({1: track})  # cross net=+1, visit_y_range=100
    assert events == []

    # El track desaparece. Death-emit DIFERIDO por grace.
    counter.check_all({})
    for _ in range(Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES + 1):
        counter.check_all({})
    assert counter.total_in == 1


def test_death_inside_counting_zone_without_crossing_no_count():
    """Track entra a la counting zone pero NUNCA cruza y después desaparece — no debe
    contar (entró pero no cruzó: persona dudó en la entrada)."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 250, 3000]])  # entry arriba (no cruzó)
    counter.check_all({1: track})
    # Desaparece sin haber cruzado.
    events = counter.check_all({})
    assert events == []
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_clean_exit_then_death_does_not_double_count():
    """Track cruza, sale de la counting zone (emite 1) y después desaparece de la dict —
    NO debe doble-contar (el snapshot post-exit tiene inside=False)."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 250, 3000]])
    counter.check_all({1: track})
    track.positions.append(np.array([300, 350, 3000], dtype=float))
    counter.check_all({1: track})
    track.positions.append(np.array([300, 520, 3000], dtype=float))  # sale
    events = counter.check_all({1: track})
    assert len(events) == 1  # emit normal de exit

    # Después se muere — sin doble-conteo.
    events = counter.check_all({})
    assert events == []
    assert counter.total_in == 1


# ---------------------------------------------------------------------------
# Gate relajado en exit: cruces extrapolados por Kalman cuentan SI el
# movimiento desde la última detección real es decisivo en la dirección del
# cruce (caso "walked → lost en zona crítica de la línea"). El drift
# estacionario (residual velocity, sin movimiento real) sigue descartado.
# ---------------------------------------------------------------------------


def test_decisive_kalman_cross_at_exit_counts():
    """Track con última posición real bien debajo de la línea (y=352), Kalman
    extrapola hacia arriba y SALE de la counting zone por arriba en frame de predicción.
    El desplazamiento desde la última real (dy=-152) es decisivo en la
    dirección del cruce (hacia arriba = side -1) → cuenta.

    Pre-cond: track viene de outside-counting-zone ABAJO de la counting zone (y>y_max) — establece
    had_outside_pos=True con snap del mismo lado que la entry (sides=+1),
    así el cross original es +1 → -1 al cruzar la línea por extrapolación.
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 0: outside counting zone por abajo (y=450 > y_max=400, side=+1) —
    # had_outside_pos=True + snap del mismo lado que la entry inside.
    track = _make_track(1, [[460, 450, 3000]])
    counter.check_all({1: track})
    # Entry real con sides=+1 (debajo de la línea, y=352).
    track.positions.append(np.array([460, 352, 3000], dtype=float))
    counter.check_all({1: track})  # entry, sets last_track_pos=(460,352)

    # Kalman push: track sale de la counting zone por arriba (y=190 < y_min=200).
    # is_real=False (track.disappeared > 0). new_side=-1 ≠ prev_side=+1
    # → cross detectado.
    track.positions.append(np.array([460, 190, 3000], dtype=float))
    track.disappeared = 5  # dentro del MAX_KALMAN_CROSS_FRAMES=15
    events = counter.check_all({1: track})

    assert len(events) == 1
    assert events[0].direction == "egress"  # cruzó hacia arriba: side -1
    assert counter.total_out == 1


def test_kalman_cross_too_old_does_not_count():
    """Misma situación que el test anterior pero con disappeared > MAX. El
    track estuvo perdido demasiado tiempo → no confiable, no cuenta."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[460, 352, 3000]])
    counter.check_all({1: track})

    track.positions.append(np.array([460, 190, 3000], dtype=float))
    track.disappeared = 30  # > MAX_KALMAN_CROSS_FRAMES=15
    events = counter.check_all({1: track})

    assert events == []
    assert counter.total_out == 0


def test_kalman_drift_at_exit_does_not_count():
    """Drift estacionario: última posición real APENAS arriba de la línea
    (y=295). El track ""drifta"" mínimamente y sale de la counting zone por el lateral —
    el desplazamiento en y desde la última real (10px) es marginal y NO
    pasa el umbral decisivo de 30px → no cuenta (anti doble-conteo del
    parado-cruzar)."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Entry justo arriba de la línea (real, sides=-1).
    track = _make_track(1, [[460, 295, 3000]])
    counter.check_all({1: track})  # last_track_pos=(460,295)
    # Drift en predicción: sale de la counting zone por x>x_max=500. dy desde última real
    # = 10 (no decisivo). En exit: prev_side=-1, new_side=+1 (y=305>300), pero
    # decisive_disp = dy*new_side = 10 < 30 → rechaza.
    track.disappeared = 5
    track.positions.append(np.array([510, 305, 3000], dtype=float))
    events = counter.check_all({1: track})

    assert events == []
    assert counter.total_in == 0


def test_real_detection_cross_at_exit_still_counts():
    """Regression: la rama de exit con detección REAL sigue funcionando igual
    (el gate relajado no rompe el camino normal)."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[460, 250, 3000]])
    counter.check_all({1: track})
    # Salta directamente abajo de la counting zone cruzando la línea — exit con detección
    # real (disappeared=0).
    track.positions.append(np.array([460, 520, 3000], dtype=float))
    events = counter.check_all({1: track})
    assert len(events) == 1
    assert events[0].direction == "ingress"


def test_kalman_exit_skipped_when_track_born_inside_counting_zone():
    """Reproduce el bug del sitter pegado a la línea de cruce.

    Escenario operativo (observado en piloto 2026-05-23 17:45): la persona
    está sentada justo dentro de la counting zone, cerca de la línea de cruce. El
    track nace inside counting zone (sin last_outside_pos previo) → had_outside_pos
    = False. La persona se mueve un toque (se acomoda en la silla):
    cruza la línea con detección real (net != 0). Después el detector la
    pierde (oclusión / cambio de pose); el Kalman extrapola la velocidad
    residual hacia el lateral de la counting zone y dispara el _decisive_kalman_cross
    en el exit. Sin el guard, el exit-por-Kalman emite un count espurio
    — la persona NUNCA salió de la tienda, solo se acomodó.

    El fix: aplicar el guard had_outside_pos también al exit branch
    cuando is_real=False (extrapolación Kalman). Coherente con el mismo
    guard en _emit_on_death (capa 3 del rescue cascade).
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 1: track aparece DIRECTAMENTE inside counting zone, lado +1 (y=350 abajo
    # de la línea y=300). No hay frame outside previo → had_outside_pos
    # será False en la entry-fresca.
    track = _make_track(1, [[460, 350, 3000]])
    assert counter._process_track(track) is None
    # Frame 2: la persona se mueve un toque cruzando la línea hacia
    # arriba (y=250, lado -1). Detección REAL — registra cross net=-1.
    assert _advance(counter, track, [460, 250, 3000]) is None
    # Frame 3: detector pierde al track. Kalman push: sale de la counting zone por
    # el lateral derecho (x=560 > x_max=500). disappeared=5 → dentro del
    # MAX_KALMAN_CROSS_FRAMES. Sin el guard, decisive_kalman_cross
    # aceptaría el cruce y emitiría egress.
    track.positions.append(np.array([560, 250, 3000], dtype=float))
    track.disappeared = 5
    events = counter.check_all({1: track})

    # Con el guard: NO count. La persona no salió de la tienda.
    assert events == []
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_kalman_exit_counts_when_track_has_outside_history():
    """Regression contra el fix anterior: el guard had_outside_pos NO
    afecta tracks que entraron a la counting zone desde afuera legítimamente. Mismo
    escenario que test_decisive_kalman_cross_at_exit_counts pero con un
    frame outside-counting-zone previo para establecer had_outside_pos=True.
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 1: track outside-counting-zone por la izquierda, debajo de la línea.
    # last_outside_pos se setea aquí → had_outside_pos=True en la entry.
    track = _make_track(1, [[50, 352, 3000]])
    assert counter._process_track(track) is None
    # Frame 2: entry inside counting zone, sides snapshoteado desde (50, 352) → +1.
    assert _advance(counter, track, [460, 352, 3000]) is None
    # Frame 3: Kalman push hacia arriba — sale de la counting zone por arriba con
    # cruce decisivo. Equivalente a test_decisive_kalman_cross.
    track.positions.append(np.array([460, 190, 3000], dtype=float))
    track.disappeared = 5
    events = counter.check_all({1: track})

    # Con el guard: SÍ cuenta (had_outside_pos=True bypassea la
    # restricción).
    assert len(events) == 1
    assert events[0].direction == "egress"
    assert counter.total_out == 1


def test_entry_fresca_skipped_when_first_inside_frame_is_kalman():
    """Reproduce el bug del entry-Kalman alucinado (piloto 2026-05-24
    09:47-09:54, tid=35): el detector emitió una FP outside counting zone, el
    Kalman la proyectó adentro, y la entry-fresca con is_real=False
    snapshoteaba sides[] + configuraba was_inside=True habilitando el
    zigzag clásico sobre la línea + un exit por Kalman emitiendo un
    COUNT IN falso.

    El fix: en la entry-fresca, si el primer frame inside es
    is_real=False, NO disparar la entry-fresca. Esperar al próximo
    frame real adentro; si la persona realmente está ahí, viene en
    1-3 ticks. Sin frame real inside nunca jamás → el track no entra
    al estado "inside" del counter y no contribuye a counts.
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 1: outside counting zone, real → establece last_outside_pos.
    track = _make_track(1, [[300, 150, 3000]])
    assert counter._process_track(track) is None

    # Frame 2: Kalman push entra a la counting zone (y=250 inside) PERO disappeared
    # > 0 (is_real=False) — Kalman alucinado. La entry-fresca debe
    # SKIPearse; was_inside permanece False; no se cuenta nada todavía.
    track.positions.append(np.array([300, 250, 3000], dtype=float))
    track.disappeared = 3
    assert counter._process_track(track) is None
    meta = track.meta.get(Counter.META_KEY, {})
    assert not meta.get("inside", False), \
        "entry-fresca con Kalman NO debe setear inside=True"

    # Frame 3: track sale de la counting zone por Kalman a (300, 520). Como nunca
    # tuvo entry-fresca legítima, was_inside=False — el exit branch
    # tampoco se dispara. No hay count.
    track.positions.append(np.array([300, 520, 3000], dtype=float))
    track.disappeared = 5
    ev = counter._process_track(track)
    assert ev is None
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_entry_fresca_deferred_until_real_detection():
    """Variante del anterior: el primer frame inside es Kalman (skipea),
    pero el SIGUIENTE frame inside es real → entry-fresca dispara ahí
    con la misma información geométrica (last_outside_pos sigue válido).
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 1: outside real → establece last_outside_pos.
    track = _make_track(1, [[300, 150, 3000]])
    counter._process_track(track)

    # Frame 2: Kalman push inside (skipea entry-fresca).
    track.positions.append(np.array([300, 250, 3000], dtype=float))
    track.disappeared = 3
    counter._process_track(track)
    assert not track.meta.get(Counter.META_KEY, {}).get("inside", False)

    # Frame 3: detección REAL inside (disappeared=0). Acá sí dispara
    # la entry-fresca. sides[] se snapshotean desde el last_outside_pos
    # capturado en Frame 1.
    track.positions.append(np.array([300, 260, 3000], dtype=float))
    track.disappeared = 0
    counter._process_track(track)
    assert track.meta[Counter.META_KEY]["inside"] is True

    # Frame 4: cruza la línea + sale de la counting zone con detección real →
    # cuenta normal (ingress).
    track.positions.append(np.array([300, 520, 3000], dtype=float))
    ev = counter._process_track(track)
    assert ev is not None
    assert ev.direction == "ingress"


def test_last_outside_pos_only_updated_with_real_detections():
    """Reproduce el bug de last_outside_pos envenenado por Kalman
    extrapolation (observado en piloto 2026-05-23 18:10).

    Escenario: sitter inside counting zone cuyo track sale por extrapolación
    Kalman alucinada hacia el lateral del frame (ej. x=850). Ese frame
    es is_inside=False AND is_real=False. Si last_outside_pos se
    actualizara con esa posición Kalman, el próximo ciclo del track
    (post ghost adoption) heredaría un last_outside_pos espurio
    (850,...) lejos del centroide real (470,...) → had_outside_pos=True
    falso → guard de capa 2 NO descarta el count espurio.

    El fix condiciona el update de last_outside_pos a is_real=True. Las
    posiciones de Kalman extrapolation NO se aceptan como evidencia de
    approach. Coherente con visit_x/y_range que también se actualiza
    solo con detecciones reales.
    """
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 1: track nace inside counting zone (snap == pos, had_outside_pos=False).
    track = _make_track(1, [[460, 350, 3000]])
    assert counter._process_track(track) is None
    # Frame 2: Kalman push hacia x=850 (outside counting zone por la derecha) —
    # SIN detección real (disappeared > 0). Esta posición NO debe
    # quedar guardada como last_outside_pos.
    track.positions.append(np.array([850, 350, 3000], dtype=float))
    track.disappeared = 5
    counter._process_track(track)
    # Frame 3: track vuelve a aparecer inside counting zone con detección real
    # (disappeared=0 emulando re-detect). Si last_outside_pos se hubiera
    # contaminado en frame 2, había_outside_pos=True ahora. Pero como
    # el fix lo filtra, sigue había_outside_pos=False.
    track.disappeared = 0
    # Simulamos un "tercer ciclo": track sale de la counting zone por la izquierda
    # — must be is_real=False para reproducir el bug original que
    # disparaba el count espurio.
    track.positions.append(np.array([460, 350, 3000], dtype=float))  # re-inside
    counter._process_track(track)
    # Cruza la línea (real, intra-visit).
    track.positions.append(np.array([460, 250, 3000], dtype=float))
    track.disappeared = 0
    counter._process_track(track)  # cross net=-1, label=egress
    # Kalman extrapola al lateral derecho (is_real=False).
    track.positions.append(np.array([560, 250, 3000], dtype=float))
    track.disappeared = 5
    events = counter.check_all({1: track})

    # Con el fix: el guard de exit-Kalman descarta porque
    # had_outside_pos sigue False (el Kalman push del frame 2 no
    # contaminó last_outside_pos).
    assert events == []
    assert counter.total_in == 0
    assert counter.total_out == 0


# ---------------------------------------------------------------------------
# Death-emit deferred grace: cuando un track desaparece de la dict, el
# death-emit NO dispara inmediato — se difiere por DEATH_EMIT_GRACE_FRAMES
# para que el tracker tenga chance de resucitarlo (ghost adoption). Si el
# track REAPARECE dentro de la ventana, el death-emit se cancela y el
# conteo emite naturalmente en el exit posterior.
# ---------------------------------------------------------------------------


def test_death_emit_deferred_then_resurrected_no_double_count():
    """Track cruza → desaparece → grace window arranca. Antes de expirar,
    REAPARECE (ghost adoption resucitó). El death-emit se cancela y el
    conteo emite cuando el track sale naturalmente — 1 sólo evento, no 2."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Frame 0: outside counting zone — fija last_outside_pos (necesario para death-emit
    # guards).
    track = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000], dtype=float))
    counter.check_all({1: track})  # entry, had_outside_pos=True
    track.positions.append(np.array([300, 350, 3000], dtype=float))
    counter.check_all({1: track})  # cross net=+1, visit_y_range=100

    # Desaparece de la dict (track murió). Death-emit NO dispara aún.
    events = counter.check_all({})
    assert events == [], "death-emit debe estar diferido por el grace"
    assert counter.total_in == 0

    # Algunos frames sin re-aparecer (todavía dentro del grace).
    for _ in range(5):
        events = counter.check_all({})
        assert events == []

    # Track resucitado (ghost adoption): mismo track_id, meta restaurada.
    # Su posición es post-cross (y=350), todavía dentro de la counting zone.
    resurrected = _make_track(1, [[300, 350, 3000]])
    # Restaurar manualmente la meta del counter como haría el ghost pool.
    resurrected.meta[Counter.META_KEY] = {
        "inside": True,
        "line_sides": [1],  # debajo de la línea
        "crossing_net": [1],
        "last_crossing_pos": (300.0, 300.0),
        "last_track_pos": (300.0, 350.0),
        "had_outside_pos": True,
        "visit_x_min": 300.0, "visit_x_max": 300.0,
        "visit_y_min": 250.0, "visit_y_max": 350.0,
    }
    events = counter.check_all({1: resurrected})
    # No emit todavía (sigue inside).
    assert events == []

    # Track sale de la counting zone → emit natural.
    resurrected.positions.append(np.array([300, 520, 3000], dtype=float))
    events = counter.check_all({1: resurrected})
    assert len(events) == 1
    assert events[0].direction == "ingress"
    assert counter.total_in == 1  # UN solo evento, no dos


def test_death_emit_fires_after_grace_expires():
    """Si el track NO reaparece dentro del grace, el death-emit dispara como
    fallback al expirar la ventana. Pre-cond: track legítimo (vino desde
    fuera de la counting zone, recorre >80 px) — pasa los guards anti-falso-positivo."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 150, 3000]])  # outside counting zone
    counter.check_all({1: track})
    track.positions.append(np.array([300, 250, 3000], dtype=float))
    counter.check_all({1: track})  # entry, had_outside_pos=True
    track.positions.append(np.array([300, 350, 3000], dtype=float))
    counter.check_all({1: track})  # cross net=+1, visit_y_range=100

    # Desaparece — death-emit diferido.
    counter.check_all({})

    # Esperar GRACE+1 frames sin re-aparición → fire.
    for _ in range(Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES + 1):
        counter.check_all({})

    # En el último, debería disparar el death-emit.
    assert counter.total_in == 1


# ---------------------------------------------------------------------------
# Stitching ratio: telemetría de fragmentación del tracker.
# ---------------------------------------------------------------------------


def test_stitching_ratio_1_when_each_track_emits_once():
    """1 track por persona, 1 evento por track → ratio = 1.0 (ideal)."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 250, 3000]])
    counter.check_all({1: track})  # entry
    track.positions.append(np.array([300, 350, 3000], dtype=float))
    counter.check_all({1: track})  # cross
    track.positions.append(np.array([300, 520, 3000], dtype=float))
    counter.check_all({1: track})  # exit, emit

    assert counter.total_in == 1
    assert counter.stitching_ratio == 1.0


def test_stitching_ratio_detects_fragmentation():
    """2 tracks distintos crossean → si solo emiten 1 conteo (uno cruzó,
    el otro se fragmentó), el ratio sube > 1."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Track 1: cruza y emite.
    t1 = _make_track(1, [[300, 250, 3000]])
    counter.check_all({1: t1})
    t1.positions.append(np.array([300, 350, 3000], dtype=float))
    counter.check_all({1: t1})
    t1.positions.append(np.array([300, 520, 3000], dtype=float))
    counter.check_all({1: t1})  # emit ingress

    # Track 2: ENTRA a la counting zone pero no cruza (fragmentación: se perdió antes
    # de cruzar, otro track tomó el relevo).
    t2 = _make_track(2, [[300, 250, 3000]])
    counter.check_all({2: t2})  # entry registrado → _seen_track_ids += 2

    # 2 tracks vistos, 1 conteo emitido → ratio = 2.0
    assert counter.stitching_ratio == 2.0


def test_stitching_ratio_zero_when_no_counts():
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    assert counter.stitching_ratio == 0.0


def test_reset_daily_clears_stitching_state():
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 250, 3000]])
    counter.check_all({1: track})
    assert len(counter._seen_track_ids) == 1
    counter.reset_daily()
    assert counter._seen_track_ids == set()
    assert counter._potential_deaths == {}


def test_death_emit_grace_default_from_class_constant():
    """Sin override, ``death_emit_grace_frames`` usa el default de clase."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    assert counter.death_emit_grace_frames == Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES


def test_death_emit_grace_override_via_constructor():
    """Pasar el param explícito sobreescribe el default — usado por main.py
    para sincronizar con ``tracker.adoption_window_frames + 2``."""
    counter = Counter(
        lines=[_line_h()], counting_zone=COUNTING_ZONE, death_emit_grace_frames=50,
    )
    assert counter.death_emit_grace_frames == 50


def test_death_emit_skipped_when_track_spawned_inside_counting_zone():
    """Guard 1: track que spawneó DENTRO de la counting zone (sin posición outside
    previa) cruza la línea por jitter y muere — NO debe emitir. Caso real:
    persona sentada cuya cabeza jitterea atravesando la línea."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Track aparece DIRECTAMENTE dentro de la counting zone (sin frame outside previo).
    track = _make_track(1, [[300, 250, 3000]])
    counter.check_all({1: track})  # entry, had_outside_pos=False
    track.positions.append(np.array([300, 350, 3000], dtype=float))
    counter.check_all({1: track})  # cross net=+1 (jitter de cabeza)

    # Desaparece. Sin guards, death-emit contaría 1 ingress. Con guard 1
    # (no_outside_history), NO emite.
    counter.check_all({})
    for _ in range(Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES + 1):
        counter.check_all({})
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_death_emit_skipped_when_visit_range_too_small():
    """Guard 2: track con outside_pos válido pero que solo se movió un
    poquito dentro de la counting zone (menos de MIN_VISIT_RANGE_FOR_DEATH_EMIT) — NO
    debe emitir. Filtra al lurker que entró, hizo un mini-cross por jitter
    cerca de la línea, y se quedó parado."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    track = _make_track(1, [[300, 150, 3000]])  # outside counting zone
    counter.check_all({1: track})  # last_outside_pos=(300,150)
    track.positions.append(np.array([300, 295, 3000], dtype=float))
    counter.check_all({1: track})  # entry justo arriba de la línea
    track.positions.append(np.array([300, 305, 3000], dtype=float))
    counter.check_all({1: track})  # cross net=+1, visit_y_range=10 (chico!)

    counter.check_all({})
    for _ in range(Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES + 1):
        counter.check_all({})
    # visit_y_range=10 < MIN=80 → skip.
    assert counter.total_in == 0


def test_death_emit_count_incremented_only_on_actual_emit():
    """Cada vez que _emit_on_death realmente emite (post-guards), incrementa
    el contador. Skipped emits NO suman."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    # Caso 1: track legítimo (outside_pos + range >= 80) → emit + count +=1.
    t1 = _make_track(1, [[300, 150, 3000]])
    counter.check_all({1: t1})
    t1.positions.append(np.array([300, 250, 3000]))
    counter.check_all({1: t1})
    t1.positions.append(np.array([300, 350, 3000]))
    counter.check_all({1: t1})
    counter.check_all({})
    for _ in range(Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES + 1):
        counter.check_all({})
    assert counter.death_emit_count == 1

    # Caso 2: track sin outside_pos (sitter) → guard 1 skip → count NO sube.
    t2 = _make_track(2, [[300, 250, 3000]])
    counter.check_all({2: t2})
    t2.positions.append(np.array([300, 350, 3000]))
    counter.check_all({2: t2})
    counter.check_all({})
    for _ in range(Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES + 1):
        counter.check_all({})
    assert counter.death_emit_count == 1  # no se incrementó


def test_reset_daily_clears_death_emit_count():
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    counter._death_emit_count = 5
    counter.reset_daily()
    assert counter.death_emit_count == 0


def test_min_visit_range_default_from_class_constant():
    """Sin override, ``min_visit_range_for_death_emit`` usa el default."""
    counter = Counter(lines=[_line_h()], counting_zone=COUNTING_ZONE)
    assert counter.min_visit_range_for_death_emit == Counter.DEFAULT_MIN_VISIT_RANGE_FOR_DEATH_EMIT


def test_min_visit_range_override_relaxes_guard_2():
    """Bajar el threshold (ej. 20 en vez de 80) deja pasar visits cortas
    que con el default fueron filtradas. Útil para sites con detector flakey
    donde la mayoría del visit cae en frames Kalman (no actualizan range)."""
    counter = Counter(
        lines=[_line_h()], counting_zone=COUNTING_ZONE, min_visit_range_for_death_emit=20.0,
    )
    track = _make_track(1, [[300, 150, 3000]])  # outside counting zone
    counter.check_all({1: track})
    track.positions.append(np.array([300, 290, 3000]))
    counter.check_all({1: track})  # entry justo arriba, had_outside_pos=True
    track.positions.append(np.array([300, 310, 3000]))
    counter.check_all({1: track})  # cruce real net=+1, visit_y_range=20

    counter.check_all({})
    for _ in range(Counter.DEFAULT_DEATH_EMIT_GRACE_FRAMES + 1):
        counter.check_all({})
    # visit_y_range=20 ≥ 20 → death-emit FIRES (con el default 80 hubiera
    # sido rechazado).
    assert counter.total_in == 1

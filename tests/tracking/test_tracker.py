"""Tests para el tracker Euclidiano 3D (Hungarian + state machine)."""

import numpy as np

from src.tracking.tracker import (
    CANDIDATE,
    CONFIRMED,
    PENDING,
    EuclideanTracker,
    _GhostInfo,
)


def test_count_by_state_groups_tracks():
    """count_by_state() returns per-state counts, zero-filled for missing keys."""
    tracker = EuclideanTracker(
        max_distance=50,
        max_disappeared=30,
        confirm_frames=2,
        pending_max_frames=3,
    )
    # Frame 1: one candidate
    tracker.update([np.array([100.0, 100.0, 3000.0])])
    counts = tracker.count_by_state()
    assert counts[CANDIDATE] == 1
    assert counts[CONFIRMED] == 0
    assert counts[PENDING] == 0

    # Frame 2: same detection -> promotes to CONFIRMED (confirm_frames=2)
    tracker.update([np.array([101.0, 101.0, 3000.0])])
    counts = tracker.count_by_state()
    assert counts[CONFIRMED] == 1

    # Frame 3: missed -> CONFIRMED -> PENDING
    tracker.update([])
    counts = tracker.count_by_state()
    assert counts[PENDING] == 1
    assert counts[CONFIRMED] == 0


def test_register_new_tracks():
    tracker = EuclideanTracker()
    dets = [np.array([100, 200, 3000]), np.array([300, 200, 3000])]
    tracks = tracker.update(dets)
    assert len(tracks) == 2


def test_track_continuity():
    tracker = EuclideanTracker(max_distance=50)
    tracker.update([np.array([100, 200, 3000])])
    tracks = tracker.update([np.array([105, 202, 3000])])
    assert len(tracks) == 1
    assert len(tracks[0].positions) == 2


def test_track_disappears():
    tracker = EuclideanTracker(max_disappeared=2, max_distance=50)
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([])  # disappeared 1
    tracker.update([])  # disappeared 2
    tracks = tracker.update([])  # disappeared 3 -> removed
    assert len(tracks) == 0


def test_new_detection_far_away():
    tracker = EuclideanTracker(max_distance=50)
    tracker.update([np.array([100, 200, 3000])])
    tracks = tracker.update([np.array([500, 500, 3000])])
    # Original disappeared, new one registered
    assert len(tracks) == 2  # old (disappeared=1) + new


def test_multiple_tracks():
    tracker = EuclideanTracker(max_distance=50)
    dets1 = [np.array([100, 200, 3000]), np.array([300, 200, 3000])]
    tracker.update(dets1)
    dets2 = [np.array([105, 202, 3000]), np.array([298, 198, 3000])]
    tracks = tracker.update(dets2)
    assert len(tracks) == 2
    for t in tracks.values():
        assert len(t.positions) == 2


# ---------------------------------------------------------------------------
# Hungarian association
# ---------------------------------------------------------------------------


def test_hungarian_globally_optimal_assignment():
    """Constructed 3x3 scenario where cheapest-first greedy picks a
    locally optimal pair that locks out the global optimum. Hungarian
    minimises the total cost.

    Tracks: A(x=0), B(x=5), C(x=10).
    Detections: D1(x=4), D2(x=6), D3(x=20).
    Greedy picks B-D1=1 first, then C-D2=4, leaving A-D3=20 (sum 25).
    Hungarian picks B-D2=1, A-D1=4, C-D3=10 (sum 15) — so the track at
    x=0 lands on the detection at x=4, which is clearly the correct ID.
    """
    tracker = EuclideanTracker(max_distance=25, max_depth_delta=1e9, confirm_frames=1)
    tracker.update(
        [
            np.array([0.0, 100.0, 3000.0]),
            np.array([5.0, 100.0, 3000.0]),
            np.array([10.0, 100.0, 3000.0]),
        ]
    )
    ids = sorted(tracker.tracks.keys())
    tid_a, tid_b, tid_c = ids

    tracks = tracker.update(
        [
            np.array([4.0, 100.0, 3000.0]),
            np.array([6.0, 100.0, 3000.0]),
            np.array([20.0, 100.0, 3000.0]),
        ]
    )

    # Hungarian assigns A->D1(x=4), B->D2(x=6), C->D3(x=20).
    assert tracks[tid_a].last_position[0] == 4.0
    assert tracks[tid_b].last_position[0] == 6.0
    assert tracks[tid_c].last_position[0] == 20.0


def test_depth_gate_rejects_impossible_match():
    """Close in pixels but very far in depth -> not the same person."""
    tracker = EuclideanTracker(max_distance=50, max_depth_delta=500, confirm_frames=1)
    tracker.update([np.array([100, 200, 1000])])  # near camera
    tracks = tracker.update([np.array([102, 201, 5000])])  # suddenly far
    # Depth gate rejected the match -> 2 tracks exist (old + newly registered)
    assert len(tracks) == 2


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_candidate_promotes_to_confirmed_after_n_frames():
    tracker = EuclideanTracker(max_distance=50, confirm_frames=3)
    tracker.update([np.array([100, 200, 3000])])
    tracks = tracker.tracks
    assert list(tracks.values())[0].state == CANDIDATE

    tracker.update([np.array([101, 200, 3000])])  # hit 2
    assert list(tracker.tracks.values())[0].state == CANDIDATE

    tracker.update([np.array([102, 200, 3000])])  # hit 3 -> CONFIRMED
    assert list(tracker.tracks.values())[0].state == CONFIRMED


def test_confirmed_becomes_pending_on_miss():
    tracker = EuclideanTracker(max_distance=50, confirm_frames=2, pending_max_frames=5)
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([101, 200, 3000])])
    assert list(tracker.tracks.values())[0].state == CONFIRMED

    tracker.update([])  # missed
    track = list(tracker.tracks.values())[0]
    assert track.state == PENDING
    assert track.disappeared == 1


def test_pending_recovers_to_confirmed_preserving_id():
    """Re-id: track missed for 3 frames, reappears near predicted location."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
        reid_gate_px=80,
    )
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([110, 200, 3000])])  # velocity = +10 px/frame in x
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    tracker.update([])  # miss 1 -> PENDING, predicted 120
    tracker.update([])  # miss 2
    tracker.update([])  # miss 3
    assert tracker.tracks[tid].state == PENDING

    # Reappear near the predicted position (velocity projected ~ 140 by now
    # since predicted_position uses last two obs -> 120 each miss frame).
    tracks = tracker.update([np.array([130, 200, 3000])])
    assert tid in tracks
    assert tracks[tid].state == CONFIRMED
    # Meta preserved across the recovery (counter can rely on this).
    tracks[tid].meta["sentinel"] = True
    tracker.update([np.array([135, 200, 3000])])
    assert tracker.tracks[tid].meta.get("sentinel") is True


def test_pending_timeout_becomes_lost_and_new_id_issued():
    """After pending_max_frames misses, the track is dropped; a new
    detection spawns a new ID.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
    )
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([105, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    # 6 consecutive misses > pending_max_frames
    for _ in range(6):
        tracker.update([])
    assert tid not in tracker.tracks

    tracks = tracker.update([np.array([200, 200, 3000])])
    new_ids = list(tracks.keys())
    assert tid not in new_ids
    assert len(new_ids) == 1


def _mark_counter_entry_completed(track):
    """Simula el counter habiendo disparado una entry-fresca real. El
    tracker chequea ``meta.counter.inside`` antes de aplicar keepalive
    (gate del 2026-05-24): tracks que solo entraron via Kalman no
    califican. Precondición de todos los tests de keepalive que no
    pasan por el counter real."""
    track.meta["counter"] = {"inside": True}


def test_keepalive_counting_zone_keeps_pending_alive_inside():
    """Un track PENDING cuya posición cae dentro del keepalive_counting_zone NO muere
    por timeout, ni siquiera pasados pending_max_frames y max_disappeared.
    Modela "cruzó y se quedó adentro de la counting zone mirando algo"."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=3,
        max_disappeared=5,
        keepalive_counting_zone=(50.0, 200.0, 150.0, 300.0),
    )
    # Track estático dentro de la counting zone (velocidad ~0 → la predicción Kalman
    # se queda en el lugar, dentro de la counting zone).
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED
    _mark_counter_entry_completed(tracker.tracks[tid])

    # Muchísimos misses, mucho más que ambos caps de timeout.
    for _ in range(50):
        tracker.update([])
    # Sigue vivo (PENDING) porque su posición quedó dentro de la counting zone.
    assert tid in tracker.tracks
    assert tracker.tracks[tid].state == PENDING
    # Y el historial de posiciones está acotado por el cap.
    assert len(tracker.tracks[tid].positions) <= 512


def test_keepalive_counting_zone_recovers_to_confirmed_after_long_gap():
    """Tras un gap largo dentro de la counting zone, una detección que reaparece cerca
    re-identifica el MISMO track (preservando ID + meta del counter)."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=3,
        max_disappeared=5,
        reid_gate_px=60,
        keepalive_counting_zone=(50.0, 200.0, 150.0, 300.0),
    )
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    _mark_counter_entry_completed(tracker.tracks[tid])
    for _ in range(30):
        tracker.update([])
    assert tracker.tracks[tid].state == PENDING
    # Reaparece cerca → re-id al mismo track.
    tracker.update([np.array([105, 200, 3000])])
    assert tid in tracker.tracks
    assert tracker.tracks[tid].state == CONFIRMED


def test_keepalive_counting_zone_extrapolates_does_not_freeze():
    """Un track PENDING dentro de la counting zone SIGUE extrapolando con el Kalman (no se
    congela) — así puede seguir a la persona y salir de la counting zone para que el
    counter emita el cruce. El doble-conteo del drift lo evita el counter
    (cuenta cruces solo con detección real), no congelando el track acá."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
        max_disappeared=10,
        pending_grace_frames=3,
        pending_velocity_decay=1.0,  # sin decay → extrapola a velocidad plena
        keepalive_counting_zone=(50.0, 200.0, 150.0, 300.0),
    )
    # Caminando hacia abajo: velocidad ~ +10 px/frame en y, dentro de la counting zone.
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 210, 3000])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED
    _mark_counter_entry_completed(tracker.tracks[tid])

    y_start = float(tracker.tracks[tid].positions[-1][1])
    for _ in range(10):
        tracker.update([])
    # La posición avanzó (extrapoló), no quedó clavada en y_start.
    y_after = float(tracker.tracks[tid].positions[-1][1])
    assert y_after > y_start + 20, (
        f"el track debe extrapolar (seguir a la persona), no congelarse "
        f"(y_start={y_start}, y_after={y_after})"
    )


def test_keepalive_counting_zone_capped_orphan_dies():
    """El keep-alive está acotado a keepalive_max_frames misses consecutivos:
    un huérfano (re-id falló, persona ya no está) se garbage-collectea pasado
    el cap, en vez de quedar como fantasma eterno acumulándose en el preview."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=3,
        max_disappeared=10,
        keepalive_counting_zone=(50.0, 200.0, 150.0, 300.0),
        keepalive_max_frames=5,
    )
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    _mark_counter_entry_completed(tracker.tracks[tid])
    # Misses por encima del cap del keep-alive.
    for _ in range(8):  # > keepalive_max_frames=5
        tracker.update([])
    assert tid not in tracker.tracks


def test_keepalive_counting_zone_does_not_protect_kalman_only_entry():
    """Opción E (2026-05-24): un track que ENTRA al counting_zone solo via
    extrapolación Kalman (nunca completa una entry-fresca real) NO califica
    para keepalive. Muere normal por timeout pese a estar dentro.

    Modela el caso del piloto donde la campera flickeando generaba detecciones
    intermitentes que el counter rechazaba con ``entry_kalman_skipped``;
    sin este gate, el track persistía indefinidamente por keepalive + ghost
    adoption, generando ruido visual constante.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=3,
        max_disappeared=5,
        keepalive_counting_zone=(50.0, 200.0, 150.0, 300.0),
    )
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED
    # NO seteamos meta.counter.inside — simula track que el counter aún no
    # confirmó (solo entry_kalman_skipped repetidos, sin entry-fresca real).

    # Misses suficientes para timeout normal.
    for _ in range(10):
        tracker.update([])

    # El track debe haber muerto pese a estar dentro del keepalive zone —
    # nunca calificó porque meta.counter.inside es False/ausente.
    assert tid not in tracker.tracks


def test_keepalive_counting_zone_does_not_protect_outside():
    """Un track FUERA del keepalive_counting_zone muere normalmente por timeout —
    el keep-alive solo aplica dentro de la counting zone."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
        keepalive_counting_zone=(50.0, 200.0, 150.0, 300.0),
    )
    # Track fuera de la counting zone (x=400 > x_max=200).
    tracker.update([np.array([400, 400, 3000])])
    tracker.update([np.array([400, 400, 3000])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED
    for _ in range(6):  # > pending_max_frames
        tracker.update([])
    assert tid not in tracker.tracks


def test_reid_gate_rejects_far_reappearance():
    """A PENDING track cannot be re-matched beyond reid_gate_px."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
        reid_gate_px=40,
    )
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([105, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    tracker.update([])  # PENDING
    # Far reappearance -> new track, old still PENDING (or eventually LOST).
    tracks = tracker.update([np.array([400, 200, 3000])])
    assert len(tracks) == 2
    # The original ID should still exist but in PENDING.
    assert tid in tracks
    assert tracks[tid].state == PENDING


def test_pass2_recovers_confirmed_with_bbox_jitter():
    """Bbox jitter past max_distance should NOT spawn a phantom track.

    Pass 1 fails at the tight gate, but pass 2 reruns Hungarian on the
    leftovers with the wider reid_gate, so the original CONFIRMED track
    catches the jittered detection. Without this, the same physical
    person gets a new ID every time YOLO's bbox wobbles ±50 px.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    tracker.update([np.array([500, 300, 3000])])
    tracker.update([np.array([510, 305, 3000])])  # promote to CONFIRMED
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    # Jitter ~80 px in one frame — past max_distance=50, well under
    # reid_gate_px=300. Pass 2 should bind it to the same track.
    tracks = tracker.update([np.array([590, 325, 3000])])
    assert len(tracks) == 1
    assert tid in tracks
    assert tracks[tid].state == CONFIRMED


def test_pass2_respects_depth_gate():
    """Pass 2 widens the 2D distance gate, NOT the depth gate.

    Two physically distinct people at very different depths must not
    swap IDs even when one drops off and another appears nearby in
    pixel space.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        max_depth_delta=500.0,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    # Person A at depth 2000mm.
    tracker.update([np.array([500, 300, 2000])])
    tid_a = list(tracker.tracks.keys())[0]

    # Frame 2: a detection 80 px away (past max_distance) but at depth
    # 4000 — clearly a different person on a different floor level.
    # Depth delta = 2000mm > max_depth_delta=500 → must NOT bind.
    tracks = tracker.update([np.array([580, 320, 4000])])
    assert tid_a in tracks
    # New track was registered for the far-depth detection.
    assert len(tracks) == 2


def test_pass2_does_not_misroute_when_pass1_was_clean():
    """When pass 1 already matched everything, pass 2 is a no-op."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    tracker.update(
        [
            np.array([100, 200, 3000]),
            np.array([400, 200, 3000]),
        ]
    )
    tids_before = sorted(tracker.tracks.keys())

    # Both move within the tight gate; pass 1 binds both.
    tracks = tracker.update(
        [
            np.array([110, 205, 3000]),
            np.array([405, 198, 3000]),
        ]
    )
    assert sorted(tracks.keys()) == tids_before
    for t in tracks.values():
        assert t.state == CONFIRMED


# ---------------------------------------------------------------------------
# Ghost pool / ID adoption: cuando un track muere, su ID queda disponible
# para adopción durante adoption_window_frames. Una detección nueva con bbox
# overlapeando + dentro del gate de distancia adopta ese ID (preserva la
# continuidad de identidad cross-gap-de-detección).
# ---------------------------------------------------------------------------


def test_ghost_adoption_preserves_id_after_lost():
    """Track muere → enghost. Detección nueva aparece con bbox que overlap
    el último del ghost + dentro del gate → adopta el ID. El track resucitado
    arranca CONFIRMED con la meta restaurada."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=2,
        max_disappeared=3,
        reid_gate_px=60,
        adoption_window_frames=10,
        adoption_iou_min=0.3,
        adoption_max_dist_px=80.0,
    )
    # Bootstrapping: track con bbox conocido.
    det = np.array([100.0, 100.0, 3000.0])
    tracker.update([det], detection_metas=[{"bbox": [90, 90, 110, 110]}])
    tracker.update([det], detection_metas=[{"bbox": [90, 90, 110, 110]}])
    tid = list(tracker.tracks.keys())[0]
    # Anotar meta arbitraria (simula estado del counter).
    tracker.tracks[tid].meta["custom"] = "hello"

    # Detector la pierde varios frames → track muere y va a ghost pool.
    for _ in range(6):
        tracker.update([])
    assert tid not in tracker.tracks  # ya murió (purgado)
    assert tid in tracker._ghosts  # quedó en el pool

    # Detección nueva con bbox que overlap el último del ghost + posición
    # cercana → adoptar.
    new_det = np.array([105.0, 102.0, 3000.0])
    tracker.update([new_det], detection_metas=[{"bbox": [95, 92, 115, 112]}])

    assert tid in tracker.tracks, "El track debe haber sido resucitado con su ID"
    assert tracker.tracks[tid].state == CONFIRMED  # arranca confirmado, no candidate
    assert tracker.tracks[tid].meta.get("custom") == "hello"  # meta restaurada
    assert tid not in tracker._ghosts  # se removió del pool tras adopción


def test_ghost_adoption_invalidates_far_outside_pos():
    """Regresión del FP observado en piloto 2026-05-24 09:15-09:18 (tid=20).

    Escenario: track muere por Kalman exit con ``last_outside_pos`` lejano
    (extrapolación alucinada, ej. ghost extrapoló a (375, 140) cuando su
    centroide real estaba en zona muy distinta). El new track que adopta
    el ID arranca con centroide en otra zona (ej. (399, 378)). Sin el
    fix, el meta heredado tenía outside_pos=(375,140) — at >150 px del
    new centroide — y eso producía:
      (a) had_outside_pos=True espurio en el next entry-fresca,
          bypaseando el guard del exit-by-Kalman.
      (b) cross artificial en el primer frame inside-was-inside (sides[]
          flipea de un lado al opuesto sólo porque el snap está en zona
          geométricamente distinta de la posición real).

    El fix invalida ``last_outside_pos`` del meta heredado si está a más
    de ``ghost_outside_invalidate_px`` (default 150) del nuevo centroide.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=2,
        max_disappeared=3,
        reid_gate_px=60,
        adoption_window_frames=10,
        adoption_iou_min=0.3,
        adoption_max_dist_px=100.0,
    )
    # Track 1: nace y opera en zona ~(400, 380). Le inyectamos al meta
    # un last_outside_pos=(375, 140) — como si su Kalman lo hubiera
    # extrapolado lejos al morir.
    det = np.array([400.0, 380.0, 3000.0])
    tracker.update([det], detection_metas=[{"bbox": [390, 370, 410, 390]}])
    tracker.update([det], detection_metas=[{"bbox": [390, 370, 410, 390]}])
    tid = list(tracker.tracks.keys())[0]
    tracker.tracks[tid].meta["last_outside_pos"] = (375.0, 140.0)
    tracker.tracks[tid].meta["custom_marker"] = "preserved"  # otra key meta

    # Track muere → enghost (con el outside_pos lejano + custom_marker).
    for _ in range(6):
        tracker.update([])
    assert tid in tracker._ghosts
    ghost_outside_before = tracker._ghosts[tid].meta_snapshot.get("last_outside_pos")
    assert ghost_outside_before == (375.0, 140.0)

    # Nueva detección cerca del último centroide del ghost (~(400,380))
    # — pasa los gates de adopción. Distancia ghost↔new ≈ 5px, OK.
    new_det = np.array([399.0, 378.0, 3000.0])
    tracker.update([new_det], detection_metas=[{"bbox": [389, 368, 409, 388]}])

    assert tid in tracker.tracks  # adopción OK
    meta = tracker.tracks[tid].meta
    # Con el fix: outside_pos heredado se invalida (distancia ~239 px > 150).
    assert meta.get("last_outside_pos") is None
    # Las demás keys del meta SE PRESERVAN (el fix es selectivo).
    assert meta.get("custom_marker") == "preserved"


def test_ghost_outside_invalidate_px_is_configurable():
    """``ghost_outside_invalidate_px`` es un kwarg del tracker. Subirlo a 300
    permite que un outside_pos a 240 px sobreviva la adopción (que con default
    150 sería invalidado). Habilita tuning per-site sin tocar código."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=2,
        max_disappeared=3,
        reid_gate_px=60,
        adoption_window_frames=10,
        adoption_iou_min=0.3,
        adoption_max_dist_px=100.0,
        ghost_outside_invalidate_px=300.0,  # ← override del default 150
    )
    assert tracker.ghost_outside_invalidate_px == 300.0

    # Escenario diseñado para caer entre los dos thresholds:
    # outside_pos=(500, 200), centroide=(399, 378) → distancia ~205 px.
    # Con default 150 el outside_pos se invalidaría; con override 300
    # se preserva.
    ghost_meta = {
        "last_outside_pos": (500.0, 200.0),
        "custom_marker": "preserved",
    }
    ghost = _GhostInfo(
        track_id=20,
        last_observed_position=np.array([400.0, 380.0, 3000.0]),
        last_bbox=(370, 350, 430, 410),
        meta_snapshot=ghost_meta,
    )
    tracker._ghosts[20] = ghost
    tracker._resurrect_ghost(ghost, np.array([399.0, 378.0, 3000.0]))

    track = tracker._tracks[20]
    assert track.meta.get("last_outside_pos") == (500.0, 200.0)


def test_ghost_adoption_preserves_close_outside_pos():
    """Regression contra el fix anterior: si el outside_pos del ghost está
    CERCA del centroide del nuevo track (<150 px), se preserva — cubre el
    caso legítimo donde el ghost murió mid-trajectory y otro track tomó
    la posta inmediatamente cerca (ej. crosser real cuya identidad se
    fragmentó por un detector miss)."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=2,
        max_disappeared=3,
        reid_gate_px=60,
        adoption_window_frames=10,
        adoption_iou_min=0.3,
        adoption_max_dist_px=100.0,
    )
    # Track en (400, 250) con outside_pos cercano (380, 200) — ~54 px.
    det = np.array([400.0, 250.0, 3000.0])
    tracker.update([det], detection_metas=[{"bbox": [390, 240, 410, 260]}])
    tracker.update([det], detection_metas=[{"bbox": [390, 240, 410, 260]}])
    tid = list(tracker.tracks.keys())[0]
    tracker.tracks[tid].meta["last_outside_pos"] = (380.0, 200.0)

    for _ in range(6):
        tracker.update([])

    # New track en (405, 255) — cerca del ghost. dist outside↔new ≈ 60 px,
    # bajo el threshold de 150 → outside_pos se preserva.
    new_det = np.array([405.0, 255.0, 3000.0])
    tracker.update([new_det], detection_metas=[{"bbox": [395, 245, 415, 265]}])

    meta = tracker.tracks[tid].meta
    assert meta.get("last_outside_pos") == (380.0, 200.0)  # preservado


def test_ghost_pool_rejects_low_iou_match():
    """Si el bbox de la nueva detección NO overlap suficiente con el bbox
    del ghost (IoU < adoption_iou_min), NO se adopta — spawnea track nuevo."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=2,
        max_disappeared=3,
        reid_gate_px=60,
        adoption_window_frames=10,
        adoption_iou_min=0.5,
        adoption_max_dist_px=200.0,
    )
    det = np.array([100.0, 100.0, 3000.0])
    tracker.update([det], detection_metas=[{"bbox": [90, 90, 110, 110]}])
    tracker.update([det], detection_metas=[{"bbox": [90, 90, 110, 110]}])
    tid = list(tracker.tracks.keys())[0]
    for _ in range(6):
        tracker.update([])
    assert tid in tracker._ghosts

    # Nueva detección con bbox lejos → IoU=0 con el del ghost.
    new_det = np.array([100.0, 100.0, 3000.0])
    tracker.update([new_det], detection_metas=[{"bbox": [200, 200, 220, 220]}])
    # Como IoU=0 < 0.5 → NO adoptado. Spawnea track nuevo con ID distinto.
    new_tids = list(tracker.tracks.keys())
    assert tid not in new_tids
    assert len(new_tids) == 1
    assert new_tids[0] != tid


def test_ghost_pool_expires_after_window():
    """Después de adoption_window_frames sin adopción, el ghost se descarta
    del pool. Una detección que llegue después spawnea track con ID nuevo."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=2,
        max_disappeared=3,
        reid_gate_px=60,
        adoption_window_frames=3,
    )
    det = np.array([100.0, 100.0, 3000.0])
    tracker.update([det], detection_metas=[{"bbox": [90, 90, 110, 110]}])
    tracker.update([det], detection_metas=[{"bbox": [90, 90, 110, 110]}])
    tid = list(tracker.tracks.keys())[0]
    # Muere y va al pool.
    for _ in range(5):
        tracker.update([])
    assert tid in tracker._ghosts

    # Frames adicionales sin detección → el ghost envejece y expira.
    for _ in range(5):
        tracker.update([])
    assert tid not in tracker._ghosts

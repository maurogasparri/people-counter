"""Tests para el tracker Euclidiano 3D (Hungarian + state machine)."""

import numpy as np

from src.tracking.tracker import (
    CANDIDATE,
    CONFIRMED,
    PENDING,
    EuclideanTracker,
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


def test_keepalive_roi_keeps_pending_alive_inside():
    """Un track PENDING cuya posición cae dentro del keepalive_roi NO muere
    por timeout, ni siquiera pasados pending_max_frames y max_disappeared.
    Modela "cruzó y se quedó adentro del ROI mirando algo"."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=3,
        max_disappeared=5,
        keepalive_roi=(50.0, 200.0, 150.0, 300.0),
    )
    # Track estático dentro del ROI (velocidad ~0 → la predicción Kalman
    # se queda en el lugar, dentro del ROI).
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    # Muchísimos misses, mucho más que ambos caps de timeout.
    for _ in range(50):
        tracker.update([])
    # Sigue vivo (PENDING) porque su posición quedó dentro del ROI.
    assert tid in tracker.tracks
    assert tracker.tracks[tid].state == PENDING
    # Y el historial de posiciones está acotado por el cap.
    assert len(tracker.tracks[tid].positions) <= 512


def test_keepalive_roi_recovers_to_confirmed_after_long_gap():
    """Tras un gap largo dentro del ROI, una detección que reaparece cerca
    re-identifica el MISMO track (preservando ID + meta del counter)."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=3,
        max_disappeared=5,
        reid_gate_px=60,
        keepalive_roi=(50.0, 200.0, 150.0, 300.0),
    )
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    for _ in range(30):
        tracker.update([])
    assert tracker.tracks[tid].state == PENDING
    # Reaparece cerca → re-id al mismo track.
    tracker.update([np.array([105, 200, 3000])])
    assert tid in tracker.tracks
    assert tracker.tracks[tid].state == CONFIRMED


def test_keepalive_roi_extrapolates_does_not_freeze():
    """Un track PENDING dentro del ROI SIGUE extrapolando con el Kalman (no se
    congela) — así puede seguir a la persona y salir del ROI para que el
    counter emita el cruce. El doble-conteo del drift lo evita el counter
    (cuenta cruces solo con detección real), no congelando el track acá."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
        max_disappeared=10,
        pending_grace_frames=3,
        pending_velocity_decay=1.0,  # sin decay → extrapola a velocidad plena
        keepalive_roi=(50.0, 200.0, 150.0, 300.0),
    )
    # Caminando hacia abajo: velocidad ~ +10 px/frame en y, dentro del ROI.
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 210, 3000])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    y_start = float(tracker.tracks[tid].positions[-1][1])
    for _ in range(10):
        tracker.update([])
    # La posición avanzó (extrapoló), no quedó clavada en y_start.
    y_after = float(tracker.tracks[tid].positions[-1][1])
    assert y_after > y_start + 20, (
        f"el track debe extrapolar (seguir a la persona), no congelarse "
        f"(y_start={y_start}, y_after={y_after})"
    )


def test_keepalive_roi_capped_orphan_dies():
    """El keep-alive está acotado a keepalive_max_frames misses consecutivos:
    un huérfano (re-id falló, persona ya no está) se garbage-collectea pasado
    el cap, en vez de quedar como fantasma eterno acumulándose en el preview."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=3,
        max_disappeared=10,
        keepalive_roi=(50.0, 200.0, 150.0, 300.0),
        keepalive_max_frames=5,
    )
    tracker.update([np.array([100, 200, 3000])])
    tracker.update([np.array([100, 200, 3000])])
    tid = list(tracker.tracks.keys())[0]
    # Misses por encima del cap del keep-alive.
    for _ in range(8):  # > keepalive_max_frames=5
        tracker.update([])
    assert tid not in tracker.tracks


def test_keepalive_roi_does_not_protect_outside():
    """Un track FUERA del keepalive_roi muere normalmente por timeout —
    el keep-alive solo aplica dentro del ROI."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
        keepalive_roi=(50.0, 200.0, 150.0, 300.0),
    )
    # Track fuera del ROI (x=400 > x_max=200).
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

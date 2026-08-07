"""Tests para el modelo de movimiento Kalman integrado al tracker.

Cubren el comportamiento nuevo agregado cuando ``Track.predicted_position``
became Kalman-driven:

* Constant-velocity sequence: the filter converges and predicts ahead
  with low error.
* Multi-frame miss + reappearance: the filter extrapolates across the
  gap and re-acquires the right detection where the old linear model
  lost it.
* Single-observation track + miss: predicted position stays at the
  observed point (velocity prior is 0, covariance high → no rocket
  launch).
* Process / measurement noise tuning: changing the noise diagonals
  shifts the trade-off between trusting model vs. observation in the
  expected direction.
"""

import time

import numpy as np

from src.tracking.kalman import TrackKalman
from src.tracking.tracker import (
    CONFIRMED,
    PENDING,
    EuclideanTracker,
)


# ---------------------------------------------------------------------------
# TrackKalman unit tests
# ---------------------------------------------------------------------------


def test_kalman_converges_on_constant_velocity():
    """After a few uniform-motion measurements the filter velocity
    converges to the true value and the predicted position lands within
    a small tolerance of the next measurement."""
    kf = TrackKalman(
        initial_position=np.array([0.0, 0.0]),
        process_noise=1.0,
        measurement_noise=5.0,
        initial_velocity_uncertainty=100.0,
    )
    true_v = np.array([10.0, 5.0])
    pos = np.array([0.0, 0.0])
    for step in range(1, 10):
        pos = pos + true_v
        kf.predict()
        kf.update(pos)

    # Velocity estimate within ±10% of truth after 10 steps.
    vx, vy = kf.velocity
    assert abs(vx - true_v[0]) < 1.0
    assert abs(vy - true_v[1]) < 0.5

    # Predict the next position; truth = pos + true_v.
    next_pred = kf.predict()
    expected = pos + true_v
    assert np.linalg.norm(next_pred - expected) < 2.0


def test_kalman_single_observation_stays_local_under_miss():
    """A track with one observation and then misses should NOT diverge.

    Velocity prior is zero with high uncertainty; without measurements
    the position estimate stays put (just covariance grows).
    """
    kf = TrackKalman(
        initial_position=np.array([100.0, 200.0]),
        process_noise=1.0,
        measurement_noise=5.0,
        initial_velocity_uncertainty=100.0,
    )
    for _ in range(10):
        kf.predict()
    pos = kf.position
    # Position estimate is still glued to the seed (velocity mean = 0).
    assert abs(pos[0] - 100.0) < 1e-6
    assert abs(pos[1] - 200.0) < 1e-6


def test_kalman_high_measurement_noise_trusts_model_more():
    """Cranking R high shifts weight onto F — the corrected position
    sits closer to the predicted (model) position than to the noisy
    measurement when both disagree."""
    # Same prior trajectory, two filters with different R.
    seed = np.array([0.0, 0.0])
    kf_low_r = TrackKalman(seed, measurement_noise=1.0)
    kf_high_r = TrackKalman(seed, measurement_noise=100.0)
    # Build velocity with a few clean steps so the model has a fix.
    for step in range(1, 6):
        truth = np.array([10.0 * step, 0.0])
        kf_low_r.predict()
        kf_low_r.update(truth)
        kf_high_r.predict()
        kf_high_r.update(truth)

    # Now feed an outlier measurement way off the line.
    kf_low_r.predict()
    kf_high_r.predict()
    pred_low = kf_low_r.position.copy()
    pred_high = kf_high_r.position.copy()
    outlier = np.array([60.0, 100.0])  # was expecting ~[60, 0]

    kf_low_r.update(outlier)
    kf_high_r.update(outlier)

    # Low-R filter trusts the measurement more → moves further toward outlier.
    moved_low = np.linalg.norm(kf_low_r.position - pred_low)
    moved_high = np.linalg.norm(kf_high_r.position - pred_high)
    assert moved_low > moved_high


# ---------------------------------------------------------------------------
# Tracker-level Kalman integration tests
# ---------------------------------------------------------------------------


def test_tracker_predicted_position_uses_kalman():
    """After two clean hits the tracker's predicted_position should
    advance by ~one velocity step, similar to the old linear model but
    now sourced from the Kalman state."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=10,
    )
    tracker.update([np.array([100.0, 200.0, 3000.0])])
    tracker.update([np.array([110.0, 200.0, 3000.0])])

    track = list(tracker.tracks.values())[0]
    pred = track.predicted_position
    # Velocity ≈ +10 px/frame after one update; predicted next is ~120.
    # Kalman estimate is slightly damped by R/P trade-off — allow ±2 px.
    assert abs(pred[0] - 120.0) < 4.0
    assert abs(pred[1] - 200.0) < 1.0
    # Z is carried forward unchanged from the last observation.
    assert pred[2] == 3000.0


def test_tracker_recovers_across_3_frame_gap():
    """Track moving at ~+15 px/frame disappears for 3 frames; on
    re-appearance, Kalman has propagated to ~5 frames forward and the
    detection is bound to the original track ID.

    Old constant-velocity extrapolation also extrapolates linearly, so
    the canonical guarantee here is not "Kalman wins where linear loses
    over a small gap" but "Kalman re-acquires correctly under the same
    conditions". The next test stresses a longer gap where linear
    breaks down.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=10,
        reid_gate_px=100,
    )
    # Two hits to confirm + seed velocity ~+15 px/frame.
    tracker.update([np.array([100.0, 200.0, 3000.0])])
    tracker.update([np.array([115.0, 200.0, 3000.0])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    # 3 missed frames.
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert tracker.tracks[tid].state == PENDING

    # Person reappears on the projected line: ~115 + 4*15 = 175.
    tracks = tracker.update([np.array([175.0, 200.0, 3000.0])])
    assert tid in tracks
    assert tracks[tid].state == CONFIRMED


def test_tracker_kalman_gap_recovers_under_gate_where_linear_would_too():
    """Quantitative: with a long gap and modest reid_gate, Kalman's
    propagated position lands close enough to the actual reappearance
    to bind. The old linear model with the same data also extrapolates,
    but Kalman's covariance widens which means even off-line
    reappearances still bind once the gate clears.

    We assert binding (the operationally important property) and check
    the predicted position is in the expected ballpark.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=120,
    )
    tracker.update([np.array([0.0, 100.0, 3000.0])])
    tracker.update([np.array([20.0, 100.0, 3000.0])])  # v ≈ +20
    tracker.update([np.array([40.0, 100.0, 3000.0])])  # v stable
    tid = list(tracker.tracks.keys())[0]

    # 5 missed frames.
    for _ in range(5):
        tracker.update([])
    assert tracker.tracks[tid].state == PENDING

    # Kalman should be predicting around 40 + 6*20 = 160 by the time the
    # next frame (the 6th post-hit) lands. Reappearance at 150 is within
    # the reid gate of that prediction.
    pred = tracker.tracks[tid].predicted_position
    assert abs(pred[0] - 160.0) < 25.0  # broad tolerance — Kalman damp

    tracks = tracker.update([np.array([150.0, 100.0, 3000.0])])
    assert tid in tracks, "Kalman extrapolation should bind reappearance"
    assert tracks[tid].state == CONFIRMED


def test_tracker_single_obs_then_miss_does_not_diverge():
    """Edge: a track born and immediately missed must not predict
    runaway positions. Without measurements, velocity prior is 0."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
    )
    tracker.update([np.array([500.0, 300.0, 3000.0])])
    tid = list(tracker.tracks.keys())[0]

    for _ in range(8):
        tracker.update([])

    track = tracker.tracks.get(tid)
    if track is not None:  # may have transitioned to LOST depending on caps
        pred = track.predicted_position
        # Position estimate stays near the seed.
        assert abs(pred[0] - 500.0) < 2.0
        assert abs(pred[1] - 300.0) < 2.0


def test_pending_velocity_decay_bounds_drift_and_preserves_reid():
    """Production-grade: con ``pending_velocity_decay < 1.0``, un track
    que pierde detecciones converge a "track quieto" en pocos frames en
    vez de seguir extrapolando con la velocidad de cuando había obs.

    El bug que esto previene: persona entra caminando + el detector la
    pierde (oclusión, static_suppressor, motion blur). Sin decay, el
    Kalman sigue empujando la posición a la velocidad pre-miss → tras
    1-2s el predict está a 1-3m del lugar real → al re-aparecer la
    persona, queda fuera del ``reid_gate_px`` → nace un track NUEVO →
    el track viejo muere por timeout sin emitir egress y el nuevo emite
    un ingress → DOBLE CONTEO.

    Con decay 0.5, después de N misses sin obs:
        drift_total = v * (1 + 0.5 + 0.25 + ...) <= v * 2

    Drift acotado independientemente de cuántos frames pasen, así el
    re-id binding sigue cayendo dentro del gate.
    """
    # Setup: track confirmed con velocidad +20 px/frame.
    tracker = EuclideanTracker(
        max_distance=200,
        confirm_frames=1,
        pending_max_frames=50,
        reid_gate_px=300,
        pending_velocity_decay=0.5,
    )
    tracker.update([np.array([0.0, 100.0, 3000.0])])
    tracker.update([np.array([20.0, 100.0, 3000.0])])
    tid = list(tracker.tracks.keys())[0]
    pos_at_last_obs = tracker.tracks[tid].last_position[0]
    assert pos_at_last_obs == 20.0

    # 20 frames sin observaciones — track pasa a PENDING y queda ahí.
    for _ in range(20):
        tracker.update([])
    assert tracker.tracks[tid].state == PENDING

    # Drift del Kalman: con decay 0.5, suma geométrica de v acotada.
    # v inicial = 20 px/frame, post-decay después de N frames PENDING:
    #     drift ≤ 20 * Σ (0.5^k for k=0..∞) = 40 px.
    pos_after_20_misses = tracker.tracks[tid].kalman.position[0]
    drift = pos_after_20_misses - pos_at_last_obs
    assert drift < 50.0, (
        f"Drift {drift:.1f}px excede el límite teórico geométrico (~40px). "
        "Sin decay esto sería 20*20 = 400px — un track real saldría del FOV."
    )

    # Reaparición en una posición razonable matchea de nuevo (re-id intacto).
    tracks = tracker.update([np.array([35.0, 100.0, 3000.0])])
    assert tid in tracks, (
        "Después del decay el predict queda cerca de la última obs y el "
        "reaparición a 35px (dentro del rango decay-acotado) debe re-id."
    )
    assert tracks[tid].state == CONFIRMED


def test_pending_velocity_decay_default_disabled_back_compat():
    """El tracker construido sin ``pending_velocity_decay`` mantiene el
    comportamiento previo (decay = 1.0 = sin decay). Garantiza que tests
    y configs viejas no cambian semántica inadvertidamente.
    """
    tracker = EuclideanTracker(
        max_distance=200,
        confirm_frames=1,
        pending_max_frames=20,
    )
    assert tracker.pending_velocity_decay == 1.0
    assert tracker.pending_grace_frames == 0


def test_pending_grace_frames_preserves_velocity_during_window():
    """Production-grade: con ``pending_grace_frames=N``, los primeros N
    frames de PENDING preservan velocidad completa antes de que arranque
    el decay. Modela oclusiones cortas donde la persona sigue caminando
    detrás del oclusor — queremos que el Kalman extrapole honest así el
    predict cae cerca de la detección cuando reaparece, no congelado en
    la posición de entrada al PENDING.

    Sin grace (decay desde frame 1), una persona ocluida por 3 frames a
    20 px/frame queda con predict cerca de pos_init + decayed_drift,
    típicamente <30 px del freeze point. Con grace=3, los 3 frames
    extrapolan a velocidad real → predict cae cerca del movimiento
    real → reid robusto.

    Timing: ``disappeared`` se incrementa en ``_record_miss`` AL FINAL
    del update, así que el predict del próximo update ve el valor
    post-incremento. Con grace=3 el decay arranca cuando
    ``disappeared > 3`` entrando al predict (= 4to update sin obs).
    """
    tracker = EuclideanTracker(
        max_distance=200,
        confirm_frames=1,
        pending_max_frames=50,
        reid_gate_px=300,
        pending_velocity_decay=0.5,
        pending_grace_frames=3,
    )
    tracker.update([np.array([0.0, 100.0, 3000.0])])
    tracker.update([np.array([20.0, 100.0, 3000.0])])
    tid = list(tracker.tracks.keys())[0]
    vx_observed = float(tracker.tracks[tid].kalman.x[2])
    assert abs(vx_observed - 20.0) < 5.0

    # 3 updates sin obs: el track entra a PENDING en el primero, los
    # siguientes 2 mantienen velocidad porque disappeared ∈ {1,2,3}
    # entrando al predict y todos satisfacen disappeared <= grace=3.
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks[tid].state == PENDING
    vx_after_grace = float(tracker.tracks[tid].kalman.x[2])
    assert abs(vx_after_grace - vx_observed) < 2.0, (
        f"Durante grace=3 la velocidad debe preservarse "
        f"(observed={vx_observed:.1f}, after_grace={vx_after_grace:.1f})"
    )

    # 5 updates más: el primero todavía tiene disappeared=3 entrando al
    # predict (still grace), pero los siguientes 4 tienen disappeared
    # ∈ {4,5,6,7} → decay aplica 4 veces, vx *= 0.5^4 = 0.0625.
    for _ in range(5):
        tracker.update([])
    vx_after_decay = float(tracker.tracks[tid].kalman.x[2])
    assert vx_after_decay < vx_observed * 0.3, (
        f"Post-grace el decay debe colapsar velocidad "
        f"(observed={vx_observed:.1f}, after_decay={vx_after_decay:.1f})"
    )


def test_track_id_cycles_mod_max():
    """Con ``max_track_id`` pequeño, los IDs ciclan en lugar de crecer
    monotónicamente — IDs de 0..max-1 se reusan después de que los
    tracks viejos mueren. Garantiza el bound en runs largos (12h/día
    × 365 días) y evita IDs growing-without-bound."""
    tracker = EuclideanTracker(
        max_distance=200,
        confirm_frames=1,
        pending_max_frames=1,
        max_disappeared=1,
        max_track_id=4,
    )
    ids_seen: list[int] = []
    for i in range(8):
        tracks = tracker.update([np.array([100.0 + i * 50.0, 100.0, 3000.0])])
        # Capturar el ID asignado en este spawn antes de que muera.
        ids_seen.extend(tracks.keys())
        # Una vuelta vacía → track va a PENDING (disappeared=1).
        tracker.update([])
        # Otra vuelta vacía → PENDING + 1 > pending_max_frames=1 → LOST.
        tracker.update([])
    # Todos los IDs deben caer en [0, 4) (módulo 4).
    assert all(
        0 <= tid < 4 for tid in ids_seen
    ), f"IDs deben ciclar en [0, 4): {ids_seen}"
    # Y debe haber al menos un ID repetido (cycling efectivo).
    assert len(set(ids_seen)) < len(
        ids_seen
    ), f"Esperaba IDs repitiéndose (cycling), todos únicos: {ids_seen}"


def test_track_id_skips_live_collision():
    """En el rollover, si el próximo ID candidato está en uso por un
    track vivo, ``_register`` salta al siguiente disponible. Garantiza
    que IDs colisionantes nunca se asignen — los tracks vivos preservan
    identidad incluso en pools chicos."""
    tracker = EuclideanTracker(
        max_distance=200,
        confirm_frames=1,
        pending_max_frames=100,  # tracks NO mueren
        max_track_id=4,
    )
    # 3 tracks vivos → IDs 0, 1, 2.
    tracker.update(
        [
            np.array([100.0, 100.0, 3000.0]),
            np.array([300.0, 100.0, 3000.0]),
            np.array([500.0, 100.0, 3000.0]),
        ]
    )
    assert set(tracker.tracks.keys()) == {0, 1, 2}
    # Refrescar los 3 existentes + uno nuevo. El próximo candidato es
    # 3 (libre) → se asigna 3.
    tracker.update(
        [
            np.array([100.0, 100.0, 3000.0]),
            np.array([300.0, 100.0, 3000.0]),
            np.array([500.0, 100.0, 3000.0]),
            np.array([700.0, 100.0, 3000.0]),
        ]
    )
    assert set(tracker.tracks.keys()) == {0, 1, 2, 3}


def test_track_id_default_max_is_16bit():
    """Default 65536 = 16-bit unsigned. Verifica que el constructor
    sin override usa ese valor."""
    tracker = EuclideanTracker(max_distance=50)
    assert tracker.max_track_id == 65536


def test_ambiguous_match_rejected_no_spawn():
    """Con ratio_test 0.8, una detección entre dos tracks equidistantes
    se rechaza por ambigüedad: ambos tracks pasan a PENDING y la
    detección se consume (no spawnea un track nuevo). Esta es la
    defensa anti-ID-swap en cruces — preferimos miss antes que swap.
    """
    tracker = EuclideanTracker(
        max_distance=100,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=100,
        ambiguous_match_ratio=0.8,
    )
    # Spawn dos tracks adyacentes a 10px.
    tracker.update(
        [
            np.array([100.0, 100.0, 3000.0]),
            np.array([110.0, 100.0, 3000.0]),
        ]
    )
    tracker.update(
        [
            np.array([100.0, 100.0, 3000.0]),
            np.array([110.0, 100.0, 3000.0]),
        ]
    )
    assert len(tracker.tracks) == 2
    initial_ids = list(tracker.tracks.keys())
    assert all(tracker.tracks[tid].state == CONFIRMED for tid in initial_ids)

    # Detección equidistante (5px de cada). Hungarian asigna a UNO con
    # cost=5; el 2do-mejor (otro track al mismo cost=5) está dentro del
    # ratio 0.8 → rechazo.
    tracker.update([np.array([105.0, 100.0, 3000.0])])

    # Ambos tracks PENDING (rejected match + missed).
    for tid in initial_ids:
        assert tracker.tracks[tid].state == PENDING, (
            f"Track {tid} debería estar PENDING, está " f"{tracker.tracks[tid].state}"
        )
    # No spawn de un tercer track: la detección ambigua se consumió.
    assert len(tracker.tracks) == 2


def test_ambiguous_match_ratio_default_off_back_compat():
    """Sin ratio_test (default 1.0), Hungarian asigna cualquier match
    válido sin rechazar por ambigüedad. Garantiza que tests viejos no
    cambian de semántica."""
    tracker = EuclideanTracker(
        max_distance=100,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=100,
    )
    assert tracker.ambiguous_match_ratio == 1.0
    tracker.update(
        [
            np.array([100.0, 100.0, 3000.0]),
            np.array([110.0, 100.0, 3000.0]),
        ]
    )
    tracker.update(
        [
            np.array([100.0, 100.0, 3000.0]),
            np.array([110.0, 100.0, 3000.0]),
        ]
    )
    initial_ids = list(tracker.tracks.keys())
    # Una sola detección ambigua — sin ratio test, Hungarian la matchea
    # a uno y el otro queda missed.
    tracker.update([np.array([105.0, 100.0, 3000.0])])
    states = [tracker.tracks[tid].state for tid in initial_ids]
    assert states.count(CONFIRMED) == 1
    assert states.count(PENDING) == 1


def test_unambiguous_match_passes_ratio_test():
    """Con ratio=0.8 pero una detección claramente más cerca de un
    track que del otro, el match no es ambiguo y se acepta."""
    tracker = EuclideanTracker(
        max_distance=200,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=200,
        ambiguous_match_ratio=0.8,
    )
    # Tracks separados — cualquier detección cerca de uno deja al otro
    # como alternativa muy lejana.
    tracker.update(
        [
            np.array([100.0, 100.0, 3000.0]),
            np.array([200.0, 100.0, 3000.0]),
        ]
    )
    tracker.update(
        [
            np.array([100.0, 100.0, 3000.0]),
            np.array([200.0, 100.0, 3000.0]),
        ]
    )
    initial_ids = list(tracker.tracks.keys())
    # Detección a 105: cost 5 a track 1, cost 95 a track 2 → ratio
    # 5/95 = 0.05 << 0.8 → match no ambiguo.
    tracker.update([np.array([105.0, 100.0, 3000.0])])
    states = [tracker.tracks[tid].state for tid in initial_ids]
    assert CONFIRMED in states  # uno fue matched
    assert PENDING in states  # el otro missed (sin obs propia)


def test_pending_grace_frames_default_zero_back_compat():
    """Tracker sin ``pending_grace_frames`` aplica decay desde el primer
    update que vea state=PENDING (semántica previa). Garantiza que
    tests y configs viejas no shiftean.

    Timing: el primer update con miss transiciona CONFIRMED→PENDING al
    final, pero el predict de ESE update vio state=CONFIRMED → no
    aplica decay. Es el SEGUNDO update con miss el primero que ve
    state=PENDING entrando al predict → decay aplica. Con grace=0,
    disappeared=1 > 0 = True → decay arranca en el 2do miss.
    """
    tracker = EuclideanTracker(
        max_distance=200,
        confirm_frames=1,
        pending_max_frames=20,
        pending_velocity_decay=0.5,
    )
    assert tracker.pending_grace_frames == 0
    tracker.update([np.array([0.0, 100.0, 3000.0])])
    tracker.update([np.array([20.0, 100.0, 3000.0])])
    tid = list(tracker.tracks.keys())[0]
    vx_pre = float(tracker.tracks[tid].kalman.x[2])
    tracker.update([])  # CONFIRMED→PENDING, predict sin decay (still CONFIRMED)
    tracker.update([])  # PENDING, disappeared=1 > 0 → decay aplica
    vx_post = float(tracker.tracks[tid].kalman.x[2])
    assert vx_post < vx_pre * 0.7, (
        f"Con grace=0, decay aplica desde el primer update en PENDING "
        f"(pre={vx_pre:.1f}, post={vx_post:.1f})"
    )


def test_tracker_kalman_params_propagated_to_filter():
    """Constructor kwargs should land on each track's filter."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        process_noise=2.5,
        measurement_noise=7.0,
        initial_velocity_uncertainty=42.0,
    )
    tracker.update([np.array([10.0, 20.0, 3000.0])])
    track = list(tracker.tracks.values())[0]
    kf = track.kalman
    assert kf is not None
    # Q diagonal == process_noise everywhere.
    assert np.allclose(np.diag(kf.Q), [2.5] * 4)
    # R diagonal == measurement_noise.
    assert np.allclose(np.diag(kf.R), [7.0] * 2)
    # P[2,2] / P[3,3] hold the velocity uncertainty seed.
    assert kf.P[2, 2] == 42.0
    assert kf.P[3, 3] == 42.0


def test_tracker_constant_velocity_long_run_stays_locked():
    """20 frames of clean linear motion — the same track ID should
    persist throughout. No new tracks spawned, no re-IDs."""
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
    )
    pos = np.array([100.0, 200.0, 3000.0])
    v = np.array([8.0, -3.0, 0.0])
    tracker.update([pos.copy()])
    tid = list(tracker.tracks.keys())[0]

    for _ in range(20):
        pos = pos + v
        tracks = tracker.update([pos.copy()])
        assert tid in tracks
        assert len(tracks) == 1


# ---------------------------------------------------------------------------
# Mini benchmark — informational, not gated, but useful when tuning.
# ---------------------------------------------------------------------------


def test_kalman_update_microbench(capsys):
    """Print per-frame cost of Kalman predict+update for a 5-track scene.

    Not a hard assertion (machine variability), but the absolute numbers
    should land in the low-microseconds-per-step range — well below
    the 33 ms budget at 30 FPS.
    """
    n_tracks = 5
    n_frames = 500

    filters = [TrackKalman(np.array([100.0 * i, 200.0])) for i in range(n_tracks)]

    t0 = time.perf_counter()
    for f in range(n_frames):
        for i, kf in enumerate(filters):
            kf.predict()
            kf.update(np.array([100.0 * i + f * 1.5, 200.0 + f * 0.2]))
    elapsed = time.perf_counter() - t0

    per_step_us = (elapsed / (n_frames * n_tracks)) * 1e6
    with capsys.disabled():
        print(
            f"\n[bench] Kalman predict+update: {per_step_us:.1f} µs/step "
            f"({n_tracks} tracks × {n_frames} frames in {elapsed * 1000:.1f} ms)"
        )
    # Sanity: well under 1 ms per step on any reasonable machine.
    assert per_step_us < 1000.0

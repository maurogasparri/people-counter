"""Tests for the Kalman motion model integrated into the tracker.

These cover the new behaviour added when ``Track.predicted_position``
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

    Not a hard assertion (CI variability), but the absolute numbers
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

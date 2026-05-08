"""Tests for the two-stage (ByteTrack-style) matching path.

When ``candidate_positions`` is supplied to ``EuclideanTracker.update``,
those low-confidence detections may re-associate with existing tracks
that did not match a high-confidence detection in the main pass, but
they NEVER spawn new tracks. When the kwargs are omitted, the tracker
degrades cleanly to its single-bucket behaviour (backward-compat).
"""

from __future__ import annotations

import numpy as np

from src.tracking.tracker import (
    CONFIRMED,
    PENDING,
    EuclideanTracker,
)


# ---------------------------------------------------------------------------
# Re-association
# ---------------------------------------------------------------------------


def test_low_conf_reassociates_existing_track():
    """A track exists; no high-conf detection this frame, but a low-conf
    detection lands within the re-id gate. The tracker re-associates
    instead of letting the track go PENDING.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    tracker.update([np.array([500.0, 300.0, 3000.0])])
    tracker.update([np.array([510.0, 305.0, 3000.0])])  # CONFIRMED
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED
    initial_disappeared = tracker.tracks[tid].disappeared

    # No high-conf this frame, but a low-conf detection nearby.
    tracks = tracker.update(
        [],
        candidate_positions=[np.array([520.0, 308.0, 3000.0])],
    )

    assert tid in tracks
    # State preserved (didn't go to PENDING because low-conf re-bound it).
    assert tracks[tid].state == CONFIRMED
    # disappeared should be 0 (record_hit resets it).
    assert tracks[tid].disappeared == 0
    # Position was updated to the low-conf detection.
    assert tracks[tid].last_position[0] == 520.0
    # Did not spawn a duplicate track.
    assert len(tracks) == 1
    # Bookkeeping sanity — disappeared decreased / stayed at 0.
    assert tracks[tid].disappeared <= initial_disappeared


def test_low_conf_does_not_spawn_new_track_when_no_tracks_alive():
    """Low-confidence detections must NOT bootstrap fresh tracks. If the
    tracker is empty, candidate detections are silently dropped.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    tracks = tracker.update(
        [],
        candidate_positions=[np.array([500.0, 300.0, 3000.0])],
    )
    assert len(tracks) == 0


def test_low_conf_does_not_spawn_when_no_tracks_match():
    """A track exists but the low-conf detection is far outside the
    re-id gate. The candidate must be discarded (not spawn a track),
    and the existing track records a miss.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=80,  # tight re-id gate
    )
    tracker.update([np.array([100.0, 100.0, 3000.0])])
    tracker.update([np.array([105.0, 102.0, 3000.0])])  # CONFIRMED
    tid = list(tracker.tracks.keys())[0]

    # Low-conf detection 600 px away — far beyond the re-id gate.
    tracks = tracker.update(
        [],
        candidate_positions=[np.array([700.0, 600.0, 3000.0])],
    )

    # Original track still exists (now PENDING from the miss), no new track.
    assert tid in tracks
    assert len(tracks) == 1
    assert tracks[tid].state == PENDING


def test_low_conf_respects_depth_gate():
    """Two-stage matching uses the wide pixel re-id gate but still
    enforces the depth gate. A low-conf detection nearby in pixels but
    on a totally different floor must NOT bind to the existing track.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        max_depth_delta=500.0,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    tracker.update([np.array([500.0, 300.0, 2000.0])])  # depth 2.0m
    tracker.update([np.array([510.0, 305.0, 2000.0])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    # Low-conf detection 80 px away (within reid gate) but at depth
    # 4.5m — depth delta = 2500 mm > max_depth_delta=500.
    tracks = tracker.update(
        [],
        candidate_positions=[np.array([530.0, 320.0, 4500.0])],
    )

    # Original is unmatched -> PENDING; low-conf was discarded (no spawn).
    assert tid in tracks
    assert len(tracks) == 1
    assert tracks[tid].state == PENDING


# ---------------------------------------------------------------------------
# Mixed buckets in a single frame
# ---------------------------------------------------------------------------


def test_high_conf_spawn_plus_low_conf_reassociation_simultaneously():
    """One frame mixes a high-conf detection that creates a new track
    AND a low-conf detection that re-associates an existing track.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    # Existing track at x=100.
    tracker.update([np.array([100.0, 200.0, 3000.0])])
    tracker.update([np.array([105.0, 200.0, 3000.0])])
    tid_existing = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid_existing].state == CONFIRMED

    # Frame N: a brand-new person at x=900 (high-conf), and a low-conf
    # detection close to the existing track at x=130.
    tracks = tracker.update(
        [np.array([900.0, 200.0, 3000.0])],  # high-conf, new
        candidate_positions=[np.array([130.0, 200.0, 3000.0])],  # low-conf
    )

    # Two tracks total: the existing one (recovered via low-conf) and
    # the new high-conf spawn.
    assert len(tracks) == 2
    assert tid_existing in tracks
    # Existing track was re-associated to (130, 200).
    assert tracks[tid_existing].last_position[0] == 130.0
    assert tracks[tid_existing].state == CONFIRMED
    # The other entry was spawned for the high-conf detection.
    other = next(t for tid, t in tracks.items() if tid != tid_existing)
    assert other.last_position[0] == 900.0


def test_high_conf_takes_priority_over_low_conf_for_same_track():
    """If a track has both a high-conf candidate (pass 1) and a low-conf
    candidate (pass 2 / candidate stage), the high-conf wins. The
    low-conf detection is then dropped (it can't spawn a new track).
    """
    tracker = EuclideanTracker(
        max_distance=200,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    tracker.update([np.array([500.0, 300.0, 3000.0])])
    tracker.update([np.array([505.0, 302.0, 3000.0])])
    tid = list(tracker.tracks.keys())[0]

    tracks = tracker.update(
        [np.array([515.0, 304.0, 3000.0])],  # high-conf -> pass 1
        candidate_positions=[np.array([522.0, 308.0, 3000.0])],  # low-conf
    )

    assert len(tracks) == 1
    assert tid in tracks
    # High-conf detection was used.
    assert tracks[tid].last_position[0] == 515.0


# ---------------------------------------------------------------------------
# Backward compatibility (feature off)
# ---------------------------------------------------------------------------


def test_no_candidate_kwargs_is_backward_compatible():
    """Without candidate_positions / candidate_metadata kwargs, behaviour
    is identical to the legacy single-bucket call. This is the no-op
    guarantee for the feature-off path.
    """
    tracker_legacy = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
    )
    tracker_legacy.update([np.array([100.0, 100.0, 3000.0])])
    tracks_a = tracker_legacy.update([np.array([105.0, 100.0, 3000.0])])
    state_a = (tracks_a[0].state, tuple(tracks_a[0].last_position.tolist()))

    tracker_new = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
    )
    tracker_new.update([np.array([100.0, 100.0, 3000.0])])
    tracks_b = tracker_new.update(
        [np.array([105.0, 100.0, 3000.0])],
        # Explicit None — same as omitting.
        candidate_positions=None,
        candidate_metadata=None,
    )
    state_b = (tracks_b[0].state, tuple(tracks_b[0].last_position.tolist()))

    assert state_a == state_b


def test_empty_candidate_list_is_no_op():
    """Empty candidate_positions list (feature on but nothing to match)
    must not change tracker behaviour vs not passing the kwarg.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=5,
    )
    tracker.update([np.array([100.0, 100.0, 3000.0])])

    # Miss this frame, with an explicit empty candidate list.
    tracks = tracker.update([], candidate_positions=[], candidate_metadata=[])
    # Replicate the legacy single-bucket path: result must match exactly.
    tracker_legacy = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=5,
    )
    tracker_legacy.update([np.array([100.0, 100.0, 3000.0])])
    legacy = tracker_legacy.update([])
    assert set(tracks.keys()) == set(legacy.keys())
    for tid_l in legacy:
        assert tracks[tid_l].state == legacy[tid_l].state
        assert tracks[tid_l].disappeared == legacy[tid_l].disappeared


def test_candidate_metadata_appended_only_on_match():
    """When a low-conf detection matches a track, its metadata appears in
    the track's detection_history. When it doesn't match (and is dropped),
    nothing leaks into the tracker state.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    tracker.update(
        [np.array([100.0, 100.0, 3000.0])],
        detection_metas=[{"tag": "high1"}],
    )
    tracker.update(
        [np.array([105.0, 102.0, 3000.0])],
        detection_metas=[{"tag": "high2"}],
    )
    tid = list(tracker.tracks.keys())[0]

    # Frame with only a low-conf detection that DOES match.
    tracker.update(
        [],
        candidate_positions=[np.array([115.0, 103.0, 3000.0])],
        candidate_metadata=[{"tag": "low_matched"}],
    )
    history_tags = [
        m.get("tag") for m in tracker.tracks[tid].meta.get("detection_history", [])
    ]
    assert history_tags == ["high1", "high2", "low_matched"]

    # Frame with only a low-conf detection FAR away (no match, dropped).
    tracker.update(
        [],
        candidate_positions=[np.array([2000.0, 2000.0, 3000.0])],
        candidate_metadata=[{"tag": "low_dropped"}],
    )
    history_tags_after = [
        m.get("tag") for m in tracker.tracks[tid].meta.get("detection_history", [])
    ]
    assert "low_dropped" not in history_tags_after


def test_candidate_metadata_length_must_match_positions():
    """Mismatched length between candidate_positions and candidate_metadata
    must raise — same contract as detection_metas.
    """
    tracker = EuclideanTracker(max_distance=50, confirm_frames=1)
    import pytest

    with pytest.raises(ValueError, match="candidate_metadata length"):
        tracker.update(
            [],
            candidate_positions=[np.array([10.0, 10.0, 3000.0])],
            candidate_metadata=[{"a": 1}, {"b": 2}],  # length mismatch
        )


# ---------------------------------------------------------------------------
# Backward compat: re-run a legacy scenario with the kwargs explicitly None
# ---------------------------------------------------------------------------


def test_legacy_pass2_jitter_recovery_still_works_with_feature_off():
    """Pass-2 (the existing wide-gate fallback) must still recover a
    jittered bbox when low-conf bucket is unused — i.e. the new code
    path doesn't disturb the old single-bucket pass-2 behaviour.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=1,
        pending_max_frames=20,
        reid_gate_px=300,
    )
    tracker.update([np.array([500.0, 300.0, 3000.0])])
    tracker.update([np.array([510.0, 305.0, 3000.0])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    # 80 px jitter — caught by pass-2, NOT by the new candidate path.
    tracks = tracker.update(
        [np.array([590.0, 325.0, 3000.0])],
        candidate_positions=None,  # feature off
    )
    assert len(tracks) == 1
    assert tid in tracks
    assert tracks[tid].state == CONFIRMED


def test_legacy_state_machine_with_no_candidates():
    """CANDIDATE → CONFIRMED → PENDING → recover works identically when
    candidate_positions is omitted.
    """
    tracker = EuclideanTracker(
        max_distance=50,
        confirm_frames=2,
        pending_max_frames=5,
        reid_gate_px=80,
    )
    tracker.update([np.array([100.0, 200.0, 3000.0])])
    tracker.update([np.array([110.0, 200.0, 3000.0])])
    tid = list(tracker.tracks.keys())[0]
    assert tracker.tracks[tid].state == CONFIRMED

    tracker.update([])
    tracker.update([])
    assert tracker.tracks[tid].state == PENDING

    tracks = tracker.update([np.array([130.0, 200.0, 3000.0])])
    assert tid in tracks
    assert tracks[tid].state == CONFIRMED

"""Tests for 3D Euclidean tracker."""
import numpy as np

from src.tracking.tracker import EuclideanTracker


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
    tracks = tracker.update([])  # disappeared 3 → removed
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

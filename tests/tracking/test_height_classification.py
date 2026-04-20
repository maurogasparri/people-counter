"""Integration test: tracker → counter → CountEvent with height_class."""

import numpy as np

from src.tracking.counter import LineCounter, ROICounter
from src.tracking.tracker import EuclideanTracker


def _run_line_crossing(tracker, counter, metas_per_frame, positions_per_frame):
    """Drive tracker + counter with synthetic detections. Returns emitted events."""
    emitted = []
    for positions, metas in zip(positions_per_frame, metas_per_frame):
        tracks = tracker.update(positions, detection_metas=metas)
        emitted.extend(counter.check_all(tracks))
    return emitted


class TestLineCounterHeightClass:
    def test_adult_track_emits_adult_event(self):
        tracker = EuclideanTracker(confirm_frames=1)
        counter = LineCounter(line_y=100.0)

        # Single track: 4 positions crossing the line; all frames tagged "adult"
        positions = [
            [np.array([50, 60, 2000])],
            [np.array([50, 80, 2000])],
            [np.array([50, 110, 2000])],  # crosses line 100
            [np.array([50, 130, 2000])],
        ]
        metas = [[{"height_class": "adult"}]] * 4
        events = _run_line_crossing(tracker, counter, metas, positions)
        assert len(events) == 1
        assert events[0].direction == "in"
        assert events[0].height_class == "adult"

    def test_child_track_emits_child_event(self):
        tracker = EuclideanTracker(confirm_frames=1)
        counter = LineCounter(line_y=100.0)

        positions = [
            [np.array([50, 60, 2500])],
            [np.array([50, 80, 2500])],
            [np.array([50, 110, 2500])],
        ]
        metas = [[{"height_class": "child"}]] * 3
        events = _run_line_crossing(tracker, counter, metas, positions)
        assert events[0].height_class == "child"

    def test_missing_meta_yields_unknown(self):
        tracker = EuclideanTracker(confirm_frames=1)
        counter = LineCounter(line_y=100.0)

        positions = [
            [np.array([50, 60, 2000])],
            [np.array([50, 80, 2000])],
            [np.array([50, 110, 2000])],
        ]
        # No metadata at all — classifier disabled path
        events = _run_line_crossing(
            tracker, counter, [None, None, None], positions,
        )
        # update() will raise if list lengths mismatch; use metas=None paths
        # → need to call with no detection_metas arg
        # Rebuild cleanly:
        tracker = EuclideanTracker(confirm_frames=1)
        counter = LineCounter(line_y=100.0)
        for pos in positions:
            tracks = tracker.update(pos)
            events = counter.check_all(tracks)
        # After final cross, run once more to capture the event
        # (the simple loop above already collected it if crossing occurred)
        # Re-run with storage
        tracker = EuclideanTracker(confirm_frames=1)
        counter = LineCounter(line_y=100.0)
        all_events = []
        for pos in positions:
            tracks = tracker.update(pos)
            all_events.extend(counter.check_all(tracks))
        assert all_events, "expected at least one crossing event"
        assert all_events[0].height_class == "unknown"

    def test_mixed_classifications_majority_wins(self):
        """Flickering classification over a track settles on the majority."""
        tracker = EuclideanTracker(confirm_frames=1)
        counter = LineCounter(line_y=100.0)

        positions = [
            [np.array([50, 60, 2000])],
            [np.array([50, 80, 2000])],
            [np.array([50, 110, 2000])],
        ]
        # 2 adult vs 1 child → adult
        metas = [
            [{"height_class": "adult"}],
            [{"height_class": "child"}],
            [{"height_class": "adult"}],
        ]
        events = _run_line_crossing(tracker, counter, metas, positions)
        assert events[0].height_class == "adult"


class TestTrackerDetectionHistory:
    def test_metas_appended_to_matched_track(self):
        tracker = EuclideanTracker(confirm_frames=1)
        positions = [np.array([50, 50, 2000])]
        metas = [{"height_class": "adult", "head_height_mm": 1700}]
        tracks = tracker.update(positions, detection_metas=metas)
        tid = next(iter(tracks))
        history = tracks[tid].meta.get("detection_history", [])
        assert len(history) == 1
        assert history[0]["height_class"] == "adult"

    def test_mismatched_metas_length_raises(self):
        tracker = EuclideanTracker(confirm_frames=1)
        positions = [np.array([50, 50, 2000]), np.array([200, 200, 2500])]
        metas = [{"height_class": "adult"}]  # only 1 meta for 2 detections
        try:
            tracker.update(positions, detection_metas=metas)
        except ValueError as e:
            assert "detection_metas length" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def test_history_capped(self):
        tracker = EuclideanTracker(confirm_frames=1)
        tracker.DETECTION_HISTORY_MAX = 5
        positions = [[np.array([50, 50 + i * 0.5, 2000])] for i in range(20)]
        metas = [[{"height_class": "adult"}]] * 20
        for pos, meta in zip(positions, metas):
            tracker.update(pos, detection_metas=meta)
        tid = next(iter(tracker.tracks))
        history = tracker.tracks[tid].meta["detection_history"]
        assert len(history) == 5

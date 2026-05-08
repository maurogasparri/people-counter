"""Test de integración: tracker → counter → CountEvent con height_class.

Maneja el Counter unificado (ROI + line) end-to-end desde sintético
detections, checking that per-frame height metadata propagates through
the tracker into the emitted CountEvent.
"""

import numpy as np

from src.tracking.counter import Counter, Line
from src.tracking.tracker import EuclideanTracker


def _make_counter() -> Counter:
    """ROI 0..200 x 50..150, single horizontal line at y=100 with both
    directions labelled. Tracks crossing top->bottom emit ``ingress``."""
    return Counter(
        lines=[Line(
            from_xy=(0, 100), to_xy=(200, 100),
            labels={"top_to_bottom": "ingress", "bottom_to_top": "egress"},
        )],
        roi={"x_min": 0, "x_max": 200, "y_min": 50, "y_max": 150},
    )


def _run_crossing(tracker, counter, metas_per_frame, positions_per_frame):
    """Drive tracker + counter with synthetic detections. Returns emitted
    events."""
    emitted = []
    for positions, metas in zip(positions_per_frame, metas_per_frame):
        tracks = tracker.update(positions, detection_metas=metas)
        emitted.extend(counter.check_all(tracks))
    return emitted


# Trajectory: y goes 60 (inside, above line) → 80 (still inside, above) →
# 110 (inside, crossed below) → 160 (outside ROI, exit triggers count).
_INGRESS_POSITIONS = [
    [np.array([50, 60, 2000])],
    [np.array([50, 80, 2000])],
    [np.array([50, 110, 2000])],
    [np.array([50, 160, 2000])],
]


class TestHeightClassPropagation:
    def test_adult_track_emits_adult_event(self):
        tracker = EuclideanTracker(confirm_frames=1)
        counter = _make_counter()
        metas = [[{"height_class": "adult"}]] * 4
        events = _run_crossing(tracker, counter, metas, _INGRESS_POSITIONS)
        assert len(events) == 1
        assert events[0].direction == "ingress"
        assert events[0].height_class == "adult"

    def test_child_track_emits_child_event(self):
        tracker = EuclideanTracker(confirm_frames=1)
        counter = _make_counter()
        metas = [[{"height_class": "child"}]] * 4
        events = _run_crossing(tracker, counter, metas, _INGRESS_POSITIONS)
        assert events[0].height_class == "child"

    def test_missing_meta_yields_unknown(self):
        """No metadata supplied — aggregator falls back to 'unknown'."""
        tracker = EuclideanTracker(confirm_frames=1)
        counter = _make_counter()
        all_events = []
        for pos in _INGRESS_POSITIONS:
            tracks = tracker.update(pos)
            all_events.extend(counter.check_all(tracks))
        assert all_events, "expected at least one crossing event"
        assert all_events[0].height_class == "unknown"

    def test_mixed_classifications_majority_wins(self):
        """Flickering classification over a track settles on the majority."""
        tracker = EuclideanTracker(confirm_frames=1)
        counter = _make_counter()
        # 3 adult vs 1 child → adult
        metas = [
            [{"height_class": "adult"}],
            [{"height_class": "child"}],
            [{"height_class": "adult"}],
            [{"height_class": "adult"}],
        ]
        events = _run_crossing(tracker, counter, metas, _INGRESS_POSITIONS)
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

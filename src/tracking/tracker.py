"""3D Euclidean distance tracker for person trajectories."""
import logging
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """A tracked person with position history."""

    track_id: int
    positions: list[np.ndarray] = field(default_factory=list)
    disappeared: int = 0

    @property
    def last_position(self) -> np.ndarray:
        return self.positions[-1]


class EuclideanTracker:
    """Assigns detections to tracks using 3D Euclidean distance."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._next_id = 0
        self._tracks: OrderedDict[int, Track] = OrderedDict()

    @property
    def tracks(self) -> dict[int, Track]:
        return dict(self._tracks)

    def update(self, detections: list[np.ndarray]) -> dict[int, Track]:
        if len(detections) == 0:
            return self._handle_no_detections()
        if len(self._tracks) == 0:
            return self._register_all(detections)
        return self._match_and_update(detections)

    def _handle_no_detections(self) -> dict[int, Track]:
        to_remove = []
        for track_id, track in self._tracks.items():
            track.disappeared += 1
            if track.disappeared > self.max_disappeared:
                to_remove.append(track_id)
        for tid in to_remove:
            del self._tracks[tid]
        return self.tracks

    def _register_all(self, detections: list[np.ndarray]) -> dict[int, Track]:
        for det in detections:
            self._register(det)
        return self.tracks

    def _register(self, centroid: np.ndarray) -> int:
        tid = self._next_id
        self._tracks[tid] = Track(track_id=tid, positions=[centroid.copy()])
        self._next_id += 1
        return tid

    def _match_and_update(self, detections: list[np.ndarray]) -> dict[int, Track]:
        track_ids = list(self._tracks.keys())
        track_centroids = np.array([self._tracks[t].last_position for t in track_ids])
        det_array = np.array(detections)

        distances = np.linalg.norm(
            track_centroids[:, np.newaxis] - det_array[np.newaxis, :], axis=2
        )

        used_dets: set[int] = set()
        matched_tracks: set[int] = set()

        for flat_idx in np.argsort(distances, axis=None):
            t_idx = flat_idx // len(detections)
            d_idx = flat_idx % len(detections)
            if t_idx in matched_tracks or d_idx in used_dets:
                continue
            if distances[t_idx, d_idx] > self.max_distance:
                break
            tid = track_ids[t_idx]
            self._tracks[tid].positions.append(detections[d_idx].copy())
            self._tracks[tid].disappeared = 0
            matched_tracks.add(t_idx)
            used_dets.add(d_idx)

        to_remove = []
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                self._tracks[tid].disappeared += 1
                if self._tracks[tid].disappeared > self.max_disappeared:
                    to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]

        for d_idx, det in enumerate(detections):
            if d_idx not in used_dets:
                self._register(det)

        return self.tracks

"""3D tracker for person trajectories.

Implements Hungarian-based association with a short-term state machine:

    CANDIDATE  -> CONFIRMED  (after N consecutive hits)
    CONFIRMED  -> PENDING    (on a missed frame; predicted with last velocity)
    PENDING    -> CONFIRMED  (re-match within re-id gate)
    PENDING    -> LOST       (after pending_max_frames consecutive misses)

The tracker exposes all tracks in any state; the counter layer filters to
CONFIRMED/PENDING when making counting decisions. Track IDs are preserved
across PENDING -> CONFIRMED recoveries so the counter's per-track state
(entry side, crossed line) survives short occlusions.

Positions are [cx, cy, z] with cx/cy in pixels and z in mm. Matching cost is
2D pixel distance; depth is used as a secondary gate (|dz| > max_depth_delta
rejects the match) because pixels and millimetres are not directly
comparable.
"""
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Track states
CANDIDATE = "candidate"
CONFIRMED = "confirmed"
PENDING = "pending"
LOST = "lost"


@dataclass
class Track:
    """A tracked person with position history and state."""

    track_id: int
    positions: list[np.ndarray] = field(default_factory=list)
    disappeared: int = 0
    state: str = CANDIDATE
    hits: int = 1
    # Counter-layer bookkeeping (set by the counter, not the tracker).
    # Stored on the track so state survives PENDING re-identification.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def last_position(self) -> np.ndarray:
        return self.positions[-1]

    @property
    def predicted_position(self) -> np.ndarray:
        """Estimate the next position from the last two observations."""
        if len(self.positions) < 2:
            return self.positions[-1].copy()
        velocity = self.positions[-1] - self.positions[-2]
        return self.positions[-1] + velocity


class EuclideanTracker:
    """Hungarian 2D pixel matching + depth gating, with a small state machine.

    Backward-compatible: the constructor still accepts ``max_disappeared``
    and ``max_distance``; new optional kwargs enable the state machine and
    re-identification behaviour. ``update(detections)`` still returns a
    ``dict[int, Track]`` of the currently-alive tracks (any state except
    LOST; LOST tracks are removed).
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 50.0,
        max_depth_delta: float = 500.0,
        confirm_frames: int = 3,
        pending_max_frames: int = 5,
        reid_gate_px: float = 60.0,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_depth_delta = max_depth_delta
        self.confirm_frames = max(1, confirm_frames)
        self.pending_max_frames = max(1, pending_max_frames)
        self.reid_gate_px = reid_gate_px
        self._next_id = 0
        self._tracks: OrderedDict[int, Track] = OrderedDict()

    @property
    def tracks(self) -> dict[int, Track]:
        return dict(self._tracks)

    def count_by_state(self) -> dict[str, int]:
        """Return a count of currently-alive tracks grouped by state.

        Keys: ``candidate``, ``confirmed``, ``pending``, ``lost``. LOST
        tracks are purged at the end of ``update()`` so the count is
        normally zero except when called mid-update.
        """
        counts = {CANDIDATE: 0, CONFIRMED: 0, PENDING: 0, LOST: 0}
        for track in self._tracks.values():
            counts[track.state] = counts.get(track.state, 0) + 1
        return counts

    # ------------------------------------------------------------------ API
    def update(
        self,
        detections: list[np.ndarray],
        detection_metas: Optional[list[dict]] = None,
    ) -> dict[int, Track]:
        """Update tracks from the current frame's detections.

        Args:
            detections: list of (x, y, z) positions in pixel / mm coords.
            detection_metas: optional parallel list of metadata dicts per
                detection. When a detection is matched (or registers a new
                track), its metadata is appended to ``track.meta["detection_history"]``
                (capped at ``DETECTION_HISTORY_MAX`` samples). Downstream
                consumers read that history for per-track aggregate labels
                (e.g. adult/child classification).
        """
        det_arr = (
            np.asarray(detections, dtype=float)
            if len(detections) > 0
            else np.empty((0, 3))
        )

        if detection_metas is not None and len(detection_metas) != len(detections):
            raise ValueError(
                "detection_metas length must equal detections length "
                f"({len(detection_metas)} != {len(detections)})"
            )

        def _meta_for(idx: int) -> Optional[dict]:
            return detection_metas[idx] if detection_metas is not None else None

        if len(self._tracks) == 0:
            for i, det in enumerate(det_arr):
                tid = self._register(det)
                self._append_detection_meta(self._tracks[tid], _meta_for(i))
            return self.tracks

        track_ids = list(self._tracks.keys())
        # For matching, use the predicted position of PENDING tracks and the
        # last observed position of everyone else.
        track_refs = np.array(
            [
                self._tracks[t].predicted_position
                if self._tracks[t].state == PENDING
                else self._tracks[t].last_position
                for t in track_ids
            ],
            dtype=float,
        )

        matches, unmatched_t, unmatched_d = self._associate(track_refs, det_arr, track_ids)

        # Apply matches
        for t_idx, d_idx in matches:
            tid = track_ids[t_idx]
            self._record_hit(self._tracks[tid], det_arr[d_idx])
            self._append_detection_meta(self._tracks[tid], _meta_for(d_idx))

        # Mark unmatched tracks as missed
        for t_idx in unmatched_t:
            tid = track_ids[t_idx]
            self._record_miss(self._tracks[tid])

        # Register new tracks for unmatched detections
        for d_idx in unmatched_d:
            tid = self._register(det_arr[d_idx])
            self._append_detection_meta(self._tracks[tid], _meta_for(d_idx))

        # Purge LOST
        to_remove = [tid for tid, t in self._tracks.items() if t.state == LOST]
        for tid in to_remove:
            del self._tracks[tid]

        return self.tracks

    DETECTION_HISTORY_MAX = 60  # ~4s at 15 FPS — enough for stable majority vote

    def _append_detection_meta(self, track: Track, meta: Optional[dict]) -> None:
        if meta is None:
            return
        history = track.meta.setdefault("detection_history", [])
        history.append(dict(meta))
        if len(history) > self.DETECTION_HISTORY_MAX:
            del history[: len(history) - self.DETECTION_HISTORY_MAX]

    # -------------------------------------------------------- association
    def _associate(
        self,
        track_refs: np.ndarray,
        det_arr: np.ndarray,
        track_ids: list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian match with gating. Returns (matches, unmatched_t, unmatched_d)."""
        n_t, n_d = len(track_refs), len(det_arr)
        if n_t == 0 or n_d == 0:
            return [], list(range(n_t)), list(range(n_d))

        dist_2d = np.linalg.norm(
            track_refs[:, np.newaxis, :2] - det_arr[np.newaxis, :, :2], axis=2
        )

        if track_refs.shape[1] > 2 and det_arr.shape[1] > 2:
            depth_delta = np.abs(
                track_refs[:, np.newaxis, 2] - det_arr[np.newaxis, :, 2]
            )
        else:
            depth_delta = np.zeros_like(dist_2d)

        # Per-track gate: PENDING tracks use the wider re-id gate, others the
        # standard max_distance. Expand to (n_t, n_d).
        per_track_gate = np.array(
            [
                self.reid_gate_px
                if self._tracks[tid].state == PENDING
                else self.max_distance
                for tid in track_ids
            ],
            dtype=float,
        )[:, np.newaxis]

        INF = 1e9
        cost = dist_2d.copy()
        cost[dist_2d > per_track_gate] = INF
        cost[depth_delta > self.max_depth_delta] = INF

        row_ind, col_ind = linear_sum_assignment(cost)

        matches: list[tuple[int, int]] = []
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= INF:
                continue
            matches.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_d.add(int(c))

        unmatched_t = [i for i in range(n_t) if i not in matched_t]
        unmatched_d = [j for j in range(n_d) if j not in matched_d]
        return matches, unmatched_t, unmatched_d

    # -------------------------------------------------- per-track updates
    def _register(self, centroid: np.ndarray) -> int:
        tid = self._next_id
        self._tracks[tid] = Track(
            track_id=tid,
            positions=[np.asarray(centroid, dtype=float).copy()],
            state=CANDIDATE,
            hits=1,
        )
        self._next_id += 1
        return tid

    def _record_hit(self, track: Track, centroid: np.ndarray) -> None:
        track.positions.append(np.asarray(centroid, dtype=float).copy())
        track.disappeared = 0
        track.hits += 1
        if track.state == CANDIDATE and track.hits >= self.confirm_frames:
            track.state = CONFIRMED
        elif track.state == PENDING:
            # Recovered; preserve ID and counter meta.
            track.state = CONFIRMED

    def _record_miss(self, track: Track) -> None:
        track.disappeared += 1
        if track.state == CANDIDATE:
            # Unconfirmed misses are cheap: drop after max_disappeared, but
            # also drop immediately on the first miss if confirm_frames
            # requires consecutive hits.
            if track.disappeared > self.max_disappeared:
                track.state = LOST
            return

        if track.state == CONFIRMED:
            track.state = PENDING

        if track.state == PENDING and track.disappeared > self.pending_max_frames:
            track.state = LOST
            return

        # Also enforce the legacy max_disappeared cap as an upper bound.
        if track.disappeared > self.max_disappeared:
            track.state = LOST

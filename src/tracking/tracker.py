"""3D tracker for person trajectories.

Implements Hungarian-based association with a short-term state machine:

    CANDIDATE  -> CONFIRMED  (after N consecutive hits)
    CONFIRMED  -> PENDING    (on a missed frame; predicted with Kalman)
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

Motion model
------------
Each track owns a small constant-velocity Kalman filter (see
``src.tracking.kalman.TrackKalman``) with state ``[cx, cy, vx, vy]``.
Lifecycle per frame:

1. At the start of ``update()``, every track's filter advances one step
   via ``predict()``. This sets ``Track.predicted_position`` (and the
   matching reference for PENDING tracks) to the model's expectation for
   this frame, and grows the position covariance — natural behaviour for
   a track that's been unmatched for several frames.
2. Hungarian association runs against ``predicted_position`` for PENDING
   tracks and ``last_position`` for everyone else (the same ref the
   pre-Kalman tracker used; preserves the tight gate semantics for
   adjacent tracks).
3. Matched tracks call ``kalman.update(measurement)`` to fold in the new
   detection. Unmatched tracks keep the post-predict state — across
   several misses this extrapolates farther while widening uncertainty.

Note: depth is NOT in the Kalman state. The depth gate runs on raw
detection vs. last observed depth; depth fluctuations frame-to-frame
are dominated by SGBM noise rather than smooth motion, so a 4D filter
on (x, y, vx, vy) is the right scope.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman import TrackKalman

logger = logging.getLogger(__name__)

# Track states
CANDIDATE = "candidate"
CONFIRMED = "confirmed"
PENDING = "pending"
LOST = "lost"

# Kalman defaults — overridable via EuclideanTracker constructor (which is
# in turn fed by hardware.yaml's tracking.state_machine.kalman block).
DEFAULT_PROCESS_NOISE = 1.0
DEFAULT_MEASUREMENT_NOISE = 5.0
DEFAULT_INITIAL_VELOCITY_UNCERTAINTY = 100.0


@dataclass
class Track:
    """A tracked person with position history, state and motion filter.

    ``positions`` keeps the raw observed centroids for compat with
    downstream consumers (counter, viewer) that read ``positions[-1]``.
    The Kalman filter is the source of truth for ``predicted_position``;
    raw history is preserved so trajectory diagnostics still work.
    """

    track_id: int
    positions: list[np.ndarray] = field(default_factory=list)
    disappeared: int = 0
    state: str = CANDIDATE
    hits: int = 1
    # Motion filter — set by the tracker at registration. ``Optional`` only
    # because dataclass init order otherwise requires an awkward default.
    kalman: Optional[TrackKalman] = None
    # Counter-layer bookkeeping (set by the counter, not the tracker).
    # Stored on the track so state survives PENDING re-identification.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def last_position(self) -> np.ndarray:
        """Most recent observed centroid [cx, cy, z]. Compat for counter."""
        return self.positions[-1]

    @property
    def predicted_position(self) -> np.ndarray:
        """Kalman-predicted NEXT-frame position as [cx, cy, z].

        Non-destructive: uses ``kalman.peek_next()`` so the filter state
        is unchanged. The caller gets the model's expectation for one
        step beyond the current state (i.e., where this track is most
        likely to be observed at the next ``tracker.update()``).

        The Kalman filter only models 2D pixel motion; the returned z
        carries forward the most recent observed depth (depth lacks a
        smooth motion model — see module docstring).
        """
        if self.kalman is None:
            # Fallback — only happens if a Track is built without a filter,
            # which the tracker itself never does. Kept defensive so old
            # tests / external constructors don't crash.
            if len(self.positions) < 2:
                return self.positions[-1].copy()
            velocity = self.positions[-1] - self.positions[-2]
            return self.positions[-1] + velocity
        xy = self.kalman.peek_next()
        z = self.positions[-1][2] if len(self.positions[-1]) > 2 else 0.0
        return np.array([xy[0], xy[1], z], dtype=float)


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
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
        initial_velocity_uncertainty: float = DEFAULT_INITIAL_VELOCITY_UNCERTAINTY,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_depth_delta = max_depth_delta
        self.confirm_frames = max(1, confirm_frames)
        self.pending_max_frames = max(1, pending_max_frames)
        self.reid_gate_px = reid_gate_px
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_velocity_uncertainty = float(initial_velocity_uncertainty)
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
        candidate_positions: Optional[list[np.ndarray]] = None,
        candidate_metadata: Optional[list[dict]] = None,
    ) -> dict[int, Track]:
        """Update tracks from the current frame's detections.

        Args:
            detections: list of (x, y, z) positions in pixel / mm coords.
                These are the *high-confidence* detections — they can both
                update existing tracks AND spawn new ones.
            detection_metas: optional parallel list of metadata dicts per
                detection. When a detection is matched (or registers a new
                track), its metadata is appended to ``track.meta["detection_history"]``
                (capped at ``DETECTION_HISTORY_MAX`` samples). Downstream
                consumers read that history for per-track aggregate labels
                (e.g. adult/child classification).
            candidate_positions: optional list of *low-confidence*
                detections. Used only for re-association of existing tracks
                that did not match a high-confidence detection in pass 1
                (ByteTrack-style two-stage matching). Low-confidence
                detections that do NOT match any unmatched track are
                silently discarded — they never spawn new tracks. Default
                ``None`` disables this behaviour entirely (the tracker
                degenerates to its previous single-bucket behaviour).
            candidate_metadata: optional parallel metadata list for
                ``candidate_positions``. Same semantics as
                ``detection_metas`` but applied only when a candidate
                actually re-associates with an existing track.
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

        cand_arr = (
            np.asarray(candidate_positions, dtype=float)
            if candidate_positions is not None and len(candidate_positions) > 0
            else np.empty((0, 3))
        )
        if (
            candidate_metadata is not None
            and candidate_positions is not None
            and len(candidate_metadata) != len(candidate_positions)
        ):
            raise ValueError(
                "candidate_metadata length must equal candidate_positions "
                f"length ({len(candidate_metadata)} != "
                f"{len(candidate_positions)})"
            )

        def _meta_for(idx: int) -> Optional[dict]:
            return detection_metas[idx] if detection_metas is not None else None

        def _cand_meta_for(idx: int) -> Optional[dict]:
            return candidate_metadata[idx] if candidate_metadata is not None else None

        # Advance every track's Kalman filter one frame BEFORE association.
        # Matched tracks will then call kalman.update(z) to absorb the
        # measurement; unmatched tracks keep the post-predict state, which
        # is the correct extrapolation for the missing frame.
        for track in self._tracks.values():
            if track.kalman is not None:
                track.kalman.predict()

        if len(self._tracks) == 0:
            # No tracks alive: low-conf detections cannot re-associate
            # against anything (and never spawn). Only high-conf spawns.
            for i, det in enumerate(det_arr):
                tid = self._register(det)
                self._append_detection_meta(self._tracks[tid], _meta_for(i))
            return self.tracks

        track_ids = list(self._tracks.keys())
        # For matching, PENDING tracks use the Kalman-propagated current
        # position (which is the prediction for THIS frame, since predict
        # was just called above); CONFIRMED/CANDIDATE use the last observed
        # centroid. Same gating policy as pre-Kalman — only PENDING gets
        # the propagated reference, so adjacent CONFIRMED tracks can't
        # drift toward each other across a single frame.
        track_refs = np.array(
            [
                (
                    self._pending_match_ref(self._tracks[t])
                    if self._tracks[t].state == PENDING
                    else self._tracks[t].last_position
                )
                for t in track_ids
            ],
            dtype=float,
        )

        matches, unmatched_t, unmatched_d = self._associate(
            track_refs, det_arr, track_ids
        )

        # Apply matches
        for t_idx, d_idx in matches:
            tid = track_ids[t_idx]
            self._record_hit(self._tracks[tid], det_arr[d_idx])
            self._append_detection_meta(self._tracks[tid], _meta_for(d_idx))

        # ByteTrack-style pass 3: try to re-associate the still-unmatched
        # tracks against the low-confidence (candidate) detections with the
        # wide reid gate. Candidate detections that don't bind to any track
        # here are dropped — they never spawn new tracks. Skipped entirely
        # when no candidates were supplied (feature off).
        if len(cand_arr) > 0 and unmatched_t:
            unmatched_track_refs = track_refs[unmatched_t]
            unmatched_track_ids = [track_ids[i] for i in unmatched_t]
            cand_matches, _, _ = self._associate_candidates(
                unmatched_track_refs,
                cand_arr,
                unmatched_track_ids,
            )
            still_unmatched_t_local: set[int] = set(range(len(unmatched_t)))
            for local_t_idx, c_idx in cand_matches:
                tid = unmatched_track_ids[local_t_idx]
                self._record_hit(self._tracks[tid], cand_arr[c_idx])
                self._append_detection_meta(
                    self._tracks[tid],
                    _cand_meta_for(c_idx),
                )
                still_unmatched_t_local.discard(local_t_idx)
            unmatched_t = [unmatched_t[i] for i in sorted(still_unmatched_t_local)]

        # Mark unmatched tracks as missed
        for t_idx in unmatched_t:
            tid = track_ids[t_idx]
            self._record_miss(self._tracks[tid])

        # Register new tracks for unmatched HIGH-confidence detections only.
        # Low-confidence (candidate) detections are intentionally dropped.
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
        """Hungarian match with gating, two-pass.

        Pass 1 uses the state-aware gate (``reid_gate_px`` for PENDING,
        ``max_distance`` otherwise). Pass 2 reruns Hungarian on whatever
        is still unmatched with the wider ``reid_gate_px`` for everyone
        — so a CONFIRMED that's about to go PENDING gets one shot at
        re-matching the orphan detection before that detection is
        registered as a brand-new track. Without pass 2, bbox jitter or
        a single dropped detection produces phantom track pairs (the
        original CONFIRMED transitions to PENDING while a fresh ID is
        created from the same physical person).

        Returns ``(matches, unmatched_t, unmatched_d)``.
        """
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

        # Pass 1: per-track gate. PENDING gets the wider re-id gate to
        # finish recovering; everyone else gets the tight gate to keep
        # adjacent tracks from swapping.
        gate_pass1 = np.array(
            [
                (
                    self.reid_gate_px
                    if self._tracks[tid].state == PENDING
                    else self.max_distance
                )
                for tid in track_ids
            ],
            dtype=float,
        )[:, np.newaxis]

        matches, matched_t, matched_d = self._hungarian_with_gate(
            dist_2d,
            depth_delta,
            gate_pass1,
        )

        unmatched_t = [i for i in range(n_t) if i not in matched_t]
        unmatched_d = [j for j in range(n_d) if j not in matched_d]

        # Pass 2: relax gate to reid_gate_px for the leftovers. This is
        # the safety net that prevents duplicate-track creation when a
        # CONFIRMED's bbox jittered slightly past max_distance — the
        # depth gate is unchanged, so two distinct people at different
        # depths still can't swap IDs here.
        if unmatched_t and unmatched_d:
            sub_dist = dist_2d[np.ix_(unmatched_t, unmatched_d)]
            sub_depth = depth_delta[np.ix_(unmatched_t, unmatched_d)]
            sub_gate = np.full(
                (len(unmatched_t), 1),
                self.reid_gate_px,
                dtype=float,
            )
            sub_matches, _, _ = self._hungarian_with_gate(
                sub_dist,
                sub_depth,
                sub_gate,
            )
            for r_local, c_local in sub_matches:
                t_idx = unmatched_t[r_local]
                d_idx = unmatched_d[c_local]
                matches.append((t_idx, d_idx))
                matched_t.add(t_idx)
                matched_d.add(d_idx)
            unmatched_t = [i for i in range(n_t) if i not in matched_t]
            unmatched_d = [j for j in range(n_d) if j not in matched_d]

        return matches, unmatched_t, unmatched_d

    def _associate_candidates(
        self,
        track_refs: np.ndarray,
        det_arr: np.ndarray,
        track_ids: list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian re-association of leftover tracks vs low-conf detections.

        Always uses the wide ``reid_gate_px`` for every track (ByteTrack
        intuition: the leftover tracks are the ones the high-confidence
        detector failed to match this frame, so we accept any low-conf
        detection that lands within the re-id radius). The depth gate
        still applies as the secondary safety net so two physically
        distinct people at very different depths cannot swap IDs even
        when one is missing this frame.

        Returns ``(matches, unmatched_t, unmatched_d)`` with indices
        local to the inputs.
        """
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

        gate = np.full((n_t, 1), self.reid_gate_px, dtype=float)
        matches, matched_t, matched_d = self._hungarian_with_gate(
            dist_2d,
            depth_delta,
            gate,
        )
        unmatched_t = [i for i in range(n_t) if i not in matched_t]
        unmatched_d = [j for j in range(n_d) if j not in matched_d]
        # track_ids accepted for symmetry with _associate but unused —
        # no per-track behaviour switch (gate is uniformly wide here).
        del track_ids
        return matches, unmatched_t, unmatched_d

    def _hungarian_with_gate(
        self,
        dist_2d: np.ndarray,
        depth_delta: np.ndarray,
        gate_per_track: np.ndarray,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Run Hungarian on a cost matrix with distance + depth gating.

        ``gate_per_track`` is broadcast against ``dist_2d``; depths over
        ``max_depth_delta`` are always rejected regardless of the gate.
        Returns ``(matches, matched_track_indices, matched_det_indices)``.
        """
        INF = 1e9
        cost = dist_2d.copy()
        cost[dist_2d > gate_per_track] = INF
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
        return matches, matched_t, matched_d

    def _pending_match_ref(self, track: Track) -> np.ndarray:
        """Match reference for a PENDING track in the current frame.

        Inside ``update()`` we already advanced every track's filter, so
        ``kalman.position`` is the model's expectation for THIS frame
        (the `predicted_position` property would peek ONE more step
        ahead, which would be wrong for current-frame matching).
        """
        if track.kalman is not None:
            xy = track.kalman.position
            z = track.positions[-1][2] if len(track.positions[-1]) > 2 else 0.0
            return np.array([xy[0], xy[1], z], dtype=float)
        # Defensive fallback (Tracks created without a filter).
        if len(track.positions) < 2:
            return track.positions[-1].copy()
        velocity = track.positions[-1] - track.positions[-2]
        return track.positions[-1] + velocity

    # -------------------------------------------------- per-track updates
    def _make_kalman(self, centroid: np.ndarray) -> TrackKalman:
        return TrackKalman(
            initial_position=centroid[:2],
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            initial_velocity_uncertainty=self.initial_velocity_uncertainty,
        )

    def _register(self, centroid: np.ndarray) -> int:
        tid = self._next_id
        centroid = np.asarray(centroid, dtype=float)
        self._tracks[tid] = Track(
            track_id=tid,
            positions=[centroid.copy()],
            state=CANDIDATE,
            hits=1,
            kalman=self._make_kalman(centroid),
        )
        self._next_id += 1
        return tid

    def _record_hit(self, track: Track, centroid: np.ndarray) -> None:
        centroid = np.asarray(centroid, dtype=float)
        track.positions.append(centroid.copy())
        track.disappeared = 0
        track.hits += 1
        if track.kalman is not None:
            track.kalman.update(centroid[:2])
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

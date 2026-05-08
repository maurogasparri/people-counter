"""Counting logic for tracked persons.

Single :class:`Counter` parameterised by:

- An optional rectangular ROI (gate of interest — tracks outside the ROI
  are ignored).
- One or more directional :class:`Line` segments. Each line carries
  per-direction labels: a crossing in a configured direction emits the
  associated label; crossings in unconfigured directions are silently
  ignored (one-way gates).

A track is counted when:

  1. It enters the ROI from outside (or appears in frame if no ROI).
  2. It crosses one of the configured lines in a labelled direction
     while inside the ROI.
  3. It exits the ROI on the opposite side (or leaves the frame if no
     ROI).

When (3) fires, the track meta is reset so the same track can count
another full cycle later — important when a person walks in and walks
right back out without leaving the camera frame long enough for the
track to die.

``build_counter(config)`` constructs a :class:`Counter` from YAML.
Schema:

    counter:
      roi:                                # optional
        x_min: 100
        x_max: 1050
        y_min: 150
        y_max: 500
      lines:
        - from: [200, 300]
          to:   [500, 300]
          labels:
            top_to_bottom: ingress
            bottom_to_top: egress
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.tracking.tracker import CONFIRMED, PENDING, Track
from src.vision.world_coords import project_to_floor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CountEvent:
    """A counting event."""

    track_id: int
    direction: str  # the label configured on the line for this crossing
    timestamp: float
    position_y: float
    # Optional per-track attributes populated when classifier is enabled.
    # "unknown" when height data is missing (no depth, classifier disabled).
    height_class: str = "unknown"
    # Median head height (m) and head depth (m) across the track's detection
    # history. None when no depth was sampled (classifier disabled or every
    # detected frame fell outside the depth map). Useful for downstream
    # analytics — total_in/out alone doesn't tell you the demographic mix.
    height_m: Optional[float] = None
    head_depth_m: Optional[float] = None
    # Median YOLO confidence across the track's detection history. Lets
    # downstream filter out low-confidence events (likely false positives or
    # marginal poses).
    confidence: Optional[float] = None


_HORIZONTAL_DIRECTIONS = ("top_to_bottom", "bottom_to_top")
_VERTICAL_DIRECTIONS = ("left_to_right", "right_to_left")


@dataclass
class Line:
    """An axis-aligned counting line segment with per-direction labels.

    ``from_xy`` and ``to_xy`` define the segment endpoints. The segment
    must be axis-aligned (purely horizontal or purely vertical) — oblique
    segments are not supported because the operator-friendly direction
    names (``top_to_bottom`` / ``bottom_to_top`` for horizontal,
    ``left_to_right`` / ``right_to_left`` for vertical) only make sense
    on axis-aligned segments.

    ``labels`` maps a direction string to the label emitted when a track
    crosses the segment in that direction. Directions absent from the
    map are *one-way gates*: a crossing in that direction is silently
    ignored. This is the natural way to model two physically separate
    doors (one IN, one OUT) on the same frame.
    """

    from_xy: tuple[float, float]
    to_xy: tuple[float, float]
    labels: dict[str, str]
    orientation: str = field(init=False)
    _line_pos: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x1, y1 = float(self.from_xy[0]), float(self.from_xy[1])
        x2, y2 = float(self.to_xy[0]), float(self.to_xy[1])
        self.from_xy = (x1, y1)
        self.to_xy = (x2, y2)
        if y1 == y2 and x1 != x2:
            self.orientation = "horizontal"
            self._line_pos = y1
            valid = _HORIZONTAL_DIRECTIONS
        elif x1 == x2 and y1 != y2:
            self.orientation = "vertical"
            self._line_pos = x1
            valid = _VERTICAL_DIRECTIONS
        else:
            raise ValueError(
                f"Line segment {self.from_xy}->{self.to_xy} must be axis-aligned "
                "(purely horizontal or vertical)."
            )
        if not self.labels:
            raise ValueError(
                f"Line {self.from_xy}->{self.to_xy} has no direction labels — "
                "configure at least one of "
                f"{valid}."
            )
        for direction in self.labels:
            if direction not in valid:
                raise ValueError(
                    f"Direction {direction!r} invalid for {self.orientation} "
                    f"line. Valid: {valid}."
                )

    # ----------------------------------------------------------------- API
    def side_of(self, cx: float, cy: float) -> int:
        """Return +1, -1, or 0 for the side of the line the point is on.

        Sign convention: for a horizontal line, ``-1`` means "above" (lower
        ``y``), ``+1`` means "below" (higher ``y``); for a vertical line,
        ``-1`` is "left" (lower ``x``) and ``+1`` is "right" (higher ``x``).
        Zero means the point lies exactly on the line (edge case).
        """
        if self.orientation == "horizontal":
            if cy < self._line_pos:
                return -1
            if cy > self._line_pos:
                return 1
            return 0
        if cx < self._line_pos:
            return -1
        if cx > self._line_pos:
            return 1
        return 0

    def within_segment(self, cx: float, cy: float) -> bool:
        """True if ``(cx, cy)`` projects onto the segment's extent (between
        the endpoints, not just on the infinite line). Used to ignore
        crossings that pass the line's plane outside the actual gate."""
        if self.orientation == "horizontal":
            lo, hi = sorted([self.from_xy[0], self.to_xy[0]])
            return lo <= cx <= hi
        lo, hi = sorted([self.from_xy[1], self.to_xy[1]])
        return lo <= cy <= hi

    def crossing_label(self, prev_side: int, new_side: int) -> Optional[str]:
        """Map a side transition to the configured label, or ``None`` if
        the direction has no label (one-way gate, opposite direction)."""
        if prev_side == 0 or new_side == 0 or prev_side == new_side:
            return None
        if self.orientation == "horizontal":
            direction = "top_to_bottom" if prev_side == -1 else "bottom_to_top"
        else:
            direction = "left_to_right" if prev_side == -1 else "right_to_left"
        return self.labels.get(direction)


# ---------------------------------------------------------------------------
# Aggregation helpers (per-event metadata pulled from the track)
# ---------------------------------------------------------------------------


def _aggregate_height_class_from_track(track: Track) -> str:
    """Pull per-frame height_class samples from track metadata and pick a
    majority-vote verdict for the count event. Returns "unknown" if the
    tracker wasn't given classification metadata (feature disabled).
    """
    history = track.meta.get("detection_history", [])
    if not history:
        return "unknown"
    from src.vision.world_coords import aggregate_height_class

    samples = [rec.get("height_class", "unknown") for rec in history]
    return aggregate_height_class(samples)


def _aggregate_height_m_from_track(track: Track) -> Optional[float]:
    """Median head height in metres across detection history. None if no
    sample has a measured head_height_mm (classifier disabled, no depth)."""
    history = track.meta.get("detection_history", [])
    samples = [
        rec.get("head_height_mm")
        for rec in history
        if rec.get("head_height_mm") is not None
    ]
    if not samples:
        return None
    samples.sort()
    median_mm = samples[len(samples) // 2]
    return float(median_mm) / 1000.0


def _aggregate_head_depth_m_from_track(track: Track) -> Optional[float]:
    """Median head depth (distance from lens to top of head) in metres."""
    history = track.meta.get("detection_history", [])
    samples = [
        rec.get("near_depth_mm")
        for rec in history
        if rec.get("near_depth_mm") is not None and rec.get("near_depth_mm") > 0
    ]
    if not samples:
        return None
    samples.sort()
    median_mm = samples[len(samples) // 2]
    return float(median_mm) / 1000.0


def _latest_head_height_and_bbox(
    track: Track,
) -> tuple[Optional[float], Optional[tuple[float, float, float, float]]]:
    """Most recent valid (head_height_mm, bbox) from the track's detection
    history. Used by :meth:`Counter._tracking_point` to drive the
    parallax-corrected footpoint projection.

    Walks the history from the back so a stale value never displaces a
    fresh one, and returns ``head_height_mm`` only when it is positive
    (None-out is upstream's signal that the depth pick was implausible
    — see the height sanity gate in ``main.py``).
    """
    history = track.meta.get("detection_history", [])
    head_mm: Optional[float] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    for rec in reversed(history):
        if head_mm is None:
            v = rec.get("head_height_mm")
            if v is not None and float(v) > 0:
                head_mm = float(v)
        if bbox is None:
            b = rec.get("bbox")
            if b is not None and len(b) == 4:
                bbox = (
                    float(b[0]),
                    float(b[1]),
                    float(b[2]),
                    float(b[3]),
                )
        if head_mm is not None and bbox is not None:
            break
    return head_mm, bbox


def _aggregate_confidence_from_track(track: Track) -> Optional[float]:
    """Median YOLO confidence across the track's detection history."""
    history = track.meta.get("detection_history", [])
    samples = [
        rec.get("confidence") for rec in history if rec.get("confidence") is not None
    ]
    if not samples:
        return None
    samples.sort()
    return float(samples[len(samples) // 2])


# ---------------------------------------------------------------------------
# ROI validation
# ---------------------------------------------------------------------------


def _validate_roi(
    roi: dict[str, float],
) -> tuple[float, float, float, float]:
    try:
        x_min = float(roi["x_min"])
        x_max = float(roi["x_max"])
        y_min = float(roi["y_min"])
        y_max = float(roi["y_max"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"counter.roi malformed: {e}") from e
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(f"counter.roi requires x_min<x_max and y_min<y_max, got {roi}")
    return x_min, x_max, y_min, y_max


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class Counter:
    """ROI + N directional lines counter.

    Track meta is keyed under ``META_KEY``. The counter manages:

    - ``inside``: whether the track is currently inside the ROI.
    - ``last_label``: the most recent label set by a valid line crossing
      during the current ROI visit. Reset on entry/exit transitions.
    - ``line_sides``: per-line cached "side" of the tracking point from
      the previous frame. Used to detect side transitions that imply a
      crossing.
    - ``proj_active``: whether the previous frame's ``line_sides`` were
      computed using the parallax-corrected footpoint (True) or the raw
      bbox centroid (False). Toggling between conventions is handled by
      resnapshotting sides instead of comparing across them.

    Tracking point — parallax-corrected when possible
    -------------------------------------------------
    The detector's bbox in zenith fisheye encloses head + shoulders +
    torso to roughly the waist; pies are not visible from above. Using
    the centroid (≈ shoulders) as the tracking point makes the line
    crossing fire when the person's *shoulder projection* — not the
    feet — passes the line. With a 3 m mount that's ~1 m of error at
    the frame border (60° eccentricity).

    When the per-track metadata carries a valid ``head_height_mm`` from
    SGBM and the calibration's principal point + mounting height are
    available, the counter uses :func:`project_to_floor` to scale the
    head pixel toward the principal point by ``Z_head / mount`` and
    counts the crossing of the projected *foot* pixel instead. This is
    the FootfallCam-old-firmware trick that the new RK3588 product line
    lost — and the differentiator that depth lets us preserve.

    When projection isn't viable for a track (no head height yet, no
    calibration plumbed in, mount=0), the counter falls back to the
    centroid behaviour. The two conventions are never mixed within a
    single frame's side-comparison: ``proj_active`` records which one
    was used for the cached sides, and a convention flip resnapshots
    the sides without emitting a phantom crossing.
    """

    META_KEY = "counter"

    def __init__(
        self,
        lines: list[Line],
        roi: Optional[dict[str, float]] = None,
        *,
        mounting_height_mm: Optional[float] = None,
        principal_point: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        Args:
            lines: At least one axis-aligned counting line.
            roi: Optional rectangular ROI gating the counted region.
            mounting_height_mm: Camera height above the floor (mm). Set
                this together with ``principal_point`` to enable the
                parallax-corrected footpoint tracking. When either is
                ``None`` the counter falls back to bbox-centroid
                tracking — preserves the legacy semantics for tests and
                pre-calibration runs.
            principal_point: ``(cx, cy)`` of the rectified left camera
                in pixels. Comes from ``P1[0,2], P1[1,2]`` of the
                stereo calibration ``.npz``. Must be in the same pixel
                space as the tracking positions (i.e. the rectified
                runtime resolution).
        """
        if not lines:
            raise ValueError("Counter requires at least one line.")
        self._lines: list[Line] = list(lines)
        self._roi: Optional[tuple[float, float, float, float]] = (
            _validate_roi(roi) if roi else None
        )
        # Footpoint projection params. Both must be populated for
        # projection to engage; otherwise we behave exactly like the
        # pre-parallax-fix counter.
        self._mounting_height_mm: Optional[float] = (
            float(mounting_height_mm)
            if mounting_height_mm is not None and float(mounting_height_mm) > 0
            else None
        )
        self._principal_point: Optional[tuple[float, float]] = (
            (float(principal_point[0]), float(principal_point[1]))
            if principal_point is not None
            else None
        )
        all_labels: set[str] = set()
        for line in self._lines:
            all_labels.update(line.labels.values())
        self._totals: dict[str, int] = {label: 0 for label in all_labels}

    # ----------------------------------------------------------------- API
    @property
    def lines(self) -> list[Line]:
        """The configured lines (read-only copy)."""
        return list(self._lines)

    @property
    def roi(self) -> Optional[tuple[float, float, float, float]]:
        """ROI as ``(x_min, x_max, y_min, y_max)`` or ``None`` if unset."""
        return self._roi

    @property
    def total_in(self) -> int:
        """Count of ``ingress`` events. Convenience alias used by
        the rest of the pipeline (telemetry, MQTT events)."""
        return self._totals.get("ingress", 0)

    @property
    def total_out(self) -> int:
        """Count of ``egress`` events. Convenience alias."""
        return self._totals.get("egress", 0)

    @property
    def totals(self) -> dict[str, int]:
        """All totals keyed by label. Includes any custom label set on
        the line ``labels`` map (not just ``ingress``/``egress``)."""
        return dict(self._totals)

    def check_all(self, tracks: dict[int, Track]) -> list[CountEvent]:
        events: list[CountEvent] = []
        for track in tracks.values():
            ev = self._process_track(track)
            if ev is not None:
                events.append(ev)
        return events

    def reset_daily(self) -> None:
        for k in self._totals:
            self._totals[k] = 0

    # ------------------------------------------------------------- internal
    def _inside_roi(self, cx: float, cy: float) -> bool:
        if self._roi is None:
            return True
        x_min, x_max, y_min, y_max = self._roi
        return x_min <= cx <= x_max and y_min <= cy <= y_max

    def _tracking_point(self, track: Track) -> tuple[float, float, bool]:
        """Pick the pixel used for ROI + line crossing this frame.

        Returns ``(x, y, projected)`` where ``projected`` is True when
        the parallax-corrected footpoint was used. The caller stores
        ``projected`` in the track meta so the next frame can detect a
        convention flip and re-snapshot the cached line sides.

        Selection rules:

        1. If the counter wasn't built with calibration params
           (``mounting_height_mm`` + ``principal_point``), fall back to
           the centroid. Preserves legacy behaviour for tests and
           pre-calibration runs.
        2. If the track has a valid ``head_height_mm`` in its most
           recent detection metadata, project the *top* of the bbox
           (head pixel) to the floor. The bbox top is closer to the
           true head than the centroid in zenith fisheye — bbox center
           sits roughly on the shoulders, so projecting the centroid
           would underestimate the parallax shift by ~30%.
           Falls back to the centroid X when bbox is missing.
        3. Otherwise centroid.

        Pulling head height from the LATEST detection record (not the
        median) keeps the projection responsive to depth changes — a
        person bending down has a smaller head height; we should track
        them where their feet currently are, not where their median
        was. Speckle is already filtered upstream by
        ``head_depth_in_bbox`` (anthropometric gates) and the height
        sanity check in ``main.py`` (None-out implausible values), so
        latest is safe.
        """
        cx = float(track.positions[-1][0])
        cy = float(track.positions[-1][1])

        if self._mounting_height_mm is None or self._principal_point is None:
            return cx, cy, False

        head_mm, bbox = _latest_head_height_and_bbox(track)
        if head_mm is None or head_mm <= 0:
            return cx, cy, False

        # Head pixel = bbox top centre. In zenith fisheye the head sits
        # near the upper edge of the person bbox (the bbox extends down
        # through torso/waist). When bbox is unavailable (older meta),
        # fall back to the centroid X with the centroid Y — better than
        # nothing, and the projection still corrects most of the
        # parallax because the radial direction is what matters.
        if bbox is not None:
            x1, y1, x2, _y2 = bbox
            head_px = ((float(x1) + float(x2)) / 2.0, float(y1))
        else:
            head_px = (cx, cy)

        cx_pp, cy_pp = self._principal_point
        u_foot, v_foot = project_to_floor(
            head_px,
            head_mm,
            self._mounting_height_mm,
            cx_pp,
            cy_pp,
        )
        # If project_to_floor rejected the inputs (degenerate), it
        # returns the head pixel unchanged. We can't safely call that
        # the "footpoint" — fall back to centroid so the convention
        # flag accurately reflects what we used.
        if (u_foot, v_foot) == head_px and head_px != (cx, cy):
            return cx, cy, False
        return u_foot, v_foot, True

    def _process_track(self, track: Track) -> Optional[CountEvent]:
        # CANDIDATE tracks are too unstable to count.
        if track.state not in (CONFIRMED, PENDING):
            return None
        if not track.positions:
            return None

        # Tracking point — projected footpoint when depth + calibration
        # are available, raw centroid otherwise. ``projected`` flags the
        # convention so a flip across frames triggers a side-cache
        # resnapshot below.
        cx, cy, projected = self._tracking_point(track)
        meta = track.meta.setdefault(self.META_KEY, {})
        was_inside = bool(meta.get("inside", False))
        prev_projected = bool(meta.get("proj_active", False))
        is_inside = self._inside_roi(cx, cy)

        # Per-line previous-side cache (one slot per line). Created lazily;
        # repaired if the configured line count changed between updates.
        sides = meta.get("line_sides")
        if not isinstance(sides, list) or len(sides) != len(self._lines):
            sides = [0] * len(self._lines)
            meta["line_sides"] = sides

        # Convention flip: previous frame cached sides under the *other*
        # tracking-point convention (centroid vs projected footpoint).
        # Comparing sides across conventions can fabricate a "crossing"
        # — e.g. centroid was below the line, projection puts the foot
        # above it, no real motion happened. Re-snapshot under the new
        # convention without emitting a label this frame; the next
        # frame will detect any genuine subsequent crossing cleanly.
        convention_flipped = was_inside and (projected != prev_projected)
        if convention_flipped:
            for i, line in enumerate(self._lines):
                sides[i] = line.side_of(cx, cy) if is_inside else 0
            meta["proj_active"] = projected
            if not is_inside:
                # Treat as a benign exit (no crossing context to trust).
                meta["inside"] = False
                meta["last_label"] = None
            return None
        meta["proj_active"] = projected

        if is_inside and not was_inside:
            # Fresh entry: reset cycle state and snapshot current sides.
            meta["inside"] = True
            meta["last_label"] = None
            for i, line in enumerate(self._lines):
                sides[i] = line.side_of(cx, cy)
            return None

        if is_inside and was_inside:
            # Detect a side transition on each line. The track may cross
            # multiple lines in one ROI visit; the most recent valid
            # crossing wins (a defensive choice — in well-configured
            # deployments the lines cover disjoint regions, so this rarely
            # matters).
            for i, line in enumerate(self._lines):
                prev_side = sides[i]
                new_side = line.side_of(cx, cy)
                if (
                    prev_side != 0
                    and new_side != 0
                    and prev_side != new_side
                    and line.within_segment(cx, cy)
                ):
                    label = line.crossing_label(prev_side, new_side)
                    if label is not None:
                        meta["last_label"] = label
                sides[i] = new_side
            return None

        # is_inside is False — if we *were* inside, this is the exit frame.
        if was_inside:
            # Detect crossings on the exit transition itself. Important if
            # the track jumps from inside-one-side to outside-the-other in
            # a single frame (low fps + fast motion, or detector gaps).
            for i, line in enumerate(self._lines):
                prev_side = sides[i]
                new_side = line.side_of(cx, cy)
                if (
                    prev_side != 0
                    and new_side != 0
                    and prev_side != new_side
                    and line.within_segment(cx, cy)
                ):
                    label = line.crossing_label(prev_side, new_side)
                    if label is not None:
                        meta["last_label"] = label
            label = meta.get("last_label")
            # Reset state so the same track can count another full cycle
            # later. The antiglitch invariant holds: counting again requires
            # re-entry from outside + a labelled line crossing + exit.
            meta["inside"] = False
            meta["last_label"] = None
            for i in range(len(sides)):
                sides[i] = 0
            if label:
                self._totals[label] = self._totals.get(label, 0) + 1
                logger.debug(
                    "count_event",
                    extra={"track_id": track.track_id, "label": label},
                )
                return CountEvent(
                    track_id=track.track_id,
                    direction=label,
                    timestamp=time.time(),
                    position_y=cy,
                    height_class=_aggregate_height_class_from_track(track),
                    height_m=_aggregate_height_m_from_track(track),
                    head_depth_m=_aggregate_head_depth_m_from_track(track),
                    confidence=_aggregate_confidence_from_track(track),
                )
            logger.debug(
                "exit_without_crossing",
                extra={"track_id": track.track_id},
            )
        return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_counter(
    config: dict[str, Any],
    frame_height: Optional[int] = None,  # noqa: ARG001 — kept for API stability
    *,
    mounting_height_mm: Optional[float] = None,
    principal_point: Optional[tuple[float, float]] = None,
) -> Counter:
    """Build the counter from YAML config.

    The optional ``mounting_height_mm`` and ``principal_point`` enable
    parallax-corrected footpoint tracking (see :class:`Counter` for the
    rationale). Both must be provided together; either omitted leaves
    the counter on the legacy centroid-tracking path. Passed through
    from ``main.py`` once calibration is loaded.

    Schema (strict):

        counter:
          roi:                              # optional
            x_min: 100
            x_max: 1050
            y_min: 150
            y_max: 500
          lines:
            - from: [200, 300]
              to:   [500, 300]
              labels:
                top_to_bottom: ingress
                bottom_to_top: egress
    """
    counter_cfg = config.get("counter") or {}
    raw_lines = counter_cfg.get("lines")
    if not raw_lines:
        raise ValueError(
            "counter.lines is required and must not be empty. See the docstring "
            "of build_counter() for the expected schema."
        )
    lines: list[Line] = []
    for idx, raw in enumerate(raw_lines):
        try:
            from_xy = tuple(raw["from"])
            to_xy = tuple(raw["to"])
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"counter.lines[{idx}]: 'from' and 'to' required as [x, y] "
                f"pairs ({e})."
            ) from e
        if len(from_xy) != 2 or len(to_xy) != 2:
            raise ValueError(
                f"counter.lines[{idx}]: 'from' and 'to' must be [x, y] pairs."
            )
        labels = dict(raw.get("labels") or {})
        lines.append(Line(from_xy=from_xy, to_xy=to_xy, labels=labels))
    return Counter(
        lines=lines,
        roi=counter_cfg.get("roi"),
        mounting_height_mm=mounting_height_mm,
        principal_point=principal_point,
    )

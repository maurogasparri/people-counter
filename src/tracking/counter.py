"""Counting logic for tracked persons.

Two implementations live here:

- :class:`LineCounter`: legacy virtual-line crossing counter (kept for
  backwards compatibility with the original pipeline wiring).
- :class:`ROICounter`: rectangular ROI + inner virtual line. A track is
  counted only when it *exits* the ROI on the opposite side of where it
  *entered*, having crossed the line in between. "Indeciso" tracks (same
  entry/exit side or no line crossing) do not count. This avoids edge cases
  at the threshold: somebody lingering on the line stays inside the ROI
  until they leave, and they leave on the side they came from.

``build_counter(config)`` returns whichever implementation the config
selects. New deployments should use the ``counter.roi`` + ``counter.line``
schema; legacy ``vision.counting_line_y`` still works (with a deprecation
warning) so the existing pipeline keeps running.
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from src.tracking.tracker import CONFIRMED, PENDING, Track

logger = logging.getLogger(__name__)


@dataclass
class CountEvent:
    """A counting event."""

    track_id: int
    direction: str  # "in" / "out" (legacy) or the configured label
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
        rec.get("head_height_mm") for rec in history
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
        rec.get("near_depth_mm") for rec in history
        if rec.get("near_depth_mm") is not None and rec.get("near_depth_mm") > 0
    ]
    if not samples:
        return None
    samples.sort()
    median_mm = samples[len(samples) // 2]
    return float(median_mm) / 1000.0


def _aggregate_confidence_from_track(track: Track) -> Optional[float]:
    """Median YOLO confidence across the track's detection history."""
    history = track.meta.get("detection_history", [])
    samples = [
        rec.get("confidence") for rec in history
        if rec.get("confidence") is not None
    ]
    if not samples:
        return None
    samples.sort()
    return float(samples[len(samples) // 2])


# ---------------------------------------------------------------------------
# Legacy line counter (preserved for backward compatibility)
# ---------------------------------------------------------------------------


class LineCounter:
    """Detects when tracks cross a virtual counting line (legacy).

    Kept intact so existing callers and tests continue to work. New
    installations should use :class:`ROICounter` instead.
    """

    def __init__(self, line_y: float) -> None:
        self.line_y = line_y
        self._counted_tracks: set[int] = set()
        self.total_in = 0
        self.total_out = 0

    def check(self, track: Track) -> Optional[CountEvent]:
        if track.track_id in self._counted_tracks:
            return None
        if len(track.positions) < 2:
            return None

        prev_y = track.positions[-2][1]
        curr_y = track.positions[-1][1]

        if prev_y < self.line_y <= curr_y:
            self._counted_tracks.add(track.track_id)
            self.total_in += 1
            logger.debug(
                "line_crossing",
                extra={"track_id": track.track_id, "direction": "in"},
            )
            return CountEvent(
                track.track_id, "in", time.time(), curr_y,
                height_class=_aggregate_height_class_from_track(track),
                height_m=_aggregate_height_m_from_track(track),
                head_depth_m=_aggregate_head_depth_m_from_track(track),
                confidence=_aggregate_confidence_from_track(track),
            )

        if prev_y > self.line_y >= curr_y:
            self._counted_tracks.add(track.track_id)
            self.total_out += 1
            logger.debug(
                "line_crossing",
                extra={"track_id": track.track_id, "direction": "out"},
            )
            return CountEvent(
                track.track_id, "out", time.time(), curr_y,
                height_class=_aggregate_height_class_from_track(track),
                height_m=_aggregate_height_m_from_track(track),
                head_depth_m=_aggregate_head_depth_m_from_track(track),
                confidence=_aggregate_confidence_from_track(track),
            )

        return None

    def check_all(self, tracks: dict[int, Track]) -> list[CountEvent]:
        events = []
        for track in tracks.values():
            ev = self.check(track)
            if ev is not None:
                events.append(ev)
        return events

    def reset_daily(self) -> None:
        self._counted_tracks.clear()
        self.total_in = 0
        self.total_out = 0


# ---------------------------------------------------------------------------
# ROI-based counter (new)
# ---------------------------------------------------------------------------


_SIDE_A = "a"
_SIDE_B = "b"


def _validate_roi(roi: dict[str, float]) -> tuple[float, float, float, float]:
    try:
        x_min = float(roi["x_min"])
        x_max = float(roi["x_max"])
        y_min = float(roi["y_min"])
        y_max = float(roi["y_max"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"counter.roi malformed: {e}") from e
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(
            f"counter.roi requires x_min<x_max and y_min<y_max, got {roi}"
        )
    return x_min, x_max, y_min, y_max


def _validate_line(
    line: dict[str, Any],
    roi: tuple[float, float, float, float],
) -> tuple[str, float]:
    orientation = line.get("orientation", "horizontal")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"counter.line.orientation must be 'horizontal' or 'vertical', got {orientation!r}"
        )
    try:
        position = float(line["position"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"counter.line.position required and numeric: {e}") from e

    x_min, x_max, y_min, y_max = roi
    if orientation == "horizontal":
        if not (y_min < position < y_max):
            raise ValueError(
                f"counter.line.position {position} must lie strictly inside ROI "
                f"y range ({y_min}, {y_max})"
            )
    else:
        if not (x_min < position < x_max):
            raise ValueError(
                f"counter.line.position {position} must lie strictly inside ROI "
                f"x range ({x_min}, {x_max})"
            )
    return orientation, position


class ROICounter:
    """Counts a track only when it exits the ROI on the opposite side of entry.

    Track state lives on ``track.meta`` under the ``roi_counter`` key so
    re-identification (PENDING -> CONFIRMED) preserves entry side and the
    "crossed the line" flag.
    """

    META_KEY = "roi_counter"

    def __init__(
        self,
        roi: dict[str, float],
        line: dict[str, Any],
        direction_labels: Optional[dict[str, str]] = None,
    ) -> None:
        self._roi = _validate_roi(roi)
        self._orientation, self._line_pos = _validate_line(line, self._roi)
        labels = direction_labels or {}
        self._label_a_to_b = labels.get("side_a_to_b", "ingress")
        self._label_b_to_a = labels.get("side_b_to_a", "egress")
        # Totals keyed by label, plus legacy aliases for back-compat.
        self._totals: dict[str, int] = {
            self._label_a_to_b: 0,
            self._label_b_to_a: 0,
        }
        self._counted: set[int] = set()

    # ------------------------------------------------------------------ API
    @property
    def total_in(self) -> int:
        """Count of 'a -> b' (ingress) events. Legacy alias."""
        return self._totals.get(self._label_a_to_b, 0)

    @property
    def total_out(self) -> int:
        """Count of 'b -> a' (egress) events. Legacy alias."""
        return self._totals.get(self._label_b_to_a, 0)

    @property
    def totals(self) -> dict[str, int]:
        return dict(self._totals)

    def check_all(self, tracks: dict[int, Track]) -> list[CountEvent]:
        # Tracks that silently disappear from the tracker (LOST) are simply
        # not re-seen here. If they were inside the ROI, no count is emitted
        # — the conservative default. Re-counts are blocked by self._counted.
        events: list[CountEvent] = []
        for track in tracks.values():
            ev = self._process_track(track)
            if ev is not None:
                events.append(ev)
        return events

    def reset_daily(self) -> None:
        self._counted.clear()
        for k in self._totals:
            self._totals[k] = 0

    # ----------------------------------------------------------- internals
    def _side(self, cx: float, cy: float) -> str:
        if self._orientation == "horizontal":
            return _SIDE_A if cy < self._line_pos else _SIDE_B
        return _SIDE_A if cx < self._line_pos else _SIDE_B

    def _inside_roi(self, cx: float, cy: float) -> bool:
        x_min, x_max, y_min, y_max = self._roi
        return x_min <= cx <= x_max and y_min <= cy <= y_max

    def _process_track(self, track: Track) -> Optional[CountEvent]:
        if track.track_id in self._counted:
            return None
        # Only act on tracks eligible for counting. CANDIDATE tracks are too
        # unstable. PENDING tracks use their last observed position and keep
        # their meta until they recover or go LOST.
        if track.state not in (CONFIRMED, PENDING):
            return None
        if not track.positions:
            return None

        cx, cy = float(track.positions[-1][0]), float(track.positions[-1][1])
        meta = track.meta.setdefault(self.META_KEY, {})
        was_inside = bool(meta.get("inside", False))
        is_inside = self._inside_roi(cx, cy)
        current_side = self._side(cx, cy)

        if is_inside:
            if not was_inside:
                meta["inside"] = True
                meta["entry_side"] = current_side
                meta["crossed"] = False
            else:
                entry_side = meta.get("entry_side")
                if entry_side is not None and current_side != entry_side:
                    meta["crossed"] = True
            return None

        # Not inside now. If we were inside, this frame is the exit frame.
        if was_inside:
            entry_side = meta.get("entry_side")
            crossed = bool(meta.get("crossed", False))
            # Determine exit side: for an exit frame we use the side the
            # centroid is now on.
            exit_side = current_side
            meta["inside"] = False
            self._counted.add(track.track_id)
            if crossed and entry_side is not None and exit_side != entry_side:
                if entry_side == _SIDE_A and exit_side == _SIDE_B:
                    direction = self._label_a_to_b
                else:
                    direction = self._label_b_to_a
                self._totals[direction] = self._totals.get(direction, 0) + 1
                logger.debug(
                    "roi_count",
                    extra={
                        "track_id": track.track_id,
                        "direction": direction,
                        "entry_side": entry_side,
                        "exit_side": exit_side,
                    },
                )
                return CountEvent(
                    track_id=track.track_id,
                    direction=direction,
                    timestamp=time.time(),
                    position_y=cy,
                    height_class=_aggregate_height_class_from_track(track),
                    height_m=_aggregate_height_m_from_track(track),
                    head_depth_m=_aggregate_head_depth_m_from_track(track),
                    confidence=_aggregate_confidence_from_track(track),
                )
            logger.debug(
                "roi_no_count_indeciso",
                extra={
                    "track_id": track.track_id,
                    "entry_side": entry_side,
                    "exit_side": exit_side,
                    "crossed": crossed,
                },
            )
        return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_counter(
    config: dict[str, Any],
    frame_height: Optional[int] = None,
):
    """Pick the appropriate counter implementation from config.

    Preference order:
      1. ``counter.roi`` + ``counter.line`` -> ROICounter
      2. legacy ``vision.counting_line_y`` -> LineCounter (warn)
    """
    counter_cfg = config.get("counter") or {}
    if "roi" in counter_cfg and "line" in counter_cfg:
        return ROICounter(
            roi=counter_cfg["roi"],
            line=counter_cfg["line"],
            direction_labels=counter_cfg.get("direction_labels"),
        )

    vision_cfg = config.get("vision", {})
    if "counting_line_y" in vision_cfg:
        logger.warning(
            "legacy_counting_line_config",
            extra={"hint": "migrate to counter.roi + counter.line"},
        )
        line_y = float(vision_cfg["counting_line_y"])
        if line_y <= 1.0 and frame_height is not None:
            line_y = line_y * frame_height
        return LineCounter(line_y=line_y)

    raise ValueError(
        "No counter configuration found. Set counter.roi + counter.line "
        "(preferred) or vision.counting_line_y (legacy)."
    )

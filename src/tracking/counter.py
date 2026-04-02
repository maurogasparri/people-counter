"""Virtual line crossing counter for ingress/egress detection."""
import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.tracking.tracker import Track

logger = logging.getLogger(__name__)


@dataclass
class CountEvent:
    """A counting event."""

    track_id: int
    direction: str  # "in" or "out"
    timestamp: float
    position_y: float


class LineCounter:
    """Detects when tracks cross a virtual counting line.

    The line is defined as a Y coordinate in image/depth space.
    Tracks moving from above to below the line = "in" (ingress).
    Tracks moving from below to above the line = "out" (egress).
    """

    def __init__(self, line_y: float) -> None:
        self.line_y = line_y
        self._counted_tracks: set[int] = set()
        self.total_in = 0
        self.total_out = 0

    def check(self, track: Track) -> Optional[CountEvent]:
        """Check if a track has crossed the counting line.

        Requires at least 2 positions. Only counts each track once.
        """
        if track.track_id in self._counted_tracks:
            return None
        if len(track.positions) < 2:
            return None

        prev_y = track.positions[-2][1]
        curr_y = track.positions[-1][1]

        if prev_y < self.line_y <= curr_y:
            self._counted_tracks.add(track.track_id)
            self.total_in += 1
            event = CountEvent(
                track_id=track.track_id,
                direction="in",
                timestamp=time.time(),
                position_y=curr_y,
            )
            logger.debug("Ingress: track %d", track.track_id)
            return event

        if prev_y > self.line_y >= curr_y:
            self._counted_tracks.add(track.track_id)
            self.total_out += 1
            event = CountEvent(
                track_id=track.track_id,
                direction="out",
                timestamp=time.time(),
                position_y=curr_y,
            )
            logger.debug("Egress: track %d", track.track_id)
            return event

        return None

    def check_all(self, tracks: dict[int, Track]) -> list[CountEvent]:
        """Check all tracks for line crossings."""
        events = []
        for track in tracks.values():
            event = self.check(track)
            if event is not None:
                events.append(event)
        return events

    def reset_daily(self) -> None:
        """Reset counters for new business day."""
        self._counted_tracks.clear()
        self.total_in = 0
        self.total_out = 0

"""World-space helpers: mount geometry → physical measurements.

The stereo calibration is installation-independent — we measure camera-to-scene
depth. To translate those depths into physically meaningful quantities (head
height, floor-plane coordinates, etc.) we combine depth with the installation's
mounting height above the floor.

Currently used to classify detections as adult vs child based on head height.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def head_height_above_floor(
    near_depth_mm: float,
    mounting_height_mm: float,
) -> Optional[float]:
    """Compute the height of the top of a tracked object above the floor.

    For a zenith-mounted stereo pair, the smallest valid depth inside a
    detection bbox corresponds to the nearest point — typically the top of a
    person's head. Height-above-floor is the complement of that depth:

        height = mounting_height - near_depth

    Args:
        near_depth_mm: Depth at the closest point in the detection (e.g.
            min_depth_at_bbox). Must be > 0 to be valid.
        mounting_height_mm: Camera-to-floor distance at install time.

    Returns:
        Head height above floor in millimetres, or None if inputs are invalid
        or the computed height is negative (would indicate the object is
        below the floor, which means bad depth).
    """
    if near_depth_mm <= 0 or mounting_height_mm <= 0:
        return None
    height_mm = mounting_height_mm - near_depth_mm
    if height_mm < 0:
        return None
    return float(height_mm)


def classify_height(
    head_height_mm: Optional[float],
    adult_min_mm: float,
) -> str:
    """Classify a person by head height: adult / child / unknown.

    Args:
        head_height_mm: Output of head_height_above_floor(). None → unknown.
        adult_min_mm: Threshold in millimetres. height >= threshold → adult.

    Returns:
        "adult" | "child" | "unknown".
    """
    if head_height_mm is None:
        return "unknown"
    return "adult" if head_height_mm >= adult_min_mm else "child"


def aggregate_height_class(samples: list[str]) -> str:
    """Stabilise per-frame classifications into a single per-track verdict.

    Uses majority vote across the track's sampled classifications, ignoring
    "unknown" samples. Ties (equal adult/child counts) resolve to the last
    non-unknown observation — biased toward most recent depth which is
    usually the cleanest (track is established, bbox is stable).

    Args:
        samples: List of per-frame classifications (values from classify_height).

    Returns:
        Final classification for the track: "adult", "child", or "unknown"
        when no non-unknown samples exist.
    """
    valid = [s for s in samples if s != "unknown"]
    if not valid:
        return "unknown"
    adult = sum(1 for s in valid if s == "adult")
    child = sum(1 for s in valid if s == "child")
    if adult > child:
        return "adult"
    if child > adult:
        return "child"
    return valid[-1]

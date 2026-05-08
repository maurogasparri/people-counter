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


def project_to_floor(
    head_pixel: tuple[float, float],
    head_height_mm: float,
    mounting_height_mm: float,
    cx_px: float,
    cy_px: float,
) -> tuple[float, float]:
    """Project a head pixel to where the feet would land in image space.

    Background — parallax in zenith fisheye
    ---------------------------------------
    With the bracket pointing straight down, the line from the principal
    point through the lens centre is the only ray with zero parallax: a
    person standing exactly under nadir projects with head and feet on
    the same pixel. Everywhere else, head and feet land on *different*
    pixels because they sit at different distances from the camera —
    head is closer, so it projects further from the principal point than
    the feet.

    The detector's bbox does NOT contain feet (they sit hidden under the
    torso in top-down view). So if we use ``bbox.center`` as the tracking
    point and chase a virtual line drawn at floor level, we trip the line
    when the *shoulder/torso projection* crosses it — which can happen
    tens of centimetres before the feet actually do, depending on FOV
    eccentricity and mounting height. At 60° eccentricity (frame border)
    that error is ~110 cm for a 1.7 m person under a 3 m mount.

    Math — pinhole projection of a vertical line
    -------------------------------------------
    Camera at height ``H`` above floor, principal point ``(cx, cy)``.
    A point at height ``h`` above the floor sits at ``Z = H - h`` from
    the camera's optical centre. Pinhole projection of a 3D point
    ``(X, Y, Z)`` is ``(cx + f*X/Z, cy + f*Y/Z)``. The same physical
    ``(X, Y)`` projected at the head's depth ``Z_head = H - h_head``
    versus the floor's depth ``Z_foot = H`` differs only by the
    multiplier ``Z_head / H`` — so the foot pixel is the head pixel
    scaled toward the principal point by that factor.

    Verification:
    - Nadir (``head_pixel == (cx, cy)``) → ``foot_pixel == (cx, cy)``.
    - Periphery → shift toward principal point, magnitude = parallax.
    - As ``h_head → 0``, ``Z_head → H``, scale → 1, no shift (the head is
      already on the floor).

    The function operates on the rectified image plane, so ``(cx_px,
    cy_px)`` are the principal-point coordinates of the rectified left
    camera (i.e. ``P1[0,2]``, ``P1[1,2]`` from the calibration ``.npz``).
    A pinhole approximation is fine here because the rectification map
    has already removed the lens distortion — the rectified image IS a
    pinhole projection of the world.

    Args:
        head_pixel: ``(u, v)`` of the head in rectified-image pixels.
        head_height_mm: Person's head height above the floor (mm).
            Coming from :func:`head_height_above_floor`.
        mounting_height_mm: Camera height above the floor (mm).
        cx_px, cy_px: Principal point of the rectified left camera (px).

    Returns:
        ``(u_foot, v_foot)`` — the pixel where the person's feet would
        project if the camera could see through the torso. On degenerate
        inputs (non-positive mount, non-positive head height, head at or
        above the camera) returns the input unchanged so the caller can
        fall back to centroid behaviour.
    """
    if mounting_height_mm <= 0 or head_height_mm <= 0:
        return (float(head_pixel[0]), float(head_pixel[1]))
    z_head = mounting_height_mm - head_height_mm
    if z_head <= 0:
        # Head above the camera plane (taller than mount, or sensor
        # noise). Projection toward principal point would invert sign;
        # the safe behaviour is to fall back to the input pixel.
        return (float(head_pixel[0]), float(head_pixel[1]))
    scale = z_head / mounting_height_mm
    u_h, v_h = float(head_pixel[0]), float(head_pixel[1])
    u_foot = cx_px + (u_h - cx_px) * scale
    v_foot = cy_px + (v_h - cy_px) * scale
    return (u_foot, v_foot)


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

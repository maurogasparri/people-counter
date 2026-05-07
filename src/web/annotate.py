"""Annotation helpers for the runtime web viewer.

Pure functions that draw onto BGR frames. No state, no I/O. Kept out of
``viewer.py`` so the streaming module is small and easy to test in
isolation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colour palette
_COLOR_CONFIRMED = (0, 255, 0)
_COLOR_PENDING = (0, 165, 255)
_COLOR_CANDIDATE = (180, 180, 180)
_COLOR_DET = (90, 90, 90)
_COLOR_ROI = (0, 0, 255)         # red
_COLOR_TEXT = (255, 255, 255)
_COLOR_OVERLAY_BG = (0, 0, 0)

# Counting-line palette by direction label. Anything else falls back to
# white so an exotic label still renders visibly. Greens for IN-side
# events, blues for OUT-side — matches the operator's mental model.
_LINE_COLOR_BY_LABEL = {
    "ingress": (0, 255, 0),       # green: IN
    "egress": (255, 0, 0),        # blue: OUT
    "in": (0, 255, 0),
    "out": (255, 0, 0),
    "enter": (0, 255, 0),
    "leave": (255, 0, 0),
}
_LINE_COLOR_FALLBACK = (255, 255, 255)


def annotate_left(
    frame: np.ndarray,
    detections: list,
    tracks: dict,
    counter: Optional[Any],
    fps: float = 0.0,
) -> np.ndarray:
    """Draw ROI, line, raw detections and tracks onto a copy of ``frame``.

    Args:
        frame: BGR left-camera frame (rectified).
        detections: Detection objects (have ``.bbox`` and ``.centroid``).
        tracks: dict[int, Track] from EuclideanTracker.
        counter: LineCounter or ROICounter instance (read for geometry +
            totals overlay). May be None.
        fps: pipeline FPS estimate for the bottom overlay.
    """
    out = frame.copy()

    # Geometry first so detections sit on top.
    _draw_counter_geometry(out, counter)

    # Raw detections in subtle gray. Tracked positions get a coloured
    # marker on top so the operator can see which detections produced
    # tracks and which didn't.
    for det in detections:
        try:
            x1, y1, x2, y2 = det.bbox
        except Exception:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      _COLOR_DET, 1)

    # Defer the import so a circular import on partial init doesn't blow
    # up the viewer module.
    from src.tracking.tracker import CONFIRMED, PENDING, CANDIDATE
    state_colour = {
        CONFIRMED: _COLOR_CONFIRMED,
        PENDING: _COLOR_PENDING,
        CANDIDATE: _COLOR_CANDIDATE,
    }
    # Larger fonts so the operator can read the overlay while walking
    # under the cameras during a piloto check (the live viewer is meant
    # for on-site debug, not for archival).
    for tid, track in tracks.items():
        colour = state_colour.get(getattr(track, "state", None))
        if colour is None:
            continue
        positions = getattr(track, "positions", None)
        if not positions:
            continue
        cx, cy = int(positions[-1][0]), int(positions[-1][1])
        cv2.circle(out, (cx, cy), 10, colour, -1)
        cv2.putText(out, f"#{tid}", (cx + 14, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2,
                    cv2.LINE_AA)
        # Height label from the most recent detection meta on the track.
        meta = getattr(track, "meta", None)
        history = meta.get("detection_history") if isinstance(meta, dict) else None
        if history:
            last = history[-1]
            head_mm = last.get("head_height_mm")
            cls = last.get("height_class") or "unknown"
            if isinstance(head_mm, (int, float)) and head_mm > 0:
                label = f"{head_mm/1000:.2f}m {cls}"
            else:
                label = cls
            cv2.putText(out, label, (cx + 14, cy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2,
                        cv2.LINE_AA)

    _draw_counter_overlay(out, counter, fps)
    return out


def depth_to_colormap(
    depth_map: Optional[np.ndarray],
    vmin_mm: float = 500.0,
    vmax_mm: float = 5000.0,
) -> np.ndarray:
    """Render a depth-in-mm map as a BGR JET colormap.

    Zero (= invalid disparity) renders black. None / empty input returns
    a small dark gray frame so the panel is visually distinct from a
    real depth map and the caller doesn't need a None check.
    """
    if depth_map is None or depth_map.size == 0:
        return np.full((100, 100, 3), 30, dtype=np.uint8)
    d = depth_map.astype(np.float32)
    invalid = d <= 0
    d = np.clip(d, vmin_mm, vmax_mm)
    norm = ((d - vmin_mm) / (vmax_mm - vmin_mm) * 255.0).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    if np.any(invalid):
        coloured[invalid] = (0, 0, 0)
    return coloured


def compose_3panel(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    target_height: int = 480,
) -> np.ndarray:
    """Two-row composite: top row is L | R side-by-side, bottom row is the
    depth panel spanning the same width.

    The L/R top row gives the operator the same view as the cameras see;
    the depth row underneath stays roughly square so the colormap is
    legible. Missing / empty panels are filled with a dark gray
    placeholder so the composite never crashes on partial input.
    """
    def _to_bgr_height(img: Optional[np.ndarray], h_target: int,
                       w_fallback: int) -> np.ndarray:
        if img is None or img.size == 0:
            return np.full((h_target, w_fallback, 3), 30, dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        if h != h_target:
            scale = h_target / h
            img = cv2.resize(
                img, (max(1, int(w * scale)), h_target),
                interpolation=cv2.INTER_AREA,
            )
        return img

    # Top row: L and R at target_height each.
    l_img = _to_bgr_height(left, target_height, target_height)
    r_img = _to_bgr_height(right, target_height, target_height)
    top = cv2.hconcat([l_img, r_img])
    top_w = top.shape[1]

    # Bottom row: depth resized to span exactly the top width. Keep it
    # roughly half the height so the layout looks balanced.
    depth_h = max(1, target_height // 2)
    if depth is None or depth.size == 0:
        bottom = np.full((depth_h, top_w, 3), 30, dtype=np.uint8)
    else:
        d = depth
        if d.ndim == 2:
            d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        bottom = cv2.resize(
            d, (top_w, depth_h), interpolation=cv2.INTER_AREA,
        )
    return cv2.vconcat([top, bottom])


# ----------------------------------------------------------------- internals
def _draw_counter_geometry(frame: np.ndarray, counter: Optional[Any]) -> None:
    """Overlay ROI rectangle (if any) and each line segment with a
    perpendicular arrow showing the counted direction.

    Each line is drawn in the colour matching its dominant label
    (``ingress`` -> green, ``egress`` -> blue, anything else white). One
    arrow per labelled direction, perpendicular to the segment, pointing
    towards the side a crossing must end on for that label to fire.
    """
    if counter is None:
        return
    roi = getattr(counter, "roi", None)
    if roi is not None:
        try:
            x_min, x_max, y_min, y_max = roi
            cv2.rectangle(frame, (int(x_min), int(y_min)),
                          (int(x_max), int(y_max)), _COLOR_ROI, 4)
        except Exception:
            logger.debug("ROI overlay failed", exc_info=True)

    lines = getattr(counter, "lines", None) or []
    for line in lines:
        try:
            _draw_line_with_arrows(frame, line)
        except Exception:
            logger.debug("Line overlay failed", exc_info=True)


def _draw_line_with_arrows(frame: np.ndarray, line: Any) -> None:
    """Draw one counting line + per-direction arrows.

    The arrow length scales with segment length so it stays visible on
    short ROIs without overflowing on big ones. If both directions are
    labelled the segment renders white (neutral) and the per-arrow colour
    encodes which side of the line is which event.
    """
    x1, y1 = int(line.from_xy[0]), int(line.from_xy[1])
    x2, y2 = int(line.to_xy[0]), int(line.to_xy[1])
    orientation = line.orientation
    labels: dict[str, str] = line.labels

    seg_len = max(1, abs(x2 - x1) + abs(y2 - y1))
    arrow_len = max(20, min(60, seg_len // 6))
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2

    if len(labels) >= 2:
        seg_color = _LINE_COLOR_FALLBACK
    else:
        only_label = next(iter(labels.values()), None)
        seg_color = _LINE_COLOR_BY_LABEL.get(
            only_label or "", _LINE_COLOR_FALLBACK,
        )
    cv2.line(frame, (x1, y1), (x2, y2), seg_color, 4)

    # The arrow's tail anchors on the line and the tip extends one
    # ``arrow_len`` away on the side the crossing must end on. This keeps
    # the arrow strictly on one side of the segment instead of straddling
    # it, which makes the "go this way" reading immediate.
    if orientation == "horizontal":
        for direction, label in labels.items():
            color = _LINE_COLOR_BY_LABEL.get(label, _LINE_COLOR_FALLBACK)
            if direction == "top_to_bottom":
                tail = (mx, my)
                tip = (mx, my + arrow_len)
            else:  # bottom_to_top
                tail = (mx, my)
                tip = (mx, my - arrow_len)
            cv2.arrowedLine(frame, tail, tip, color, 4, tipLength=0.35)
    else:
        for direction, label in labels.items():
            color = _LINE_COLOR_BY_LABEL.get(label, _LINE_COLOR_FALLBACK)
            if direction == "left_to_right":
                tail = (mx, my)
                tip = (mx + arrow_len, my)
            else:  # right_to_left
                tail = (mx, my)
                tip = (mx - arrow_len, my)
            cv2.arrowedLine(frame, tail, tip, color, 4, tipLength=0.35)


def _draw_counter_overlay(
    frame: np.ndarray, counter: Optional[Any], fps: float,
) -> None:
    h, w = frame.shape[:2]
    in_n = getattr(counter, "total_in", 0) if counter else 0
    out_n = getattr(counter, "total_out", 0) if counter else 0
    text = f"IN: {in_n}  OUT: {out_n}  FPS: {fps:.1f}"
    bar_h = 56
    cv2.rectangle(frame, (0, h - bar_h), (w, h), _COLOR_OVERLAY_BG, -1)
    cv2.putText(frame, text, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, _COLOR_TEXT, 3,
                cv2.LINE_AA)
